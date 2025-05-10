import collections
import cv2
import math
import numpy as np
import optparse
import string

def get_contour_params(contour):
    (x_pos, y_pos), radius = cv2.minEnclosingCircle(contour)
    return {
        "center": (int(x_pos), int(y_pos)),
        "radius": radius,
        "area": cv2.contourArea(contour)
    }

def binarize_image(img, threshold_value):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, img_thresh = cv2.threshold(img_blur, threshold_value, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    img_thresh = cv2.dilate(img_thresh, kernel, iterations=3)
    return img_thresh

def check_counter(counter):
    return len(counter.most_common()) != 0 and counter.most_common(1)[0][1] == 15  # Augmenté à 15

def compute_distances(center, letters):
    return [l2_norm(center, letter["position"]) for letter in letters]

def l2_norm(point_1, point_2):
    return math.sqrt((point_2[1] - point_1[1])**2 + (point_2[0] - point_1[0])**2)

def draw_letters(frame):
    letters_position = []
    rows = [(50, 70), (50, 150), (50, 230)]  # Déplacement des lettres
    letter_list = list(string.ascii_lowercase)
    row_counts = [10, 10, 6]
    
    idx = 0
    for row, count in zip(rows, row_counts):
        x_pos, y_pos = row
        for i in range(count):
            letter = letter_list[idx]
            x_pos += 60
            cv2.rectangle(frame, (x_pos - 25, y_pos - 30), (x_pos + 25, y_pos + 30), (0, 0, 0), 2)  # Noir
            cv2.putText(frame, letter, (x_pos - 10, y_pos + 10), font_params["type"], font_params["size"], (0, 0, 0), font_params["width"])  # Noir
            letters_position.append({"letter": letter, "position": (x_pos, y_pos)})
            idx += 1

    # Placement des boutons à droite
    right_x = 720  # Position X tout à droite
    space_y, delete_y = 100, 180

    # Bouton "Espace"
    cv2.rectangle(frame, (right_x - 60, space_y - 30), (right_x + 60, space_y + 30), (255, 0, 0), 2)
    cv2.putText(frame, "Espace", (right_x - 40, space_y + 10), font_params["type"], font_params["size"], (255, 0, 0), font_params["width"])
    letters_position.append({"letter": " ", "position": (right_x, space_y)})

    # Bouton "Effacer"
    cv2.rectangle(frame, (right_x - 50, delete_y - 30), (right_x + 50, delete_y + 30), (0, 0, 255), 2)
    cv2.putText(frame, "Effacer", (right_x - 30, delete_y + 10), font_params["type"], font_params["size"], (0, 0, 255), font_params["width"])
    letters_position.append({"letter": "delete", "position": (right_x, delete_y)})

    return letters_position

def draw_crosshair(img, params):
    color = (0, 0, 0)
    cv2.circle(img, params["center"], int(params["radius"]), color, thickness=5)
    cv2.line(img, (params["center"][0] - 50, params["center"][1]), (params["center"][0] + 50, params["center"][1]), color, 3)
    cv2.line(img, (params["center"][0], params["center"][1] - 50), (params["center"][0], params["center"][1] + 50), color, 3)

def create_trackbar(named_window):
    def nothing(x): pass
    cv2.namedWindow(named_window)
    cv2.createTrackbar("threshold", named_window, 0, 255, nothing)
    cv2.setTrackbarPos("threshold", named_window, options.threshold)

def get_image(cap):
    flag, img = cap.read()
    img = cv2.flip(img, 1)
    return cv2.resize(img.copy(), (800, 600))

def main():
    named_window = "Image"
    cap = cv2.VideoCapture(0)
    cnt_th_low, cnt_th_high = 700, 2000

    if options.calibrate:
        create_trackbar(named_window)
    
    if not options.calibrate:
        counter = collections.Counter()
        selected_letters = ""

    threshold_value = options.threshold

    while cap.isOpened():
        composite = get_image(cap)
        keyboard_section = composite.copy()[20:350, :composite.shape[1]]
        letters = draw_letters(composite)

        if options.calibrate:
            threshold_value = cv2.getTrackbarPos("threshold", "Image")
            cv2.putText(composite, f"Threshold value: {threshold_value}", (20, composite.shape[0] - 20), font_params["type"], 1, (255, 255, 255), 2)

        img_thresh = binarize_image(keyboard_section, threshold_value)
        contours, hierarchy = cv2.findContours(img_thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            contour_params = get_contour_params(contour)
            if options.calibrate:
                cv2.putText(composite, f"Contour Area: {contour_params['area']}", (20, composite.shape[0] - 50), font_params["type"], 1, (255, 255, 255), 2)
                continue

            if cnt_th_low < contour_params["area"] < cnt_th_high:
                draw_crosshair(composite, contour_params)
                distances = compute_distances(contour_params["center"], letters)
                selected_letter = letters[distances.index(min(distances))]["letter"]
                counter[selected_letter] += 1

        if not options.calibrate:
            if check_counter(counter):
                selected_letter = counter.most_common(1)[0][0]
                if selected_letter == "delete":
                    selected_letters = selected_letters[:-1]
                elif selected_letter == " ":
                    selected_letters += " "
                else:
                    selected_letters += selected_letter
                counter.clear()

            cv2.putText(composite, selected_letters, (20, 400), font_params["type"], 1, (0, 0, 0), 2)  # Texte en noir

        out_img = composite if not options.calibrate else np.vstack([img_thresh, cv2.cvtColor(composite, cv2.COLOR_BGR2GRAY)])
        cv2.imshow(named_window, out_img)
        
        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            selected_letters = selected_letters[:-1]
        elif key == ord('c'):
            selected_letters = ""

    print("Done")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('-c', '--calibrate', dest='calibrate', action='store_true', default=False)
    parser.add_option('-t', '--threshold', dest='threshold', type='int', default=70)
    options, remainder = parser.parse_args()

    font_params = {"type": cv2.FONT_HERSHEY_SIMPLEX, "color": (0, 0, 0), "size": 1, "width": 2}  # Noir
    main()
