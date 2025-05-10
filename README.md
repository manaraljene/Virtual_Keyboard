# Virtual Keyboard Using Contour Detection

This project is a **virtual keyboard** implemented in Python using **OpenCV**. It allows users to "type" by placing an object (like a finger or pointer) over a live webcam feed. The application detects the object's position and maps it to virtual keys drawn on the screen.

## 🛠 Features

- Webcam-based virtual keyboard interaction
- Letter grid + space and delete buttons
- Real-time contour detection using thresholding
- Calibrate mode for threshold adjustment
- Tracks selection based on object stability

## 🎥 How It Works

- The camera feed is displayed with a drawn keyboard.
- You move a visible object (e.g., finger) into the keyboard area.
- When the object stays over a letter for a certain time, that letter is typed.
- "Espace" adds a space, and "Effacer" deletes the last character.

## 🧰 Requirements

- Python 3.x
- OpenCV (`cv2`)
- NumPy

Install the required dependencies using:

pip install opencv-python numpy
