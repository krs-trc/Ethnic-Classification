import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tkinter as tk
from PIL import Image, ImageTk
import threading

# Load the trained model
model_path = 'model_finetuned2.h5'
model = load_model(model_path)

# Define class labels
class_labels = {1: 'Indian', 2: 'Malay', 3: 'Chinese'}

# Load OpenCV's pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to preprocess image for model input
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

class EthnicityClassificationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ethnicity Classification Using Deep Learning")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")

        # Header
        self.header = tk.Label(
            root,
            text="Ethnicity Classification Using Deep Learning",
            font=("Helvetica", 20, "bold"),
            bg="#2c3e50",
            fg="white",
            pady=10
        )
        self.header.pack(fill=tk.X)

        # Video feed canvas
        self.canvas = tk.Canvas(root, width=640, height=480, bg="black", highlightthickness=2, highlightbackground="#34495e")
        self.canvas.pack(pady=20)
        self.canvas.create_text(
            320, 240,
            text="Press Start to begin",
            font=("Helvetica", 16),
            fill="white"
        )

        # Button frame
        self.button_frame = tk.Frame(root, bg="#f0f0f0")
        self.button_frame.pack(pady=10)

        # Start button
        self.start_button = tk.Button(
            self.button_frame,
            text="Start",
            command=self.start_feed,
            font=("Helvetica", 12),
            bg="#27ae60",
            fg="white",
            activebackground="#2ecc71",
            padx=10,
            pady=5
        )
        self.start_button.pack(side=tk.LEFT, padx=5)

        # Stop button
        self.stop_button = tk.Button(
            self.button_frame,
            text="Stop",
            command=self.stop_feed,
            font=("Helvetica", 12),
            bg="#c0392b",
            fg="white",
            activebackground="#e74c3c",
            padx=10,
            pady=5
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # Initialize webcam (not opened until Start is clicked)
        self.cap = None
        self.running = False
        self.thread = None

        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def start_feed(self):
        if not self.running:
            self.cap = cv2.VideoCapture(2)
            if not self.cap.isOpened():
                tk.Label(
                    self.button_frame,
                    text="Error: Could not open laptop webcam.",
                    font=("Helvetica", 12),
                    bg="#f0f0f0",
                    fg="red"
                ).pack()
                return
            self.running = True
            self.thread = threading.Thread(target=self.update_frame)
            self.thread.daemon = True
            self.thread.start()

    def stop_feed(self):
        if self.running:
            self.running = False
            if self.cap:
                self.cap.release()
                self.cap = None
            self.canvas.delete("all")
            self.canvas.create_text(
                320, 240,
                text="Webcam Stopped",
                font=("Helvetica", 16),
                fill="white"
            )

    def update_frame(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                self.canvas.delete("all")
                self.canvas.create_text(
                    320, 240,
                    text="Error: Failed to capture frame.",
                    font=("Helvetica", 16),
                    fill="white"
                )
                break

            # Mirror the frame horizontally
            frame = cv2.flip(frame, 1)

            # Convert frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Process each detected face
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                # Validate face region before processing
                if face.size > 0 and w >= 30 and h >= 30:
                    face_processed = preprocess_image(face)
                    predictions = model.predict(face_processed, verbose=0)
                    predicted_class = np.argmax(predictions, axis=1)[0] + 1
                    predicted_label = class_labels.get(predicted_class, 'Unknown')
                    confidence = np.max(predictions) * 100

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    label = f'{predicted_label} ({confidence:.1f}%)'
                    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Convert frame to RGB and display on canvas
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
            self.canvas.image = img

    def on_closing(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = EthnicityClassificationApp(root)
    root.mainloop()