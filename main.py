import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import threading
import queue
import speech_recognition as sr
from textblob import TextBlob
import sounddevice as sd
import wave
import numpy as np
import time


class UdayAI:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion & Sentiment Analyser")
        self.root.geometry("850x820")
        self.root.configure(bg="#1e1e1e")

        self.queue = queue.Queue()
        self.running = True

        
        self.face = "Detecting..."
        self.audio = "Not Recorded"
        self.conf = 0
        self.history = []

        
        self.last_update_time = 0
        self.freeze_duration = 4
        self.manual_freeze = False

        
        self.prev_positions = []
        self.prev_x = None
        self.hold_start = None
        self.last_gesture_time = 0
        self.cooldown = 4

        
        self.video_label = tk.Label(root, bg="black")
        self.video_label.pack(pady=10)

        self.result_label = tk.Label(
            root,
            text="System Ready",
            fg="#00ff00",
            bg="#1e1e1e",
            font=("Arial", 12),
            justify="left"
        )
        self.result_label.pack(pady=10)

        tk.Button(root, text="🎤 Record Audio",
                  command=self.record_audio).pack(pady=5)

        tk.Button(root, text="📸 Upload Image",
                  command=self.upload_image).pack(pady=5)

        tk.Button(root, text="🧊 Freeze / Unfreeze",
                  command=self.toggle_freeze).pack(pady=5)

        # CAMERA
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.result_label.config(text="❌ Camera not detected")
            self.running = False
            return

        # MODELS
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_smile.xml')

        self.update_video()
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    #  FREEZE 
    def toggle_freeze(self):
        self.manual_freeze = not self.manual_freeze

    # SENTIMENT 
    def sentiment(self, text):
        score = TextBlob(text).sentiment.polarity
        if score > 0.2:
            return "Positive 😊"
        elif score < -0.2:
            return "Negative 😠"
        return "Neutral 😐"

    #  FACE ANALYSIS 
    def analyze_face(self, face):
        face = cv2.equalizeHist(face)
        brightness = np.mean(face)
        contrast = np.std(face)

        smiles = self.smile_cascade.detectMultiScale(
            face, scaleFactor=1.5, minNeighbors=20)

        if len(smiles) > 0:
            return "Happy 😊", 90
        elif brightness < 50:
            return "Sad 😢", 75
        elif contrast < 25:
            return "Neutral 😐", 70
        else:
            return "Neutral 🙂", 65

    #  FINAL DECISION 
    def final_decision(self):
        if "Happy" in self.face and "Positive" in self.audio:
            return "Excited 😄"
        elif "Sad" in self.face and "Negative" in self.audio:
            return "Depressed 😞"
        elif "Negative" in self.audio:
            return "Stressed 😓"
        return "Balanced 🙂"

    #  AUDIO 
    def record_audio(self):
        def process():
            try:
                fs = 44100
                seconds = 4
                self.queue.put(("audio", "🎙 Recording..."))

                rec = sd.rec(int(seconds * fs),
                             samplerate=fs,
                             channels=1,
                             dtype='int16')
                sd.wait()

                with wave.open("temp.wav", 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(fs)
                    wf.writeframes(rec.tobytes())

                r = sr.Recognizer()
                with sr.AudioFile("temp.wav") as source:
                    audio = r.record(source)

                text = r.recognize_google(audio)
                result = self.sentiment(text)

            except:
                result = "Audio Error"

            self.queue.put(("audio", result))

        threading.Thread(target=process, daemon=True).start()

    #  IMAGE PROCESS 
    def process_image(self, file):
        img = cv2.imread(file)
        if img is None:
            return

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces):
            (x, y, w, h) = faces[0]
            face = gray[y:y+h, x:x+w]
            result, conf = self.analyze_face(face)
        else:
            result, conf = "No face", 0

        self.queue.put(("face", (result, conf)))

    #  UPLOAD 
    def upload_image(self):
        file = filedialog.askopenfilename()
        if file:
            self.process_image(file)

    #  GESTURE 
    def detect_gesture(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower = np.array([0, 20, 70])
        upper = np.array([20, 255, 255])

        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.GaussianBlur(mask, (7, 7), 0)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)

        if area < 8000:
            return None

        x, y, w, h = cv2.boundingRect(c)
        cx = x + w // 2

        
        self.prev_positions.append(cx)
        if len(self.prev_positions) > 5:
            self.prev_positions.pop(0)

        cx = int(sum(self.prev_positions) / len(self.prev_positions))

        gesture = None

        if self.prev_x is not None:
            dx = cx - self.prev_x
            if dx > 120:
                gesture = "RIGHT"
            elif dx < -120:
                gesture = "LEFT"

        if area > 30000:
            gesture = "PALM"
        elif area < 12000:
            gesture = "FIST"

        # HOLD
        if self.prev_x is not None and abs(cx - self.prev_x) < 5:
            if self.hold_start is None:
                self.hold_start = time.time()
            elif time.time() - self.hold_start > 2:
                gesture = "HOLD"
        else:
            self.hold_start = None

        self.prev_x = cx
        return gesture

    
    def update_video(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)

            gesture = self.detect_gesture(frame)

            if gesture and time.time() - self.last_gesture_time > self.cooldown:
                self.last_gesture_time = time.time()

                if gesture == "PALM":
                    self.record_audio()

                elif gesture == "RIGHT":
                    cv2.imwrite("gesture.jpg", frame)
                    self.process_image("gesture.jpg")

                elif gesture == "HOLD":
                    cv2.imwrite("capture.jpg", frame)
                    self.process_image("capture.jpg")

            # FACE DETECTION
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                emotion, conf = self.analyze_face(face)

                cv2.rectangle(frame, (x, y),
                              (x+w, y+h), (0, 255, 0), 2)

                cv2.putText(frame,
                            f"{emotion} ({conf}%)",
                            (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2)

                self.queue.put(("face", (emotion, conf)))
                break

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(img)

            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        while not self.queue.empty():
            t, data = self.queue.get()

            if t == "audio":
                self.audio = data

            elif t == "face":
                self.face, self.conf = data
                self.history.append(self.face)

                if len(self.history) > 5:
                    self.history.pop(0)

        
        current_time = time.time()

        if (not self.manual_freeze and
                current_time - self.last_update_time > self.freeze_duration):

            self.last_update_time = current_time

            final = self.final_decision()
            history_text = " → ".join(self.history)

            self.result_label.config(
                text=f"Face: {self.face} ({self.conf}%)\n"
                     f"Voice: {self.audio}\n"
                     f"Final: {final}\n"
                     f"History: {history_text}\n\n⏳ Stable Mode"
            )

        self.root.after(20, self.update_video)

    # CLOSE 
    def close(self):
        self.running = False
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = UdayAI(root)
    root.mainloop()