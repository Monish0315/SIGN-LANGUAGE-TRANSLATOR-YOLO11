import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import os
import string
import numpy as np
import requests
from PIL import Image as PILImage
import pyttsx3  
import speech_recognition as sr
import os


class SignLanguageApp(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("Sign Language Translator")
        self.geometry("1200x800")

        # Initialize the main container frame
        container = tk.Frame(self)
        container.pack(fill="both", expand=True)

        # Store frames in a dictionary
        self.frames = {}

        for F in (StartPage, SignToVoice, TextToSign):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        # Show the start page
        self.show_frame("StartPage")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()

        # Start camera only when SignToVoice page is shown
        if page_name == "SignToVoice":
            frame.start_camera()

class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.config(bg='#f0f0f0')

        # Title Label
        label = tk.Label(self, text="Sign Language Translator", font=("Bahnscript", 24, "bold"), bg='#f0f0f0')
        label.pack(pady=20)

        # Button frame for options
        button_frame = tk.Frame(self, bg='#f0f0f0')
        button_frame.pack(pady=10)

        button = tk.Button(button_frame, text="Sign to Voice", 
                           command=lambda: controller.show_frame("SignToVoice"),
                           bg='#4CAF50', fg='white', font=("Arial", 12), borderwidth=0, padx=20)
        button.grid(row=0, column=0, padx=10)

        button2 = tk.Button(button_frame, text="Text to Sign", 
                            command=lambda: controller.show_frame("TextToSign"),
                            bg='#2196F3', fg='white', font=("Arial", 12), borderwidth=0, padx=20)
        button2.grid(row=0, column=1, padx=10)

        # Load and display image
        load = Image.open(r"C:\Users\DELL\Downloads\1 Desktop\sdp\SDP-Sign-Language-Translator\homepage image.png")  # Replace with the actual image path
        load = load.resize((520, 350), Image.LANCZOS)
        render = ImageTk.PhotoImage(load)
        img = Label(self, image=render, bg='#f0f0f0')
        img.image = render
        img.place(x=100, y=200)


class SignToVoice(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.cap = None  # Placeholder for the camera
        self.model = None  # Placeholder for the YOLO model
        self.detected_text = ""
        self.sign_start_time = None  # Time when the sign was first detected
        self.tts_engine = pyttsx3.init()  # Initialize pyttsx3 TTS engine
        # Split the frame into left and right sections
        self.left_frame = tk.Frame(self, width=600, height=800, bg='black')
        self.left_frame.grid(row=0, column=0, sticky="nsew")
        self.left_frame.grid_propagate(False)

        self.right_frame = tk.Frame(self, width=600, height=800, bg='#f0f0f0')
        self.right_frame.grid(row=0, column=1, sticky="nsew")
        self.right_frame.grid_propagate(False)

        # Webcam feed on the left side
        self.label = tk.Label(self.left_frame, bg="black")
        self.label.pack(fill="both", expand=True)

        # Buttons and text box on the right side
        self.text_box = tk.Text(self.right_frame, height=15, width=40, font=("Arial", 12))
        self.text_box.pack(pady=20)

        self.sentence_button = tk.Button(self.right_frame, text="Generate Sentence",
                                         command=self.generate_sentence,
                                         bg='#2196F3', fg='white', font=("Arial", 12))
        self.sentence_button.pack(pady=10)

        self.speech_button = tk.Button(self.right_frame, text="Read Detected Text Aloud",
                                       command=self.speak_detected_text,
                                       bg='#4CAF50', fg='white', font=("Arial", 12))
        self.speech_button.pack(pady=10)

        back_button = tk.Button(self.right_frame, text="Back",
                                command=lambda: self.stop_camera(controller),
                                bg='#FF5733', fg='white', font=("Arial", 12))
        back_button.pack(pady=10)

        # Adjust the grid weights to maintain proportions
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

    def start_camera(self):
        if not self.cap:
            self.cap = cv2.VideoCapture(0)  # Open the camera
        if not self.model:
            self.model = YOLO(r'C:\Users\DELL\Downloads\1 Desktop\sdp\SDP-Sign-Language-Translator\best1.pt')  # Replace with your YOLO model path
        
        # Start the video capturing process
        self.update_frame()

    def update_frame(self):
        success, frame = self.cap.read()
        if success:
            results = self.model.track(frame, persist=True)
            annotated_frame = results[0].plot()

            current_detected_text = ""
            # Loop through each result to get detected labels
            for result in results:
                if result.boxes:
                    for box in result.boxes:
                        class_id = int(box.cls[0])  # Assuming single detection
                        label = result.names[class_id]
                        current_detected_text += label + " "

            current_detected_text = current_detected_text.strip()

            if current_detected_text:
                if self.sign_start_time is None:
                    self.sign_start_time = cv2.getTickCount()
                else:
                    elapsed_time = (cv2.getTickCount() - self.sign_start_time) / cv2.getTickFrequency()
                    if elapsed_time > 1:
                        if current_detected_text not in self.detected_text:
                            self.detected_text += " " + current_detected_text
                            self.text_box.delete("1.0", tk.END)
                            self.text_box.insert(tk.END, self.detected_text)
                            self.sign_start_time = None
            else:
                self.sign_start_time = None

            cv_img = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(cv_img)
            imgtk = ImageTk.PhotoImage(image=img_pil)

            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)

        self.after(10, self.update_frame)

    def stop_camera(self, controller):
        if self.cap:
            self.cap.release()
        controller.show_frame("StartPage")

    def generate_sentence(self):
        # Open a new window for correcting detected text
        correction_window = tk.Toplevel(self)
        correction_window.title("Correct Detected Text")
        correction_window.geometry("400x300")

        # Text box for editing detected text
        edit_text = tk.Text(correction_window, height=10, width=40, font=("Arial", 12))
        edit_text.insert(tk.END, self.text_box.get("1.0", tk.END).strip())  # Load detected text
        edit_text.pack(pady=10)

        # Function to submit corrected text
        def submit_corrected_text():
             # Hugging Face API key
            input_text = edit_text.get("1.0", tk.END).strip()  # Get the input text

            if input_text:
                headers = {
                    "Authorization": f"Bearer HUGGING_API_KEY"  # Replace with your Hugging Face API key
                }
                payload = {
                    "inputs": f"grammar: {input_text}",  # Prefix for grammar correction
                    "parameters": {"max_length": 100},  # Adjust max_length as needed
                    "options": {"wait_for_model": True},
                }
                response = requests.post(
                    "https://api-inference.huggingface.co/models/vennify/t5-base-grammar-correction",  # Grammar correction model
                    headers=headers,
                    json=payload
                )
                if response.status_code == 200:
                    corrected_sentence = response.json()[0]["generated_text"]
                    self.text_box.delete("1.0", tk.END)  # Clear the text box
                    self.text_box.insert(tk.END, corrected_sentence)  # Insert corrected sentence
                else:
                    self.text_box.insert(tk.END, "\nError correcting grammar.")
                correction_window.destroy()  # Close the correction window after submission

        # Button to submit corrected text
        submit_button = tk.Button(correction_window, text="Correct Grammar", command=submit_corrected_text, bg='#2196F3', fg='white', font=("Arial", 12))
        submit_button.pack(pady=10)

    def speak_detected_text(self):
        # Get detected text from the text box
        detected_text = self.text_box.get("1.0", tk.END).strip()
        if detected_text:
            # Use the TTS engine to speak the detected text
            self.tts_engine.say(detected_text)
            self.tts_engine.runAndWait()



class TextToSign(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg='#f0f0f0')
        self.controller = controller
        label = tk.Label(self, text="Text to Sign", font=("Verdana", 16, "bold"), bg='#f0f0f0')
        label.pack(pady=10)

        self.text_input = tk.Entry(self, width=50, font=("Arial", 14))
        self.text_input.pack(pady=10)

        # Button to translate text to sign
        translate_button = tk.Button(self, text="Translate to Sign", command=self.convert_text_to_sign, 
                                     bg='#4CAF50', fg='white', font=("Arial", 12), padx=20)
        translate_button.pack(pady=10)

        # Button to record voice
        record_button = tk.Button(self, text="Record Voice", command=self.record_voice_to_text, 
                                  bg='#FF9800', fg='white', font=("Arial", 12), padx=20)
        record_button.pack(pady=10)

        self.gif_box = tk.Label(self, bg='#f0f0f0')
        self.gif_box.pack(pady=10)

        # Button frame for navigation
        button_frame = tk.Frame(self, bg='#f0f0f0')
        button_frame.pack(pady=10)

        button1 = tk.Button(button_frame, text="Back to Home", command=lambda: controller.show_frame("StartPage"), 
                            bg='#f44336', fg='white', font=("Arial", 12), borderwidth=0, padx=20)
        button1.grid(row=0, column=0, padx=10)

        button2 = tk.Button(button_frame, text="Sign to Voice", command=lambda: controller.show_frame("SignToVoice"), 
                            bg='#2196F3', fg='white', font=("Arial", 12), borderwidth=0, padx=20)
        button2.grid(row=0, column=1, padx=10)

    def convert_text_to_sign(self):
        input_text = self.text_input.get().lower()
        if not input_text:
            print("No text entered!")
            return

        alpha_dest = r"C:\Users\DELL\Downloads\1 Desktop\sdp\SDP-Sign-Language-Translator\alphabet"  # Update to your GIF folder path
        all_frames = []

        for letter in input_text:
            if letter in string.ascii_lowercase:  # Convert only lowercase letters
                gif_path = rf"{alpha_dest}\{letter}_small.gif"  # Update path for small GIFs
                if os.path.exists(gif_path):
                    try:
                        gif = PILImage.open(gif_path)
                        frameCnt = gif.n_frames
                        for frame_num in range(frameCnt):
                            gif.seek(frame_num)
                            img = PILImage.fromarray(np.array(gif.convert('RGBA')))
                            img = img.resize((189, 189), PILImage.LANCZOS)
                            all_frames.append(img)
                    except Exception as e:
                        print(f"Error loading GIF for '{letter}': {e}")
                else:
                    print(f"No GIF found for '{letter}'")

        self.display_gif_frames(all_frames)

    def display_gif_frames(self, all_frames):
        if not all_frames:
            print("No frames to display.")
            return

        def update_frame(idx):
            frame = all_frames[idx]
            img_tk = ImageTk.PhotoImage(frame)
            self.gif_box.configure(image=img_tk)
            self.gif_box.image = img_tk
            idx += 1
            if idx < len(all_frames):
                self.after(200, update_frame, idx)

        update_frame(0)

    def record_voice_to_text(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Recording... Please speak.")
            audio = recognizer.listen(source)

        try:
            # Recognize the speech using Google Web Speech API
            text = recognizer.recognize_google(audio)
            print("You said:", text)
            self.text_input.delete(0, tk.END)  # Clear existing text
            self.text_input.insert(tk.END, text)  # Insert recognized text
        except sr.UnknownValueError:
            print("Could not understand the audio.")
        except sr.RequestError:
            print("Could not request results from the speech recognition service.")

if __name__ == "__main__":
    app = SignLanguageApp()
    app.mainloop()
