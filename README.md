😴 Drowsiness and Driver State Monitoring System
This is a real-time driver monitoring system that uses a webcam to detect signs of drowsiness and inattentiveness. 
It monitors for multiple indicators to ensure the driver is alert and focused on the road.

📸 Features
Drowsiness Detection: Detects when a driver's eyes are closed for an extended period, triggering an audible alarm and an email alert.

Yawn Detection: Counts yawns as an early indicator of fatigue, sending an email if the frequency is too high.

Eye Hypnotism / Fixed Gaze Alert: Warns the driver if their gaze is fixed for too long with a low blink rate, a condition known as "highway hypnosis."

Tiredness Detection: Uses subtle facial cues like a slightly open mouth and drooping eyebrows to detect general fatigue.

Camera Blockage Detection: Identifies if the camera is intentionally or accidentally covered.

Email Notifications: Sends automated email alerts to a designated recipient.

User-Friendly Interface: Displays real-time status messages on the video feed and includes an on-screen "EXIT" button.

🛠️ Prerequisites
Before running the program, you need to install the necessary libraries. You can do this by running the following command in your terminal or command prompt:

Bash

pip install opencv-python dlib scipy numpy
Note: winsound is a built-in Python module for Windows and does not require a separate installation.

⚙️ Setup and Configuration
Download the dlib Model: You must download the shape_predictor_68_face_landmarks.
dat file from the dlib website and place it in the same directory as your project's main script.

Email Configuration: To enable email notifications, you must set up an App Password for your Gmail account.

Go to your Google Account settings.

Navigate to Security.

Under "Signing in to Google," select App passwords.

Select Mail for the app and Other (Custom name) for the device.

Name it something like "Drowsiness Detector" and click Generate.

Copy the 16-character password.

Update the Code: Open the main script and replace the placeholder values in the send_email function with your actual email address and the generated app password:

Python

def send_email(subject, message):
    from_email = "your_email@gmail.com" # Replace with your email
    from_password = "your_app_password" # Replace with your generated App Password
    to_email = "recipient_email@example.com" # Replace with the recipient's email
    # ... rest of the code
▶️ How to Run
Place the shape_predictor_68_face_landmarks.dat file in the same folder as your Python script.

Open a terminal or command prompt.

Navigate to the project directory.

Run the script using the following command:

Bash

python drowsiness_detector.py

The webcam will activate and start monitoring for signs of drowsiness. 
You can press the q key on your keyboard or click the "EXIT" button on the screen to quit the application.
