import cv2
import dlib
import time
from scipy.spatial import distance
import winsound
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from collections import deque

# ------------- EMAIL FUNCTION -------------
def send_email(subject, message):
    from_email = "YouremailID@gmail.com"
    from_password = "xxxxxxxxxxxxx" # This is an App Password, not your regular Gmail password
    to_email = "SendersEmailID"

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(message, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, from_password)
        server.send_message(msg)
        server.quit()
        print(f"Email sent: {subject}")
    except Exception as e:
        print("Email failed:", e)

# -------- EAR CALCULATION FUNCTION --------
def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# -------- MAR CALCULATION FUNCTION (for Yawn Detection) --------
def calculate_MAR(mouth):
    # 68-point facial landmark indices for mouth:
    # Outer mouth: 48-60
    # Inner mouth: 60-68
    A = distance.euclidean(mouth[3], mouth[9]) # 52, 58
    B = distance.euclidean(mouth[2], mouth[10]) # 51, 59
    C = distance.euclidean(mouth[4], mouth[8])  # 53, 57
    L = distance.euclidean(mouth[0], mouth[6]) # 48, 54 (corners of the mouth)
    if L == 0: # Avoid division by zero if mouth is perfectly closed horizontally
        return 0.001
    return (A + B + C) / (3.0 * L)

# -------- Tiredness Calculation (Eyebrow-to-Eye Distance) --------
def calculate_eyebrow_eye_dist(landmarks):
    # Eyebrow landmarks: 17-21 (left eyebrow), 22-26 (right eyebrow)
    # Eye landmarks: 36-41 (left eye), 42-47 (right eye)

    # Left eyebrow points
    left_eyebrow_inner = (landmarks.part(19).x, landmarks.part(19).y) # Point 19 is roughly center-left of eyebrow
    left_eyebrow_outer = (landmarks.part(17).x, landmarks.part(17).y) # Point 17 is leftmost of eyebrow

    # Right eyebrow points
    right_eyebrow_inner = (landmarks.part(24).x, landmarks.part(24).y) # Point 24 is roughly center-right of eyebrow
    right_eyebrow_outer = (landmarks.part(26).x, landmarks.part(26).y) # Point 26 is rightmost of eyebrow

    # Corresponding top eyelid points
    left_eye_top_inner = (landmarks.part(37).x, landmarks.part(37).y)
    left_eye_top_outer = (landmarks.part(38).x, landmarks.part(38).y) # Use 38, or average of 37,38,40,41 for upper eyelid

    right_eye_top_inner = (landmarks.part(43).x, landmarks.part(43).y)
    right_eye_top_outer = (landmarks.part(44).x, landmarks.part(44).y) # Use 44, or average of 43,44,46,47 for upper eyelid


    # Calculate vertical distances
    dist1 = distance.euclidean(left_eyebrow_inner, left_eye_top_inner)
    dist2 = distance.euclidean(left_eyebrow_outer, left_eye_top_outer)
    dist3 = distance.euclidean(right_eyebrow_inner, right_eye_top_inner)
    dist4 = distance.euclidean(right_eyebrow_outer, right_eye_top_outer)

    return (dist1 + dist2 + dist3 + dist4) / 4.0


# ----------- SETUP -----------
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # Make sure this file is in the same directory

# Indices for specific facial landmarks (from dlib's 68-point model)
left_eye_idx = list(range(36, 42))
right_eye_idx = list(range(42, 48))
mouth_idx = list(range(48, 68)) # All mouth landmarks

cap = cv2.VideoCapture(0)

# Drowsiness/Eye Cover Thresholds
EYE_AR_THRESH = 0.25
DROWSY_DURATION = 4
COVER_DURATION = 4
eye_closed_start = None
cover_start = None
drowsy_email_sent = False
eye_cover_email_sent = False
camera_block_email_sent = False

# Yawn Detection Thresholds
YAWN_THRESH = 0.7  # This value might need tuning
YAWN_CONSEC_FRAMES = 5 # Number of consecutive frames MAR must be above threshold
yawn_counter = 0
yawn_start_time = time.time()
yawn_email_sent_this_hour = False
yawn_consec_frames = 0 # Initialize here

# Eye Hypnotism (Fixed Gaze / Blink Rate)
BLINK_THRESH = 0.20 # A slightly lower EAR for definite blink
MIN_BLINKS_PER_PERIOD = 8 # Minimum blinks expected in 60 seconds (approx 10-20 blinks/min is normal)
BLINK_MONITOR_PERIOD = 60 # seconds
is_blinking_this_frame = False # To prevent counting a single blink multiple times across frames
blinks_in_period = deque() # Stores timestamps of blinks
HYPNOTISM_DURATION = 10 # Seconds of fixed gaze/low blinks before alert
road_hypnotism_email_sent = False

# Variables to track eye movement for gaze stability
prev_left_eye_center = None
prev_right_eye_center = None
FIXED_GAZE_DISTANCE_THRESH = 5 # Max pixel movement for "fixed" gaze
fixed_gaze_start_time = None

# Tiredness Detection Variables
TIREDNESS_EYEBROW_THRESH = 25 # Average eyebrow-to-eye distance (tune this)
TIREDNESS_MAR_THRESH = 0.2 # A slightly open mouth (not a yawn, tune this)
TIREDNESS_DURATION = 5 # Seconds of consistent tiredness signs
tiredness_start_time = None
tiredness_email_sent = False
# We'll use a short deque to smooth tiredness detection over frames
tiredness_metrics_history = deque(maxlen=30) # Store last 30 frames of tiredness metrics

# -------- Exit Button Function --------
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if the click is within the "EXIT" button rectangle
        if 450 <= x <= 640 and 10 <= y <= 50:
            cap.release()
            cv2.destroyAllWindows()
            exit()

# ----------- MAIN LOOP -----------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    contrast = gray.std()
    faces = face_detector(gray)
    current_time = time.time() # Get current time once per loop

    # Reset detection states if a face is detected (to avoid stale alerts for camera block)
    if len(faces) > 0:
        camera_block_email_sent = False # Camera is not blocked if face is found

    # ------ CAMERA BLOCK DETECTION ------
    # If no faces are detected AND the image is very dark/low contrast
    if len(faces) == 0 and (brightness < 40 or contrast < 15):
        if cover_start is None:
            cover_start = current_time
        elif current_time - cover_start >= COVER_DURATION:
            cv2.putText(frame, "CAMERA BLOCKED!", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
            winsound.Beep(1200, 1000)
            if not camera_block_email_sent:
                send_email("Camera Blocked", "The camera appears to be blocked or covered completely.")
                camera_block_email_sent = True
    else:
        cover_start = None
        # camera_block_email_sent is handled above when faces > 0

    # Initialize current frame blink status
    is_blinking_this_frame = False
    face_detected_this_frame = False # Flag to know if we processed a face

    # ------ EYE STATUS, YAWN, ROAD HYPNOTISM, AND TIREDNESS DETECTION ------
    if len(faces) > 0: # Only proceed if a face is detected
        face_detected_this_frame = True
        for face in faces:
            landmarks = landmark_predictor(gray, face)
            left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in left_eye_idx]
            right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in right_eye_idx]
            mouth = [(landmarks.part(i).x, landmarks.part(i).y) for i in mouth_idx]

            left_EAR = calculate_EAR(left_eye)
            right_EAR = calculate_EAR(right_eye)
            avg_EAR = (left_EAR + right_EAR) / 2.0

            mar = calculate_MAR(mouth) # Calculate Mouth Aspect Ratio
            eyebrow_eye_dist = calculate_eyebrow_eye_dist(landmarks)


            # --- Drowsiness / Eyes Covered ---
            if avg_EAR < EYE_AR_THRESH:
                if eye_closed_start is None:
                    eye_closed_start = current_time
                elif current_time - eye_closed_start >= DROWSY_DURATION:
                    winsound.Beep(1000, 1000)
                    if avg_EAR < 0.1:  # extremely low EAR means covered, not just closed
                        cv2.putText(frame, "EYES COVERED!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                        if not eye_cover_email_sent:
                            send_email("Eyes Covered", "Driver's eyes appear to be covered with an object.")
                            eye_cover_email_sent = True
                    else:
                        cv2.putText(frame, "DROWSY (Eyes Closed)!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                        if not drowsy_email_sent:
                            send_email("Drowsiness Detected", "Driver appears to be drowsy or eyes are closed.")
                            drowsy_email_sent = True
            else:
                eye_closed_start = None
                drowsy_email_sent = False
                eye_cover_email_sent = False
                cv2.putText(frame, "Eyes Open", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)


            # --- Yawn Detection ---
            if mar > YAWN_THRESH:
                cv2.putText(frame, "YAWNING!", (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                yawn_consec_frames += 1
            else:
                if yawn_consec_frames >= YAWN_CONSEC_FRAMES:
                    yawn_counter += 1 # A full yawn was detected
                    winsound.Beep(800, 300) # Short beep for a detected yawn
                    print(f"Yawn detected! Total yawns: {yawn_counter}")
                yawn_consec_frames = 0 # Reset for next yawn

            # Check yawn count for email (every frame)
            if current_time - yawn_start_time >= 3600: # 1 hour
                yawn_counter = 0 # Reset yawn counter after an hour
                yawn_start_time = current_time # Reset the start time for the next hour
                yawn_email_sent_this_hour = False # Allow sending email again for the new hour

            if yawn_counter >= 3 and not yawn_email_sent_this_hour:
                send_email("Excessive Yawning Detected", f"Driver has yawned {yawn_counter} times in the last hour. This indicates severe drowsiness.")
                yawn_email_sent_this_hour = True # Prevent sending multiple emails for the same period

            cv2.putText(frame, f"Yawns: {yawn_counter}", (400, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)


            # --- Eye Hypnotism (Gaze Stability & Blink Rate) ---
            # 1. Blink Rate
            if avg_EAR < BLINK_THRESH and not is_blinking_this_frame:
                # This is a new blink
                blinks_in_period.append(current_time)
                is_blinking_this_frame = True # Mark that a blink was registered for this frame

            # Remove old blinks from the deque
            while blinks_in_period and blinks_in_period[0] < current_time - BLINK_MONITOR_PERIOD:
                blinks_in_period.popleft()

            current_blink_count = len(blinks_in_period)
            cv2.putText(frame, f"Blinks/Min: {current_blink_count}", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)


            # 2. Gaze Stability
            left_eye_center = np.mean(left_eye, axis=0).astype("int")
            right_eye_center = np.mean(right_eye, axis=0).astype("int")

            if prev_left_eye_center is not None and prev_right_eye_center is not None:
                dist_left = distance.euclidean(left_eye_center, prev_left_eye_center)
                dist_right = distance.euclidean(right_eye_center, prev_right_eye_center)

                # If both eyes are relatively stable
                if dist_left < FIXED_GAZE_DISTANCE_THRESH and dist_right < FIXED_GAZE_DISTANCE_THRESH:
                    if fixed_gaze_start_time is None:
                        fixed_gaze_start_time = current_time
                    elif current_time - fixed_gaze_start_time >= HYPNOTISM_DURATION:
                        # Check if blink rate is also low
                        if current_blink_count < MIN_BLINKS_PER_PERIOD:
                            cv2.putText(frame, "EYE HYPNOTISM!", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                            winsound.Beep(1500, 1000)
                            if not road_hypnotism_email_sent:
                                send_email("Eye Hypnotism Detected", "Driver appears to be experiencing eye hypnotism (fixed gaze, low blink rate).")
                                road_hypnotism_email_sent = True
                else:
                    fixed_gaze_start_time = None
                    road_hypnotism_email_sent = False # Reset if gaze moves

            prev_left_eye_center = left_eye_center
            prev_right_eye_center = right_eye_center

            # --- Tiredness by Facial Expressions ---
            # Evaluate current frame for tiredness signs
            is_tired_this_frame = False
            if eyebrow_eye_dist < TIREDNESS_EYEBROW_THRESH and mar < TIREDNESS_MAR_THRESH and mar > 0.05: # MAR not too low (closed) but slightly open/relaxed
                 is_tired_this_frame = True

            tiredness_metrics_history.append(is_tired_this_frame)

            # Check for consistent tiredness over TIREDNESS_DURATION
            # We assume a frame rate of roughly 30 FPS for 30 frames history ~ 1 second
            # So, for TIREDNESS_DURATION seconds, we need TIREDNESS_DURATION * 30 frames of history
            # If the majority of recent frames show tiredness
            frames_for_tiredness_check = int(TIREDNESS_DURATION * (cap.get(cv2.CAP_PROP_FPS) or 30))
            if len(tiredness_metrics_history) == tiredness_metrics_history.maxlen and \
               sum(tiredness_metrics_history) >= frames_for_tiredness_check * 0.75: # 75% of frames indicate tiredness
                if tiredness_start_time is None:
                    tiredness_start_time = current_time
                elif current_time - tiredness_start_time >= TIREDNESS_DURATION:
                    cv2.putText(frame, "TIREDNESS DETECTED!", (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 100, 255), 2)
                    winsound.Beep(900, 700)
                    if not tiredness_email_sent:
                        send_email("Tiredness Detected", "Driver appears to be tired based on facial expressions.")
                        tiredness_email_sent = True
            else:
                tiredness_start_time = None
                tiredness_email_sent = False
                tiredness_metrics_history.clear() # Clear history if not consistently tired


    if not face_detected_this_frame: # No face detected in current frame
        # Reset all time-based counters/flags when no face is present to avoid false positives
        eye_closed_start = None
        drowsy_email_sent = False
        eye_cover_email_sent = False
        yawn_consec_frames = 0 # Reset yawn consecutive frames
        fixed_gaze_start_time = None
        road_hypnotism_email_sent = False
        tiredness_start_time = None
        tiredness_email_sent = False
        tiredness_metrics_history.clear()
        # Do NOT reset yawn_counter or blinks_in_period here, as they are for cumulative monitoring.


    # -------- Exit Button --------
    cv2.rectangle(frame, (450, 10), (640, 50), (50, 50, 50), -1)
    cv2.putText(frame, "EXIT", (500, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # -------- Show Frame --------
    cv2.imshow("Drowsiness & Driver State Monitoring", frame)
    cv2.setMouseCallback("Drowsiness & Driver State Monitoring", click_event)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
