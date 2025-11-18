import cv2
import face_recognition
import pickle
import numpy as np

# --- CONFIGURATION: Personalization Database ---
# This simulates the "Shopping Experience" logic
customer_data = {
    "Bishwa":   "VIP Member - Hello, Bishwa there is 20% Discount on Electronics",
    "Dipendra": "Returning Customer - Recommended: Winter Jackets",
    "Prabin":   "New Member - Welcome! Try our Free Coffee",
    "Richesh":  "Gold Tier - Personal Shopper is Ready",
    "Unknown":  "Welcome to our Store! Please Register."
}
# -----------------------------------------------

# 1. Load the learned data
print("Loading known faces database...")
try:
    with open("known_encodings.pkl", "rb") as f:
        data = pickle.load(f)
        known_encodings = data["encodings"]
        known_names = data["names"]
        print(f"Loaded {len(known_names)} known faces.")
except FileNotFoundError:
    print("Error: 'known_encodings.pkl' not found. Run create_encodings.py first!")
    exit()

# 2. Start Webcam
# '0' is usually the default webcam. If you have an external one, try '1'.
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("System Ready. Press 'q' to quit.")

while True:
    # 3. Capture Frame
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame.")
        break
    
    # Resize frame to 1/4 size for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Convert BGR (OpenCV standard) to RGB (face_recognition standard)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # 4. Detect & Encode Faces in the Live Frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    
    # 5. Match Faces
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]
        
        face_names.append(name)

    # 6. Display Results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up x4 (since we resized to 1/4)
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Set Color: Green for known, Red for unknown
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

        # Draw Box around face
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Draw Name Label
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        # --- PERSONALIZATION FEATURE (SMART FIT) ---
        msg = customer_data.get(name, customer_data["Unknown"])
        
        # 1. Start with a default font size & thickness
        font_scale = 0.8
        thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX

        # 2. Calculate how wide the text is
        (text_width, text_height), baseline = cv2.getTextSize(msg, font, font_scale, thickness)
        
        # 3. If text is wider than the screen (minus 20px padding), shrink it!
        screen_width = frame.shape[1]
        while text_width > (screen_width - 20) and font_scale > 0.3:
            font_scale -= 0.1
            (text_width, text_height), baseline = cv2.getTextSize(msg, font, font_scale, thickness)

        # 4. Draw a background bar that fits the text height
        # We add some padding (20px) to the height
        bar_height = text_height + 40
        cv2.rectangle(frame, (0, 0), (screen_width, bar_height), (0, 0, 0), cv2.FILLED)

        # 5. Draw the text centered vertically in the bar
        text_y = (bar_height // 2) + (text_height // 2)
        cv2.putText(frame, msg, (10, text_y), font, font_scale, (255, 255, 255), thickness)

    cv2.imshow('Personalized Shopping Experience', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()