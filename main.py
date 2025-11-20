import cv2
import face_recognition
import pickle
import numpy as np
import time # Imported for potential future use or logging

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

# --- TRACKING VARIABLES ---
# These variables are initialized outside the loop and will be modified within it.
next_face_id = 0          # The next ID to assign to a new, unknown face (FIXED: Assignment moved here)
tracked_faces = {}        # Dictionary to store ID: [Face_Encoding, Name, Bbox Center]
max_distance = 0.6        # Max distance for a match (0.6 is standard)
# --------------------------

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
    
    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # 4. Detect & Encode Faces in the Live Frame
    live_face_locations = face_recognition.face_locations(rgb_small_frame)
    live_face_encodings = face_recognition.face_encodings(rgb_small_frame, live_face_locations)
    
    # List to track which live faces have been matched to a known ID
    matched_live_indices = set()
    
    # --- 5. Matching & Tracking (NEW LOGIC) ---
    
    current_frame_data = []

    # 5a. Try to match live faces to already tracked IDs (to prevent flickering)
    for i, live_encoding in enumerate(live_face_encodings):
        best_match_id = -1
        min_dist = float('inf')

        for face_id, (tracked_encoding, name, center) in tracked_faces.items():
            # Calculate distance to tracked face
            distance = face_recognition.face_distance([tracked_encoding], live_encoding)[0]
            
            if distance < min_dist and distance < max_distance:
                min_dist = distance
                best_match_id = face_id

        if best_match_id != -1:
            # Found a match to an existing tracked ID
            matched_live_indices.add(i)
            tracked_faces[best_match_id][0] = live_encoding  # Update encoding for better tracking
            
            # Now, check against the known database (Prabin's Core Logic)
            matches = face_recognition.compare_faces(known_encodings, live_encoding)
            face_distances = face_recognition.face_distance(known_encodings, live_encoding)
            
            final_name = "Unknown"
            best_db_match_index = np.argmin(face_distances)
            
            if matches[best_db_match_index]:
                final_name = known_names[best_db_match_index]
            
            # Store data for drawing
            top, right, bottom, left = live_face_locations[i]
            current_frame_data.append({
                'loc': (top * 4, right * 4, bottom * 4, left * 4),
                'name': final_name,
                'id': best_match_id
            })

    # 5b. Assign new IDs to unmatched live faces
    # The 'next_face_id' variable is handled correctly here.
    for i, live_encoding in enumerate(live_face_encodings):
        if i not in matched_live_indices:
            # New face detected, assign an ID
            new_id = next_face_id
            next_face_id += 1 # Update the global counter

            # Check against the known database (Prabin's Core Logic)
            matches = face_recognition.compare_faces(known_encodings, live_encoding)
            face_distances = face_recognition.face_distance(known_encodings, live_encoding)
            
            final_name = "Unknown"
            best_db_match_index = np.argmin(face_distances)
            
            if matches[best_db_match_index]:
                final_name = known_names[best_db_match_index]
            
            # Add to tracker dictionary
            tracked_faces[new_id] = [live_encoding, final_name, None]
            
            # Store data for drawing
            top, right, bottom, left = live_face_locations[i]
            current_frame_data.append({
                'loc': (top * 4, right * 4, bottom * 4, left * 4),
                'name': final_name,
                'id': new_id
            })

    # --- 6. Display Results ---
    
    # We will now loop through the current_frame_data list
    for data in current_frame_data:
        top, right, bottom, left = data['loc']
        name = data['name']
        face_id = data['id']

        # Set Color based on name 
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

        # Draw Box around face, including the ID
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, f"ID: {face_id}", (left, top - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        # Draw Name Label
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        # --- PERSONALIZATION FEATURE (Smart Fit Logic) ---
        msg = customer_data.get(name, customer_data["Unknown"])
        
        # Simple Draw Message 
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, msg, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


    cv2.imshow('Personalized Shopping Experience', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()