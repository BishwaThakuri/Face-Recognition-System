import cv2
import face_recognition
import pickle
import numpy as np
import time 

# --- CONFIGURATION: Personalization Database ---
customer_data = {
    "Bishwa":   "VIP Member - Hello, Bishwa there is 20% Discount on Electronics",
    "Dipendra": "Returning Customer - Recommended: Winter Jackets",
    "Prabin":   "New Member - Welcome! Try our Free Coffee",
    "Richesh":  "Gold Tier - Personal Shopper is Ready",
    "Unknown":  "Welcome to our Store! Please Register."
}
# -----------------------------------------------

# --- TRACKING VARIABLES ---
next_face_id = 0          
tracked_faces = {}        
max_distance = 0.6
inactive_timeout = 3      
prev_time = time.time()   # For FPS calculation
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
    
    # NEW ERROR HANDLING CHECK
    if not ret: 
        print("ERROR: Failed to grab frame from webcam. Checking connection...")
        video_capture = cv2.VideoCapture(0)
        time.sleep(0.5) 
        continue 
    
    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # NEW TRACKER CLEANUP
    current_time = time.time()
    ids_to_delete = []

    for face_id, data_list in tracked_faces.items():
        last_seen = data_list[3]
        if current_time - last_seen > inactive_timeout:
            ids_to_delete.append(face_id)

    for face_id in ids_to_delete:
        del tracked_faces[face_id]
        
    # -------------------------------------------

    # 4. Detect & Encode Faces in the Live Frame
    live_face_locations = face_recognition.face_locations(rgb_small_frame)
    live_face_encodings = face_recognition.face_encodings(rgb_small_frame, live_face_locations)
    
    matched_live_indices = set()
    current_frame_data = []

    # 5a. Try to match live faces to already tracked IDs
    for i, live_encoding in enumerate(live_face_encodings):
        best_match_id = -1
        min_dist = float('inf')

        for face_id, data_list in tracked_faces.items():
            tracked_encoding = data_list[0]
            distance = face_recognition.face_distance([tracked_encoding], live_encoding)[0]
            
            if distance < min_dist and distance < max_distance:
                min_dist = distance
                best_match_id = face_id

        if best_match_id != -1:
            matched_live_indices.add(i)
            tracked_faces[best_match_id][0] = live_encoding      
            tracked_faces[best_match_id][3] = time.time()        
            
            matches = face_recognition.compare_faces(known_encodings, live_encoding)
            face_distances = face_recognition.face_distance(known_encodings, live_encoding)
            
            final_name = "Unknown"
            best_db_match_index = np.argmin(face_distances)
            
            if matches[best_db_match_index]:
                final_name = known_names[best_db_match_index]
            
            tracked_faces[best_match_id][1] = final_name         
            
            top, right, bottom, left = live_face_locations[i]
            current_frame_data.append({
                'loc': (top * 4, right * 4, bottom * 4, left * 4),
                'name': final_name,
                'id': best_match_id
            })

    # 5b. Assign new IDs to unmatched live faces
    for i, live_encoding in enumerate(live_face_encodings):
        if i not in matched_live_indices:
            new_id = next_face_id
            next_face_id += 1 

            matches = face_recognition.compare_faces(known_encodings, live_encoding)
            face_distances = face_recognition.face_distance(known_encodings, live_encoding)
            
            final_name = "Unknown"
            best_db_match_index = np.argmin(face_distances)
            
            if matches[best_db_match_index]:
                final_name = known_names[best_db_match_index]
            
            tracked_faces[new_id] = [live_encoding, final_name, None, time.time()] 
            
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

    # 🛑 NEW DYNAMIC TEXT SCALING FOR CUSTOMER MESSAGE 🛑
    msg = customer_data.get(current_frame_data[0]['name'] if current_frame_data else "Unknown", customer_data["Unknown"])
    
    # Initial parameters
    font_scale = 0.8
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    screen_width = frame.shape[1]
    
    # 1. Calculate text size and automatically shrink font if too wide
    (text_width, text_height), baseline = cv2.getTextSize(msg, font, font_scale, thickness)
    
    # Shrink loop: ensures message fits within screen width minus 20px padding
    while text_width > (screen_width - 20) and font_scale > 0.3:
        font_scale -= 0.05
        (text_width, text_height), baseline = cv2.getTextSize(msg, font, font_scale, thickness)

    # 2. Draw the black background bar (Dynamic height)
    bar_height = text_height + 40 # 20px padding top/bottom
    cv2.rectangle(frame, (0, 0), (screen_width, bar_height), (0, 0, 0), cv2.FILLED)

    # 3. Draw the scaled text (Top Left Corner)
    text_y = (bar_height // 2) + (text_height // 2) # Center text vertically in the bar
    cv2.putText(frame, msg, (10, text_y), font, font_scale, (255, 255, 255), thickness)
    
    # --------------------------------------------------------------------
    
    # 🛑 NEW FPS CALCULATION AND DISPLAY (Bottom Right Corner) 🛑
    curr_time = time.time()
    
    # Calculate FPS (if we have a previous time marker)
    if prev_time != 0:
        fps = 1 / (curr_time - prev_time)
        
        # --- Drawing the FPS Text with an Outline ---
        fps_text = f"FPS: {int(fps)}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        text_color = (0, 255, 255)  # Bright Yellow/Cyan (BGR format)
        outline_color = (0, 0, 0)   # Black outline
        
        # Position: Bottom Right, 40px from right edge, 40px from bottom edge
        text_x = screen_width - 120
        text_y = frame.shape[0] - 40
        
        # 1. Draw Black Outline (Draws a slightly thicker black copy first)
        cv2.putText(frame, fps_text, (text_x, text_y), 
                    font, font_scale, outline_color, 4) # Thickness 4 for outline

        # 2. Draw Bright Yellow Text on top
        cv2.putText(frame, fps_text, (text_x, text_y), 
                    font, font_scale, text_color, 2) # Thickness 2 for main text

    prev_time = curr_time
    # -------------------------------------------------------------

    cv2.imshow('Personalized Shopping Experience', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()