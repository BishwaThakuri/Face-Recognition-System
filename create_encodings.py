import face_recognition
import pickle
import os

# --- CONFIGURATION ---
folder_path = "photos"
encoding_file = "known_encodings.pkl"
# ---------------------

encodings_list = []
names_list = []

# Check if photos folder exists
if not os.path.exists(folder_path):
    print(f"Error: The folder '{folder_path}' does not exist. Please create it and add photos.")
    exit()

print("Step 1: Processing images...")

# Loop through every file in the photos folder
for filename in os.listdir(folder_path):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        name = os.path.splitext(filename)[0].capitalize()
        image_path = os.path.join(folder_path, filename)
        
        print(f"   > Analyzing {filename}...")
        
        try:
            # Load image and find the face
            image = face_recognition.load_image_file(image_path)
            # Get the 128-dimension face encoding
            # We assume there is only 1 face per photo, so we take index [0]
            encoding = face_recognition.face_encodings(image)[0]
            
            encodings_list.append(encoding)
            names_list.append(name)
            print(f"     - Verified: {name}")
            
        except IndexError:
            print(f"     - WARNING: No face found in {filename}. Skipping.")
        except Exception as e:
            print(f"     - Error processing {filename}: {e}")

# Save the data to a file
print("\nStep 2: Saving database...")
data = {"encodings": encodings_list, "names": names_list}

with open(encoding_file, "wb") as f:
    pickle.dump(data, f)

print(f"Success! '{encoding_file}' has been created with {len(names_list)} faces.")