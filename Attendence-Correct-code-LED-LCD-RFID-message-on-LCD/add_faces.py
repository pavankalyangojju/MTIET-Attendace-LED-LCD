import cv2
import pickle
import numpy as np
import os
import RPi.GPIO as GPIO
from mfrc522 import SimpleMFRC522
from skimage.metrics import structural_similarity as ssim

# Initialize RFID Reader
reader = SimpleMFRC522()

# Open the camera
video = cv2.VideoCapture(0)

if not video.isOpened():
    print("Error: Could not open camera.")
    exit()

# Load Haarcascade for face detection
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create data directory if not exists
if not os.path.exists('data'):
    os.makedirs('data')

faces_data = []
rfid_data = []
i = 0

# Read RFID Card
print("Scan your RFID card...")
try:
    card_id, card_text = reader.read()
    print(f"RFID Card ID: {card_id}")
    name = input("Enter Your Name: ")
except Exception as e:
    print(f"RFID Error: {e}")
    exit()

# File paths
faces_file = 'data/faces_data.pkl'
names_file = 'data/names.pkl'
rfid_file = 'data/rfid_data.pkl'

# Load existing face data if available
if os.path.exists(faces_file):
    with open(faces_file, 'rb') as f:
        stored_faces = pickle.load(f)
        stored_faces = stored_faces.reshape(-1, 50, 50, 3)
else:
    stored_faces = np.empty((0, 50, 50, 3), dtype=np.uint8)

# Load existing RFID data
if os.path.exists(rfid_file):
    with open(rfid_file, 'rb') as f:
        stored_rfid = pickle.load(f)
else:
    stored_rfid = []

def mse(image1, image2):
    """Mean Squared Error (MSE) between two images"""
    err = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    err /= float(image1.shape[0] * image1.shape[1])
    return err

def is_duplicate(face):
    """Check if the captured face already exists in stored_faces."""
    if stored_faces.shape[0] == 0:
        return False  # No stored faces yet

    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    for stored_face in stored_faces:
        stored_face_gray = cv2.cvtColor(stored_face, cv2.COLOR_BGR2GRAY)
        
        if face_gray.shape != stored_face_gray.shape:
            continue

        mse_score = mse(face_gray, stored_face_gray)
        ssim_score = ssim(face_gray, stored_face_gray)

        if mse_score < 200 and ssim_score > 0.8:
            return True

    return False

while True:
    ret, frame = video.read()
    
    if not ret:
        print("Error: Could not capture frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    duplicate_detected = False

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]

        if crop_img.size == 0:
            print("Error: Empty face image.")
            continue

        resized_img = cv2.resize(crop_img, (50, 50))

        if is_duplicate(resized_img):
            duplicate_detected = True
            cv2.putText(frame, "Duplicate Face Detected!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (0, 0, 255), 2)
        elif len(faces_data) < 100 and i % 10 == 0:
            faces_data.append(resized_img)
            rfid_data.append(card_id)  # Store RFID along with face data

        i += 1
        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)

    if k == ord('q') or len(faces_data) == 100:
        break

video.release()
cv2.destroyAllWindows()

# Convert to NumPy array and reshape dynamically
faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(len(faces_data), 50, 50, 3)

# Save names
if not os.path.exists(names_file):
    names = [name] * len(faces_data)
    with open(names_file, 'wb') as f:
        pickle.dump(names, f)
else:
    with open(names_file, 'rb') as f:
        names = pickle.load(f)
    names.extend([name] * len(faces_data))
    with open(names_file, 'wb') as f:
        pickle.dump(names, f)

# Save RFID data
if not os.path.exists(rfid_file):
    with open(rfid_file, 'wb') as f:
        pickle.dump(rfid_data, f)
else:
    with open(rfid_file, 'rb') as f:
        stored_rfid = pickle.load(f)
    stored_rfid.extend(rfid_data)
    with open(rfid_file, 'wb') as f:
        pickle.dump(stored_rfid, f)

# Save face data only if not duplicate
if not os.path.exists(faces_file):
    with open(faces_file, 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open(faces_file, 'rb') as f:
        stored_faces = pickle.load(f)
        stored_faces = stored_faces.reshape(-1, 50, 50, 3)

    stored_faces = np.vstack((stored_faces, faces_data))
    with open(faces_file, 'wb') as f:
        pickle.dump(stored_faces, f)

print("Face and RFID data saved successfully!")
