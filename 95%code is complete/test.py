import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
import pyttsx3
import RPi.GPIO as GPIO
import smbus2  # For I2C communication
import requests  # For sending attendance data to API
from mfrc522 import SimpleMFRC522  # RFID module
from sklearn.neighbors import KNeighborsClassifier

# GPIO Setup
GPIO.setmode(GPIO.BCM)
BUZZER_PIN = 17  # Change according to your wiring
GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.output(BUZZER_PIN, GPIO.LOW)

# Initialize pyttsx3 engine for text-to-speech
engine = pyttsx3.init()

# Initialize RFID Reader
reader = SimpleMFRC522()

# LCD I2C Setup
LCD_ADDRESS = 0x27  # Check with 'sudo i2cdetect -y 1'
LCD_WIDTH = 16
LCD_CHR = 1
LCD_CMD = 0
LINE_1 = 0x80
LINE_2 = 0xC0
LCD_BACKLIGHT = 0x08
ENABLE = 0b00000100

bus = smbus2.SMBus(1)  # I2C channel 1

def lcd_byte(bits, mode):
    high_bits = mode | (bits & 0xF0) | LCD_BACKLIGHT
    low_bits = mode | ((bits << 4) & 0xF0) | LCD_BACKLIGHT
    bus.write_byte(LCD_ADDRESS, high_bits)
    lcd_toggle_enable(high_bits)
    bus.write_byte(LCD_ADDRESS, low_bits)
    lcd_toggle_enable(low_bits)

def lcd_toggle_enable(bits):
    time.sleep(0.0005)
    bus.write_byte(LCD_ADDRESS, bits | ENABLE)
    time.sleep(0.0005)
    bus.write_byte(LCD_ADDRESS, bits & ~ENABLE)
    time.sleep(0.0005)

def lcd_init():
    lcd_byte(0x33, LCD_CMD)
    lcd_byte(0x32, LCD_CMD)
    lcd_byte(0x06, LCD_CMD)
    lcd_byte(0x0C, LCD_CMD)
    lcd_byte(0x28, LCD_CMD)
    lcd_byte(0x01, LCD_CMD)
    time.sleep(0.005)

def lcd_display(message, line):
    lcd_byte(line, LCD_CMD)
    message = message.ljust(LCD_WIDTH, " ")
    for char in message:
        lcd_byte(ord(char), LCD_CHR)

lcd_init()  # Initialize LCD

def speak(text):
    engine.say(text)
    engine.runAndWait()

def send_attendance_api(name, date, timestamp):
    api_url = "http://localhost:5000/attendance"  # Replace with actual API
    payload = {"name": name, "date": date, "time": timestamp}
    try:
        response = requests.post(api_url, json=payload)
        if response.status_code == 200:
            print("Attendance successfully sent to API.")
        else:
            print("Failed to send attendance. Status:", response.status_code)
    except Exception as e:
        print("API Error:", e)

# Load Face Recognition Model
video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Error: Camera not found.")
    exit()

facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

try:
    with open('data/names.pkl', 'rb') as w:
        LABELS = pickle.load(w)
    with open('data/faces_data.pkl', 'rb') as f:
        FACES = pickle.load(f)
    with open('data/rfid_data.pkl', 'rb') as r:
        RFID_LIST = pickle.load(r)  # Load RFID numbers
except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
    print("Error loading face data:", e)
    exit()

FACES = np.array(FACES).reshape(FACES.shape[0], -1)
print('Faces matrix shape:', FACES.shape)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

COL_NAMES = ['NAME', 'RFID', 'DATE', 'TIME']
if not os.path.exists("Attendance"):
    os.makedirs("Attendance")

attended = set()
camera_off_time = None  # Track when the camera was turned off
camera_active = True  # Camera state

def buzzer_and_lcd_message(name):
    for _ in range(2):
        GPIO.output(BUZZER_PIN, GPIO.HIGH)
        time.sleep(1)
        GPIO.output(BUZZER_PIN, GPIO.LOW)
        time.sleep(0.5)

    lcd_display("Attendance Taken", LINE_1)
    lcd_display(f"Name: {name}", LINE_2)
    time.sleep(3)
    lcd_display("", LINE_1)
    lcd_display("", LINE_2)

while True:
    if not camera_active:
        # Check if 10 seconds have passed
        if time.time() - camera_off_time >= 10:
            video = cv2.VideoCapture(0)
            if video.isOpened():
                print("Camera turned back on.")
                camera_active = True
            else:
                print("Error: Could not reopen camera.")

    lcd_display("Put Face in Front", LINE_1)
    lcd_display("of Camera", LINE_2)

    if not camera_active:
        time.sleep(1)
        continue  # Skip iteration if camera is off

    ret, frame = video.read()
    if not ret:
        print("Error: Frame not captured.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        if w == 0 or h == 0:
            continue

        crop_img = frame[y:y+h, x:x+w]
        if crop_img.size == 0:
            continue

        try:
            resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        except Exception as e:
            print("Error resizing image:", e)
            continue

        try:
            output = knn.predict(resized_img)
            recognized_name = output[0]
            print("Recognized:", recognized_name)
        except Exception as e:
            print("Prediction Error:", e)
            continue

        lcd_display("Put RFID Card", LINE_1)
        lcd_display("", LINE_2)
        speak("Put your RFID Card")

        try:
            card_id, _ = reader.read()  # Read RFID card
            print(f"RFID Card Detected: {card_id}")
        except Exception as e:
            print("RFID Error:", e)
            continue

        if recognized_name not in attended and str(card_id).strip() in [str(rfid).strip() for rfid in RFID_LIST]:
            ts = time.time()
            date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
            timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
            attendance_file = f"Attendance/Attendance_{date}.csv"

            file_exists = os.path.isfile(attendance_file)
            with open(attendance_file, "a", newline='') as csvfile:
                writer = csv.writer(csvfile)
                if not file_exists:
                    writer.writerow(COL_NAMES)
                writer.writerow([recognized_name, card_id, date, timestamp])

            attended.add(recognized_name)
            lcd_display("Attendance Taken", LINE_1)
            lcd_display(f"Name: {recognized_name}", LINE_2)
            speak("Attendance Taken")
            send_attendance_api(recognized_name, date, timestamp)
            buzzer_and_lcd_message(recognized_name)

            # Turn off the camera after attendance is taken
            video.release()
            camera_active = False
            camera_off_time = time.time()
            print("Camera turned off for 10 seconds.")

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        lcd_display("", LINE_1)
        lcd_display("", LINE_2)
        break

video.release()
cv2.destroyAllWindows()
GPIO.cleanup()
