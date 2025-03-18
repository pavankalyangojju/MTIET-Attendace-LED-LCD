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
BUZZER_PIN = 17  
GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.output(BUZZER_PIN, GPIO.LOW)

# Initialize pyttsx3 engine for text-to-speech
engine = pyttsx3.init()

# Initialize RFID Reader
reader = SimpleMFRC522()

# LCD I2C Setup
LCD_ADDRESS = 0x27  
LCD_WIDTH = 16
LCD_CHR = 1
LCD_CMD = 0
LINE_1 = 0x80
LINE_2 = 0xC0
LCD_BACKLIGHT = 0x08
ENABLE = 0b00000100

bus = smbus2.SMBus(1)  

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

lcd_init()  

def speak(text):
    engine.say(text)
    engine.runAndWait()

def send_attendance_api(name, date, timestamp):
    api_url = "http://localhost:5000/attendance"
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
        RFID_LIST = pickle.load(r)
except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
    print("Error loading face data:", e)
    exit()

FACES = np.array(FACES).reshape(FACES.shape[0], -1)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

COL_NAMES = ['NAME', 'RFID', 'DATE', 'TIME']
if not os.path.exists("Attendance"):
    os.makedirs("Attendance")

attended = set()  

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
    lcd_display("Put Face in Front", LINE_1)
    lcd_display("of Camera", LINE_2)

    ret, frame = video.read()
    if not ret:
        print("Error: Frame not captured.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)

        try:
            output = knn.predict(resized_img)
            recognized_name = output[0]
            print("Recognized:", recognized_name)
        except Exception as e:
            print("Prediction Error:", e)
            continue

        lcd_display("Put RFID Card", LINE_1)
        speak("Put your RFID Card")

        try:
            card_id, card_text = reader.read()  
            if not card_id:
                raise ValueError("No RFID card detected!")
            print(f"RFID Card Detected: {card_id}")
        except Exception as e:
            lcd_display("RFID Read Error!", LINE_1)
            lcd_display("Try Again", LINE_2)
            print(f"RFID Error: {e}")
            time.sleep(2)
            lcd_display("", LINE_1)
            lcd_display("", LINE_2)
            continue  

        # ✅ CHECK IF ATTENDANCE ALREADY TAKEN
        date_today = datetime.now().strftime("%d-%m-%Y")
        attendance_file = f"Attendance/Attendance_{date_today}.csv"

        file_exists = os.path.isfile(attendance_file)
        already_taken = False

        if file_exists:
            with open(attendance_file, "r") as f:
                reader = csv.reader(f)
                next(reader)  
                for row in reader:
                    if row[0] == recognized_name and row[1] == str(card_id):
                        already_taken = True
                        break

        if already_taken:
            lcd_display("Attendance", LINE_1)
            lcd_display("Already Taken!", LINE_2)
            speak("Attendance already taken, no need again.")
            time.sleep(3)
            lcd_display("", LINE_1)
            lcd_display("", LINE_2)
            continue  

        ts = time.time()
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")

        with open(attendance_file, "a", newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(COL_NAMES)
            writer.writerow([recognized_name, card_id, date_today, timestamp])

        lcd_display("Attendance Taken", LINE_1)
        lcd_display(f"Name: {recognized_name}", LINE_2)
        speak("Attendance Taken")
        send_attendance_api(recognized_name, date_today, timestamp)
        buzzer_and_lcd_message(recognized_name)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        lcd_display("", LINE_1)
        lcd_display("", LINE_2)
        break

video.release()
cv2.destroyAllWindows()
GPIO.cleanup()
