import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
import pyttsx3
import RPi.GPIO as GPIO
import smbus2
import requests
from mfrc522 import SimpleMFRC522
from sklearn.neighbors import KNeighborsClassifier

# --- Telegram Config ---
BOT_TOKEN = "8129064480:AAFZZjw7UTUrPgwUW33xu_B51MyJPg3WneY"
CHAT_ID = "1367693706"

# GPIO Setup
GPIO.setmode(GPIO.BCM)
BUZZER_PIN = 17
LED_WRONG_PIN = 27
LED_ATTENDANCE_PIN = 22

GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.setup(LED_WRONG_PIN, GPIO.OUT)
GPIO.setup(LED_ATTENDANCE_PIN, GPIO.OUT)

GPIO.output(BUZZER_PIN, GPIO.LOW)
GPIO.output(LED_WRONG_PIN, GPIO.LOW)
GPIO.output(LED_ATTENDANCE_PIN, GPIO.LOW)

# TTS
engine = pyttsx3.init()

# RFID
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
lcd_display("HI,Welcome to", LINE_1)
lcd_display("AttendanceSystem", LINE_2)
time.sleep(3)
lcd_display("", LINE_1)
lcd_display("", LINE_2)

def speak(text):
    engine.say(text)
    engine.runAndWait()

def send_attendance_api(name, date, timestamp):
    try:
        requests.post("http://localhost:5000/attendance", json={"name": name, "date": date, "time": timestamp})
    except Exception as e:
        print("API Error:", e)

def send_telegram_photo(image_path, caption=""):
    try:
        with open(image_path, 'rb') as photo:
            files = {'photo': photo}
            data = {'chat_id': CHAT_ID, 'caption': caption}
            requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto", files=files, data=data)
    except Exception as e:
        print("Telegram Error:", e)

# Load Face Data
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
except Exception as e:
    print("Error loading face data:", e)
    exit()

FACES = np.array(FACES).reshape(FACES.shape[0], -1)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

COL_NAMES = ['NAME', 'RFID', 'DATE', 'TIME']
os.makedirs("Attendance", exist_ok=True)

try:
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
            except Exception as e:
                print("Prediction Error:", e)
                continue

            lcd_display("Put RFID Card", LINE_1)
            speak("Put your RFID Card")

            # Turn OFF camera
            video.release()
            cv2.destroyAllWindows()

            try:
                card_id, card_text = reader.read()
            except Exception as e:
                lcd_display("RFID Read Error!", LINE_1)
                lcd_display("Try Again", LINE_2)
                print("RFID Error:", e)
                time.sleep(2)
                lcd_display("", LINE_1)
                lcd_display("", LINE_2)
                continue

            if recognized_name not in LABELS:
                lcd_display("Unknown Face", LINE_1)
                lcd_display("Try Again", LINE_2)
                continue

            expected_rfid = RFID_LIST[LABELS.index(recognized_name)]
            if str(card_id) != str(expected_rfid):
                lcd_display("Put Correct Card", LINE_1)
                speak("Please put correct card")
                GPIO.output(LED_WRONG_PIN, GPIO.HIGH)
                for _ in range(2):
                    GPIO.output(BUZZER_PIN, GPIO.HIGH)
                    time.sleep(1)
                    GPIO.output(BUZZER_PIN, GPIO.LOW)
                    time.sleep(0.5)
                GPIO.output(LED_WRONG_PIN, GPIO.LOW)
                time.sleep(5)

                # Turn camera back ON after 5s
                video = cv2.VideoCapture(0)
                if not video.isOpened():
                    lcd_display("Camera Error!", LINE_1)
                    print("Error: Could not reopen camera.")
                    GPIO.cleanup()
                    exit()

                GPIO.output(LED_WRONG_PIN, GPIO.HIGH)
                time.sleep(3.5)
                GPIO.output(LED_WRONG_PIN, GPIO.LOW)
                lcd_display("", LINE_1)
                lcd_display("", LINE_2)
                continue

            date_today = datetime.now().strftime("%d-%m-%Y")
            attendance_file = f"Attendance/Attendance_{date_today}.csv"
            file_exists = os.path.isfile(attendance_file)
            attendance_count = 0
            if file_exists:
                with open(attendance_file, "r") as f:
                    reader_csv = csv.reader(f)
                    next(reader_csv)
                    for row in reader_csv:
                        if row[0] == recognized_name and row[1] == str(card_id):
                            attendance_count += 1

            if attendance_count >= 2:
                lcd_display("Two Time Is Over", LINE_1)
                lcd_display("Not Allowed", LINE_2)
                speak("Two time is over, not allowed")
                time.sleep(3)
                continue

            timestamp = datetime.now().strftime("%H:%M:%S")
            with open(attendance_file, "a", newline='') as csvfile:
                writer = csv.writer(csvfile)
                if not file_exists:
                    writer.writerow(COL_NAMES)
                writer.writerow([recognized_name, card_id, date_today, timestamp])

            photo_filename = f"Attendance/images/{recognized_name}_{datetime.now().strftime('%H%M%S')}.jpg"
            os.makedirs("Attendance/images", exist_ok=True)
            cv2.imwrite(photo_filename, frame)

            send_telegram_photo(photo_filename, caption=f"Attendance Marked\nName: {recognized_name}\nDate: {date_today}\nTime: {timestamp}")
            send_attendance_api(recognized_name, date_today, timestamp)

            lcd_display("Attendance Taken", LINE_1)
            lcd_display(f"Name: {recognized_name}", LINE_2)
            speak("Attendance Taken")

            GPIO.output(LED_ATTENDANCE_PIN, GPIO.HIGH)
            GPIO.output(BUZZER_PIN, GPIO.HIGH)
            time.sleep(1)
            GPIO.output(BUZZER_PIN, GPIO.LOW)
            time.sleep(10)
            GPIO.output(LED_ATTENDANCE_PIN, GPIO.LOW)

            # Turn camera ON again
            video = cv2.VideoCapture(0)
            if not video.isOpened():
                lcd_display("Camera Error!", LINE_1)
                print("Error: Could not reopen camera.")
                GPIO.cleanup()
                exit()

        if 'video' in locals() and video.isOpened():
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    lcd_display("", LINE_1)
    lcd_display("", LINE_2)
    if 'video' in locals():
        video.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()
