import cv2
from deepface import DeepFace
from collections import Counter
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import threading
import uvicorn
import time
import redis
import json
import base64

r = redis.Redis(host='localhost', port=6379, db=0)

def save_face_to_redis(face_img, face_data):
    try:
        # encoding slike u base64
        _, buffer = cv2.imencode('.jpg', face_img)
        face_b64 = base64.b64encode(buffer).decode('utf-8')

        # opcionalno: vektor lica
        embedding = DeepFace.represent(face_img, model_name='Facenet', enforce_detection=False)
        if embedding:
            face_data["embedding"] = embedding[0]["embedding"]

        # dodaj sliku
        face_data["image"] = face_b64
        face_data["timestamp"] = time.time()

        # generiraj ID
        key = f"face:{int(time.time() * 1000)}"

        # spremi kao JSON
        r.set(key, json.dumps(face_data))

    except Exception as e:
        print("Greška pri spremanju u Redis:", e)

app = FastAPI()
latest_summary = {
    "total_faces": 0,
    "gender": {"Woman": 0, "Man": 0},
    "age_distribution": {
        "18-25": 0,
        "26-35": 0,
        "36-50": 0,
        "50+": 0
    },
    "emotions": {
        "happy": 0,
        "neutral": 0,
        "sad": 0,
        "angry": 0
    }
}

# RTSP URL
rtsp_url = 'rtsp://admin:Prekidac202x@185.98.0.105:9910/stream'

def categorize_custom_age(age):
    try:
        age = int(age)
    except:
        return None
    if 18 <= age <= 25:
        return "18-25"
    elif 26 <= age <= 35:
        return "26-35"
    elif 36 <= age <= 50:
        return "36-50"
    elif age > 50:
        return "50+"
    return None

def analyze_stream():
    global latest_summary
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("Greška pri otvaranju streama.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(1)
            continue

        try:
            results = DeepFace.analyze(
                frame,
                actions=['age', 'gender', 'emotion'],
                detector_backend='opencv',
                enforce_detection=False,
                silent=True
            )

            if isinstance(results, dict):
                results = [results]

            # Inicijalizacija lokalnih brojača
            gender_count = {"Woman": 0, "Man": 0}
            age_distribution = {"18-25": 0, "26-35": 0, "36-50": 0, "50+": 0}
            emotions = {"happy": 0, "neutral": 0, "sad": 0, "angry": 0}

            for face in results:

                emotion = str(face.get("dominant_emotion", "neutral")).lower()
                age = face.get("age", 0)
                raw_gender = face.get("gender", "unknown")

                # Ako je dict, uzmi onaj s većim postotkom
                if isinstance(raw_gender, dict):
                    gender = max(raw_gender, key=raw_gender.get)
                else:
                    gender = str(raw_gender).lower()

                if gender.lower() == "woman":
                    gender_count["Woman"] += 1
                elif gender.lower() == "man":
                    gender_count["Man"] += 1

                age_group = categorize_custom_age(age)
                if age_group:
                    age_distribution[age_group] += 1

                if emotion in emotions:
                    emotions[emotion] += 1

                # Spremi lice u Redis
                region = face.get("region", {})
                x, y, w, h = region.get("x", 0), region.get("y", 0), region.get("w", 0), region.get("h", 0)
                face_img = frame[y:y+h, x:x+w]

                face_data = {
                    "gender": gender,
                    "age": age,
                    "emotion": emotion,
                    "age_group": age_group
                }

                save_face_to_redis(face_img, face_data)


            # Ažuriraj globalni sažetak
            latest_summary = {
                "total_faces": len(results),
                "gender": gender_count,
                "age_distribution": age_distribution,
                "emotions": emotions
            }

        except Exception as e:
            print("Greška u analizi:", e)

        time.sleep(1)

    cap.release()

@app.get("/status")
def get_status():
    return JSONResponse(content=latest_summary)

def start_server():
    threading.Thread(target=analyze_stream, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8081)

if __name__ == "__main__":
    start_server()
