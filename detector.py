import cv2
import numpy as np
import time
from keras.models import Sequential, Model
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from keras.applications.resnet50 import ResNet50, preprocess_input
from ultralytics import YOLO
from datetime import datetime
import os

class Detector:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        self.feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)
        self.autoencoder = self.build_autoencoder()
        self.trained = False
        self.threshold = None
        self.training = False
        self.training_progress = "⚠️ Not trained"
        self.anomaly = False
        self.yolo = YOLO("yolov8m.pt")
        self.save_anomalies = True
        self.knife_detected = False
        self.last_loss = 0.0  # For UI display

        if self.save_anomalies and not os.path.exists("anomaly_snapshots"):
            os.makedirs("anomaly_snapshots")

    def build_autoencoder(self):
        model = Sequential([
            LSTM(128, activation='relu', input_shape=(1, 2048), return_sequences=False),
            RepeatVector(1),
            LSTM(128, activation='relu', return_sequences=True),
            TimeDistributed(Dense(2048))
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def train_model_from_camera(self, cap, duration=60, threshold_std=5):
        self.training = True
        self.training_progress = "🚀 Collecting normal frames for training..."
        frames = []
        start_time = time.time()

        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                continue
            resized = cv2.resize(frame, (224, 224))
            norm = preprocess_input(np.expand_dims(resized.astype("float32"), axis=0))
            features = self.feature_extractor.predict(norm, verbose=0)
            frames.append(features)

        features = np.array(frames).reshape((-1, 1, 2048))
        self.training_progress = "🧠 Training LSTM Autoencoder..."
        self.autoencoder.fit(features, features, epochs=20, batch_size=32, verbose=0)

        recon = self.autoencoder.predict(features, verbose=0)
        loss = np.mean(np.power(features - recon, 2), axis=(1, 2))
        self.threshold = np.mean(loss) + threshold_std * np.std(loss)

        self.trained = True
        self.training = False
        self.training_progress = f"✅ Training complete. Threshold = {self.threshold:.4f}"

    def generate_frames(self):
        while True:
            success, frame = self.cap.read()
            if not success:
                break

            resized = cv2.resize(frame, (224, 224))
            norm = preprocess_input(np.expand_dims(resized.astype("float32"), axis=0))
            features = self.feature_extractor.predict(norm, verbose=0)
            features = features.reshape((1, 1, 2048))
            recon = self.autoencoder.predict(features, verbose=0)
            loss = np.mean(np.abs(features - recon))

            self.last_loss = loss  # Store loss for frontend UI

            # YOLO Detection
            results = self.yolo(frame, verbose=False)[0]
            self.knife_detected = False
            for result in results.boxes.data:
                x1, y1, x2, y2, conf, cls = result
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                label = self.yolo.names[int(cls)]
                confidence = float(conf)
                if confidence > 0.4:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    if "knife" in label.lower():
                        self.knife_detected = True

            # Anomaly Detection
            if self.trained and self.threshold is not None:
                if loss > self.threshold:
                    self.anomaly = True
                    if self.save_anomalies:
                        filename = datetime.now().strftime("anomaly_snapshots/%Y%m%d_%H%M%S.jpg")
                        cv2.imwrite(filename, frame)
                else:
                    self.anomaly = False

            # Overlay alerts on screen
            y_offset = 30
            if self.knife_detected:
                cv2.putText(frame, "Knife Detected!", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                y_offset += 30
            if self.anomaly:
                cv2.putText(frame, "Anomaly Detected!", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Encode for web stream
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
