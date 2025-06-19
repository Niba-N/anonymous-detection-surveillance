# Anonymous Threat Detection using Neural Networks for Smart Surveillance


This project is a real-time surveillance system that detects **weapons** and **anomalous human behavior** using deep learning models. Built with a Flask backend and integrated computer vision modules, it offers smart, privacy-aware monitoring for dynamic environments.

---

## 🚀 Features

* 🔍 **Weapon Detection** using YOLOv8
* 🤖 **Anomaly Detection** using LSTM Autoencoder
* 🧠 **ResNet-based Feature Extraction**
* ⚡ Real-time alerts via Flask API
* 🔒 Designed to ensure anonymity and privacy

---

## 🧠 Tech Stack

| Task               | Tool              |
| ------------------ | ----------------- |
| Object Detection   | YOLOv8 (PyTorch)  |
| Feature Extraction | ResNet50          |
| Anomaly Detection  | LSTM Autoencoder  |
| Backend API        | Flask             |
| Frontend           | HTML (index.html) |
| Video Processing   | OpenCV            |

---

## 📁 Folder Structure

```
intelligent-surveillance-system/
│
├── anomaly_project/
│   ├── templates/
│   │   └── index.html          # Frontend UI
│   ├── anomaly_snapshots/     # Detected anomalies
│
├── models/
│   ├── yolov8n.pt              # YOLOv8 lightweight model
│   ├── yolov8m.pt              # YOLOv8 medium model
│   ├── lstm_autoencoder.h5     # Trained LSTM model
│   └── lstm_autoencoder.keras  # Keras version (backup)
│
├── src/
│   ├── app.py                  # Flask app
│   ├── detector.py             # Detection logic
│   ├── detection_module.py     # Core module
│   ├── before_resnet.py        # ResNet preprocessing
│   └── train_autoencoder.py    # Train anomaly model
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ▶️ How to Run

1. **Install dependencies**
   Make sure Python ≥ 3.8 is installed.

```bash
pip install -r requirements.txt
```

2. **Run the Flask App**

```bash
python src/app.py
```

3. **Access in Browser**

```
http://127.0.0.1:5000
```

4. **Upload or stream video** via the web UI to trigger live detection.

---

## 📷 Demo

> *(Insert a GIF or link to demo video here)*
> Anomaly detection (loitering, sudden panic) and weapon detection in real-time CCTV footage.

---

## 🔐 Privacy & Ethics

* No facial recognition used
* Only motion and object anomalies are detected
* No data stored permanently

---

## 📚 Future Improvements

* 📩 Email/SMS alert system
* 📊 Admin dashboard
* 🌐 Deployable to Raspberry Pi / Jetson Nano

---

## 🤝 Contributing

Pull requests are welcome! Feel free to suggest features or improvements by opening an issue.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).



