from flask import Flask, render_template, Response, jsonify
from detector import Detector
import threading

app = Flask(__name__)
detector = Detector()

@app.route('/')
def index():
    return render_template('index.html', training_status=detector.training_progress)

@app.route('/video_feed')
def video_feed():
    return Response(detector.generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/train', methods=['GET'])
def train():
    if not detector.training:
        threading.Thread(target=detector.train_model_from_camera, args=(detector.cap,)).start()
        return jsonify({'status': 'Training started'})
    else:
        return jsonify({'status': 'Training already in progress'})

@app.route('/status', methods=['GET'])
def status():
    return jsonify({'status': detector.training_progress})

@app.route('/metrics', methods=['GET'])
def metrics():
    return jsonify({
        'loss': f"{detector.last_loss:.4f}",
        'threshold': f"{detector.threshold:.4f}" if detector.threshold else "Not Set"
    })

if __name__ == '__main__':
    app.run(debug=True)
