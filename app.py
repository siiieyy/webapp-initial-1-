from flask import Flask, render_template, Response, jsonify
import cv2
from detection.model import detect_open_beak_from_frame

app = Flask(__name__)
camera = cv2.VideoCapture(0)
open_beak_state = {"count": 0}

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        count, annotated_frame = detect_open_beak_from_frame(frame)
        open_beak_state["count"] = count  # 🧠 This MUST be inside the loop

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify(open_beak_state)

if __name__ == "__main__":
    app.run(debug=True)
