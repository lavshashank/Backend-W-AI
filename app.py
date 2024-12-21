from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import cv2
import os
import time
import torch
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
from ultralytics import YOLO

# Define ESRGAN model architecture
class ResidualDenseBlock(torch.nn.Module):
    def __init__(self, num_feat=64, growth_rate=32):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(num_feat, growth_rate, 3, 1, 1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(growth_rate, growth_rate, 3, 1, 1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(growth_rate, growth_rate, 3, 1, 1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(growth_rate, num_feat, 3, 1, 1)
        )
        self.res_scale = 0.2

    def forward(self, x):
        return x + self.layers(x) * self.res_scale


class RRDB(torch.nn.Module):
    def __init__(self, num_feat=64, growth_rate=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, growth_rate)
        self.rdb2 = ResidualDenseBlock(num_feat, growth_rate)
        self.rdb3 = ResidualDenseBlock(num_feat, growth_rate)
        self.res_scale = 0.2

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + out * self.res_scale


class RRDBNet(torch.nn.Module):
    def __init__(self, in_ch, out_ch, num_feat, num_block):
        super().__init__()
        self.conv_first = torch.nn.Conv2d(in_ch, num_feat, 3, 1, 1)
        self.body = torch.nn.Sequential(*[RRDB(num_feat) for _ in range(num_block)])
        self.conv_body = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.upconv1 = torch.nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.pixel_shuffle1 = torch.nn.PixelShuffle(2)
        self.upconv2 = torch.nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.pixel_shuffle2 = torch.nn.PixelShuffle(2)
        self.conv_last = torch.nn.Conv2d(num_feat, out_ch, 3, 1, 1)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat)) + feat
        upsampled = self.pixel_shuffle1(self.upconv1(body_feat))
        upsampled = self.pixel_shuffle2(self.upconv2(upsampled))
        return self.conv_last(upsampled)


# Initialize Flask
app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLOv8 model
try:
    model = YOLO('yolov8n.pt')
except Exception as e:
    print(f"Error loading YOLOv8 model: {e}")
    model = None

# Load ESRGAN model
def load_esrgan_model(model_path):
    try:
        esrgan_model = RRDBNet(in_ch=3, out_ch=3, num_feat=64, num_block=23)
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        esrgan_model.load_state_dict(state_dict)
        esrgan_model.eval()
        return esrgan_model
    except Exception as e:
        print(f"Error loading ESRGAN model: {e}")
        return None

ESRGAN_MODEL_PATH = './ESRGAN/models/RRDB_ESRGAN_x4.pth'
upsampler = load_esrgan_model(ESRGAN_MODEL_PATH)

# Helper Functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def enhance_resolution(frame, model):
    try:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = (
            torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        )
        with torch.no_grad():
            output_tensor = model(input_tensor)
        sr_frame = (output_tensor.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return sr_frame
    except Exception as e:
        print(f"Error during resolution enhancement: {e}")
        return frame

def generate_event_stream():
    while True:
        time.sleep(1)
        yield f"data:{realtime_data['collision_warning']}|{realtime_data['traffic_light']}\n\n"

realtime_data = {"collision_warning": "None", "traffic_light": "Unknown"}

def cleanup_uploads_folder():
    try:
        for file in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print("Uploads folder cleaned up successfully.")
    except Exception as e:
        print(f"Error cleaning uploads folder: {e}")

@app.route('/')
def homepage():
    return "Welcome to the homepage!"

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided.'}), 400

    video_file = request.files['video']
    if video_file and allowed_file(video_file.filename):
        filename = secure_filename(video_file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            video_file.save(file_path)
            return jsonify({'message': 'File uploaded successfully.', 'file_path': file_path}), 200
        except Exception as e:
            return jsonify({'error': f'Error saving file: {str(e)}'}), 500

    return jsonify({'error': 'Invalid file type.'}), 400

@app.route('/video_feed/<path:file_path>')
def video_feed(file_path):
    def generate_frames():
        try:
            cap = cv2.VideoCapture(file_path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if upsampler is not None:
                    frame = enhance_resolution(frame, upsampler)

                if model is not None:
                    try:
                        results = model.predict(source=frame, save=False, conf=0.8)
                        frame = results[0].plot()
                    except Exception as e:
                        print(f"YOLO prediction error: {e}")

                _, buffer = cv2.imencode('.jpg', frame)
                frame_data = buffer.tobytes()
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

            cap.release()
        except Exception as e:
            print(f"Video feed error: {e}")

    response = Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    response.call_on_close(cleanup_uploads_folder)
    return response

if __name__ == '__main__':
    app.run()