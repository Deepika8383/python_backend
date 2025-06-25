import os
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify
import face_recognition
import numpy as np
from flask_cors import CORS
from PIL import Image
import io
import time
import cv2

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}}, supports_credentials=True)


def calculate_confidence(distance):
    if distance > 0.6:
        return round((1.0 - distance) * 100, 2)
    else:
        confidence = (1.0 - distance) ** 2
        return round(confidence * 100, 2)

@app.route('/match', methods=['POST'])
def match_faces():
    aadhaar_path = selfie_path = None
    try:
        aadhaar_file = request.files.get('aadhaar')
        selfie_file = request.files.get('selfie')

        if not aadhaar_file or not selfie_file:
            return jsonify({'error': 'Both Aadhaar and Selfie images are required'}), 400

        # Save files locally
        aadhaar_path = 'aadhaar_image.jpg'
        selfie_path = 'selfie_image.jpg'
        aadhaar_file.save(aadhaar_path)
        selfie_file.save(selfie_path)

        print("✅ Files saved locally")

        # Load images from saved files
        aadhaar_img = face_recognition.load_image_file(aadhaar_path)
        selfie_img = face_recognition.load_image_file(selfie_path)

        # Face detection with error handling
        try:
            start_aadhaar = time.time()
            aadhaar_locations = face_recognition.face_locations(aadhaar_img)
            print(f"Aadhaar face_locations took {round(time.time() - start_aadhaar, 2)}s")
        except Exception as e:
            print(f"🔴 Error while detecting face in Aadhaar image: {str(e)}")
            return jsonify({'error': 'Face detection failed on Aadhaar image', 'details': str(e)}), 500

        try:
            start_selfie = time.time()
            selfie_locations = face_recognition.face_locations(selfie_img)
            print(f"Selfie face_locations took {round(time.time() - start_selfie, 2)}s")
        except Exception as e:
            print(f"🔴 Error while detecting face in Selfie image: {str(e)}")
            return jsonify({'error': 'Face detection failed on Selfie image', 'details': str(e)}), 500

        if not aadhaar_locations:
            return jsonify({'match': False, 'confidence_percent': 0, 'reason': 'No face found in Aadhaar image'}), 200

        if not selfie_locations:
            return jsonify({'match': False, 'confidence_percent': 0, 'reason': 'No face found in Selfie image'}), 200

        # Aadhaar encoding
        aadhaar_face_location = max(
            aadhaar_locations, key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3])
        )
        aadhaar_encoding_list = face_recognition.face_encodings(aadhaar_img, known_face_locations=[aadhaar_face_location])
        if not aadhaar_encoding_list:
            return jsonify({'match': False, 'confidence_percent': 0, 'reason': 'Unable to encode face from Aadhaar image'}), 200
        aadhaar_encoding = aadhaar_encoding_list[0]

        # Selfie encoding
        selfie_encoding_list = face_recognition.face_encodings(selfie_img, known_face_locations=[selfie_locations[0]])
        if not selfie_encoding_list:
            return jsonify({'match': False, 'confidence_percent': 0, 'reason': 'Unable to encode face from Selfie image'}), 200
        selfie_encoding = selfie_encoding_list[0]

        # Comparison
        face_dist = face_recognition.face_distance([aadhaar_encoding], selfie_encoding)[0]
        match_result = face_recognition.compare_faces([aadhaar_encoding], selfie_encoding, tolerance=0.6)[0]
        confidence = calculate_confidence(face_dist)

        return jsonify({
            'match': bool(match_result),
            'distance': round(float(face_dist), 4),
            'confidence_percent': confidence
        })

    except Exception as e:
        print(f"🔴 Error occurred: {str(e)}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

    finally:
        # Always clean up files
        if aadhaar_path and os.path.exists(aadhaar_path):
            os.remove(aadhaar_path)
            print("✅ Aadhaar image deleted")

        if selfie_path and os.path.exists(selfie_path):
            os.remove(selfie_path)
            print("✅ Selfie image deleted")
app.config['UPLOAD_FOLDER'] = 'uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def is_blurry(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var < threshold, lap_var

def is_poor_lighting(image, brightness_threshold=50):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    return mean_brightness < brightness_threshold, mean_brightness

@app.route('/check-quality', methods=['POST'])
def check_image_quality():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    image = cv2.imread(filepath)

    blurry, blur_score = is_blurry(image)
    poor_light, light_score = is_poor_lighting(image)

    os.remove(filepath)

    message = []
    if blurry:
        message.append("Image is blurry")
    if poor_light:
        message.append("Image has poor lighting")
    if not message:
        message.append("Image is clear and well lit")

    return jsonify({
        "blurry": bool(blurry),
        "blur_score": float(round(blur_score, 2)),
        "poorLighting": bool(poor_light),
        "brightness": float(round(light_score, 2)),
        "message": ' & '.join(message)
    })



if __name__ == '__main__':
    app.run(port=5000, debug=False)
