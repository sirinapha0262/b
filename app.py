# app.py
import os
import json
import numpy as np
from io import BytesIO
from PIL import Image
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ---------- Config ----------
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'tif', 'webp'}
IMG_SIZE = (128, 128)
# กันไฟล์ใหญ่เกินจน proxy ตัด (ปรับได้ตามต้องการ)
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "10"))

# ถ้าคุณโฮสต์ Front แยกที่ Netlify แล้ว ไม่จำเป็นต้อง serve static จาก Flask
# ถ้ายังอยากคงไว้ ให้ตั้ง static_folder ตามโครงของคุณ
app = Flask(__name__, static_folder='front/dist', static_url_path='')
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_MB * 1024 * 1024

# ตั้ง CORS ให้ชัดเจน: origin ของ Netlify
CORS(app,
     resources={r"/*": {"origins": [
         "https://creative-llama-52be6a.netlify.app"
     ]}},
     methods=["GET", "POST", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"],
     expose_headers=["Content-Type"])

# ---------- Model Load (load once) ----------
# ใช้ tf.keras เฉพาะตอนโหลด เพื่อลด import หนัก ๆ ตอน cold start
import tensorflow as tf
from tensorflow.keras.models import load_model

model_path = "EyeModel_v3_best.h5"
if not os.path.exists(model_path):
    model_path = "EyeModel_v3_final.h5"

model = None
try:
    model = load_model(model_path, compile=False)
    print(f"[INIT] Model loaded: {model_path}")
except Exception as e:
    print(f"[INIT] Model load error: {e}")

# ---------- Class Names ----------
try:
    with open('class_names.json', 'r', encoding='utf-8') as f:
        class_names = json.load(f)
    print(f"[INIT] Class names loaded: {len(class_names)}")
except FileNotFoundError:
    class_names = [
        'Central Serous Chorioretinopathy',
        'Diabetic Retinopathy',
        'Disc Edema',
        'Glaucoma',
        'Healthy',
        'Macular Scar',
        'Myopia',
        'Pterygium',
        'Retinal Detachment',
        'Retinitis Pigmentosa'
    ]
    print("[INIT] Using fallback class names")

# ---------- Class Info ----------
class_info = {
    'Central Serous Chorioretinopathy': {
        'en': 'Central Serous Chorioretinopathy',
        'th': 'โรคจอตาชั้นกลางบวม (Central Serous Chorioretinopathy)',
        'description': {
            'en': 'A condition where fluid builds up under the retina, causing vision problems.',
            'th': 'ภาวะที่มีของเหลวสะสมใต้จอตา ทำให้มีปัญหาการมองเห็น'
        }
    },
    'Diabetic Retinopathy': {
        'en': 'Diabetic Retinopathy',
        'th': 'จอตาเสื่อมจากเบาหวาน (Diabetic Retinopathy)',
        'description': {
            'en': 'Eye damage caused by diabetes complications. Requires immediate medical attention.',
            'th': 'ความเสียหายของตาจากภาวะแทรกซ้อนของเบาหวาน ต้องรีบพบแพทย์'
        }
    },
    'Disc Edema': {
        'en': 'Disc Edema',
        'th': 'จานประสาทตาบวม (Disc Edema)',
        'description': {
            'en': 'Swelling of the optic disc, often indicating serious underlying conditions.',
            'th': 'การบวมของจานประสาทตา มักเป็นสัญญาณของโรคร้ายแรง'
        }
    },
    'Glaucoma': {
        'en': 'Glaucoma',
        'th': 'ต้อหิน (Glaucoma)',
        'description': {
            'en': 'High eye pressure causing optic nerve damage. Requires urgent treatment.',
            'th': 'ความดันในตาสูงทำลายประสาทตา ต้องรักษาด่วน'
        }
    },
    'Healthy': {
        'en': 'Healthy',
        'th': 'ปกติ (Healthy)',
        'description': {
            'en': 'Your eyes appear healthy. No disease detected.',
            'th': 'ดวงตาของคุณดูปกติดี ไม่พบโรค'
        }
    },
    'Macular Scar': {
        'en': 'Macular Scar',
        'th': 'รอยแผลเป็นที่จุดเหี่ยง (Macular Scar)',
        'description': {
            'en': 'Scarring in the central vision area, may affect detailed vision.',
            'th': 'รอยแผลเป็นในบริเวณจุดเหี่ยง อาจส่งผลต่อการมองเห็นรายละเอียด'
        }
    },
    'Myopia': {
        'en': 'Myopia',
        'th': 'สายตาสั้น (Myopia)',
        'description': {
            'en': 'Nearsightedness - difficulty seeing distant objects clearly.',
            'th': 'สายตาสั้น - มองวัตถุไกลไม่ชัด'
        }
    },
    'Pterygium': {
        'en': 'Pterygium',
        'th': 'เนื้องอกตาปลา (Pterygium)',
        'description': {
            'en': 'Growth of tissue over the eye, often caused by sun exposure.',
            'th': 'เนื้อเยื่อเจริญเติบโตปิดตา มักเกิดจากแสงแดด'
        }
    },
    'Retinal Detachment': {
        'en': 'Retinal Detachment',
        'th': 'จอตาลอก (Retinal Detachment)',
        'description': {
            'en': 'Serious condition where retina separates from eye wall. Emergency treatment needed.',
            'th': 'จอตาหลุดจากผนังตา เป็นภาวะฉุกเฉิน ต้องรักษาทันที'
        }
    },
    'Retinitis Pigmentosa': {
        'en': 'Retinitis Pigmentosa',
        'th': 'จอตาเสื่อมทางพันธุกรรม (Retinitis Pigmentosa)',
        'description': {
            'en': 'Genetic disorder causing progressive vision loss.',
            'th': 'โรคทางพันธุกรรมที่ทำให้สายตาเสื่อมไปเรื่อยๆ'
        }
    }
}

# ---------- Utils ----------
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_pil(img: Image.Image) -> np.ndarray:
    img = img.convert('RGB').resize(IMG_SIZE)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=0)
    return x

# ---------- Routes ----------
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'total_classes': len(class_names),
        'classes': class_names
    })

@app.route('/prewarm', methods=['POST', 'GET'])
def prewarm():
    """เรียก endpoint นี้หลัง deploy เพื่อวอร์มโมเดล/หลบ cold start"""
    if model is None:
        return jsonify({'ok': False, 'message': 'Model not loaded'}), 500
    dummy = np.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
    _ = model.predict(dummy)
    return jsonify({'ok': True, 'message': 'Model warmed'})

@app.route('/classes', methods=['GET'])
def get_classes():
    classes_with_info = []
    for class_name in class_names:
        info = class_info.get(class_name, {
            'en': class_name,
            'th': class_name,
            'description': {'en': 'No description available', 'th': 'ไม่มีข้อมูลอธิบาย'}
        })
        classes_with_info.append({
            'name': {'en': info['en'], 'th': info['th']},
            'description': info['description']
        })
    return jsonify({'status': 'success', 'total_classes': len(class_names), 'classes': classes_with_info})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    # รองรับทั้ง multipart/form-data และ direct binary
    if 'image' in request.files:
        file = request.files['image']
        filename = secure_filename(file.filename or 'upload')
        if not filename or not allowed_file(filename):
            return jsonify({'error': 'Invalid file type. Supported: ' + ', '.join(sorted(ALLOWED_EXTENSIONS))}), 400
        try:
            img = Image.open(file.stream)
        except Exception as e:
            return jsonify({'error': f'Cannot read image: {e}'}), 400
    else:
        # เผื่อกรณี client ส่ง binary ตรง ๆ (เช่น fetch ส่ง Blob)
        if not request.data:
            return jsonify({'error': 'No image uploaded'}), 400
        try:
            img = Image.open(BytesIO(request.data))
        except Exception as e:
            return jsonify({'error': f'Cannot read image: {e}'}), 400

    try:
        x = preprocess_pil(img)
        preds = model.predict(x)[0]
        pred_index = int(np.argmax(preds))
        predicted_class = class_names[pred_index]
        confidence = round(float(preds[pred_index]) * 100, 2)

        # รายละเอียดทุกคลาส
        details = []
        for i, prob in enumerate(preds):
            class_name = class_names[i]
            info = class_info.get(class_name, {
                'en': class_name, 'th': class_name,
                'description': {'en': 'No description available', 'th': 'ไม่มีข้อมูลอธิบาย'}
            })
            details.append({
                'class': {'en': info['en'], 'th': info['th']},
                'probability': round(float(prob) * 100, 2)
            })
        details.sort(key=lambda x: x['probability'], reverse=True)

        predicted_info = class_info.get(predicted_class, {
            'en': predicted_class, 'th': predicted_class,
            'description': {'en': 'No description available', 'th': 'ไม่มีข้อมูลอธิบาย'}
        })

        return jsonify({
            'status': 'success',
            'prediction': {'en': predicted_info['en'], 'th': predicted_info['th']},
            'confidence': confidence,
            'details': details,
            'recommendation': predicted_info['description'],
            'total_classes': len(class_names)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error processing image: {str(e)}'}), 500

# ---------- Error Handlers ----------
@app.errorhandler(413)
def too_large(e):
    return jsonify({'status': 'error', 'message': f'File too large. Max {MAX_UPLOAD_MB}MB'}), 413

# ---------- (Optional) Serve React build if needed ----------
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    # ถ้าคุณโฮสต์ Front ที่ Netlify อยู่แล้ว เส้นทางนี้แทบไม่ได้ใช้
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    index_path = os.path.join(app.static_folder, 'index.html')
    if os.path.exists(index_path):
        return send_from_directory(app.static_folder, 'index.html')
    return jsonify({'ok': True, 'message': 'API service'}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', '5000'))
    app.run(debug=False, host='0.0.0.0', port=port)
