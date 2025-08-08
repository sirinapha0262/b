# app.py
import numpy as np
import os
import json
import gc
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix
import logging
from PIL import Image
import io

# ตั้งค่า logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ตั้งค่า CORS สำหรับ Netlify
CORS(app, origins=[
    "http://localhost:3000",
    "http://localhost:5173", 
    "https://*.netlify.app",
    "https://creative-llama-52be6a.netlify.app"  # เปลี่ยนเป็น URL จริงของคุณ
], supports_credentials=True)

# ตั้งค่าสำหรับ Proxy (สำหรับ Render)
app.wsgi_app = ProxyFix(app.wsgi_app)

# ตั้งค่า limits
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = '/tmp'  # ใช้ /tmp บน Render

# โหลดโมเดล (ให้โหลดครั้งเดียวตอนเริ่ม)
model = None
class_names = []

def load_ai_model():
    global model, class_names
    try:
        model_path = "EyeModel_v3_best.h5"
        if not os.path.exists(model_path):
            model_path = "EyeModel_v3_final.h5"
        
        logger.info(f"Loading model from {model_path}")
        model = load_model(model_path, compile=False)
        logger.info("Model loaded successfully")
        
        # โหลดรายชื่อคลาส
        try:
            with open('class_names.json', 'r', encoding='utf-8') as f:
                class_names = json.load(f)
            logger.info(f"Loaded {len(class_names)} class names")
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
            logger.info("Using fallback class names")
            
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model = None

# โหลดโมเดลตอนเริ่มต้น
load_ai_model()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image(file_stream):
    """ตรวจสอบและปรับขนาดรูป"""
    try:
        img = Image.open(file_stream)
        
        # ตรวจสอบขนาดไฟล์
        if img.size[0] * img.size[1] > 4000 * 4000:  # ไม่เกิน 4000x4000
            return None, "Image too large"
            
        # แปลงเป็น RGB ถ้าจำเป็น
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # ปรับขนาดถ้าใหญ่เกินไป
        if img.size[0] > 1024 or img.size[1] > 1024:
            img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            
        return img, None
    except Exception as e:
        return None, f"Invalid image: {str(e)}"

# ข้อมูลคลาสและคำแนะนำ
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

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint สำหรับ Render"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'classes_count': len(class_names)
    })

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    # Handle preflight request
    if request.method == 'OPTIONS':
        return '', 200
        
    if model is None:
        return jsonify({
            'status': 'error',
            'message': 'AI model not available. Please try again later.'
        }), 503
        
    if 'image' not in request.files:
        return jsonify({
            'status': 'error',
            'message': 'No image uploaded'
        }), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({
            'status': 'error',
            'message': 'Empty filename'
        }), 400
        
    if not allowed_file(file.filename):
        return jsonify({
            'status': 'error',
            'message': 'Invalid file type. Supported: PNG, JPG, JPEG, GIF, BMP, TIFF'
        }), 400

    try:
        logger.info(f"Processing image: {file.filename}")
        
        # ตรวจสอบและปรับแต่งรูป
        file_stream = io.BytesIO(file.read())
        img_pil, error = validate_image(file_stream)
        
        if error:
            return jsonify({
                'status': 'error',
                'message': error
            }), 400
        
        # แปลงรูปสำหรับ model
        img_array = np.array(img_pil.resize((128, 128)))
        x = img_array.astype(np.float32) / 255.0
        x = np.expand_dims(x, axis=0)
        
        logger.info("Running prediction...")
        
        # ทำนาย
        preds = model.predict(x, verbose=0)[0]
        pred_index = np.argmax(preds)
        
        # ดึงข้อมูลคลาสที่ทำนายได้
        predicted_class = class_names[pred_index]
        confidence = round(float(preds[pred_index]) * 100, 2)
        
        logger.info(f"Prediction: {predicted_class} ({confidence}%)")
        
        # สร้างรายละเอียดทุกคลาส
        details = []
        for i, prob in enumerate(preds):
            class_name = class_names[i]
            class_data = class_info.get(class_name, {
                'en': class_name,
                'th': class_name,
                'description': {
                    'en': 'No description available', 
                    'th': 'ไม่มีข้อมูลอธิบาย'
                }
            })
            
            details.append({
                'class': {
                    'en': class_data['en'],
                    'th': class_data['th']
                },
                'probability': round(float(prob) * 100, 2)
            })
        
        # เรียงลำดับจากมากไปน้อย
        details = sorted(details, key=lambda x: x['probability'], reverse=True)
        
        # ข้อมูลคลาสที่ทำนายได้
        predicted_info = class_info.get(predicted_class, {
            'en': predicted_class,
            'th': predicted_class,
            'description': {
                'en': 'No description available', 
                'th': 'ไม่มีข้อมูลอธิบาย'
            }
        })
        
        # ทำความสะอาด memory
        del x, preds, img_array, img_pil
        gc.collect()
        
        return jsonify({
            'status': 'success',
            'prediction': {
                'en': predicted_info['en'],
                'th': predicted_info['th']
            },
            'confidence': confidence,
            'details': details,
            'recommendation': predicted_info['description'],
            'total_classes': len(class_names)
        })
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Error processing image. Please try again with a different image.'
        }), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'status': 'error',
        'message': 'File too large. Maximum size is 16MB.'
    }), 413

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Internal server error. Please try again later.'
    }), 500

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting app on port {port}")
    app.run(debug=False, threaded=True, host='0.0.0.0', port=port)
