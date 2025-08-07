# app.py
import numpy as np
import os
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# โหลดโมเดลใหม่
model_path = "EyeModel_v3_best.h5"
if not os.path.exists(model_path):
    model_path = "EyeModel_v3_final.h5"

try:
    model = load_model(model_path, compile=False)
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# โหลดรายชื่อคลาส
try:
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
    print(f"Loaded {len(class_names)} class names: {class_names}")
except FileNotFoundError:
    # Fallback: ใช้ชื่อคลาสเริ่มต้นตามที่เห็นในรูป
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
    print("Using fallback class names")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ข้อมูลคลาสและคำแนะนำ (ภาษาไทยและอังกฤษ)
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

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
        
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Supported: PNG, JPG, JPEG, GIF, BMP, TIFF'}), 400

    try:
        # สร้างโฟลเดอร์ uploads ถ้ายังไม่มี
        basepath = os.path.dirname(__file__)
        upload_dir = os.path.join(basepath, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        
        # บันทึกไฟล์
        filepath = os.path.join(upload_dir, file.filename)
        file.save(filepath)
        
        # โหลดและประมวลผลภาพ
        img = image.load_img(filepath, target_size=(128, 128))
        x = image.img_to_array(img)
        x = x / 255.0  # normalize เหมือนตอน training
        x = np.expand_dims(x, axis=0)
        
        # ทำนาย
        preds = model.predict(x)[0]
        pred_index = np.argmax(preds)
        
        # ดึงข้อมูลคลาสที่ทำนายได้
        predicted_class = class_names[pred_index]
        confidence = round(float(preds[pred_index]) * 100, 2)
        
        # สร้างรายละเอียดทุกคลาส
        details = []
        for i, prob in enumerate(preds):
            class_name = class_names[i]
            class_data = class_info.get(class_name, {
                'en': class_name,
                'th': class_name,
                'description': {'en': 'No description available', 'th': 'ไม่มีข้อมูลอธิบาย'}
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
            'description': {'en': 'No description available', 'th': 'ไม่มีข้อมูลอธิบาย'}
        })
        
        # ลบไฟล์ที่อัปโหลดหลังจากประมวลผลเสร็จ
        try:
            os.remove(filepath)
        except:
            pass
        
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
        return jsonify({
            'status': 'error', 
            'message': f'Error processing image: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'total_classes': len(class_names),
        'classes': class_names
    })

@app.route('/classes', methods=['GET'])
def get_classes():
    """Get all available classes"""
    classes_with_info = []
    for class_name in class_names:
        info = class_info.get(class_name, {
            'en': class_name,
            'th': class_name,
            'description': {'en': 'No description available', 'th': 'ไม่มีข้อมูลอธิบาย'}
        })
        classes_with_info.append({
            'name': {
                'en': info['en'],
                'th': info['th']
            },
            'description': info['description']
        })
    
    return jsonify({
        'status': 'success',
        'total_classes': len(class_names),
        'classes': classes_with_info
    })

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    # ใช้ PORT จาก environment variable
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, threaded=True, host='0.0.0.0', port=port)