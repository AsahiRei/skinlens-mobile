import os
# Suppress TF warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path

# Constants
IMG_SIZE = (224, 224)
CLASS_NAMES = ["acne", "eczema"]

# Get model path
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BACKEND_DIR, "ml", "models", "detection", "skinlens_detection.keras")

_model = None

def load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model

def preprocess_image(image_input):
    # Load image if path is provided
    if isinstance(image_input, (str, Path)):
        image = Image.open(image_input).convert("RGB")
    else:
        image = image_input.convert("RGB")

    image = image.resize(IMG_SIZE)
    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
    
    return image_array

def detect_skin_condition(image_input, threshold=0.5):
    try:
        model = load_model()
        
        processed_image = preprocess_image(image_input)
        predictions = model.predict(processed_image, verbose=0)
        prediction_scores = predictions[0]
        
        predicted_class_idx = np.argmax(prediction_scores)
        confidence = float(prediction_scores[predicted_class_idx])
        predicted_class = CLASS_NAMES[predicted_class_idx]
        
        # Build all predictions dict
        all_predictions = {
            CLASS_NAMES[i]: float(prediction_scores[i]) 
            for i in range(len(CLASS_NAMES))
        }
        
        return {
            'class': predicted_class,
            'confidence': confidence,
            'all_predictions': all_predictions,
            'is_confident': confidence >= threshold
        }
    
    except Exception as e:
        raise RuntimeError(f"Detection failed: {str(e)}")

def batch_detect(image_paths, threshold=0.5):
    results = []
    for image_path in image_paths:
        try:
            result = detect_skin_condition(image_path, threshold)
            results.append(result)
        except Exception as e:
            results.append({'error': str(e), 'image': str(image_path)})
    
    return results
