import tensorflow as tf
import numpy as np
import json
import os

def load_class_names(json_path='model/class_indices.json'):
    with open(json_path, 'r') as f:
        class_names = json.load(f)
    return class_names

def preprocess_image(image):
    # Expects a numpy array, normalized float32
    image = np.expand_dims(image, axis=0).astype(np.float32)
    return image

def normalize_label(label):
    # Return label as-is (no forced lowercase or space replacement)
    return label

def predict(image, model_path='model/plant_disease_model.tflite', idx2label=None, disease_info_path='disease_info.json'):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data = preprocess_image(image)

    # Handle input type
    input_type = input_details[0]['dtype']
    if input_type == np.uint8:
        input_data = (input_data * 255).astype(np.uint8)
    elif input_type == np.float32:
        input_data = input_data.astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_index = int(np.argmax(output_data))
    confidence = float(np.max(output_data))

    # Load default mapping if not provided
    if idx2label is None:
        class_names = load_class_names()
        idx2label = {int(v): k for k, v in class_names.items()}

    raw_label = idx2label.get(predicted_index, str(predicted_index))
    label = normalize_label(raw_label)

    # Load disease info
    if os.path.exists(disease_info_path):
        with open(disease_info_path, 'r', encoding='utf-8') as f:
            disease_info = json.load(f)
    else:
        disease_info = {}

    # Debug output for troubleshooting
    print(f"DEBUG: Predicted label: '{label}'")
    print(f"DEBUG: Available keys: {list(disease_info.keys())}")

    description = disease_info.get(label, {}).get('description', 'No description available.')
    treatment = disease_info.get(label, {}).get('treatment', 'No treatment advice available.')

    return label, confidence, description, treatment
