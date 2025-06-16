
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the saved model
model_path = 'mobilenetv2_metal_defect_final_model.h5'  # or use the best model if preferred
model = load_model(model_path)
print(f"✅ Model '{model_path}' loaded successfully!")

# Define class names based on your training directories
class_names = ['Crazing', 'Inclusion', 'Patches', 'Pitted', 'Rolled', 'Scratches']

# Function to preprocess a single image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Prediction function
def predict_image(img_path):
    processed_image = preprocess_image(img_path)
    predictions = model.predict(processed_image)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    print(f"\nPredicted Class: {predicted_class}")
    print(f"Confidence: {confidence * 100:.2f}%")

# Example: Provide image path here
if __name__ == "__main__":
    img_path = input("Enter the image path: ").strip()

    if os.path.exists(img_path):
        predict_image(img_path)
    else:
        print("❌ Image path does not exist. Please check the path and try again.")