import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# 1. Load the Saved Model (Relative Path)
# =========================
model_path = os.path.join(os.getcwd(), 'mobilenetv2_metal_defect_best_model.h5')
model = load_model(model_path)
print("✅ Model loaded successfully!")

# =========================
# 2. Load Test Data (Relative Path)
# =========================
test_dir = os.path.join(os.getcwd(), 'Preprocessed Data', 'test')

image_size = (224, 224)
batch_size = 32

test_data = ImageDataGenerator(rescale=1./255).flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# =========================
# 3. Make Predictions
# =========================
y_pred_probs = model.predict(test_data)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true = test_data.classes

# Get class labels
class_labels = list(test_data.class_indices.keys())

# =========================
# 4. Print Metrics
# =========================
# Accuracy
accuracy = accuracy_score(y_true, y_pred_classes)
print(f'\n✅ Test Accuracy: {accuracy * 100:.2f}%')

# Classification Report
print('\n✅ Classification Report:')
print(classification_report(y_true, y_pred_classes, target_names=class_labels))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)

# =========================
# 5. Plot Confusion Matrix
# =========================
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
