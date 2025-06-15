import os
from PIL import Image

# Base directories using relative paths
source_base = os.path.join(os.getcwd(), 'NEU Metal Surface Defects Data')
target_base = os.path.join(os.getcwd(), 'Preprocessed Data')

# Image size for model input
image_size = (224, 224)


# Function to preprocess and save images
def preprocess_and_save(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for class_folder in os.listdir(source_dir):
        class_source = os.path.join(source_dir, class_folder)
        class_target = os.path.join(target_dir, class_folder)

        if not os.path.exists(class_target):
            os.makedirs(class_target)

        for img_file in os.listdir(class_source):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(class_source, img_file)
                image = Image.open(img_path).convert('RGB')
                image = image.resize(image_size)

                # Save the processed image
                image.save(os.path.join(class_target, img_file))


# Preprocess train, valid, and test folders
for folder in ['train', 'valid', 'test']:
    preprocess_and_save(
        os.path.join(source_base, folder),
        os.path.join(target_base, folder)
    )

print("✅ Preprocessing complete. All images saved successfully!")
