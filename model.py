import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# =========================
# 1. Load Preprocessed Data (Relative Paths)
# =========================
base_dir = os.path.join(os.getcwd(), 'Preprocessed Data')

train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'valid')
test_dir = os.path.join(base_dir, 'test')

image_size = (224, 224)
batch_size = 32

train_data = ImageDataGenerator(rescale=1./255).flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

valid_data = ImageDataGenerator(rescale=1./255).flow_from_directory(
    valid_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_data = ImageDataGenerator(rescale=1./255).flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# =========================
# 2. Build MobileNetV2 Model
# =========================
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze base model

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
predictions = Dense(6, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# =========================
# 3. Set Up Model Saving
# =========================
checkpoint = ModelCheckpoint('mobilenetv2_metal_defect_best_model.h5',
                             monitor='val_accuracy',
                             save_best_only=True,
                             mode='max',
                             verbose=1)

# =========================
# 4. Train the Model
# =========================
history = model.fit(
    train_data,
    epochs=10,
    validation_data=valid_data,
    callbacks=[checkpoint]
)

# Save the final model after training
model.save('mobilenetv2_metal_defect_final_model.h5')
print("Final model saved successfully!")

# =========================
# 5. Evaluate the Model
# =========================
test_loss, test_accuracy = model.evaluate(test_data)
print(f'\nTest Accuracy: {test_accuracy * 100:.2f}%')
