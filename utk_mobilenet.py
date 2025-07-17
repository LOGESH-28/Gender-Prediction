import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
import cv2

# ========== UTKFace Data Generator ==========
class UTKFaceDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, labels, batch_size=32, augment=False, shuffle=True):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return len(self.image_paths) // self.batch_size

    def __getitem__(self, index):
        batch_paths = self.image_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]
        images = [self._load_image(p) for p in batch_paths]
        return np.array(images), np.array(batch_labels)

    def on_epoch_end(self):
        if self.shuffle:
            temp = list(zip(self.image_paths, self.labels))
            np.random.shuffle(temp)
            self.image_paths, self.labels = zip(*temp)

    def _load_image(self, path):
        image = cv2.imread(path)
        image = cv2.resize(image, (224, 224))
        image = image / 255.0
        if self.augment:
            if np.random.rand() < 0.5:
                image = cv2.flip(image, 1)
        return image

# ========== Load and Balance UTKFace Dataset ==========
def load_utkface_data(data_dir):
    image_paths = []
    labels = []
    for fname in os.listdir(data_dir):
        if fname.endswith(".jpg") or fname.endswith(".png"):
            try:
                gender = int(fname.split("_")[1])
                if gender in [0, 1]:
                    image_paths.append(os.path.join(data_dir, fname))
                    labels.append(gender)
            except:
                continue

    image_paths = np.array(image_paths)
    labels = np.array(labels)

    male_idx = np.where(labels == 1)[0]
    female_idx = np.where(labels == 0)[0]
    min_count = min(len(male_idx), len(female_idx))

    selected_idx = np.concatenate([
        np.random.choice(male_idx, min_count, replace=False),
        np.random.choice(female_idx, min_count, replace=False)
    ])
    np.random.shuffle(selected_idx)

    balanced_paths = image_paths[selected_idx]
    balanced_labels = labels[selected_idx]
    return balanced_paths.tolist(), balanced_labels.tolist()

# ========== Load and Prepare Data ==========
print("Loading UTKFace data...")
image_paths, genders = load_utkface_data("E:/UTK Face dataset/UTKFace")

X_train, X_val, y_train, y_val = train_test_split(
    image_paths, genders, test_size=0.2, random_state=42
)

train_gen = UTKFaceDataGenerator(X_train, y_train, batch_size=32, augment=True)
val_gen = UTKFaceDataGenerator(X_val, y_val, batch_size=32, augment=False)

# ========== Build the MobileNetV2 Model ==========
print("Building model...")
base_model = MobileNetV2(input_shape=(224, 224, 3), weights='imagenet', include_top=False)

# Freeze early layers
for layer in base_model.layers[:-20]:
    layer.trainable = False
for layer in base_model.layers[-20:]:
    layer.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.6)(x)
x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ========== Train the Model ==========
print("Training model...")
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=[early_stop]
)

# ========== Save the Model ==========
os.makedirs("saved_models", exist_ok=True)
model.save("saved_models/UTKFace_mobilenetv2_gender.h5")
print("✅ Model saved to saved_models/UTKFace_mobilenetv2_gender.h5")
