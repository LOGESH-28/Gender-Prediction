import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
import scipy.io
import cv2

# ========== IMDb Data Generator ==========
class IMDbDataGenerator(tf.keras.utils.Sequence):
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

# ========== Load IMDb Dataset ==========

def load_imdb_data(mat_path, img_dir, limit=20000):  # 10k male + 10k female = 20k
    mat = scipy.io.loadmat(mat_path)
    meta = mat["imdb"]
    photo_paths = meta[0][0][2][0]
    genders = meta[0][0][3][0]

    valid_idx = ~np.isnan(genders)
    photo_paths = photo_paths[valid_idx]
    genders = genders[valid_idx].astype(int)
    full_paths = [os.path.join(img_dir, p[0]) for p in photo_paths]

    # Balance dataset: 10,000 male and 10,000 female
    male_idx = np.where(genders == 1)[0]
    female_idx = np.where(genders == 0)[0]
    min_len = min(len(male_idx), len(female_idx), limit // 2)

    selected_idx = np.concatenate([
        np.random.choice(male_idx, min_len, replace=False),
        np.random.choice(female_idx, min_len, replace=False)
    ])
    np.random.shuffle(selected_idx)

    selected_paths = [full_paths[i] for i in selected_idx]
    selected_labels = genders[selected_idx]

    return selected_paths, selected_labels

# ========== Load and Prepare Data ==========
print("Loading IMDb data...")
image_paths, genders = load_imdb_data(
    mat_path="E:/imdb Dataset/imdb.mat",
    img_dir="E:/imdb Dataset/",
    limit=20000
)

X_train, X_val, y_train, y_val = train_test_split(
    image_paths, genders, test_size=0.2, random_state=42
)

train_gen = IMDbDataGenerator(X_train, y_train, batch_size=32, augment=True)
val_gen = IMDbDataGenerator(X_val, y_val, batch_size=32, augment=False)

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
model.save("saved_models/mobilenetv2_gender.h5")
print("✅ Model saved to saved_models/mobilenetv2_gender.h5")
