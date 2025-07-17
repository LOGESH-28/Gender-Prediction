import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization, Activation
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import scipy.io

# =====================
# Data Loader Generator
# =====================
class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, labels, batch_size=32, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.augment = augment
        self.on_epoch_end()

    def __len__(self):
        return len(self.image_paths) // self.batch_size

    def __getitem__(self, idx):
        batch_paths = self.image_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        images = [self._load_image(p) for p in batch_paths]
        return np.array(images), np.array(batch_labels)

    def on_epoch_end(self):
        data = list(zip(self.image_paths, self.labels))
        np.random.shuffle(data)
        self.image_paths, self.labels = zip(*data)

    def _load_image(self, path):
        img = cv2.imread(path)
        img = cv2.resize(img, (64, 64))
        img = img / 255.0
        if self.augment and np.random.rand() < 0.5:
            img = cv2.flip(img, 1)
        return img

# =====================
# SSR-Net-like Architecture
# =====================
def build_ssrnet(input_shape=(64, 64, 3)):
    inp = Input(shape=input_shape)

    x = Conv2D(32, (3,3), padding='same')(inp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(64, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(128, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inp, outputs=output)
    return model

# =====================
# Load IMDb Dataset
# =====================
def load_imdb_data(mat_path, img_dir, limit=5000):
    mat = scipy.io.loadmat(mat_path)
    meta = mat["imdb"]
    photo_paths = meta[0][0][2][0]
    genders = meta[0][0][3][0]

    valid_idx = ~np.isnan(genders)
    photo_paths = photo_paths[valid_idx]
    genders = genders[valid_idx].astype(int)

    full_paths = [os.path.join(img_dir, p[0]) for p in photo_paths]

    male_idx = np.where(genders == 1)[0]
    female_idx = np.where(genders == 0)[0]
    min_len = min(len(male_idx), len(female_idx), limit // 2)

    selected_idx = np.concatenate([male_idx[:min_len], female_idx[:min_len]])
    np.random.shuffle(selected_idx)

    return [full_paths[i] for i in selected_idx], genders[selected_idx]

# =====================
# Load Data
# =====================
print("Loading IMDb data...")
image_paths, genders = load_imdb_data("E:/imdb Dataset/imdb.mat", "E:/imdb Dataset/", limit=5000)
X_train, X_val, y_train, y_val = train_test_split(image_paths, genders, test_size=0.2, random_state=42)

train_gen = CustomDataGenerator(X_train, y_train, batch_size=32, augment=True)
val_gen = CustomDataGenerator(X_val, y_val, batch_size=32, augment=False)

# =====================
# Compile and Train SSR-Net
# =====================
model = build_ssrnet()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model.fit(train_gen, validation_data=val_gen, epochs=20, callbacks=[early_stop])

# =====================
# Save Model
# =====================
os.makedirs("saved_models", exist_ok=True)
model.save("saved_models/ssrnet_gender.h5")
print("SSR-Net model saved to saved_models/ssrnet_gender.h5")
