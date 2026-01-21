import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# =========================
# 1. Paths + load CSV
# =========================

# TODO: ADD YOUR CSV FILE PATH HERE
# Example (Windows):
# CSV_PATH = r"C:\path\to\your\Dataset.csv"
# Example (Mac/Linux):
# CSV_PATH = "/path/to/your/Dataset.csv"
CSV_PATH = r""  

# TODO: ADD YOUR IMAGE DIRECTORY PATH HERE
# This directory should contain class folders (labels)
# Example:
# IMG_DIR/
#   class1/
#     image1.jpg
#   class2/
#     image2.jpg
IMG_DIR = r""  

df = pd.read_csv(CSV_PATH)

# =========================
# 2. Train / val / test split
# =========================

train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["label"],
    random_state=42
)

train_df = train_df.reset_index(drop=True)
val_df   = val_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)

# =========================
# 3. Build full file paths
# =========================

def build_path(row):
    return os.path.join(IMG_DIR, row["label"], row["id"])

train_df["filepath"] = train_df.apply(build_path, axis=1)
val_df["filepath"]   = val_df.apply(build_path, axis=1)
test_df["filepath"]  = test_df.apply(build_path, axis=1)

# =========================
# 4. ImageDataGenerator
# =========================

IMG_SIZE = (128, 128)
BATCH_SIZE = 8

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=None,
    x_col="filepath",
    y_col="label",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_generator = test_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=None,
    x_col="filepath",
    y_col="label",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=None,
    x_col="filepath",
    y_col="label",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# =========================
# 5. Class info
# =========================

class_indices = train_generator.class_indices
num_classes = len(class_indices)

# This code below that is commented out is is the old version before using MobileNetV2
# =========================
# 6. Build CNN model
# =========================

#model = Sequential([
#    Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
#    MaxPooling2D((2, 2)),
#
#    Conv2D(64, (3, 3), activation="relu"),
#   MaxPooling2D((2, 2)),
#
#    Conv2D(128, (3, 3), activation="relu"),
#    MaxPooling2D((2, 2)),
#
#
#   Flatten(),
#    Dense(128, activation="relu"),
#    Dropout(0.5),
#    Dense(num_classes, activation="softmax")
#])

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D

base = MobileNetV2(include_top=False, input_shape=(128,128,3), weights="imagenet")
base.trainable = False  # freeze base model

model = Sequential([
    base,
    GlobalAveragePooling2D(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(num_classes, activation="softmax")
])


model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# =========================
# 7. Train model
# =========================

EPOCHS = 70

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)

# =========================
# 8. Evaluate on test set
# =========================

print("\n=== Predicting on test set ===")
y_prob = model.predict(test_generator)
y_pred = np.argmax(y_prob, axis=1)

y_true = test_df["label"].map(class_indices).values

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

print("\n=== Test Metrics ===")
print("Accuracy :", acc)
print("Precision:", prec)
print("Recall   :", rec)
print("F1-score :", f1)

print("\n=== Classification Report ===")
idx_to_class = {v: k for k, v in class_indices.items()}
target_names = [idx_to_class[i] for i in range(num_classes)]
print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

print("\n=== Confusion Matrix ===")
cm = confusion_matrix(y_true, y_pred)
print(cm)
