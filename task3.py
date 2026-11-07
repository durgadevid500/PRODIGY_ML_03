
import os
import numpy as np
import matplotlib.pyplot as plt
import joblib

from PIL import Image
from skimage.feature import hog

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Dataset path
DATASET_PATH = r"C:\Users\hp\Downloads\train\train"

# 2. Initialize lists
images = []
labels = []

cat_count = 0
dog_count = 0
max_images = 500

print("Loading images...")

# 3. Load images
for filename in os.listdir(DATASET_PATH):

    if filename.lower().startswith("cat."):
        if cat_count >= max_images:
            continue

        label = 0
        cat_count += 1

    elif filename.lower().startswith("dog."):
        if dog_count >= max_images:
            continue

        label = 1
        dog_count += 1

    else:
        continue

    try:
        image_path = os.path.join(DATASET_PATH, filename)

        img = Image.open(image_path).convert("L")
        img = img.resize((64, 64))

        images.append(np.array(img))
        labels.append(label)

    except Exception as e:
        print("Error:", filename, e)

print("\nCats loaded:", cat_count)
print("Dogs loaded:", dog_count)
print("Total images:", len(images))

# 4. Convert to NumPy arrays
X = np.array(images)
y = np.array(labels)

print("Image shape:", X.shape)

# 5. Extract HOG features
print("\nExtracting HOG features...")

hog_features = []

for i, img in enumerate(X):

    features = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True
    )

    hog_features.append(features)

    if (i + 1) % 100 == 0:
        print("Processed:", i + 1, "/", len(X))

X_hog = np.array(hog_features)

print("HOG feature shape:", X_hog.shape)

# 6. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_hog,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining images:", len(X_train))
print("Testing images:", len(X_test))

# 7. Create SVM model
model = LinearSVC(
    C=1.0,
    dual=False,
    max_iter=10000,
    random_state=42
)

# 8. Train model
print("\nTraining HOG + SVM model...")

model.fit(X_train, y_train)

print("Training completed!")

# 9. Predictions
y_pred = model.predict(X_test)

# 10. Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nHOG + SVM Accuracy:",
      round(accuracy * 100, 2), "%")

# 11. Classification report
print("\nClassification Report:")

print(classification_report(
    y_test,
    y_pred,
    target_names=["Cat", "Dog"]
))

# 12. Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# 13. Save model
joblib.dump(
    model,
    "cat_dog_hog_svm_model.pkl"
)

print("\nModel saved successfully!")

# 14. Display sample predictions
plt.figure(figsize=(12, 5))

for i in range(5):

    plt.subplot(1, 5, i + 1)

    plt.imshow(X_test[i].reshape(-1, 1)
               if False else X_test[i][:1].reshape(1, 1),
               cmap="gray") if False else None

    # Display original test image
    actual_image_index = np.where(
        np.all(X_hog == X_test[i], axis=1)
    )[0][0]

    plt.imshow(X[actual_image_index], cmap="gray")

    prediction = "Dog" if y_pred[i] == 1 else "Cat"
    actual = "Dog" if y_test[i] == 1 else "Cat"

    plt.title(f"Pred: {prediction}\nActual: {actual}")
    plt.axis("off")

plt.suptitle("Cats vs Dogs - HOG + SVM")
plt.tight_layout()
plt.show()

print("\nTASK 03 COMPLETED!")
