import os
import cv2 as cv
import numpy as np
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score

from SVM import SVM 

BASE_DIR = "chest_xray/chest_xray"

def collect_data(split="train"):
    normal = "NORMAL"
    pneumonia = "PNEUMONIA"

    images = []
    labels = []

    # Tải ảnh Normal
    normal_dir = os.path.join(BASE_DIR, split, normal)
    if os.path.exists(normal_dir):
        for img_file in tqdm(os.listdir(normal_dir), desc=f"Loading {split} - Normal"):
            img_path = os.path.join(normal_dir, img_file)
            img = cv.imread(img_path)
            if img is not None:
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                img = cv.resize(img, (128, 128), interpolation=cv.INTER_LINEAR_EXACT)
                images.append(img.flatten())
                labels.append(-1)

    # Tải ảnh Pneumonia
    pneumonia_dir = os.path.join(BASE_DIR, split, pneumonia)
    if os.path.exists(pneumonia_dir):
        for img_file in tqdm(os.listdir(pneumonia_dir), desc=f"Loading {split} - Pneumonia"):
            img_path = os.path.join(pneumonia_dir, img_file)
            img = cv.imread(img_path)
            if img is not None:
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                img = cv.resize(img, (128, 128), interpolation=cv.INTER_LINEAR_EXACT)
                images.append(img.flatten())
                labels.append(1)

    X = np.stack(images, axis=0)
    X = (X - X.mean()) / X.std()
    y = np.array(labels)

    return X, y

if __name__ == "__main__":
    X_train, y_train = collect_data("train")
    X_test, y_test = collect_data("test")


    # ASSIGNMENT 1:
    print("\nASSIGNMENT 1")
    model = SVM(
        C=1.0,
        lr=0.01,
        n_iterations=1000
    )

    model.fit(X_train, y_train)

    scores = model.get_metrics(X_test, y_test)
    for scorer in scores:
        score = scores[scorer]
        print(f"{scorer}: {score}")

    # ASSIGNMENT 2
    print("\nASSIGNMENT 2")
    sklearn_model = SVC(kernel='linear', C=1.0)
    sklearn_model.fit(X_train, y_train)

    y_pred_sklearn = sklearn_model.predict(X_test)
    
    p_sk = precision_score(y_test, y_pred_sklearn, zero_division=0)
    r_sk = recall_score(y_test, y_pred_sklearn, zero_division=0)
    f1_sk = f1_score(y_test, y_pred_sklearn, zero_division=0)

    print(f"Precision: {p_sk}")
    print(f"Recall: {r_sk}")
    print(f"F1: {f1_sk}")