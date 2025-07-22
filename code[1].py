import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.utils import to_categorical

def load_images_and_labels(data_dir, image_size=(224, 224)):
    X, y = [], []
    label_map = {}
    label_idx = 0

    if not os.path.exists(data_dir):
        raise ValueError(f"Dataset path does not exist: {data_dir}")

    for class_folder in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_folder)
        if os.path.isdir(class_path):
            if class_folder not in label_map:
                label_map[class_folder] = label_idx
                label_idx += 1
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.resize(img, image_size)
                X.append(img)
                y.append(label_map[class_folder])

    if not X or not y:
        raise ValueError("No valid images found.")

    return np.array(X), to_categorical(np.array(y)), label_map

def scale_attention(x):
    filters = x.shape[-1]
    attn = Conv2D(filters, kernel_size=1, activation='sigmoid')(x)
    return x * attn

def axis_attention(x):
    v = Conv2D(x.shape[-1], (1, 3), padding='same', activation='relu')(x)
    h = Conv2D(x.shape[-1], (3, 1), padding='same', activation='relu')(x)
    return concatenate([v, h], axis=-1)

def build_customizednet121(input_shape=(224, 224, 3), num_classes=25):
    base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.output
    x = scale_attention(x)
    x = axis_attention(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=base_model.input, outputs=output)

def train_model(model, X_train, y_train, X_test, y_test):
    model.compile(optimizer=Adam(learning_rate=0.001, decay=0.0005),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test))
    return model, history

def evaluate_model(model, X_test, y_test):
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    cm = confusion_matrix(y_true, y_pred)
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    TN = cm.sum() - (TP + FP + FN)

    specificity = np.mean(TN / (TN + FP + 1e-10))
    misdetection_rate = np.mean(FN / (TP + FN + 1e-10))
    confidence_score = np.mean(np.max(y_pred_probs, axis=1))
    growth_index = (recall + specificity) / 2
    npv = np.mean(TN / (TN + FN + 1e-10))

    print("\n=== Final Evaluation Metrics ===")
    print(f"Accuracy                 : {acc * 100:.2f}%")
    print(f"Precision (Macro Avg)    : {precision * 100:.2f}%")
    print(f"Recall (Sensitivity)     : {recall * 100:.2f}%")
    print(f"F1-Score (Macro Avg)     : {f1 * 100:.2f}%")
    print(f"Specificity              : {specificity * 100:.2f}%")
    print(f"Misdetection Rate        : {misdetection_rate * 100:.2f}%")
    print(f"Confidence Score (Avg)   : {confidence_score * 100:.2f}%")
    print(f"Growth Development Index : {growth_index * 100:.2f}%")
    print(f"Negative Predictive Value: {npv * 100:.2f}%")

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "specificity": specificity,
        "misdetection_rate": misdetection_rate,
        "confidence_score": confidence_score,
        "growth_index": growth_index,
        "npv": npv
    }

if __name__ == "__main__":
    data_dir = r'"Enter\your\dataset\path\here\FNNPK'

    X, y, label_map = load_images_and_labels(data_dir)
    X = X.astype('float32') / 255.0
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_customizednet121(input_shape=(224, 224, 3), num_classes=y.shape[1])
    model, history = train_model(model, X_train, y_train, X_test, y_test)
    evaluate_model(model, X_test, y_test)
    model.save("JO_CustomizedNet121_Hydroponic.h5") 