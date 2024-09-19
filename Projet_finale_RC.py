import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import VGG16
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import cv2
import matplotlib.pyplot as plt


# Charger le modèle VGG16 pré-entrainé, sans les couches fully-connected
def load_feature_extractor():
    base_model = VGG16(weights='imagenet', include_top=False)
    return base_model


# Extraire les caractéristiques d'une image à partir de VGG16
def extract_features(model, img_path):
    img = cv2.imread(img_path)

    if img is None:
        raise ValueError(f"L'image à {img_path} n'a pas pu être chargée.")

    # Affichage de l'image avant redimensionnement
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # OpenCV charge en BGR, donc converti en RGB pour Matplotlib
    plt.title(f"Image: {img_path.split('/')[-1]}")
    plt.axis('off')  # Pas d'axes pour une image propre
    plt.show()

    img = cv2.resize(img, (224, 224))  # Assurez-vous que l'image est de la bonne taille pour VGG16
    img = img.astype('float32')
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.vgg16.preprocess_input(img)

    # Extraire les caractéristiques
    features = model.predict(img)
    return features.flatten()


# Préparer le modèle SVM pour la reconnaissance faciale
def prepare_svm_model():
    scaler = StandardScaler()
    clf = SVC(kernel='linear', probability=True)
    return clf, scaler


# Entraîner le modèle avec des caractéristiques
def train_model(clf, scaler, features, labels):
    # Normaliser les caractéristiques
    scaled_features = scaler.fit_transform(features)
    clf.fit(scaled_features, labels)
    return clf, scaler


# Prédire le label à partir des caractéristiques faciales
def predict_face(clf, scaler, face_features):
    # Convertir les caractéristiques en NumPy
    face_features_np = np.array(face_features).reshape(1, -1)

    # Normaliser les caractéristiques
    face_features_np = scaler.transform(face_features_np)

    # Prédire le label
    prediction = clf.predict(face_features_np)
    confidence = clf.predict_proba(face_features_np)

    return prediction, confidence


# Charger le modèle d'extraction de caractéristiques
feature_extractor = load_feature_extractor()

# Extraire des caractéristiques des images d'entraînement
image_paths = [
    r'C:\Users\Latitude 5490\PycharmProjects\pythonProject\Projet simity\simity.jpeg',
    r'C:\Users\Latitude 5490\PycharmProjects\pythonProject\Projet simity\issa.jpeg'
]  # Liste des images
features = []
labels = ['Simity', 'Issa']  # Labels correspondants

for img_path in image_paths:
    features.append(extract_features(feature_extractor, img_path))

# Préparer le modèle SVM et scaler
clf, scaler = prepare_svm_model()

# Entraîner le modèle SVM
clf, scaler = train_model(clf, scaler, features, labels)

# Extraire les caractéristiques de la nouvelle image pour la prédiction
new_image_path = r'C:\Users\Latitude 5490\PycharmProjects\pythonProject\Projet simity\inconnu.jpeg'
new_face_features = extract_features(feature_extractor, new_image_path)

# Prédire le label de la nouvelle image
prediction, confidence = predict_face(clf, scaler, new_face_features)

print(f"Prediction: {prediction[0]}")
print(f"Confidence: {confidence}")
