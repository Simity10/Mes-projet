import cv2
import numpy as np
import face_recognition
from tensorflow.keras.models import load_model

# 1. Collecte et prétraitement des données
face_cascade = cv2.CascadeClassifier(r'C:\Users\Latitude 5490\PycharmProjects\pythonProject\Projet simity\haarcascade_frontalface_default .xml')

if face_cascade.empty():
    raise ValueError("Le fichier de cascade n'a pas été chargé correctement. Veuillez vérifier le chemin.")

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur lors de la capture de la vidéo")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 2. Caractéristiques d’extraction avec FaceNet/VGG16
model = load_model(r'C://Users//Latitude 5490//PycharmProjects//pythonProject//Projet simity//facenet_keras.h5', compile=False)

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Erreur lors du chargement de l'image : {image_path}")
    img = cv2.resize(img, (160, 160))
    img = img.astype('float32')
    mean, std = img.mean(), img.std()
    img = (img - mean) / std
    img = np.expand_dims(img, axis=0)
    return img

def get_embedding(model, image_path):
    img = preprocess_image(image_path)
    embedding = model.predict(img)
    return embedding

embedding_1 = get_embedding(model, 'mes_images/simity.jpeg')
embedding_2 = get_embedding(model, 'mes_images/issa.jpeg')

# 3. Modélisation et déploiement
distance = np.linalg.norm(embedding_1 - embedding_2)
print("Distance entre les embeddings :", distance)

threshold = 0.5
if distance < threshold:
    print("Les visages correspondent")
else:
    print("Les visages ne correspondent pas")
