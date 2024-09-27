from PIL import Image, ImageDraw, ImageFont
import face_recognition
import numpy as np

image_simity = face_recognition.load_image_file("simity.jpeg")
encodage_visage_simity = face_recognition.face_encodings(image_simity)[0]
image_issa = face_recognition.load_image_file("issa.jpeg")
encodage_visage_issa = face_recognition.face_encodings(image_issa)[0]

encodage_visage_connu = [encodage_visage_simity, encodage_visage_issa]
nom_visage_connu = ["simity.jpeg", "issa.jpeg"]

image_inconnu = face_recognition.load_image_file("inconnu.jpeg")
emplacement_visage_inconnu = face_recognition.face_locations(image_inconnu)
encodage_visage_inconnu = face_recognition.face_encodings(image_inconnu, emplacement_visage_inconnu)

image_pil = Image.fromarray(image_inconnu)
draw = ImageDraw.Draw(image_pil)

for (haut, droite, bas, gauche), encodage_visage in zip(emplacement_visage_inconnu, encodage_visage_inconnu):
    correspond = face_recognition.compare_faces(encodage_visage_connu, encodage_visage)
    nom = "Inconnu"
    distance_visage = face_recognition.face_distance(encodage_visage_connu, encodage_visage)
    meilleur_indice = np.argmin(distance_visage)
    if correspond[meilleur_indice]:
        nom = nom_visage_connu[meilleur_indice]

    draw.rectangle(((gauche, haut), (droite, bas)), outline=(0, 0, 255))
    largeur_texte, hauteur_texte = draw.textbbox((0, 0), nom, font=ImageFont.truetype("arial.ttf", 16))[2:]
    draw.text((gauche + 6, bas - hauteur_texte - 5), nom, fill=(255, 255, 255))

image_pil.show()
