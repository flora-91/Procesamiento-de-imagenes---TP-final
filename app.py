import mediapipe as mp



# Procesamiento de imágenes
import cv2
import numpy as np

# MediaPipe
import mediapipe as mp

# Interfaz
import gradio as gr

# Utilidades matemáticas
import math

# Inicializar MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Cargar la imagen
imagen = cv2.imread('persona.jpg')
imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
print("Pose test")
# Detectar pose
with mp_pose.Pose(static_image_mode=True) as pose:
    resultados = pose.process(imagen_rgb)

    if resultados.pose_landmarks:
        mp_drawing.draw_landmarks(
            imagen,
            resultados.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )
# if resultados.pose_landmarks:

#      cadera = resultados.pose_landmarks.landmark[
#         mp_pose.PoseLandmark.LEFT_HIP
#     ]

# rodilla = resultados.pose_landmarks.landmark[
#         mp_pose.PoseLandmark.LEFT_KNEE
#     ]

# tobillo = resultados.pose_landmarks.landmark[
#         mp_pose.PoseLandmark.LEFT_ANKLE
#     ]

# print("Cadera:")
# print("x =", cadera.x)
# print("y =", cadera.y)

# print("\nRodilla:")
# print("x =", rodilla.x)
# print("y =", rodilla.y)

# print("\nTobillo:")
# print("x =", tobillo.x)
# print("y =", tobillo.y)

if resultados.pose_landmarks:

    cadera = [
        resultados.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x,
        resultados.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y
    ]

    rodilla = [
        resultados.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x,
        resultados.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y
    ]

    tobillo = [
        resultados.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x,
        resultados.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y
    ]

    print("Cadera:", cadera)
    print("Rodilla:", rodilla)
    print("Tobillo:", tobillo)

print("Pose detectada")
cv2.imshow("Pose detectada", imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()

