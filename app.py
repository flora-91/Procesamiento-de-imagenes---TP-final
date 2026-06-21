import cv2
import numpy as np
import mediapipe as mp
import gradio as gr

# =====================================
# Función para calcular ángulos
# =====================================
def calcular_angulo(a, b, c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(
        c[1] - b[1],
        c[0] - b[0]
    ) - np.arctan2(
        a[1] - b[1],
        a[0] - b[0]
    )

    angulo = np.abs(radians * 180.0 / np.pi)

    if angulo > 180:
        angulo = 360 - angulo

    return angulo


# =====================================
# Inicializar MediaPipe
# =====================================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# =====================================
# Función principal de análisis
# =====================================
def analizar_sentadilla(imagen):

    imagen = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(
        static_image_mode=True,
        min_detection_confidence=0.5
    ) as pose:

        resultados = pose.process(imagen_rgb)

        if not resultados.pose_landmarks:

            return imagen_rgb, "❌ No se detectó ninguna persona"

        mp_drawing.draw_landmarks(
            imagen,
            resultados.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        cadera = [
            resultados.pose_landmarks.landmark[
                mp_pose.PoseLandmark.LEFT_HIP
            ].x,
            resultados.pose_landmarks.landmark[
                mp_pose.PoseLandmark.LEFT_HIP
            ].y
        ]

        rodilla = [
            resultados.pose_landmarks.landmark[
                mp_pose.PoseLandmark.LEFT_KNEE
            ].x,
            resultados.pose_landmarks.landmark[
                mp_pose.PoseLandmark.LEFT_KNEE
            ].y
        ]

        tobillo = [
            resultados.pose_landmarks.landmark[
                mp_pose.PoseLandmark.LEFT_ANKLE
            ].x,
            resultados.pose_landmarks.landmark[
                mp_pose.PoseLandmark.LEFT_ANKLE
            ].y
        ]

        angulo_rodilla = calcular_angulo(
            cadera,
            rodilla,
            tobillo
        )

        if angulo_rodilla < 70:

            estado = "⚠️ Sentadilla muy profunda"
            consejo = (
                "Controlá el descenso para evitar "
                "una flexión excesiva."
            )

        elif 70 <= angulo_rodilla <= 110:

            estado = "✅ Sentadilla correcta"
            consejo = (
                "Excelente técnica. "
                "La profundidad es adecuada."
            )

        elif 110 < angulo_rodilla <= 140:

            estado = "⚠️ Sentadilla incompleta"
            consejo = (
                "Intentá bajar un poco más "
                "hasta acercarte a los 90°."
            )

        else:

            estado = "❌ No se detecta una sentadilla"
            consejo = (
                "Flexioná más las rodillas "
                "y llevá la cadera hacia atrás."
            )

        resultado = f"""
### Resultado

**Ángulo de rodilla:** {angulo_rodilla:.1f}°

**Estado:** {estado}

**Consejo:** {consejo}
"""

        imagen_salida = cv2.cvtColor(
            imagen,
            cv2.COLOR_BGR2RGB
        )

        return imagen_salida, resultado


# =====================================
# Interfaz Gradio
# =====================================

demo = gr.Interface(
    fn=analizar_sentadilla,
    inputs=gr.Image(type="numpy"),
    outputs=[
        gr.Image(label="Pose detectada"),
        gr.Markdown(label="Análisis")
    ],
    title="🏋️ Analizador de Sentadillas con MediaPipe",
    description="""
Subí una imagen realizando una sentadilla.
La aplicación analizará la pose, calculará el ángulo de la rodilla
y te dará una devolución sobre la técnica.
"""
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860
    )