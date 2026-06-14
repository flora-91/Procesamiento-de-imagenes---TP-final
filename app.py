import cv2
import numpy as np
import mediapipe as mp
import gradio as gr

# =====================================
# MEDIAPIPE
# =====================================

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# =====================================
# UTILIDADES
# =====================================

def calcular_angulo(a, b, c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = (
        np.arctan2(c[1] - b[1], c[0] - b[0])
        - np.arctan2(a[1] - b[1], a[0] - b[0])
    )

    angulo = np.abs(radians * 180 / np.pi)

    if angulo > 180:
        angulo = 360 - angulo

    return angulo


def obtener_punto(landmarks, punto):

    return [
        landmarks.landmark[punto].x,
        landmarks.landmark[punto].y
    ]


# =====================================
# DETECCIÓN DE EJERCICIOS
# =====================================

def detectar_ejercicio(landmarks):

    hombro = obtener_punto(
        landmarks,
        mp_pose.PoseLandmark.LEFT_SHOULDER
    )

    cadera = obtener_punto(
        landmarks,
        mp_pose.PoseLandmark.LEFT_HIP
    )

    rodilla = obtener_punto(
        landmarks,
        mp_pose.PoseLandmark.LEFT_KNEE
    )

    tobillo = obtener_punto(
        landmarks,
        mp_pose.PoseLandmark.LEFT_ANKLE
    )

    codo = obtener_punto(
        landmarks,
        mp_pose.PoseLandmark.LEFT_ELBOW
    )

    muñeca = obtener_punto(
        landmarks,
        mp_pose.PoseLandmark.LEFT_WRIST
    )

    # ==========================
    # Ángulos
    # ==========================

    angulo_rodilla = calcular_angulo(
        cadera,
        rodilla,
        tobillo
    )

    angulo_codo = calcular_angulo(
        hombro,
        codo,
        muñeca
    )

    angulo_cuerpo = calcular_angulo(
        hombro,
        cadera,
        tobillo
    )

    # ==========================
    # DEBUG
    # ==========================

    print("\n========== DEBUG ==========")
    print("Rodilla:", round(angulo_rodilla, 1))
    print("Codo:", round(angulo_codo, 1))
    print("Cuerpo:", round(angulo_cuerpo, 1))
    print("===========================\n")

    # ==========================
    # EJERCICIOS HORIZONTALES
    # ==========================

    if angulo_cuerpo > 150:

        # PLANCHA

        if angulo_codo > 150:
            return "Plancha"

        # FLEXION

        if angulo_codo <= 150:
            return "Flexión"

    # ==========================
    # EJERCICIOS VERTICALES
    # ==========================

    distancia_piernas = abs(
        tobillo[0] - rodilla[0]
    )

    # ESTOCADA

    if (
        distancia_piernas > 0.15
        and angulo_rodilla < 130
    ):
        return "Estocada"

    # SENTADILLA

    if angulo_rodilla < 150:
        return "Sentadilla"

    return "No reconocido"


# =====================================
# ANÁLISIS SENTADILLA
# =====================================

def analizar_sentadilla(angulo_rodilla):

    if angulo_rodilla < 70:

        return (
            "⚠️ Sentadilla muy profunda",
            "Controlá un poco más el descenso."
        )

    elif angulo_rodilla <= 110:

        return (
            "✅ Sentadilla correcta",
            "Excelente técnica."
        )

    else:

        return (
            "⚠️ Sentadilla incompleta",
            "Intentá bajar un poco más."
        )


# =====================================
# ANÁLISIS ESTOCADA
# =====================================

def analizar_estocada(angulo_rodilla):

    if 80 <= angulo_rodilla <= 110:

        return (
            "✅ Estocada correcta",
            "Buena profundidad."
        )

    return (
        "⚠️ Estocada mejorable",
        "Intentá acercarte a los 90°."
    )


# =====================================
# ANÁLISIS FLEXIÓN
# =====================================

def analizar_flexion(angulo_codo):

    if angulo_codo < 90:

        return (
            "✅ Flexión correcta",
            "Excelente profundidad."
        )

    return (
        "⚠️ Flexión incompleta",
        "Bajá un poco más el torso."
    )


# =====================================
# ANÁLISIS PLANCHA
# =====================================

def analizar_plancha():

    return (
        "✅ Plancha detectada",
        "Mantené la alineación corporal."
    )


# =====================================
# FUNCIÓN PRINCIPAL
# =====================================

def analizar_ejercicio(imagen):

    imagen_bgr = cv2.cvtColor(
        imagen,
        cv2.COLOR_RGB2BGR
    )

    with mp_pose.Pose(
        static_image_mode=True,
        min_detection_confidence=0.5
    ) as pose:

        resultados = pose.process(imagen)

        if not resultados.pose_landmarks:

            return imagen, """
# ❌ No se detectó ninguna persona
"""

        mp_drawing.draw_landmarks(
            imagen_bgr,
            resultados.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        landmarks = resultados.pose_landmarks

        ejercicio = detectar_ejercicio(
            landmarks
        )

        hombro = obtener_punto(
            landmarks,
            mp_pose.PoseLandmark.LEFT_SHOULDER
        )

        cadera = obtener_punto(
            landmarks,
            mp_pose.PoseLandmark.LEFT_HIP
        )

        rodilla = obtener_punto(
            landmarks,
            mp_pose.PoseLandmark.LEFT_KNEE
        )

        tobillo = obtener_punto(
            landmarks,
            mp_pose.PoseLandmark.LEFT_ANKLE
        )

        codo = obtener_punto(
            landmarks,
            mp_pose.PoseLandmark.LEFT_ELBOW
        )

        muñeca = obtener_punto(
            landmarks,
            mp_pose.PoseLandmark.LEFT_WRIST
        )

        angulo_rodilla = calcular_angulo(
            cadera,
            rodilla,
            tobillo
        )

        angulo_codo = calcular_angulo(
            hombro,
            codo,
            muñeca
        )

        # --------------------------------

        if ejercicio == "Sentadilla":

            estado, consejo = analizar_sentadilla(
                angulo_rodilla
            )

        elif ejercicio == "Estocada":

            estado, consejo = analizar_estocada(
                angulo_rodilla
            )

        elif ejercicio == "Flexión":

            estado, consejo = analizar_flexion(
                angulo_codo
            )

        elif ejercicio == "Plancha":

            estado, consejo = analizar_plancha()

        else:

            estado = "❌ Ejercicio no reconocido"
            consejo = (
                "Probá con una sentadilla, "
                "estocada, flexión o plancha."
            )

        resultado = f"""
# 🏋️ Analizador de Ejercicios

### Ejercicio detectado
**{ejercicio}**

### Resultado
{estado}

### Recomendación
{consejo}
"""

        imagen_salida = cv2.cvtColor(
            imagen_bgr,
            cv2.COLOR_BGR2RGB
        )

        return imagen_salida, resultado


# =====================================
# INTERFAZ
# =====================================

css = """
footer {
    display:none;
}
"""

with gr.Blocks(
    css=css,
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown(
        "# 🏋️ Analizador de Ejercicios"
    )

    with gr.Row():

        entrada = gr.Image(
            type="numpy",
            label="📸 Imagen"
        )

        salida_imagen = gr.Image(
            label="🎯 Resultado"
        )

    salida_texto = gr.Markdown()

    boton = gr.Button(
        "Analizar",
        variant="primary"
    )

    boton.click(
        analizar_ejercicio,
        inputs=entrada,
        outputs=[
            salida_imagen,
            salida_texto
        ]
    )

demo.launch()