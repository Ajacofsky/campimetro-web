import streamlit as st
import cv2
import numpy as np
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(layout="wide", page_title="Peritaje Pro")
st.title("⚖️ Analizador de Incapacidad Visual")

# Configuración en el sidebar
st.sidebar.header("⚙️ Configuración")
img_file = st.sidebar.file_uploader("Subir Campo Visual", type=['jpg', 'png', 'jpeg'])

if img_file is not None:
    # Leer y decodificar imagen
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    h, w = img.shape[:2]
    
    # 1. Interacción para el centro
    st.write("### 1. Ajuste de Centro: Clic en el cruce central")
    coords = streamlit_image_coordinates(img, key="peritaje")
    cx = coords["x"] if coords else w // 2
    cy = coords["y"] if coords else h // 2
    
    # Parámetros ajustables
    radio_10 = st.sidebar.slider("Tamaño de Grilla (Pixeles por 10°)", 10, 200, 55)
    umbral = st.sidebar.slider("Sensibilidad (Umbral de Negros)", 50, 255, 110)
    
    # Procesamiento de imagen
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binaria = cv2.threshold(gris, umbral, 255, cv2.THRESH_BINARY_INV)
    img_viz = img.copy()
    total_grados = 0
    
    # Dibujar y calcular
    for i in range(6):
        r_ext = (i + 1) * radio_10
        r_int = i * radio_10
        for a_idx in range(8):
            alpha = a_idx * 45
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(mask, (cx, cy), (int(r_ext), int(r_ext)), 0, alpha, alpha + 45, 255, -1)
            cv2.circle(mask, (cx, cy), int(r_int), 0, -1)
            
            # Evaluación de puntos (solo hasta 50 grados según baremo)
            if i < 5:
                area_negra = np.count_nonzero(cv2.bitwise_and(binaria, mask))
                area_total = np.count_nonzero(mask)
                if area_total > 0 and (area_negra / area_total) > 0.05:
                    puntos = 10 if (area_negra / area_total) >= 0.60 else 5
                    total_grados += puntos
                    # Pintar sector con transparencia
                    color_bgr = (0, 0, 255) if puntos == 10 else (0, 165, 255)
                    sub_img = img_viz.copy()
                    cv2.ellipse(sub_img, (cx, cy), (int(r_ext), int(r_ext)), 0, alpha, alpha + 45, color_bgr, -1)
                    cv2.circle(sub_img, (cx, cy), int(r_int), (0,0,0), -1)
                    img_viz = cv2.addWeighted(sub_img, 0.3, img_viz, 0.7, 0)
            
            # Dibujar bordes de la grilla
            cv2.ellipse(img_viz, (cx, cy), (int(r_ext), int(r_ext)), 0, alpha, alpha + 45, (150, 150, 150), 1)

    # Marcador central
    cv2.drawMarker(img_viz, (cx, cy), (255, 0, 0), cv2.MARKER_CROSS, 30, 2)
    
    # MOSTRAR IMAGEN PROCESADA
    st.image(img_viz, caption="Análisis de Sectores Detectados", use_container_width=True)
    
    # MOSTRAR CÁLCULOS (Asegurando que sal
