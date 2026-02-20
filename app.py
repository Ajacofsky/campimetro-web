import streamlit as st
import cv2
import numpy as np
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(layout="wide", page_title="Peritaje Pro")
st.title("⚖️ Analizador de Incapacidad Visual")

st.sidebar.header("⚙️ Configuración")
img_file = st.sidebar.file_uploader("Subir Campo Visual", type=['jpg', 'png', 'jpeg'])

if img_file is not None:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    h, w = img.shape[:2]
    
    st.write("### 1. Ajuste de Centro: Clic en el cruce central del informe")
    coords = streamlit_image_coordinates(img, key="peritaje")
    cx = coords["x"] if coords else w // 2
    cy = coords["y"] if coords else h // 2
    
    # Controles laterales
    radio_10 = st.sidebar.slider("Tamaño de Grilla (Pixeles por 10°)", 10, 200, 55)
    umbral = st.sidebar.slider("Sensibilidad (Umbral de Negros)", 50, 255, 110)
    st.sidebar.info("Ajustá el 'Umbral' hasta que las zonas rojas coincidan con los puntos negros del informe.")

    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binaria = cv2.threshold(gris, umbral, 255, cv2.THRESH_BINARY_INV)
    img_viz = img.copy()
    total_grados = 0
    
    # Procesar 6 anillos (hasta 60 grados)
    for i in range(6):
        r_ext = (i + 1) * radio_10
        r_int = i * radio_10
        
        for a_idx in range(8):
            alpha = a_idx * 45
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(mask, (cx, cy), (int(r_ext), int(r_ext)), 0, alpha, alpha + 45, 255, -1)
            cv2.circle(mask, (cx, cy), int(r_int), 0, -1)
            
            # Dibujar la grilla base tenue
            cv2.ellipse(img_viz, (cx, cy), (int(r_ext), int(r_ext)), 0, alpha, alpha + 45, (200, 200, 200), 1)
            
            # Solo sumar puntos en los primeros 5 anillos (hasta 50°)
