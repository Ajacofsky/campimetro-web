import streamlit as st
import cv2
import numpy as np
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(layout="wide", page_title="Peritaje Pro")
st.title("Analizador de Incapacidad")

img_file = st.sidebar.file_uploader("Subir Campo Visual", type=['jpg', 'png', 'jpeg'])

    if not img_file:
st.info("Sube una imagen para empezar")
st.stop()

file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
img = cv2.imdecode(file_bytes, 1)
h, w = img.shape[:2]

st.write("Clic en el centro (0,0)")
coords = streamlit_image_coordinates(img, key="peritaje")

cx = coords["x"] if coords else w // 2
cy = coords["y"] if coords else h // 2

radio_10 = st.sidebar.slider("Pixeles por cada 10 unidades", 10, 200, 55)
umbral_negro = st.sidebar.slider("Umbral Negros", 50, 255, 110)

gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binaria = cv2.threshold(gris, umbral_negro, 255, cv2.THRESH_BINARY_INV)
img_viz = img.copy()
total_grados = 0

    for i in range(6):
r_ext = (i + 1) * radio_10
r_int = i * radio_10
    for a_idx in range(8):
alpha = a_idx * 45
mask = np.zeros((h, w), dtype=np.uint8)
cv2.ellipse(mask, (cx, cy), (int(r_ext), int(r_ext)), 0, alpha, alpha + 45, 255, -1)
cv2.circle(mask, (cx, cy), int(r_int), 0, -1)
    if i < 5:
a_n = np.count_nonzero(cv2.bitwise_and(binaria, mask))
a_t = np.count_nonzero(mask)
    if a_t > 0 and (a_n / a_t) > 0.04:
puntos = 10 if (a_n / a_t) >= 0.70 else 5
total_grados += puntos
color = (0, 0, 255) if puntos == 10 else (0, 165, 255)
overlay = img_viz.copy()
cv2.ellipse(overlay, (cx, cy), (int(r_ext), int(r_ext)), 0, alpha, alpha + 45, color, -1)
img_viz = cv2.addWeighted(overlay, 0.4, img_viz, 0.6, 0)
cv2.ellipse(img_viz, (cx, cy), (int(r_ext), int(r_ext)), 0, alpha, alpha + 45, (130, 130, 130), 1)

cv2.drawMarker(img_viz, (cx, cy), (255, 0, 0), cv2.MARKER_CROSS, 40, 2)
st.image(img_viz, use_container_width=True)

st.metric("Suma Base de Grados", str(total_grados))
