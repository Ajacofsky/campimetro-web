import streamlit as st
import cv2
import numpy as np
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_config = st.set_page_config(layout="wide", page_title="Peritaje Pro")
st.title("⚖️ Analizador de Incapacidad (Grilla 5°/10°)")
st.sidebar.header("⚙️ Configuración")
img_file = st.sidebar.file_uploader("Subir Campo Visual", type=['jpg', 'png', 'jpeg'])

if img_file is None:
st.info("Subí una imagen para comenzar")
st.stop()

file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
img = cv2.imdecode(file_bytes, 1)
h, w = img.shape[:2]
st.write("### 1. Centrado: Clic en el centro (0,0)")
coords = streamlit_image_coordinates(img, key="peritaje")

cx = coords["x"] if coords else w // 2
cy = coords["y"] if coords else h // 2

radio_10 = st.sidebar.slider("Píxeles por cada 10°", 10, 200, 55)
umbral_negro = st.sidebar.slider("Umbral Negros", 50, 255, 110)
fn_input = st.sidebar.text_input("Falsos Negativos (ej. 0/4)", "0/4")

gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binaria = cv2.threshold(gris, umbral_negro, 255, cv2.THRESH_BINARY_INV)
img_viz = img.copy()
total_grados = 0
Procesamiento de 6 anillos (10° a 60°) y 8 sectores por anillo

for i in range(6):
r_ext = (i + 1) * radio_10
r_int = i * radio_10
bg_val = 255 if i % 2 == 0 else 225
for a_idx in range(8):
alpha = a_idx * 45
mask = np.zeros((h, w), dtype=np.uint8)
cv2.ellipse(mask, (cx, cy), (int(r_ext), int(r_ext)), 0, alpha, alpha + 45, 255, -1)
cv2.circle(mask, (cx, cy), int(r_int), 0, -1)

cv2.drawMarker(img_viz, (cx, cy), (255, 0, 0), cv2.MARKER_CROSS, 40, 2)
st.image(img_viz, use_container_width=True)
Lógica de Falsos Negativos y Cálculo Final

f = 1.0
if "/" in fn_input:
pts = fn_input.split('/')
if len(pts) == 2 and float(pts[1]) > 0:
f = (float(pts[0])/float(pts[1])) + 1

g_final = min(total_grados * f, 320)
p_perdida = (g_final / 320) * 100
inc = p_perdida * 0.25

st.divider()
c1, c2, c3 = st.columns(3)
c1.metric("Suma Base", f"{total_grados}°")
c2.metric("Pérdida de Campo", f"{round(p_perdida, 2)}%")
c3.metric("Incapacidad Final", f"{round(inc, 2)}%")
