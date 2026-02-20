import streamlit as st
import cv2
import numpy as np
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(layout="wide", page_title="Peritaje Pro")
st.title("⚖️ Analizador de Incapacidad (Grilla 5°/10°)")

st.sidebar.header("⚙️ Configuración")
img_file = st.sidebar.file_uploader("Subir Campo Visual", type=['jpg', 'png', 'jpeg'])

if img_file is not None:
file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
img = cv2.imdecode(file_bytes, 1)
h, w = img.shape[:2]
st.write("### 1. Centrado: Clic en el cruce de los ejes (0,0)")
coords = streamlit_image_coordinates(img, key="pil")
if coords:
cx, cy = coords["x"], coords["y"]
else:
cx, cy = w // 2, h // 2
radio_10 = st.sidebar.slider("Píxeles por cada 10°", 10, 200, 55)
umbral_negro = st.sidebar.slider("Umbral Negros", 50, 255, 110)
fn_input = st.sidebar.text_input("Falsos Negativos (ej. 0/4)", "0/4")
gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binaria = cv2.threshold(gris, umbral_negro, 255, cv2.THRESH_BINARY_INV)
img_viz = img.copy()
radios_px = [i * radio_10 for i in range(1, 7)]
angulos = np.arange(0, 360, 45)
total_grados = 0
for i in range(len(radios_px)):
r_ext = radios_px[i]
r_int = 0 if i == 0 else radios_px[i-1]
bg_val = 255 if i % 2 == 0 else 225
for alpha in angulos:
mask = np.zeros((h, w), dtype=np.uint8)
cv2.ellipse(mask, (cx, cy), (int(r_ext), int(r_ext)), 0, alpha, alpha + 45, 255, -1)
if r_int > 0:
cv2.circle(mask, (cx, cy), int(r_int), 0, -1)
mask_indices = np.where(mask > 0)
img_viz[mask_indices] = cv2.addWeighted(img_viz[mask_indices], 0.85, np.full(img_viz[mask_indices].shape, bg_val, dtype=np.uint8), 0.15, 0)
if i < 5:
area_n = np.count_nonzero(cv2.bitwise_and(binaria, mask))
area_t = np.count_nonzero(mask)
if area_t > 0:
ocu = (area_n / area_t) * 100
if ocu > 4:
puntos = 10 if ocu >= 70 else 5
total_grados += puntos
color = (0, 0, 255) if ocu >= 70 else (0, 165, 255)
overlay = img_viz.copy()
cv2.ellipse(overlay, (cx, cy), (int(r_ext), int(r_ext)), 0, alpha, alpha + 45, color, -1)
if r_int > 0:
cv2.circle(overlay, (cx, cy), int(r_int), (0,0,0), -1)
img_viz = cv2.addWeighted(overlay, 0.4, img_viz, 0.6, 0)
cv2.ellipse(img_viz, (cx, cy), (int(r_ext), int(r_ext)), 0, alpha, alpha + 45, (130, 130, 130), 1)
cv2.drawMarker(img_viz, (cx, cy), (255, 0, 0), cv2.MARKER_CROSS, 40, 2)
st.image(img_viz, use_container_width=True)
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
