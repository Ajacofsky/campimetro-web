import streamlit as st
import cv2
import numpy as np
from PIL import Image

def analizar_campo(imagen_file):
    if imagen_file is None:
        return None, 0
    file_bytes = np.asarray(bytearray(imagen_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Umbral fijo para detectar los puntos negros
    _, binaria = cv2.threshold(gris, 127, 255, cv2.THRESH_BINARY_INV)
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    puntos = 0
    for c in contornos:
        area = cv2.contourArea(c)
        if 15 < area < 1000: # Filtro de tamaño para evitar ruido
            puntos += 1
    return binaria, puntos

def calcular_incapacidad(puntos, fn_str):
    try:
        num, den = map(int, fn_str.split('/'))
        factor = num / den if num > 0 else 1
    except:
        factor = 1
    grados = (puntos * 10) * factor
    if grados > 320: grados = 320
    perdida = (grados / 320) * 100
    incapacidad = perdida * 0.25
    return grados, perdida, incapacidad

st.set_page_config(page_title="Peritaje", layout="wide")
st.title("⚖️ Analizador de Incapacidad Campimétrica")

img_file = st.file_uploader("Subir imagen del Campo Visual", type=['png', 'jpg', 'jpeg'])
fn = st.text_input("Falsos Negativos (ej. 0/8)", value="0/8")

if st.button("CALCULAR INCAPACIDAD"):
    if img_file is not None:
        img_bin, total_puntos = analizar_campo(img_file)
        
        st.subheader("Vista de Diagnóstico")
        st.image(img_bin, caption="Lo que la App detecta (en blanco)", width=400)
        
        g, p, inc = calcular_incapacidad(total_puntos, fn)
        
        st.divider()
        st.metric("Puntos Negros Detectados", total_puntos)
        st.metric("Incapacidad Resultante", f"{round(inc, 2)}%")
    else:
        st.error("Por favor, subí una imagen primero.")
