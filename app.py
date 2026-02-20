import streamlit as st
import cv2
import numpy as np
from PIL import Image

def obtener_sectores_geometricos(centro, radio_max):
    # Define la red de 8 bisectrices (cada 45°)
    angulos = np.linspace(0, 315, 8)
    # Define los anillos concéntricos (10°, 20°, 30°, 60°)
    radios = [radio_max * 0.16, radio_max * 0.33, radio_max * 0.5, radio_max]
    return angulos, radios

def analizar_con_grilla(imagen_file, centro_x, centro_y, escala):
    file_bytes = np.asarray(bytearray(imagen_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binaria = cv2.threshold(gris, 110, 255, cv2.THRESH_BINARY_INV)
    
    h, w = binaria.shape
    centro = (int(w * centro_x), int(h * centro_y))
    radio_max = int(min(h, w) * 0.4 * escala)
    
    angulos, radios = obtener_sectores_geometricos(centro, radio_max)
    sectores_fallados = 0
    
    # Dibujar para diagnóstico
    img_viz = cv2.cvtColor(binaria, cv2.COLOR_GRAY2BGR)
    
    # Lógica de detección por sector geométrico
    for i in range(len(radios)):
        r_int = 0 if i == 0 else radios[i-1]
        r_ext = radios[i]
        for alpha in angulos:
            # Creamos una máscara para cada sector de la red
            mask = np.zeros_like(binaria)
            # Dibujamos el sector circular
            cv2.ellipse(mask, centro, (int(r_ext), int(r_ext)), 0, alpha, alpha + 45, 255, -1)
            if r_int > 0:
                cv2.circle(mask, centro, int(r_int), 0, -1)
            
            # Si hay suficiente negro (blanco en la binaria) en este sector, se cuenta
            resultado = cv2.bitwise_and(binaria, mask)
            if np.sum(resultado) > 500: # Umbral de área por sector
                sectores_fallados += 1
                cv2.ellipse(img_viz, centro, (int(r_ext), int(r_ext)), 0, alpha, alpha + 45, (0, 255, 0), 2)
                
    return img_viz, sectores_fallados

st.title("Analizador de Incapacidad Campimétrica")

img_file = st.file_uploader("Subir imagen del estudio", type=['jpg', 'png', 'jpeg'])

if img_file:
    # Controles para centrar la red manualmente sobre la cruz del estudio
    st.sidebar.header("Ajuste de la Red Geométrica")
    cx = st.sidebar.slider("Centro Horizontal", 0.0, 1.0, 0.5)
    cy = st.sidebar.slider("Centro Vertical", 0.0, 1.0, 0.5)
    esc = st.sidebar.slider("Escala de la Red", 0.1, 2.0, 1.0)
    fn = st.sidebar.text_input("Falsos Negativos", "0/8")

    if st.button("CALCULAR SEGÚN PARÁMETROS"):
        img_res, n_sectores = analizar_con_grilla(img_file, cx, cy, esc)
        
        st.image(img_res, caption="Red de sectores proyectada")
        
        # Cálculo basado en la suma de 320° totales
        grados_perdidos = n_sectores * 10 # Cada sector de la red suma 10°
        
        # Factor FN
        try:
            num, den = map(float, fn.split('/'))
            factor = (num/den) + 1 if den > 0 else 1
        except: factor = 1
        
        grados_finales = min(grados_perdidos * factor, 320)
        porcentaje = (grados_finales / 320) * 100
        incapacidad = porcentaje * 0.25
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Sectores Afectados", n_sectores)
        c2.metric("Suma de Grados", f"{grados_finales}°")
        c3.metric("Incapacidad Final", f"{round(incapacidad, 2)}%")
