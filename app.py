import streamlit as st
import cv2
import numpy as np

def analizar_con_grilla(img_original, cx, cy, esc):
    h, w = img_original.shape[:2]
    img_viz = img_original.copy()
    
    # 1. Definir Centro y Radio
    centro = (int(w * cx), int(h * cy))
    radio_max = int(min(h, w) * 0.4 * esc)
    
    # 2. Definir Geometría (8 bisectrices y 4 anillos)
    angulos = np.linspace(0, 315, 8)
    radios = [radio_max * 0.16, radio_max * 0.33, radio_max * 0.5, radio_max]
    
    # Pre-procesar para detección (esto ocurre por detrás)
    gris = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    _, binaria = cv2.threshold(gris, 110, 255, cv2.THRESH_BINARY_INV)
    
    sectores_fallados = 0
    
    # 3. Dibujar Red y Detectar en tiempo real
    for i in range(len(radios)):
        r_int = 0 if i == 0 else radios[i-1]
        r_ext = radios[i]
        for alpha in angulos:
            # Máscara del sector
            mask = np.zeros(gris.shape, dtype=np.uint8)
            cv2.ellipse(mask, centro, (int(r_ext), int(r_ext)), 0, alpha, alpha + 45, 255, -1)
            if r_int > 0:
                cv2.circle(mask, centro, int(r_int), 0, -1)
            
            # Detección
            overlap = cv2.bitwise_and(binaria, mask)
            if np.sum(overlap) > 400: # Sensibilidad del sector
                sectores_fallados += 1
                color = (0, 0, 255) # Rojo si está afectado
            else:
                color = (0, 255, 0) # Verde si está libre
            
            # Dibujar sector en la previsualización
            cv2.ellipse(img_viz, centro, (int(r_ext), int(r_ext)), 0, alpha, alpha + 45, color, 1)

    # Dibujar ejes centrales
    cv2.line(img_viz, (centro[0]-20, centro[1]), (centro[0]+20, centro[1]), (255,0,0), 2)
    cv2.line(img_viz, (centro[0], centro[1]-20), (centro[0], centro[1]+20), (255,0,0), 2)
                
    return img_viz, sectores_fallados

# --- Interfaz de Streamlit ---
st.set_page_config(layout="wide")
st.title("Analizador de Incapacidad Campimétrica")

img_file = st.file_uploader("Subir imagen", type=['jpg', 'png', 'jpeg'])

if img_file:
    # Convertir a formato OpenCV
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img_original = cv2.imdecode(file_bytes, 1)
    
    # Sidebar con sliders que disparan el cambio inmediato
    st.sidebar.header("Ajuste de Precisión")
    cx = st.sidebar.slider("Posición X", 0.0, 1.0, 0.5, 0.01)
    cy = st.sidebar.slider("Posición Y", 0.0, 1.0, 0.5, 0.01)
    esc = st.sidebar.slider("Escala", 0.1, 2.0, 1.0, 0.05)
    fn = st.sidebar.text_input("Falsos Negativos", "0/8")
    
    # PROCESAMIENTO EN TIEMPO REAL
    img_resultado, n_sectores = analizar_con_grilla(img_original, cx, cy, esc)
    
    # Mostrar imagen principal
    st.image(img_resultado, use_container_width=True)
    
    # Cálculos matemáticos (Suma de 320°)
    try:
        num, den = map(float, fn.split('/'))
        factor = (num/den) + 1 if den > 0 else 1
    except: factor = 1
    
    grados_finales = min((n_sectores * 10) * factor, 320)
    inc_final = ((grados_finales / 320) * 100) * 0.25
    
    # Métricas
    c1, c2, c3 = st.columns(3)
    c1.metric("Sectores Afectados", n_sectores)
    c2.metric("Suma Grados", f"{round(grados_finales, 1)}°")
    c3.metric("Incapacidad Baremo", f"{round(inc_final, 2)}%")
