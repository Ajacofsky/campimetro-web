import streamlit as st
import cv2
import numpy as np
from PIL import Image

# --- MOTOR DE CÁLCULO ---
def analizar_campo(imagen_file):
    if imagen_file is None:
        return 0
    
    file_bytes = np.asarray(bytearray(imagen_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # CAMBIO 1: Umbral adaptativo (se ajusta solo a la luz de la foto)
    binaria = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, 11, 2)
    
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    puntos_no_vistos = 0
    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w)/h
        area = cv2.contourArea(c)
        
        # CAMBIO 2: Bajamos el área mínima de 50 a 10 para detectar puntos más pequeños
        if 0.7 <= aspect_ratio <= 1.3 and area > 10:
            puntos_no_vistos += 1
            
    return puntos_no_vistos
def calcular_incapacidad(puntos, fn_str):
    try:
        num, den = map(int, fn_str.split('/'))
        factor = num / den if num > 0 else 1
    except:
        factor = 1
        
    grados = (puntos * 10) * factor # Regla de 10° simplificada para el motor
    if grados > 320: grados = 320
    
    perdida = (grados / 320) * 100
    incapacidad = perdida * 0.25
    return grados, perdida, incapacidad

# --- INTERFAZ WEB ---
st.set_page_config(page_title="Peritaje Campimétrico", layout="wide")
st.title("⚖️ Analizador de Incapacidad Campimétrica")
st.markdown("---")

modo = st.sidebar.radio("Evaluación:", ["Unilateral", "Bilateral"])

if modo == "Bilateral":
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ojo Derecho (OD)")
        img_od = st.file_uploader("Cargar CVC OD", type=['png', 'jpg', 'jpeg'], key="od")
        fn_od = st.text_input("FN OD (ej. 1/8)", value="0/8")
    with col2:
        st.subheader("Ojo Izquierdo (OI)")
        img_oi = st.file_uploader("Cargar CVC OI", type=['png', 'jpg', 'jpeg'], key="oi")
        fn_oi = st.text_input("FN OI (ej. 1/8)", value="0/8")

    if st.button("CALCULAR INCAPACIDAD BILATERAL"):
        puntos_od = analizar_campo(img_od)
        puntos_oi = analizar_campo(img_oi)
        
        g_od, p_od, i_od = calcular_incapacidad(puntos_od, fn_od)
        g_oi, p_oi, i_oi = calcular_incapacidad(puntos_oi, fn_oi)
        
        st.divider()
        st.write(f"**Resultado OD:** {i_od}% | **Resultado OI:** {i_oi}%")
        # El resultado final en bilateral suele ser la suma de ambos o según baremo específico
        st.metric("INCAPACIDAD TOTAL COMBINADA", f"{round(i_od + i_oi, 2)}%")

else:
    st.subheader("Evaluación Unilateral")
    img = st.file_uploader("Cargar Campo Visual", type=['png', 'jpg', 'jpeg'])
    fn = st.text_input("Falsos Negativos (ej. 0/8)", value="0/8")
    
    if st.button("CALCULAR INCAPACIDAD"):
        puntos = analizar_campo(img)
        g, p, inc = calcular_incapacidad(puntos, fn)
        st.success(f"Resultado: {inc}% de incapacidad.")
