import streamlit as st
import cv2
import numpy as np

def analizar_campo(imagen_file):
    if imagen_file is None:
        return None, 0
    file_bytes = np.asarray(bytearray(imagen_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Suavizamos la imagen para eliminar el "ruido" de los puntos chiquitos
    suave = cv2.GaussianBlur(gris, (5, 5), 0)
    
    # Umbral más fuerte para que solo lo MUY negro se vuelva blanco
    _, binaria = cv2.threshold(suave, 80, 255, cv2.THRESH_BINARY_INV)
    
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    puntos = 0
    
    for c in contornos:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        proporcion = float(w)/h
        
        # FILTRO CLAVE: 
        # 1. El área debe ser mayor (para ignorar el ruido) pero no gigante (para ignorar el texto)
        # 2. Debe ser más o menos cuadrado (proporción entre 0.8 y 1.2)
        if 80 < area < 800 and 0.7 < proporcion < 1.3:
            puntos += 1
            # Dibujamos un rectángulo verde para debug (opcional en el futuro)
            
    return binaria, puntos

def calcular_incapacidad(puntos, fn_str):
    try:
        num, den = map(int, fn_str.split('/'))
        factor = (num / den) + 1 if num > 0 else 1
    except:
        factor = 1
    
    # Cada punto son 10 grados
    grados = puntos * 10
    grados_corregidos = grados * factor
    
    if grados_corregidos > 320: grados_corregidos = 320
    
    perdida = (grados_corregidos / 320) * 100
    incapacidad = perdida * 0.25
    return grados_corregidos, perdida, incapacidad

st.set_page_config(page_title="Peritaje", layout="wide")
st.title("⚖️ Analizador de Incapacidad Campimétrica")

img_file = st.file_uploader("Subir imagen del Campo Visual", type=['png', 'jpg', 'jpeg'])
fn = st.text_input("Falsos Negativos (ej. 0/8)", value="0/8")

if st.button("CALCULAR INCAPACIDAD"):
    if img_file is not None:
        img_bin, total_puntos = analizar_campo(img_file)
        
        st.subheader("Vista de Diagnóstico")
        st.image(img_bin, caption="Solo los bloques blancos grandes deberían ser contados", width=500)
        
        g, p, inc = calcular_incapacidad(total_puntos, fn)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Cuadrados Detectados", total_puntos)
        col2.metric("Suma de Grados", f"{round(g, 1)}°")
        col3.metric("Incapacidad Final", f"{round(inc, 2)}%")
    else:
        st.error("Por favor, subí una imagen.")
