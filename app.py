import streamlit as st
import cv2
import numpy as np

def analizar_campo(imagen_file):
    if imagen_file is None:
        return None, 0
    file_bytes = np.asarray(bytearray(imagen_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Filtro para limpiar puntos aislados (ruido)
    suave = cv2.medianBlur(gris, 5)
    
    # 2. Umbralización fuerte
    _, binaria = cv2.threshold(suave, 100, 255, cv2.THRESH_BINARY_INV)
    
    # 3. Operación Morfológica: Une puntos cercanos y limpia motas de polvo
    kernel = np.ones((3,3), np.uint8)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel)
    
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    puntos = 0
    
    for c in contornos:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        proporcion = float(w)/h
        
        # FILTRO DE PRECISIÓN:
        # Los escotomas en tu imagen tienen un área de entre 60 y 400 píxeles.
        # El texto y el ruido suelen ser menores a 50 o muy alargados.
        if 60 < area < 500 and 0.7 < proporcion < 1.3:
            puntos += 1
            
    return binaria, puntos

def calcular_resultado_final(puntos, fn_str):
    grados_brutos = puntos * 10
    try:
        if "/" in fn_str:
            num, den = map(float, fn_str.split('/'))
            factor = (num / den) + 1 if den > 0 else 1
        else:
            factor = 1
    except:
        factor = 1
    
    grados_finales = grados_brutos * factor
    if grados_finales > 320: grados_finales = 320
    
    porcentaje_perdida = (grados_finales / 320) * 100
    incapacidad_final = porcentaje_perdida * 0.25
    
    return round(grados_finales, 1), round(porcentaje_perdida, 2), round(incapacidad_final, 2)

# --- Interfaz ---
st.set_page_config(page_title="Peritaje", layout="wide")
st.title("⚖️ Analizador de Incapacidad Campimétrica")

img_file = st.file_uploader("Subir imagen", type=['png', 'jpg', 'jpeg'])
fn_input = st.text_input("Falsos Negativos", value="0/8")

if st.button("CALCULAR"):
    if img_file:
        img_bin, n_puntos = analizar_campo(img_file)
        st.subheader("Vista de Diagnóstico")
        st.image(img_bin, caption=f"Sectores reales detectados: {n_puntos}", width=500)
        
        deg, perd, inc = calcular_resultado_final(n_puntos, fn_input)
        
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Escotomas (10°)", n_puntos)
        c2.metric("Grados Totales", f"{deg}°")
        c3.metric("Incapacidad", f"{inc}%")
    else:
        st.warning("Cargá una imagen.")
