import streamlit as st
import cv2
import numpy as np

def analizar_campo(imagen_file):
    if imagen_file is None:
        return None, 0
    file_bytes = np.asarray(bytearray(imagen_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Suavizado leve para no borrar puntos pequeños
    suave = cv2.GaussianBlur(gris, (3, 3), 0)
    
    # Umbralización (Lo que se ve negro pasa a blanco)
    _, binaria = cv2.threshold(suave, 120, 255, cv2.THRESH_BINARY_INV)
    
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    puntos = 0
    
    for c in contornos:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        proporcion = float(w)/h
        
        # AJUSTE: Bajamos el área mínima a 10 para capturar esos puntos
        # y permitimos una forma un poco más libre (proporción 0.5 a 1.5)
        if 10 < area < 1200 and 0.5 < proporcion < 1.5:
            puntos += 1
            
    return binaria, puntos

def calcular_resultado_final(puntos, fn_str):
    grados_brutos = puntos * 10
    try:
        if "/" in fn_str:
            num, den = map(float, fn_str.split('/'))
            # Factor: (Falsos Negativos / Total) + 1
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

st.set_page_config(page_title="Peritaje Campimétrico", layout="wide")
st.title("⚖️ Analizador de Incapacidad Campimétrica")

img_file = st.file_uploader("Subir Campo Visual", type=['png', 'jpg', 'jpeg'])
fn_input = st.text_input("Falsos Negativos (ejemplo: 1/8)", value="0/8")

if st.button("CALCULAR AHORA"):
    if img_file:
        img_bin, n_puntos = analizar_campo(img_file)
        
        st.subheader("Vista de Diagnóstico")
        st.image(img_bin, caption=f"Análisis visual: {n_puntos} sectores detectados", width=500)
        
        deg, perd, inc = calcular_resultado_final(n_puntos, fn_input)
        
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Puntos Detectados", n_puntos)
        c2.metric("Suma de Grados", f"{deg}°")
        c3.metric("Incapacidad Final", f"{inc}%")
        
        st.info(f"Lógica aplicada: ({n_puntos} pts * 10°) * factor FN {fn_input} = {deg}°. Incapacidad = {perd}% de pérdida visual * 0.25 de coeficiente.")
    else:
        st.warning("Por favor, cargá una imagen.")
