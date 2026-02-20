import streamlit as st
import cv2
import numpy as np

def detectar_geometria_campo(img_gris):
    # 1. Detectar líneas para hallar el centro
    bordes = cv2.Canny(img_gris, 50, 150)
    lineas = cv2.HoughLinesP(bordes, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    
    centro_x, centro_y = img_gris.shape[1]//2, img_gris.shape[0]//2 # Default
    
    if lineas is not None:
        horizontales = []
        verticales = []
        for l in lineas:
            x1, y1, x2, y2 = l[0]
            if abs(y1 - y2) < 5: horizontales.append(y1)
            if abs(x1 - x2) < 5: verticales.append(x1)
        
        if horizontales and verticales:
            centro_y = int(np.median(horizontales))
            centro_x = int(np.median(verticales))
            
    return centro_x, centro_y

def analizar_peritaje(imagen_file, fn_str):
    file_bytes = np.asarray(bytearray(imagen_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detección automática del centro
    cx, cy = detectar_geometria_campo(gris)
    
    # Umbralización para ver los escotomas
    _, binaria = cv2.threshold(gris, 110, 255, cv2.THRESH_BINARY_INV)
    
    # Definir radios basados en la escala estándar del informe (estimación inicial)
    # Se puede ajustar si el informe es de 30 o 60 grados
    radio_max = int(img.shape[1] * 0.35)
    radios = [radio_max * 0.16, radio_max * 0.33, radio_max * 0.5, radio_max]
    angulos = np.linspace(0, 315, 8)
    
    sectores_fallados = 0
    img_viz = img.copy()
    
    for i, r_ext in enumerate(radios):
        r_int = 0 if i == 0 else radios[i-1]
        for alpha in angulos:
            mask = np.zeros(gris.shape, dtype=np.uint8)
            cv2.ellipse(mask, (cx, cy), (int(r_ext), int(r_ext)), 0, alpha, alpha + 45, 255, -1)
            if r_int > 0:
                cv2.circle(mask, (cx, cy), int(r_int), 0, -1)
            
            overlap = cv2.bitwise_and(binaria, mask)
            if np.sum(overlap) > 500: # Presencia de cuadrado negro
                sectores_fallados += 1
                cv2.ellipse(img_viz, (cx, cy), (int(r_ext), int(r_ext)), 0, alpha, alpha + 45, (0, 0, 255), 2)
            else:
                cv2.ellipse(img_viz, (cx, cy), (int(r_ext), int(r_ext)), 0, alpha, alpha + 45, (0, 255, 0), 1)

    # Dibujar ejes detectados
    cv2.line(img_viz, (cx-50, cy), (cx+50, cy), (255, 0, 0), 2)
    cv2.line(img_viz, (cx, cy-50), (cx, cy+50), (255, 0, 0), 2)
                
    return img_viz, sectores_fallados

# --- Interfaz ---
st.set_page_config(layout="wide")
st.title("⚖️ Analizador de Incapacidad (Centrado Automático)")

img_file = st.file_uploader("Subir Campo Visual", type=['jpg', 'png', 'jpeg'])

if img_file:
    fn = st.sidebar.text_input("Falsos Negativos", "0/8")
    
    # Procesar con detección de ejes
    img_res, n_sectores = analizar_peritaje(img_file, fn)
    
    st.image(img_res, use_container_width=True)
    
    # Cálculo Final
    try:
        num, den = map(float, fn.split('/'))
        factor = (num/den) + 1 if den > 0 else 1
    except: factor = 1
    
    grados = min((n_sectores * 10) * factor, 320)
    incapacidad = ((grados / 320) * 100) * 0.25
    
    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric("Sectores Afectados", n_sectores)
    col2.metric("Suma de Grados", f"{round(grados, 1)}°")
    col3.metric("Incapacidad Final", f"{round(incapacidad, 2)}%")
