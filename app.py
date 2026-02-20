import streamlit as st
import cv2
import numpy as np
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(layout="wide", page_title="Peritaje Pro")
st.title("⚖️ Analizador de Incapacidad Visual (Precisión 70%)")

img_file = st.sidebar.file_uploader("Subir Campo Visual", type=['jpg', 'png', 'jpeg'])

if img_file is not None:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    h, w = img.shape[:2]
    
    st.write("### 1. Centrado: Clic en el cruce central (0,0)")
    coords = streamlit_image_coordinates(img, key="peritaje")
    cx = coords["x"] if coords else w // 2
    cy = coords["y"] if coords else h // 2
    
    radio_10 = st.sidebar.slider("Tamaño Grilla (Pixeles/10°)", 10, 200, 55)
    umbral = st.sidebar.slider("Sensibilidad de Negros", 50, 255, 120)
    
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detectamos lo negro (cuadrados)
    _, binaria = cv2.threshold(gris, umbral, 255, cv2.THRESH_BINARY_INV)
    
    img_final = img.copy()
    total_grados = 0
    
    for i in range(5): # Evaluamos hasta los 50 grados
        r_ext = (i + 1) * radio_10
        r_int = i * radio_10
        for a_idx in range(8):
            alpha = a_idx * 45
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(mask, (cx, cy), (int(r_ext), int(r_ext)), 0, alpha, alpha + 45, 255, -1)
            cv2.circle(mask, (cx, cy), int(r_int), 0, -1)
            
            area_negra = np.count_nonzero(cv2.bitwise_and(binaria, mask))
            area_total = np.count_nonzero(mask)
            
            if area_total > 0:
                porcentaje_ocupado = area_negra / area_total
                
                # LÓGICA DE COLOR Y PUNTOS
                if porcentaje_ocupado >= 0.70:
                    # AMARILLO: Pérdida mayor al 70% (10 grados)
                    total_grados += 10
                    color = (0, 255, 255) # BGR Amarillo
                    overlay = img_final.copy()
                    cv2.ellipse(overlay, (cx, cy), (int(r_ext), int(r_ext)), 0, alpha, alpha + 45, color, -1)
                    cv2.circle(overlay, (cx, cy), int(r_int), (0,0,0), -1)
                    img_final = cv2.addWeighted(overlay, 0.4, img_final, 0.6, 0)
                
                elif 0.05 < porcentaje_ocupado < 0.70:
                    # CELESTE: Pérdida menor al 70% (5 grados)
                    total_grados += 5
                    color = (255, 255, 0) # BGR Celeste
                    overlay = img_final.copy()
                    cv2.ellipse(overlay, (cx, cy), (int(r_ext), int(r_ext)), 0, alpha, alpha + 45, color, -1)
                    cv2.circle(overlay, (cx, cy), int(r_int), (0,0,0), -1)
                    img_final = cv2.addWeighted(overlay, 0.4, img_final, 0.6, 0)
                
                # Si el porcentaje es ínfimo (círculos blancos/buena visión), no se pinta.

            # Dibujar bordes de la grilla
            cv2.ellipse(img_final, (cx, cy), (int(r_ext), int(r_ext)), 0, alpha, alpha + 45, (180, 180, 180), 1)

    # Convertir a RGB para Streamlit
    img_rgb = cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB)
    
    # Cálculos finales
    perdida_campo = (total_grados / 500) * 100 # 500 es el máximo teórico en este modelo de 50 sectores
    incapacidad = (total_grados / 320) * 100 # Según AMA para campo visual
    
    st.markdown(f"## **Resultados: Suma {total_grados}° | Incapacidad {round(incapacidad, 1)}%**")
    st.image(img_rgb, caption="Amarillo: 10 pts (>70%) | Celeste: 5 pts (<70%)", use_container_width=True)

else:
    st.info("Subí el campo visual para aplicar la nueva lógica de peritaje.")
