import streamlit as st
import cv2
import numpy as np
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(layout="wide", page_title="Peritaje Pro - Detector de Masas")
st.title("⚖️ Analizador de Incapacidad (Filtro de Cuadrados Sólidos)")

img_file = st.sidebar.file_uploader("Subir Campo Visual", type=['jpg', 'png', 'jpeg'])

if img_file is not None:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    h, w = img.shape[:2]
    
    # Parámetros de calibración
    radio_10 = st.sidebar.slider("Tamaño Grilla (Pixeles/10°)", 10, 200, 55)
    umbral = st.sidebar.slider("Sensibilidad (Umbral)", 50, 255, 140)

    st.write("### 1. Ubicar Centro (0,0)")
    coords = streamlit_image_coordinates(img, key="peritaje")
    cx = coords["x"] if coords else w // 2
    cy = coords["y"] if coords else h // 2
    
    # --- FILTRADO CRUCIAL PARA DISTINGUIR FORMAS ---
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binaria = cv2.threshold(gris, umbral, 255, cv2.THRESH_BINARY_INV)
    
    # EL COLADOR: Borra líneas finas (círculos) y mantiene bloques (cuadrados)
    # Usamos un kernel de 5x5 que es más grande que el grosor de la línea del círculo
    kernel = np.ones((5,5), np.uint8)
    solo_cuadrados = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)
    
    img_final = img.copy()
    puntos_totales = 0
    
    for i in range(5): # Anillos de 10 a 50 grados
        r_ext = (i + 1) * radio_10
        r_int = i * radio_10
        
        for s in range(8): # 8 sectores de 45°
            angulo = s * 45
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(mask, (cx, cy), (int(r_ext), int(r_ext)), 0, angulo, angulo + 45, 255, -1)
            cv2.circle(mask, (cx, cy), int(r_int), 0, -1)
            
            # Contamos área de "bloques" detectados por el colador
            masa_cuadrados = np.count_nonzero(cv2.bitwise_and(solo_cuadrados, mask))
            area_sector = np.count_nonzero(mask)
            
            if area_sector > 0:
                ocupacion = (masa_cuadrados / area_sector) * 100
                
                color = None
                puntos = 0
                
                # REGLA SOLICITADA
                if ocupacion >= 15: # Celeste: Ocupación alta de masa sólida
                    color = (255, 255, 0) # Celeste (BGR)
                    puntos = 10
                elif 2 < ocupacion < 15: # Amarillo: Presencia de algún bloque sólido
                    color = (0, 255, 255) # Amarillo (BGR)
                    puntos = 5
                
                if color:
                    puntos_totales += puntos
                    overlay = img_final.copy()
                    cv2.ellipse(overlay, (cx, cy), (int(r_ext), int(r_ext)), 0, angulo, angulo + 45, color, -1)
                    cv2.circle(overlay, (cx, cy), int(r_int), (255,255,255), -1) 
                    img_final = cv2.addWeighted(overlay, 0.4, img_final, 0.6, 0)

            cv2.ellipse(img_final, (cx, cy), (int(r_ext), int(r_ext)), 0, angulo, angulo + 45, (220, 220, 220), 1)

    img_rgb = cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB)
    st.markdown(f"## 📊 Suma Total: {puntos_totales} pts")
    st.image(img_rgb, use_container_width=True)

else:
    st.info("Subí la imagen para aplicar el filtro morfológico.")
