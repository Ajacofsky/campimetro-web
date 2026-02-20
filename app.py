import streamlit as st
import cv2
import numpy as np
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(layout="wide", page_title="Peritaje Pro V6")
st.title("⚖️ Analizador de Incapacidad - Filtro de Solidez")

img_file = st.sidebar.file_uploader("Subir Campo Visual", type=['jpg', 'png', 'jpeg'])

if img_file is not None:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    h, w = img.shape[:2]
    
    st.sidebar.markdown("### 🛠️ Calibración")
    radio_10 = st.sidebar.slider("Tamaño Grilla (Pixeles/10°)", 10, 200, 55)
    umbral = st.sidebar.slider("Umbral de Negros", 50, 255, 155)

    st.write("### 1. Ubicar Centro (0,0)")
    coords = streamlit_image_coordinates(img, key="peritaje")
    cx = coords["x"] if coords else w // 2
    cy = coords["y"] if coords else h // 2
    
    # --- PROCESAMIENTO PARA AISLAR CUADRADOS ---
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binaria = cv2.threshold(gris, umbral, 255, cv2.THRESH_BINARY_INV)
    
    # FILTRO DE SOLIDEZ: Borramos lo que no sea un bloque sólido
    # Esto elimina los círculos y las líneas de la grilla
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))
    mascara_bloques = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)
    
    img_final = img.copy()
    puntos_totales = 0
    
    # 5 anillos (0-50°)
    for i in range(5): 
        r_ext = (i + 1) * radio_10
        r_int = i * radio_10
        
        # 8 sectores (45° cada uno)
        for s in range(8):
            angulo = s * 45
            mask_sector = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(mask_sector, (cx, cy), (int(r_ext), int(r_ext)), 0, angulo, angulo + 45, 255, -1)
            cv2.circle(mask_sector, (cx, cy), int(r_int), 0, -1)
            
            # Buscamos cuánta "masa sólida" hay en este sector
            masa_en_sector = cv2.bitwise_and(mascara_bloques, mask_sector)
            puntos_negros = np.count_nonzero(masa_en_sector)
            area_sector = np.count_nonzero(mask_sector)
            
            if area_sector > 0:
                ocupacion = (puntos_negros / area_sector) * 100
                
                # REGLA: CELESTE (>70% cuadrados) | AMARILLO (<70% cuadrados)
                color = None
                puntos = 0
                
                if ocupacion >= 12: # Umbral de área para detectar bloques masivos
                    color = (255, 255, 0) # Celeste
                    puntos = 10
                elif 1.5 < ocupacion < 12: # Umbral para detectar presencia de algunos cuadrados
                    color = (0, 255, 255) # Amarillo
                    puntos = 5
                
                if color:
                    puntos_totales += puntos
                    overlay = img_final.copy()
                    cv2.ellipse(overlay, (cx, cy), (int(r_ext), int(r_ext)), 0, angulo, angulo + 45, color, -1)
                    # Mantener el interior del anillo limpio (transparente)
                    cv2.circle(overlay, (cx, cy), int(r_int), (255,255,255), -1)
                    img_final = cv2.addWeighted(overlay, 0.4, img_final, 0.6, 0)

            # Grilla guía
            cv2.ellipse(img_final, (cx, cy), (int(r_ext), int(r_ext)), 0, angulo, angulo + 45, (220, 220, 220), 1)

    img_rgb = cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB)
    st.markdown(f"## 📊 Suma: {puntos_totales} pts | Incapacidad: {round((puntos_totales/320)*100, 1)}%")
    st.image(img_rgb, use_container_width=True)

else:
    st.info("Subí el informe. Esta versión usa un 'Filtro de Solidez' para ignorar los círculos.")
