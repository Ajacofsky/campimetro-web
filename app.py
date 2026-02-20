import streamlit as st
import cv2
import numpy as np
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(layout="wide", page_title="Peritaje Pro V3")
st.title("⚖️ Analizador de Sectores y Cuadrados Negros")

img_file = st.sidebar.file_uploader("Subir Campo Visual", type=['jpg', 'png', 'jpeg'])

if img_file is not None:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    h, w = img.shape[:2]
    
    st.sidebar.markdown("### 🛠️ Ajuste de Lectura")
    radio_10 = st.sidebar.slider("Tamaño Grilla (Pixeles/10°)", 10, 200, 55)
    umbral = st.sidebar.slider("Sensibilidad de Negros", 50, 255, 160)

    st.write("### 1. Marcar el Centro Exacto (Cruce de Ejes)")
    coords = streamlit_image_coordinates(img, key="peritaje")
    cx = coords["x"] if coords else w // 2
    cy = coords["y"] if coords else h // 2
    
    # --- PROCESAMIENTO DE IMAGEN PARA DETECTAR MASA ---
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Filtro para resaltar solo los cuadrados
    _, binaria = cv2.threshold(gris, umbral, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5,5), np.uint8)
    # Cerramos los huecos en los cuadrados para que la lectura de área sea real
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel)
    
    img_final = img.copy()
    puntos_totales = 0
    
    # Solo anillos de 10° a 50° (5 anillos)
    for i in range(5): 
        r_ext = (i + 1) * radio_10
        r_int = i * radio_10
        
        # 8 sectores por anillo (divididos por ejes y bisectrices cada 45°)
        for s in range(8):
            angulo_inicio = s * 45
            
            # Crear mascara del SECTOR
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(mask, (cx, cy), (int(r_ext), int(r_ext)), 0, angulo_inicio, angulo_inicio + 45, 255, -1)
            cv2.circle(mask, (cx, cy), int(r_int), 0, -1)
            
            # Calcular ocupación real de los cuadrados
            area_negra = np.count_nonzero(cv2.bitwise_and(binaria, mask))
            area_total = np.count_nonzero(mask)
            
            if area_total > 0:
                porcentaje = (area_negra / area_total) * 100
                
                color = None
                puntos_sector = 0
                
                if porcentaje >= 70:
                    # CELESTE: Presencia masiva de cuadrados
                    puntos_sector = 10
                    color = (255, 255, 0) # BGR Celeste
                elif 8 < porcentaje < 70:
                    # AMARILLO: Presencia parcial
                    puntos_sector = 5
                    color = (0, 255, 255) # BGR Amarillo
                
                if color:
                    puntos_totales += puntos_sector
                    overlay = img_final.copy()
                    cv2.ellipse(overlay, (cx, cy), (int(r_ext), int(r_ext)), 0, angulo_inicio, angulo_inicio + 45, color, -1)
                    cv2.circle(overlay, (cx, cy), int(r_int), (0,0,0), -1)
                    img_final = cv2.addWeighted(overlay, 0.45, img_final, 0.55, 0)

            # Dibujar lineas de la grilla para verificar punteria
            cv2.ellipse(img_final, (cx, cy), (int(r_ext), int(r_ext)), 0, angulo_inicio, angulo_inicio + 45, (180, 180, 180), 1)

    img_rgb = cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB)
    
    # 32 sectores evaluados de 10 pts cada uno = 320 pts max.
    incapacidad = (puntos_totales / 320) * 100
    
    st.divider()
    st.markdown(f"## 📊 Suma: {puntos_totales} pts | Incapacidad Visual: {round(incapacidad, 1)}%")
    st.image(img_rgb, caption="Celeste: >70% Cuadrados (10 pts) | Amarillo: <70% Cuadrados (5 pts)", use_container_width=True)
    
else:
    st.info("Sube la imagen para detectar los cuadrados negros por sector.")
