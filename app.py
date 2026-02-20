import streamlit as st
import cv2
import numpy as np
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(layout="wide", page_title="Peritaje Pro V4")
st.title("⚖️ Analizador de Incapacidad (Fase de Colores Exactos)")

img_file = st.sidebar.file_uploader("Subir Campo Visual", type=['jpg', 'png', 'jpeg'])

if img_file is not None:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    h, w = img.shape[:2]
    
    st.sidebar.markdown("### 🛠️ Ajuste de Lectura")
    radio_10 = st.sidebar.slider("Tamaño Grilla (Pixeles/10°)", 10, 200, 55)
    umbral = st.sidebar.slider("Sensibilidad de Negros", 50, 255, 180)

    st.write("### 1. Ubicar Centro (0,0)")
    coords = streamlit_image_coordinates(img, key="peritaje")
    cx = coords["x"] if coords else w // 2
    cy = coords["y"] if coords else h // 2
    
    # --- PROCESAMIENTO PARA DETECTAR SOLO CUADRADOS ---
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binaria = cv2.threshold(gris, umbral, 255, cv2.THRESH_BINARY_INV)
    
    # Este paso borra los redondeles y deja solo los cuadrados sólidos
    kernel = np.ones((3,3), np.uint8)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)
    
    img_final = img.copy()
    puntos_totales = 0
    
    # Solo anillos de 10° a 50° (5 anillos)
    for i in range(5): 
        r_ext = (i + 1) * radio_10
        r_int = i * radio_10
        
        # 8 sectores por anillo (45° cada uno)
        for s in range(8):
            angulo_inicio = s * 45
            
            # Máscara del sector específico
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(mask, (cx, cy), (int(r_ext), int(r_ext)), 0, angulo_inicio, angulo_inicio + 45, 255, -1)
            # Restamos el anillo interior para no tocar el centro
            cv2.circle(mask, (cx, cy), int(r_int), 0, -1)
            
            # Contar cuadraditos negros
            area_negra = np.count_nonzero(cv2.bitwise_and(binaria, mask))
            area_total = np.count_nonzero(mask)
            
            if area_total > 0:
                porcentaje = (area_negra / area_total) * 100
                
                # Definir color según ocupación (Celeste > 70%, Amarillo < 70%)
                color = None
                puntos_sector = 0
                
                if porcentaje >= 70:
                    color = (255, 255, 0) # Celeste (BGR)
                    puntos_sector = 10
                elif 5 < porcentaje < 70:
                    color = (0, 255, 255) # Amarillo (BGR)
                    puntos_sector = 5
                
                if color:
                    puntos_totales += puntos_sector
                    overlay = img_final.copy()
                    # Pintamos solo el sector detectado
                    cv2.ellipse(overlay, (cx, cy), (int(r_ext), int(r_ext)), 0, angulo_inicio, angulo_inicio + 45, color, -1)
                    cv2.circle(overlay, (cx, cy), int(r_int), (255, 255, 255), -1) # Mantener interior limpio
                    img_final = cv2.addWeighted(overlay, 0.4, img_final, 0.6, 0)

            # Dibujar la cuadrícula de referencia (gris suave)
            cv2.ellipse(img_final, (cx, cy), (int(r_ext), int(r_ext)), 0, angulo_inicio, angulo_inicio + 45, (220, 220, 220), 1)

    # Convertir a RGB para que Streamlit lo muestre bien
    img_rgb = cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB)
    
    incapacidad = (puntos_totales / 320) * 100
    
    st.divider()
    st.markdown(f"## 📊 Suma: {puntos_totales} pts | Incapacidad Visual: {round(incapacidad, 1)}%")
    st.image(img_rgb, caption="Celeste: >70% (10 pts) | Amarillo: <70% (5 pts) | Blanco: Sin pérdida", use_container_width=True)
    
else:
    st.info("Sube la imagen para aplicar la detección morfológica de cuadrados.")
