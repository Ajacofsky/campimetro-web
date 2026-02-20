import streamlit as st
import cv2
import numpy as np
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(layout="wide", page_title="Peritaje Pro - Auditoría")
st.title("⚖️ Analizador de Incapacidad - Fase Final")

img_file = st.sidebar.file_uploader("Subir Campo Visual", type=['jpg', 'png', 'jpeg'])

if img_file is not None:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    h, w = img.shape[:2]
    
    st.sidebar.markdown("### 🛠️ Calibración")
    radio_10 = st.sidebar.slider("Tamaño Grilla (Pixeles/10°)", 10, 200, 55)
    umbral = st.sidebar.slider("Sensibilidad de Negros", 50, 255, 180) # Subí el default

    st.write("### 1. Ubicar Centro (0,0)")
    coords = streamlit_image_coordinates(img, key="peritaje")
    cx = coords["x"] if coords else w // 2
    cy = coords["y"] if coords else h // 2
    
    # --- PROCESAMIENTO PARA DETECTAR SOLO CUADRADOS ---
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binaria = cv2.threshold(gris, umbral, 255, cv2.THRESH_BINARY_INV)
    
    # FILTRO MORFOLÓGICO: Elimina líneas finas (círculos) y resalta bloques (cuadrados)
    kernel = np.ones((3,3), np.uint8)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)
    
    img_final = img.copy()
    total_puntos = 0
    
    # Definimos los 5 anillos de interés (0 a 50°)
    for i in range(5): 
        r_ext = (i + 1) * radio_10
        r_int = i * radio_10
        
        # 8 sectores por anillo (divididos por ejes y bisectrices)
        for a_idx in range(8):
            angulo_inicio = a_idx * 45
            
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(mask, (cx, cy), (int(r_ext), int(r_ext)), 0, angulo_inicio, angulo_inicio + 45, 255, -1)
            cv2.circle(mask, (cx, cy), int(r_int), 0, -1)
            
            area_negra = np.count_nonzero(cv2.bitwise_and(binaria, mask))
            area_total = np.count_nonzero(mask)
            
            if area_total > 0:
                ocupacion = (area_negra / area_total) * 100
                
                color = None
                puntos_sector = 0
                
                if ocupacion >= 70:
                    puntos_sector = 10
                    color = (0, 255, 255) # AMARILLO (BGR)
                elif 10 < ocupacion < 70: # Subí el piso a 10% para ignorar círculos
                    puntos_sector = 5
                    color = (255, 255, 0) # CELESTE (BGR)
                
                if color:
                    total_puntos += puntos_sector
                    overlay = img_final.copy()
                    cv2.ellipse(overlay, (cx, cy), (int(r_ext), int(r_ext)), 0, angulo_inicio, angulo_inicio + 45, color, -1)
                    cv2.circle(overlay, (cx, cy), int(r_int), (0,0,0), -1)
                    img_final = cv2.addWeighted(overlay, 0.4, img_final, 0.6, 0)

            # Dibujar la grilla de referencia
            cv2.ellipse(img_final, (cx, cy), (int(r_ext), int(r_ext)), 0, angulo_inicio, angulo_inicio + 45, (150, 150, 150), 1)

    img_rgb = cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB)
    
    # Cálculos finales según tu regla
    # 8 sectores por cuadrante * 4 cuadrantes = 32 sectores de 10°
    # Máximo teórico en 50°: 320 puntos.
    incapacidad = (total_puntos / 320) * 100
    
    st.markdown(f"## 📊 SUMA TOTAL: {total_puntos} pts | INCAPACIDAD: {round(incapacidad, 1)}%")
    st.image(img_rgb, caption="Amarillo: 10 pts (>70%) | Celeste: 5 pts (<70%)", use_container_width=True)
    
else:
    st.info("Subí el informe para procesar los sectores.")
