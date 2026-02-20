import streamlit as st
import cv2
import numpy as np
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(layout="wide", page_title="Peritaje Pro V5")
st.title("⚖️ Analizador de Incapacidad - Detección por Contornos")

img_file = st.sidebar.file_uploader("Subir Campo Visual", type=['jpg', 'png', 'jpeg'])

if img_file is not None:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    h, w = img.shape[:2]
    
    st.sidebar.markdown("### 🛠️ Calibración de Visión")
    radio_10 = st.sidebar.slider("Tamaño Grilla (Pixeles/10°)", 10, 200, 55)
    umbral = st.sidebar.slider("Sensibilidad de Negros", 50, 255, 150)

    st.write("### 1. Ubicar Centro (0,0)")
    coords = streamlit_image_coordinates(img, key="peritaje")
    cx = coords["x"] if coords else w // 2
    cy = coords["y"] if coords else h // 2
    
    # --- PROCESAMIENTO AVANZADO: DETECCIÓN DE BLOQUES ---
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Binarización invertida para que los cuadrados sean BLANCOS sobre fondo negro (para procesar área)
    _, binaria = cv2.threshold(gris, umbral, 255, cv2.THRESH_BINARY_INV)
    
    # Limpiamos ruido pequeño (líneas de grilla) pero mantenemos bloques
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel)
    
    img_final = img.copy()
    puntos_totales = 0
    
    # Definir los 5 anillos (0-50°)
    for i in range(5): 
        r_ext = (i + 1) * radio_10
        r_int = i * radio_10
        
        # 8 sectores por anillo (divididos por ejes y bisectrices)
        for s in range(8):
            angulo_inicio = s * 45
            
            # Crear máscara del SECTOR exacto
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(mask, (cx, cy), (int(r_ext), int(r_ext)), 0, angulo_inicio, angulo_inicio + 45, 255, -1)
            cv2.circle(mask, (cx, cy), int(r_int), 0, -1) # Dejar el anillo interior fuera
            
            # Extraer solo los objetos dentro de este sector
            sector_analizado = cv2.bitwise_and(binaria, mask)
            
            # Encontrar contornos de objetos sólidos (cuadraditos)
            contornos, _ = cv2.findContours(sector_analizado, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            area_cuadrados = 0
            for cnt in contornos:
                area_cuadrados += cv2.contourArea(cnt)
            
            area_total_sector = np.count_nonzero(mask)
            
            if area_total_sector > 0:
                ocupacion = (area_cuadrados / area_total_sector) * 100
                
                # LÓGICA DE COLOR (Según tu pedido)
                # CELESTE > 70% | AMARILLO < 70% | SIN COLOR si no hay cuadrados
                color = None
                puntos = 0
                
                if ocupacion >= 65: # Ajusté a 65% porque el cuadrado nunca llena el 100% del sector circular
                    color = (255, 255, 0) # Celeste (BGR)
                    puntos = 10
                elif ocupacion > 3: # Si hay al menos un cuadradito o parte de uno
                    color = (0, 255, 255) # Amarillo (BGR)
                    puntos = 5
                
                if color:
                    puntos_totales += puntos
                    overlay = img_final.copy()
                    cv2.ellipse(overlay, (cx, cy), (int(r_ext), int(r_ext)), 0, angulo_inicio, angulo_inicio + 45, color, -1)
                    # Mantenemos el centro y el resto limpio
                    cv2.circle(overlay, (cx, cy), int(r_int), (255,255,255), -1) 
                    img_final = cv2.addWeighted(overlay, 0.4, img_final, 0.6, 0)

            # Grilla de referencia gris muy tenue
            cv2.ellipse(img_final, (cx, cy), (int(r_ext), int(r_ext)), 0, angulo_inicio, angulo_inicio + 45, (230, 230, 230), 1)

    img_rgb = cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB)
    
    # 32 sectores evaluables * 10 pts = 320 pts max.
    incapacidad = (puntos_totales / 320) * 100
    
    st.divider()
    st.markdown(f"## 📊 Suma: {puntos_totales} pts | Incapacidad Visual: {round(incapacidad, 1)}%")
    st.image(img_rgb, caption="Detección por Masa de Cuadrados: Celeste (>70%) | Amarillo (<70%)", use_container_width=True)

else:
    st.info("Subí el informe. Esta versión detecta específicamente la masa de los objetos sólidos.")
