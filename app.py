import streamlit as st
import cv2
import numpy as np
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(layout="wide", page_title="Peritaje Pro V7")
st.title("⚖️ Analizador de Incapacidad - Motor de Conteo de Objetos")

img_file = st.sidebar.file_uploader("Subir Campo Visual", type=['jpg', 'png', 'jpeg'])

if img_file is not None:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    h, w = img.shape[:2]
    
    st.sidebar.markdown("### 🛠️ Ajuste Quirúrgico")
    radio_10 = st.sidebar.slider("Tamaño Grilla (Pixeles/10°)", 10, 200, 55)
    umbral = st.sidebar.slider("Sensibilidad de Negros", 50, 255, 145)

    st.write("### 1. Ubicá el Centro (0,0)")
    coords = streamlit_image_coordinates(img, key="peritaje")
    cx = coords["x"] if coords else w // 2
    cy = coords["y"] if coords else h // 2
    
    # --- PROCESAMIENTO PARA DETECTAR CUADRADOS SOLIDOS ---
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binaria = cv2.threshold(gris, umbral, 255, cv2.THRESH_BINARY_INV)
    
    # Este paso es clave: elimina círculos y líneas finas
    kernel = np.ones((4,4), np.uint8)
    solo_bloques = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)
    
    img_final = img.copy()
    puntos_totales = 0
    
    # Analizamos 5 anillos (10° a 50°)
    for i in range(5): 
        r_ext = (i + 1) * radio_10
        r_int = i * radio_10
        
        for s in range(8): # 8 sectores de 45°
            angulo = s * 45
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(mask, (cx, cy), (int(r_ext), int(r_ext)), 0, angulo, angulo + 45, 255, -1)
            cv2.circle(mask, (cx, cy), int(r_int), 0, -1)
            
            # Recortamos el sector en la imagen de bloques
            sector_bloques = cv2.bitwise_and(solo_bloques, mask)
            
            # CONTAMOS CUÁNTOS OBJETOS (CUADRADOS) HAY REALMENTE
            contornos, _ = cv2.findContours(sector_bloques, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filtramos contornos por tamaño para asegurar que sean cuadrados
            cuadrados_reales = [c for c in contornos if cv2.contourArea(c) > 10]
            cantidad = len(cuadrados_reales)
            
            color = None
            puntos = 0
            
            # LÓGICA DE DETECCIÓN BASADA EN CANTIDAD (Más fiable que el %)
            if cantidad >= 3:
                color = (255, 255, 0) # Celeste (BGR)
                puntos = 10
            elif cantidad >= 1:
                color = (0, 255, 255) # Amarillo (BGR)
                puntos = 5
            
            if color:
                puntos_totales += puntos
                overlay = img_final.copy()
                cv2.ellipse(overlay, (cx, cy), (int(r_ext), int(r_ext)), 0, angulo, angulo + 45, color, -1)
                cv2.circle(overlay, (cx, cy), int(r_int), (255,255,255), -1) 
                img_final = cv2.addWeighted(overlay, 0.4, img_final, 0.6, 0)

            # Dibujamos grilla tenue
            cv2.ellipse(img_final, (cx, cy), (int(r_ext), int(r_ext)), 0, angulo, angulo + 45, (220, 220, 220), 1)

    img_rgb = cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB)
    
    st.divider()
    st.markdown(f"## 📊 SUMA: {puntos_totales} pts | INCAPACIDAD: {round((puntos_totales/320)*100, 1)}%")
    st.image(img_rgb, caption="Celeste: Pérdida Masiva | Amarillo: Pérdida Parcial", use_container_width=True)

else:
    st.info("Subí el informe. Esta versión 'cuenta' cuadrados uno por uno.")
