import streamlit as st
import cv2
import numpy as np
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(layout="wide", page_title="Peritaje Pro - Detección")
st.title("⚖️ Analizador: Fase 1 - Identificación de Cuadrados Negros")

img_file = st.sidebar.file_uploader("Subir Campo Visual", type=['jpg', 'png', 'jpeg'])

if img_file is not None:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    h, w = img.shape[:2]
    
    st.write("### 1. Ubicar Centro (0,0)")
    coords = streamlit_image_coordinates(img, key="peritaje")
    cx = coords["x"] if coords else w // 2
    cy = coords["y"] if coords else h // 2
    
    # Ajustes de precisión
    radio_10 = st.sidebar.slider("Tamaño Grilla (Pixeles/10°)", 10, 200, 55)
    umbral = st.sidebar.slider("Sensibilidad de Negros (Umbral)", 50, 255, 110)
    
    # PROCESAMIENTO AVANZADO
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Suavizamos un poco para eliminar el ruido de los círculos finos
    desenfoque = cv2.medianBlur(gris, 5)
    _, binaria = cv2.threshold(desenfoque, umbral, 255, cv2.THRESH_BINARY_INV)
    
    img_final = img.copy()
    total_grados = 0
    
    # Recorremos los sectores
    for i in range(5): # 5 anillos de 10 grados cada uno
        r_ext = (i + 1) * radio_10
        r_int = i * radio_10
        for a_idx in range(8):
            alpha = a_idx * 45
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(mask, (cx, cy), (int(r_ext), int(r_ext)), 0, alpha, alpha + 45, 255, -1)
            cv2.circle(mask, (cx, cy), int(r_int), 0, -1)
            
            # Calculamos la densidad de negro en el sector
            area_negra = np.count_nonzero(cv2.bitwise_and(binaria, mask))
            area_total = np.count_nonzero(mask)
            
            if area_total > 0:
                ocupacion = area_negra / area_total
                
                # UMBRAL MÍNIMO: Si hay menos de 5% de negro, es un círculo o ruido (No pintar)
                if ocupacion > 0.05:
                    total_grados += 10 # Por ahora sumamos 10 para probar detección
                    
                    # SOMBREADO AMARILLO TENUE
                    overlay = img_final.copy()
                    cv2.ellipse(overlay, (cx, cy), (int(r_ext), int(r_ext)), 0, alpha, alpha + 45, (0, 255, 255), -1)
                    cv2.circle(overlay, (cx, cy), int(r_int), (0,0,0), -1)
                    img_final = cv2.addWeighted(overlay, 0.35, img_final, 0.65, 0)
            
            # Dibujar líneas de la grilla para referencia
            cv2.ellipse(img_final, (cx, cy), (int(r_ext), int(r_ext)), 0, alpha, alpha + 45, (200, 200, 200), 1)

    # Convertir para mostrar en web
    img_rgb = cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB)
    
    st.markdown(f"## **Suma Detectada: {total_grados}°**")
    st.image(img_rgb, caption="Sectores Amarillos = Cuadrados Negros detectados", use_container_width=True)
    st.info("Si los círculos se pintan de amarillo, bajá la 'Sensibilidad de Negros'. Si los cuadrados no se pintan, subila.")

else:
    st.info("Subí la imagen para iniciar la Fase 1.")
