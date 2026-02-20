import streamlit as st
import cv2
import numpy as np
from streamlit_image_coordinates import streamlit_image_coordinates

def app():
    st.set_page_config(layout="wide", page_title="Peritaje Campimétrico Pro")
    st.title("⚖️ Analizador de Incapacidad por Sectores Geométricos")
    
    img_file = st.sidebar.file_uploader("Subir Campo Visual", type=['jpg', 'png', 'jpeg'])
    
    if img_file:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        h, w = img.shape[:2]

        st.write("### 1. Centrado: Clic en el centro (0,0)")
        value = streamlit_image_coordinates(img, key="pil")
        cx, cy = (value["x"], value["y"]) if value else (w // 2, h // 2)

        st.sidebar.subheader("📏 Calibración")
        radio_10 = st.sidebar.slider("Píxeles por cada 10°", 10, 200, 50)
        fn_input = st.sidebar.text_input("Falsos Negativos (ej. 0/4)", "0/4")

        # --- PROCESAMIENTO ---
        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binaria = cv2.threshold(gris, 120, 255, cv2.THRESH_BINARY_INV)
        
        img_viz = img.copy()
        # Definimos 6 anillos: 10, 20, 30, 40, 50, 60
        radios_px = [i * radio_10 for i in range(1, 7)]
        # 8 sectores cada 45 grados (bisectrices)
        angulos = np.arange(0, 360, 45)
        
        total_grados_afectados = 0

        # Dibujar Grilla y Evaluar Sectores
        for i, r_ext in enumerate(radios_px):
            r_int = 0 if i == 0 else radios_px[i-1]
            # Color alternante para la grilla
            color_sector = (240, 240, 240) if i % 2 == 0 else (200, 200, 200)
            
            for alpha in angulos:
                mask = np.zeros(gris.shape, dtype=np.uint8)
                cv2.ellipse(mask, (cx, cy), (int(r_ext), int(r_ext)), 0, alpha, alpha + 45, 255, -1)
                if r_int > 0:
                    cv2.circle(mask, (cx, cy), int(r_int), 0, -1)
                
                # Visualización de la grilla (Blanco/Gris)
                overlay = img_viz.copy()
                cv2.addWeighted(overlay, 0.7, img_viz, 0.3, 0, img_viz) # Efecto transparencia
                
                # Evaluación: Solo hasta los 50 grados (los primeros 5 anillos)
                if i < 5: 
                    sector_res = cv2.bitwise_and(binaria, mask)
                    area_total_sector = np.count_nonzero(mask)
                    area_ocupada = np.count_nonzero(sector_res)
                    porcentaje_ocupacion = (area_ocupada / area_total_sector) * 100
                    
                    if porcentaje_ocupacion > 5: # Umbral mínimo para detectar algo
                        if porcentaje_ocupacion >= 70:
                            total_grados_afectados += 10
                            color_alerta = (0, 0, 255) # Rojo: 10 grados
                        else:
                            total_grados_afectados += 5
                            color_alerta = (0, 165, 255) # Naranja: 5 grados
                        
                        # Pintar sector detectado
                        cv2.ellipse(img_viz, (cx, cy), (int(r_ext), int(r_ext)), 0, alpha, alpha + 45, color_alerta, -1)

                # Dibujar Bisectrices y Círculos
                cv2.ellipse(img_viz, (cx, cy), (int(r_ext), int(r_ext)), 0, alpha, alpha + 45, (150,150,150), 1)

        # Dibujar cruz azul central
        cv2.line(img_viz, (cx-20, cy), (cx+20, cy), (255, 0, 0), 2)
        cv2.line(img_viz, (cx, cy-20), (cx, cy+20), (255, 0, 0), 2)
        
        st.image(img_viz, use_container_width=True)

        # --- CÁLCULO FINAL ---
        try:
            num, den = map(float, fn_input.split('/'))
            factor = (num/den) + 1 if den > 0 else 1
        except: factor = 1
        
        grados_finales = min(total_grados_afectados * factor, 320)
        pje_perdida = (grados_finales / 320) * 100
        inc_final = pje_perdida * 0.25

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Suma de Grados Afectados", f"{total_grados_afectados}°")
        c2.metric("Pérdida de Campo", f"{round(pje_perdida, 2)}%")
        c3.metric("Incapacidad Final", f"{round(inc_final, 2)}%")
