import streamlit as st
import cv2
import numpy as np
from streamlit_image_coordinates import streamlit_image_coordinates

def app():
    st.set_page_config(layout="wide", page_title="Peritaje de Precisión")
    st.title("⚖️ Analizador de Incapacidad Campimétrica")
    
    st.sidebar.header("⚙️ Configuración")
    img_file = st.sidebar.file_uploader("Subir Campo Visual", type=['jpg', 'png', 'jpeg'])
    
    if img_file:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        h, w = img.shape[:2]

        st.write("### 1. Centrado: Hacé clic en la unión de los ejes (0,0)")
        value = streamlit_image_coordinates(img, key="pil")
        cx, cy = (value["x"], value["y"]) if value else (w // 2, h // 2)

        st.sidebar.subheader("📏 Calibración de Escala")
        radio_10 = st.sidebar.slider("Píxeles por cada 10°", 10, 200, 50)
        
        st.sidebar.subheader("🎯 Filtro de Cuadrados")
        sensibilidad = st.sidebar.slider("Sensibilidad de Detección", 100, 10000, 2000, step=100)
        
        fn_input = st.sidebar.text_input("Falsos Negativos (ej. 0/4)", "0/4")

        # --- PROCESAMIENTO ---
        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binaria = cv2.threshold(gris, 120, 255, cv2.THRESH_BINARY_INV)
        
        img_viz = img.copy()
        radios_px = [10 * (radio_10/10), 20 * (radio_10/10), 30 * (radio_10/10), 60 * (radio_10/10)]
        angulos = np.linspace(0, 315, 8)
        
        # INICIALIZAMOS EL CONTADOR DE SECTORES
        conteo_sectores_reales = 0

        for i, r_ext in enumerate(radios_px):
            r_int = 0 if i == 0 else radios_px[i-1]
            for alpha in angulos:
                mask = np.zeros(gris.shape, dtype=np.uint8)
                cv2.ellipse(mask, (cx, cy), (int(r_ext), int(r_ext)), 0, alpha, alpha + 45, 255, -1)
                if r_int > 0:
                    cv2.circle(mask, (cx, cy), int(r_int), 0, -1)
                
                # Analizar este sector específico
                sector_res = cv2.bitwise_and(binaria, mask)
                pixels_en_sector = np.sum(sector_res) / 255
                
                if pixels_en_sector > (sensibilidad / 10):
                    # SI EL SECTOR TIENE UN CUADRADO, SUMAMOS 1
                    conteo_sectores_reales += 1
                    
                    # Dibujar sombreado de detección
                    overlay = img_viz.copy()
                    cv2.ellipse(overlay, (cx, cy), (int(r_ext), int(r_ext)), 0, alpha, alpha + 45, (255, 0, 0), -1)
                    if r_int > 0:
                        cv2.circle(overlay, (cx, cy), int(r_int), (0,0,0), -1)
                    cv2.addWeighted(overlay, 0.3, img_viz, 0.7, 0, img_viz)
                else:
                    cv2.ellipse(img_viz, (cx, cy), (int(r_ext), int(r_ext)), 0, alpha, alpha + 45, (0, 255, 0), 1)

        cv2.drawMarker(img_viz, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 40, 2)
        st.image(img_viz, use_container_width=True)

        # --- CÁLCULO FINAL BASADO EN SUMA DE SECTORES ---
        try:
            num, den = map(float, fn_input.split('/'))
            factor = (num/den) + 1 if den > 0 else 1
        except: factor = 1
        
        # Cada sector vale 10 grados (Total 320)
        grados_perdidos = conteo_sectores_reales * 10
        grados_finales = min(grados_perdidos * factor, 320)
        pje_perdida = (grados_finales / 320) * 100
        inc_final = pje_perdida * 0.25

        st.divider()
        c1, c2, c3 = st.columns(3)
        # AQUÍ SE MOSTRARÁ EL NÚMERO REAL DE CUADRADOS (Ej: 11)
        c1.metric("Sectores con Cuadrados", conteo_sectores_reales)
        c2.metric("Suma de Grados", f"{round(grados_finales, 1)}°")
        c3.metric("Incapacidad Final", f"{round(inc_final, 2)}%")

if __name__ == "__main__":
    app()
