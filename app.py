import streamlit as st
import cv2
import numpy as np

def app():
    st.set_page_config(layout="wide", page_title="Peritaje de Precisión")
    st.title("⚖️ Analizador de Incapacidad Campimétrica")
    st.write("Ajustá la red geométrica sobre el centro del informe para un cálculo real.")

    img_file = st.file_uploader("Subir Campo Visual", type=['jpg', 'png', 'jpeg'])

    if img_file:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        h, w = img.shape[:2]

        # --- PANEL DE CONTROL LATERAL ---
        st.sidebar.header("🕹️ Control de Posición")
        # Estos sliders permiten mover el centro pixel por pixel
        cx = st.sidebar.slider("Posición Centro X", 0, w, w//2)
        cy = st.sidebar.slider("Posición Centro Y", 0, h, h//2)
        
        st.sidebar.header("📏 Calibración")
        escala = st.sidebar.slider("Escala de la Red", 50, w//2, w//4)
        fn_input = st.sidebar.text_input("Falsos Negativos (ej. 0/7)", "0/7")

        # --- PROCESAMIENTO ---
        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binaria = cv2.threshold(gris, 110, 255, cv2.THRESH_BINARY_INV)
        
        img_viz = img.copy()
        
        # Red geométrica: 8 bisectrices y círculos de 10°, 20°, 30°, 60°
        # El radio_max se ajusta con el slider de escala
        radios = [escala * 0.16, escala * 0.33, escala * 0.5, escala]
        angulos = np.linspace(0, 315, 8)
        sectores_fallados = 0

        for i, r_ext in enumerate(radios):
            r_int = 0 if i == 0 else radios[i-1]
            for alpha in angulos:
                mask = np.zeros(gris.shape, dtype=np.uint8)
                cv2.ellipse(mask, (cx, cy), (int(r_ext), int(r_ext)), 0, alpha, alpha + 45, 255, -1)
                if r_int > 0:
                    cv2.circle(mask, (cx, cy), int(r_int), 0, -1)
                
                # Si el sector tiene un cuadrado negro (blanco en binaria)
                overlap = cv2.bitwise_and(binaria, mask)
                if np.sum(overlap) > 400: # Umbral de detección
                    sectores_fallados += 1
                    cv2.ellipse(img_viz, (cx, cy), (int(r_ext), int(r_ext)), 0, alpha, alpha + 45, (0, 0, 255), 2)
                else:
                    cv2.ellipse(img_viz, (cx, cy), (int(r_ext), int(r_ext)), 0, alpha, alpha + 45, (0, 255, 0), 1)

        # Dibujar Cruz de Centrado
        cv2.line(img_viz, (cx-20, cy), (cx+20, cy), (255, 0, 0), 2)
        cv2.line(img_viz, (cx, cy-20), (cx, cy+20), (255, 0, 0), 2)

        st.image(img_viz, use_container_width=True)

        # --- CÁLCULO FINAL ---
        try:
            num, den = map(float, fn_input.split('/'))
            factor = (num/den) + 1 if den > 0 else 1
        except: factor = 1
        
        grados_totales = min((sectores_fallados * 10) * factor, 320)
        incapacidad = ((grados_totales / 320) * 100) * 0.25

        st.divider()
        col1, col2, col3 = st.columns(3)
        col1.metric("Sectores Detectados", sectores_fallados)
        col2.metric("Grados Totales", f"{round(grados_totales, 1)}°")
        col3.metric("Incapacidad Final", f"{round(incapacidad, 2)}%")

if __name__ == "__main__":
    app()
