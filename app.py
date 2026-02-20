import streamlit as st
import cv2
import numpy as np
from streamlit_image_coordinates import streamlit_image_coordinates

def app():
    st.set_page_config(layout="wide", page_title="Peritaje de Precisión")
    st.title("⚖️ Analizador de Incapacidad Campimétrica")
    st.info("Instrucciones: 1. Clic en el centro exacto. 2. Ajustá el 'Radio por cada 10°' hasta que los círculos toquen las muescas del eje X.")

    img_file = st.file_uploader("Subir Campo Visual", type=['jpg', 'png', 'jpeg'])

    if img_file:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        h, w = img.shape[:2]

        # 1. Posicionamiento del Centro
        st.write("### Centrado: Hacé clic en la unión de los ejes X e Y")
        value = streamlit_image_coordinates(img, key="pil")
        cx, cy = (value["x"], value["y"]) if value else (w // 2, h // 2)

        # --- PANEL DE CALIBRACIÓN GEOMÉTRICA ---
        st.sidebar.header("📏 Calibración de Escala")
        # Este es el valor clave: cuántos píxeles equivalen a 10 grados en TU imagen
        radio_10_grados = st.sidebar.slider("Radio por cada 10° (Ajustar a muescas)", 10, 150, 40)
        
        st.sidebar.header("🎯 Detección")
        sensibilidad = st.sidebar.slider("Sensibilidad (Rojo = Detectado)", 100, 5000, 1500)
        fn_input = st.sidebar.text_input("Falsos Negativos", "0/8")

        # --- PROCESAMIENTO GEOMÉTRICO ---
        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binaria = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        img_viz = img.copy()
        
        # Definimos los radios como múltiplos de la escala de 10°: 10, 20, 30 y 60
        radios_grados = [10, 20, 30, 60]
        radios_pixeles = [r * (radio_10_grados / 10) for r in radios_grados]
        
        angulos = np.linspace(0, 315, 8)
        sectores_fallados = 0

        for i, r_ext in enumerate(radios_pixeles):
            r_int = 0 if i == 0 else radios_pixeles[i-1]
            for alpha in angulos:
                mask = np.zeros(gris.shape, dtype=np.uint8)
                # Dibujamos el sector circular exacto
                cv2.ellipse(mask, (cx, cy), (int(r_ext), int(r_ext)), 0, alpha, alpha + 45, 255, -1)
                if r_int > 0:
                    cv2.circle(mask, (cx, cy), int(r_int), 0, -1)
                
                overlap = cv2.bitwise_and(binaria, mask)
                if np.sum(overlap) > sensibilidad:
                    sectores_fallados += 1
                    cv2.ellipse(img_viz, (cx, cy), (int(r_ext), int(r_ext)), 0, alpha, alpha + 45, (0, 0, 255), 2) # Rojo
                else:
                    cv2.ellipse(img_viz, (cx, cy), (int(r_ext), int(r_ext)), 0, alpha, alpha + 45, (0, 255, 0), 1) # Verde

        # Dibujar marcas de centro y muescas guía
        cv2.drawMarker(img_viz, (cx, cy), (255, 0, 0), cv2.MARKER_CROSS, 30, 2)
        
        st.image(img_viz, use_container_width=True)

        # --- CÁLCULO ---
        try:
            num, den = map(float, fn_input.split('/'))
            factor = (num/den) + 1 if den > 0 else 1
        except: factor = 1
        
        # Suma de sectores geométricos (10° cada uno)
        grados_totales = min((sectores_fallados * 10) * factor, 320)
        inc_final = ((grados_totales / 320) * 100) * 0.25

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Sectores en zona de falla", sectores_fallados)
        c2.metric("Grados Totales (Máx 320°)", f"{round(grados_totales, 1)}°")
        c3.metric("Incapacidad Baremo", f"{round(inc_final, 2)}%")

if __name__ == "__main__":
    app()
