import streamlit as st
import cv2
import numpy as np
from streamlit_image_coordinates import streamlit_image_coordinates

def app():
    st.set_page_config(layout="wide", page_title="Peritaje de Precisión")
    st.title("⚖️ Analizador de Incapacidad Campimétrica")
    st.write("1. Subí la imagen. 2. Hacé **clic** en el centro. 3. Ajustá la **Sensibilidad** hasta que coincidan los sectores rojos.")

    img_file = st.file_uploader("Subir Campo Visual", type=['jpg', 'png', 'jpeg'])

    if img_file:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        h, w = img.shape[:2]

        # Captura de clic para el centro
        st.write("### Hacé clic en la cruz central del informe:")
        value = streamlit_image_coordinates(img, key="pil")
        cx, cy = (value["x"], value["y"]) if value else (w // 2, h // 2)

        # --- PANEL DE CONTROL ---
        st.sidebar.header("📏 Calibración Pericial")
        escala = st.sidebar.slider("Escala (Ajustar a marcas 60°)", 50, w//2, w//4)
        
        # ESTE ES EL AJUSTE CLAVE:
        sensibilidad = st.sidebar.slider("Sensibilidad de Detección", 100, 5000, 1000, step=100)
        
        fn_input = st.sidebar.text_input("Falsos Negativos (ej. 0/7)", "0/7")
        
        st.sidebar.divider()
        adj_x = st.sidebar.slider("Ajuste fino X", -50, 50, 0)
        adj_y = st.sidebar.slider("Ajuste fino Y", -50, 50, 0)
        cx, cy = cx + adj_x, cy + adj_y

        # --- PROCESAMIENTO ---
        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Umbralización adaptativa para manejar sombras en la foto
        binaria = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        img_viz = img.copy()
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
                
                overlap = cv2.bitwise_and(binaria, mask)
                sumatoria = np.sum(overlap)
                
                # Usamos la sensibilidad del slider
                if sumatoria > sensibilidad:
                    sectores_fallados += 1
                    cv2.ellipse(img_viz, (cx, cy), (int(r_ext), int(r_ext)), 0, alpha, alpha + 45, (0, 0, 255), 2)
                else:
                    cv2.ellipse(img_viz, (cx, cy), (int(r_ext), int(r_ext)), 0, alpha, alpha + 45, (0, 255, 0), 1)

        cv2.drawMarker(img_viz, (cx, cy), (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=30, thickness=2)
        st.image(img_viz, use_container_width=True)

        # --- CÁLCULO ---
        try:
            num, den = map(float, fn_input.split('/'))
            factor = (num/den) + 1 if den > 0 else 1
        except: factor = 1
        
        grados_perdidos = sectores_fallados * 10
        grados_finales = min(grados_perdidos * factor, 320)
        porcentaje_perdida = (grados_finales / 320) * 100
        inc_final = porcentaje_perdida * 0.25

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Sectores Afectados", sectores_fallados)
        c2.metric("Suma de Grados", f"{round(grados_finales, 1)}°")
        c3.metric("Incapacidad Final", f"{round(inc_final, 2)}%")

if __name__ == "__main__":
    app()
