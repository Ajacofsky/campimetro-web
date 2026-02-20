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

        # 1. Instrucciones y Captura de Centro
        st.write("### 1. Hacé clic en el cruce de los ejes (0,0)")
        value = streamlit_image_coordinates(img, key="pil")
        cx, cy = (value["x"], value["y"]) if value else (w // 2, h // 2)

        # 2. Parámetros en Sidebar
        st.sidebar.subheader("📏 Calibración de Escala")
        # Ajustá esto mirando las muescas de 10, 20, 30, 60 en el eje X
        radio_10 = st.sidebar.slider("Píxeles por cada 10°", 10, 200, 50)
        
        st.sidebar.subheader("🎯 Filtro de Detección")
        # Subí este valor si te da 25% constante para "limpiar" el ruido
        sensibilidad = st.sidebar.slider("Umbral de Sensibilidad", 500, 8000, 2000, step=100)
        
        fn_input = st.sidebar.text_input("Falsos Negativos (ej. 0/7)", "0/7")

        # --- Lógica de Procesamiento ---
        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Umbralización para resaltar solo lo muy negro (cuadrados)
        _, binaria = cv2.threshold(gris, 100, 255, cv2.THRESH_BINARY_INV)
        
        img_viz = img.copy()
        radios_grados = [10, 20, 30, 60]
        radios_px = [r * (radio_10 / 10) for r in radios_grados]
        angulos = np.linspace(0, 315, 8)
        sectores_fallados = 0

        for i, r_ext in enumerate(radios_px):
            r_int = 0 if i == 0 else radios_px[i-1]
            for alpha in angulos:
                mask = np.zeros(gris.shape, dtype=np.uint8)
                cv2.ellipse(mask, (cx, cy), (int(r_ext), int(r_ext)), 0, alpha, alpha + 45, 255, -1)
                if r_int > 0:
                    cv2.circle(mask, (cx, cy), int(r_int), 0, -1)
                
                res = cv2.bitwise_and(binaria, mask)
                puntos_negros = np.sum(res) / 255 # Cantidad de píxeles negros reales
                
                if puntos_negros > (sensibilidad / 10): 
                    sectores_fallados += 1
                    cv2.ellipse(img_viz, (cx, cy), (int(r_ext), int(r_ext)), 0, alpha, alpha + 45, (0, 0, 255), 2)
                else:
                    cv2.ellipse(img_viz, (cx, cy), (int(r_ext), int(r_ext)), 0, alpha, alpha + 45, (0, 255, 0), 1)

        # Guía visual
        cv2.drawMarker(img_viz, (cx, cy), (255, 0, 0), cv2.MARKER_CROSS, 40, 2)
        st.image(img_viz, use_container_width=True)

        # --- Cálculos Finales ---
        try:
            num, den = map(float, fn_input.split('/'))
            factor = (num/den) + 1 if den > 0 else 1
        except: factor = 1
        
        grados_perdidos = sectores_fallados * 10
        grados_finales = min(grados_perdidos * factor, 320)
        pje_perdida = (grados_finales / 320) * 100
        incapacidad = pje_perdida * 0.25

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Sectores en Falla", sectores_fallados)
        c2.metric("Suma de Grados", f"{round(grados_finales, 1)}°")
        c3.metric("Incapacidad Final", f"{round(incapacidad, 2)}%")

if __name__ == "__main__":
    app()
