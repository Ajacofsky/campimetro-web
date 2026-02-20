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

        # 1. Captura de Centro
        st.write("### 1. Hacé clic en el cruce exacto de los ejes")
        value = streamlit_image_coordinates(img, key="pil")
        cx, cy = (value["x"], value["y"]) if value else (w // 2, h // 2)

        # 2. Sidebar
        st.sidebar.subheader("📏 Calibración de Escala")
        radio_10 = st.sidebar.slider("Píxeles por cada 10° (Mirar eje X)", 10, 200, 50)
        
        st.sidebar.subheader("🎯 Filtro de Cuadrados")
        # Este slider ahora controla qué tan "grande" debe ser la mancha para contarla
        umbral_area = st.sidebar.slider("Tamaño de Mancha Detectada", 10, 500, 100)
        
        fn_input = st.sidebar.text_input("Falsos Negativos", "0/8")

        # --- PROCESAMIENTO AVANZADO ---
        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Umbralización fuerte para separar el negro del blanco
        _, binaria = cv2.threshold(gris, 120, 255, cv2.THRESH_BINARY_INV)
        
        # FILTRO MORFOLÓGICO: Elimina texto y puntos, deja solo cuadrados
        kernel = np.ones((3,3), np.uint8)
        binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)
        
        img_viz = img.copy()
        radios_grados = [10, 20, 30, 60]
        radios_px = [r * (radio_10 / 10) for r in radios_grados]
        angulos = np.linspace(0, 315, 8)
        sectores_fallados = 0

        # Dibujar Red y Detectar
        for i, r_ext in enumerate(radios_px):
            r_int = 0 if i == 0 else radios_px[i-1]
            for alpha in angulos:
                mask = np.zeros(gris.shape, dtype=np.uint8)
                cv2.ellipse(mask, (cx, cy), (int(r_ext), int(r_ext)), 0, alpha, alpha + 45, 255, -1)
                if r_int > 0:
                    cv2.circle(mask, (cx, cy), int(r_int), 0, -1)
                
                # Contar píxeles blancos en la máscara (que representan negro en el original)
                res = cv2.bitwise_and(binaria, mask)
                conteo = np.count_nonzero(res)
                
                if conteo > umbral_area:
                    sectores_fallados += 1
                    # Pintar sector afectado en Rojo semi-transparente
                    overlay = img_viz.copy()
                    cv2.ellipse(overlay, (cx, cy), (int(r_ext), int(r_ext)), 0, alpha, alpha + 45, (0, 0, 255), -1)
                    if r_int > 0:
                        cv2.circle(overlay, (cx, cy), int(r_int), (0,0,0), -1)
                    cv2.addWeighted(overlay, 0.4, img_viz, 0.6, 0, img_viz)
                else:
                    # Dibujar borde verde para sectores sanos
                    cv2.ellipse(img_viz, (cx, cy), (int(r_ext), int(r_ext)), 0, alpha, alpha + 45, (0, 255, 0), 1)

        cv2.drawMarker(img_viz, (cx, cy), (255, 0, 0), cv2.MARKER_CROSS, 40, 2)
        st.image(img_viz, use_container_width=True)

        # --- CÁLCULOS (Suma de 320°) ---
        try:
            num, den = map(float, fn_input.split('/'))
            factor = (num/den) + 1 if den > 0 else 1
        except: factor = 1
        
        grados_perdidos = sectores_fallados * 10
        grados_finales = min(grados_perdidos * factor, 320)
        incapacidad = (grados_finales / 320) * 100 * 0.25

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Cuadrados Detectados", sectores_fallados)
        c2.metric("Grados Totales", f"{round(grados_finales, 1)}°")
        c3.metric("Incapacidad Final", f"{round(incapacidad, 2)}%")

if __name__ == "__main__":
    app()
