import streamlit as st
import cv2
import numpy as np

def detectar_ejes_mejorado(img_gris):
    # Filtro para resaltar líneas negras
    img_inv = cv2.bitwise_not(img_gris)
    bordes = cv2.Canny(img_inv, 50, 150)
    # Buscamos líneas muy largas (los ejes)
    lineas = cv2.HoughLinesP(bordes, 1, np.pi/180, threshold=150, minLineLength=200, maxLineGap=20)
    
    cx, cy = img_gris.shape[1]//2, img_gris.shape[0]//2
    
    if lineas is not None:
        v_lines = []
        h_lines = []
        for l in lineas:
            x1, y1, x2, y2 = l[0]
            if abs(x1 - x2) < 10: v_lines.append((x1 + x2) / 2)
            if abs(y1 - y2) < 10: h_lines.append((y1 + y2) / 2)
        
        if v_lines and h_lines:
            cx = int(np.median(v_lines))
            cy = int(np.median(h_lines))
            
    return cx, cy

def app():
    st.set_page_config(layout="wide", page_title="Peritaje Campimétrico")
    st.title("⚖️ Analizador de Incapacidad (Precisión Geométrica)")

    img_file = st.file_uploader("Subir Campo Visual", type=['jpg', 'png', 'jpeg'])

    if img_file:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Detección automática base
        auto_cx, auto_cy = detectar_ejes_mejorado(gris)

        # 2. Sidebar para ajuste manual (Si el auto falla, movés esto)
        st.sidebar.header("Calibración de Ejes")
        adj_x = st.sidebar.slider("Ajuste Centro X", -200, 200, 0)
        adj_y = st.sidebar.slider("Ajuste Centro Y", -200, 200, 0)
        cx, cy = auto_cx + adj_x, auto_cy + adj_y
        
        # Escala: Buscamos que el círculo externo toque las marcas de 60°
        escala = st.sidebar.slider("Escala (Ajustar a marcas 60°)", 0.5, 2.0, 1.0, 0.01)
        radio_max = int(img.shape[1] * 0.38 * escala)
        
        fn_input = st.sidebar.text_input("Falsos Negativos", "0/8")

        # 3. Procesamiento y Dibujo de la Red
        _, binaria = cv2.threshold(gris, 120, 255, cv2.THRESH_BINARY_INV)
        img_viz = img.copy()
        
        # Definición de sectores geométricos (Red de 320° totales)
        radios = [radio_max * 0.16, radio_max * 0.33, radio_max * 0.5, radio_max]
        angulos = np.linspace(0, 315, 8)
        sectores_fallados = 0

        for i, r_ext in enumerate(radios):
            r_int = 0 if i == 0 else radios[i-1]
            for alpha in angulos:
                mask = np.zeros(gris.shape, dtype=np.uint8)
                cv2.ellipse(mask, (cx, cy), (int(r_ext), int(r_ext)), 0, alpha, alpha + 45, 255, -1)
                if r_int > 0:
                    cv2.circle(mask, (cx, cy), int(r_int), 0, -1)
                
                # Detectar si el sector contiene un escotoma (cuadrado negro)
                overlap = cv2.bitwise_and(binaria, mask)
                if np.sum(overlap) > 450: # Umbral de sensibilidad
                    sectores_fallados += 1
                    cv2.ellipse(img_viz, (cx, cy), (int(r_ext), int(r_ext)), 0, alpha, alpha + 45, (0, 0, 255), 2)
                else:
                    cv2.ellipse(img_viz, (cx, cy), (int(r_ext), int(r_ext)), 0, alpha, alpha + 45, (0, 255, 0), 1)

        # Dibujar Ejes de Referencia
        cv2.line(img_viz, (cx-40, cy), (cx+40, cy), (255, 0, 0), 2)
        cv2.line(img_viz, (cx, cy-40), (cx, cy+40), (255, 0, 0), 2)

        st.image(img_viz, use_container_width=True, caption="Verificá que el centro azul coincida con la cruz del informe")

        # 4. Cálculo Final
        try:
            num, den = map(float, fn_input.split('/'))
            factor = (num/den) + 1 if den > 0 else 1
        except: factor = 1
        
        # Cada sector de la red geométrica suma 10°
        grados_perdidos = sectores_fallados * 10
        grados_finales = min(grados_perdidos * factor, 320)
        porcentaje_perdida = (grados_finales / 320) * 100
        incapacidad_baremo = porcentaje_perdida * 0.25

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Sectores Afectados", sectores_fallados)
        c2.metric("Suma de Grados", f"{round(grados_finales, 1)}°")
        c3.metric("Incapacidad Final", f"{round(incapacidad_baremo, 2)}%")
        
        st.info(f"Lógica pericial: {sectores_fallados} sectores detectados en la red geométrica. Total grados: {grados_finales}° / 320°. Coeficiente: 0.25.")

if __name__ == "__main__":
    app()
