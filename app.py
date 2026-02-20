import streamlit as st
import cv2
import numpy as np
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(layout="wide", page_title="Peritaje Pro")
st.title("Analizador de Incapacidad")

img_file = st.sidebar.file_uploader("Subir Campo Visual", type=['jpg', 'png', 'jpeg'])

if img_file is not None:
file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
img = cv2.imdecode(file_bytes, 1)
h, w = img.shape[:2]
