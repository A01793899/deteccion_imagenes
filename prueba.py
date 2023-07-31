import streamlit as st
import torch
import cv2
from PIL import Image
import numpy as np
import pandas as pd

picture = st.camera_input("Take a picture")

if picture:
    st.image(picture)
