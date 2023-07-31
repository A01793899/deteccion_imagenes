

import streamlit as st
import torch
import cv2
from PIL import Image
import numpy as np
import pandas as pd

#Cargar el modelo
def detector(image, conf, iou):
    #Leer el modelo
    model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path = 'C:/Users/julian.lopez/OneDrive - SOCODA S.A.S/Escritorio/__video_vamos_de_nuevo/model/mi_modelo.pt')


    model.conf = conf
    model.iou = iou
    detect = model(image)

    return(detect)

#Configuración del sitio

#Configuración del sitio
st.title("App para verificación de componentes")

conf = st.slider('% Confidence', 0, 99, 1)/100
iou = st.slider('% Overlap (IoU)', 0, 99, 1)/100

#Cargar la imagen
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    # To read file as bytes:

    st.image(uploaded_file)
    #st.write(uploaded_file)

    #Preprocesamiento de imágenes para cambiarla a un arreglo de numpy.ndarray
    bytes_data = uploaded_file.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)


    #Voy a mandar la imagen al modelo y mostrar los resultados

    detect = detector(cv2_img, conf, iou)
    #st.write(detect)

    #detect.ims # array of original images (as np array) passed to model for inference
    #detect.render() # updates results.ims with boxes and labels


    ##La siguiente línea convierte el resultado a json.
    #vector = detect.pandas().xyxy[0].to_json(orient = "records")


    #La siguiente expresión convierte el resultado en un arreglo.
    arreglo = detect.pandas().xyxy[0]

    #Como es un dataframe
    # **********El siguiente renglón imprime la matríz completa
    #st.write(arreglo)

    #De aquí en adelante voy a comenzar sacar las categorías distintas
    #st.write(arreglo.shape) #para determinar el tamaño del arreglo con los resultados

    #En esta sección voy a crear un dataframe con la cuenta de los resultados.


    #***************************************************************
    #nombres = pd.DataFrame({"name": arreglo["name"].unique()})

    #nombres_lista = nombres.values.tolist()

   # data = {"name": nombres_lista, "count": 0*len(nombres_lista)}

    data = {"name": arreglo["name"].unique(), "count": 0 * len(arreglo["name"].unique())}

    resultados = pd.DataFrame(data)


    #Ahora voy a llenar el arreglo anterior con la cuenta de los resultados que cumplen con la confianza.


    filas1 , columnas1 = arreglo.shape
    filas2 , columnas2 = resultados.shape

    #En esta variable voy a determinar el umbral a partir del cual va a contar las clases distintas
    confianza = 0.90

    #st.write(arreglo["name"][1])
    #st.write(resultados["name"][1])

    for i in range(0, filas1):

        if arreglo["confidence"][i] >= confianza:

            for j in range(0, filas2):

                if resultados['name'][j] == arreglo['name'][i]:
                    resultados['count'][j] = resultados['count'][j] + 1



    st.write(resultados)
