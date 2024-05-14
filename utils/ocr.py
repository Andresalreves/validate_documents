import cv2
import numpy as np
import imutils
import easyocr
import pytesseract
from pytesseract import Output
from PIL import Image
from matplotlib import pyplot as plt
import re

def extract_text_front(image_path):
    detectar_orientacion_corregir(image_path)
    img=cv2.imread(image_path)
    reader = easyocr.Reader(["es"], gpu=False)
    results = reader.readtext(img)
    datos_cedula = {}
    cedula = re.compile(r"\d{3}(\.\d{3})*")
    appellidos = False
    nombres = False
    for result in results:
        if(len(result[1]) > 4):
            if result[1].strip() == "CEDULA DE CIUDADANIA":
                datos_cedula['title'] = "REPUBLICA DE COLOMBIA IDENTIFICACION PERSONAL"
                datos_cedula['tipo_documento'] = result[1].strip()
            elif cedula.search(result[1]):
                datos_cedula['numero']= result[1].strip()
                appellidos = True
            elif appellidos:
                datos_cedula['apellidos'] = result[1].strip()
                appellidos = False
            if result[1].strip() == "APELLIDOS":
                nombres = True
            elif nombres:
                datos_cedula['nombres'] = result[1].strip()
                nombres = False
    print(datos_cedula)     
    return datos_cedula

def extract_text_back(image_path):
    detectar_orientacion_corregir(image_path)
    img=cv2.imread(image_path) 
    reader = easyocr.Reader(["es"], gpu=False)
    results = reader.readtext(img)
    datos_cedula = {}
    data = {}
    contador = 0
    estatura = re.compile(r"^[0-2][.][0-9][0-9]$")
    rh = re.compile(r"^(A|B|AB|O)([+-]{1})$")
    sexo = re.compile(r"^[FM]$")
    fecha_expedicion = re.compile(r"^(?:3[01]|[12][0-9]|0?[1-9])-(?:ENE|FEB|MAR|ABR|MAY|JUN|JUL|AGO|SEP|OCT|NOV|DIC)-\d{4}\s+(.*)$")
    fecha_nacimiento = False
    for result in results:
        if result[1].strip() == "FECHA DE NACIMIENTO":
            fecha_nacimiento = True
        elif fecha_nacimiento:
            datos_cedula['fecha_nacimiento'] = result[1].strip()
            fecha_nacimiento = False
        elif result[1].strip() == "LUGAR DE NACIMIENTO":
            datos_cedula['lugar_de_nacimiento'] = data[(contador - 2)].strip()+" "+data[(contador - 1)].strip()
        elif estatura.search(result[1]):
            datos_cedula['estatura'] = result[1].strip()
        elif rh.search(result[1]):
            datos_cedula['rh'] = result[1].strip()
        elif fecha_expedicion.search(result[1]):
            datos_cedula['fecha_y_lugar_de_expedicion'] = result[1].strip()
        data[contador] = result[1].strip()
        contador += 1 
    print(datos_cedula)
    return datos_cedula

def extract_text_pytesserac(img_file, boxes):
    # Lee la imagen usando OpenCV
    img = cv2.imread(img_file)
    response = [] 
    # Itera sobre los cuadros delimitadores y extrae el texto usando Tesseract
    #pytesseract.pytesseract.config['tessedit_char_whitelist'] = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

    # Configurar el idioma del texto a español
    #pytesseract.pytesseract.config['lang'] = 'spa'
    for i, box in enumerate(boxes):
        poly = np.array(box).astype(np.int32).reshape((-1, 2))
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        roi = cv2.bitwise_and(img, img, mask=mask)

        # Convierte la región a escala de grises
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)


        # Aplica umbral
        thresh = cv2.adaptiveThreshold(gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, 25)

        # Invierte el umbral para que el texto sea blanco sobre negro
        inverted_thresh = cv2.bitwise_not(thresh)

        # Convierte la región a imagen PIL para usar pytesseract
        pil_img = Image.fromarray(inverted_thresh)
        extracted_text = pytesseract.image_to_string(gray_roi, config='--psm 6')
        # Extrae el texto usando Tesseract
        # extracted_text = pytesseract.image_to_string(pil_img, config='--psm 6')  # --psm 6 para tratar como región de lista
        #if extracted_text.strip() != "":
            # Guardar el texto en el array
            #response.append(extracted_text.strip())
        print(f"Texto extraído de la región {i + 1}: {extracted_text.strip()}")
    return response
    print("Extracción de texto completada.")


def detectar_orientacion_corregir(path_image):
    image = Image.open(path_image)
    data = pytesseract.image_to_osd(np.asarray(image),output_type=Output.DICT)
    print(data)
    print(data["orientation"])
    print(data["rotate"])
    if  data["orientation"] != 0:
        new_orientation = image.rotate(data["rotate"],expand=True)
        # Redimensionar la imagen manteniendo la relación de aspecto
        new_orientation.save(path_image)

def es_flotante(cadena):
  try:
    float(cadena)
    return True
  except ValueError:
    return False