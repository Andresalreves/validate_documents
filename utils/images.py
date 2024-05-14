from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from sklearn import preprocessing
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from fastai.learner import load_learner
from .haar2D import fwdHaarDWT2D, crop_to_min_shape
from matplotlib import pyplot as plt
from pywt import wavedec2
import imagehash
import requests
import numpy as np
from fastapi import HTTPException
from PIL import Image
import pytesseract
import pyclamd
import shutil
import pywt
import os
import io
import cv2

def is_image(upload_file):
    """Verifica si el archivo es una imagen basándose en su contenido."""
    allowed_extensions = {'image/jpeg', 'image/jpg', 'image/png'}  # Puedes ajustar según tus necesidades

    # Verifica si la extensión está en la lista de extensiones permitidas
    if upload_file.content_type.lower() in allowed_extensions:
        if(upload_file.size > 1):#2800000):
            return True
        else:
            raise HTTPException(status_code=404,detail=f"El archivo {upload_file.filename} no tiene la calidad suficiente para realizar la comprobacion, por favor toma la foto de nuevo en mejor resolucion.")
    else:
        raise HTTPException(status_code=404,detail=f"El archivo {upload_file.filename} no es un archivo valido, por favor utilice imagenes en formato .jpg, jpeg o png.")

def convert_to_opencv_extract_text(image):
    """
    image_pil = Image.open(image.file)
    # Binarizar la imagen con PIL
    image_pil = image_pil.convert("L")  # Convertir a escala de grises
    image_bin = image_pil.point(lambda x: 0 if x < 128 else 255, '1')  # Binarizar
    # Realizar OCR con PyTesseract
    text = pytesseract.image_to_string(image_bin)
    # Cerrar el archivo de la imagen
    image_pil.close()
    print(text)
    return text
    """
    # Leer la imagen con OpenCV
    image_opencv = cv2.imread(image)

    # Preprocesamiento
    gray = cv2.cvtColor(image_opencv, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    blur = cv2.GaussianBlur(thresh, (5, 5), 0)
    contrasted = cv2.equalizeHist(blur)

    # Detección de contornos
    contours, _ = cv2.findContours(contrasted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrar contornos y realizar OCR
    text = ""
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 100 and h > 20:
            roi = contrasted[y:y+h, x:x+w]
            text += pytesseract.image_to_string(roi, lang="spa")
    print(text)
    # Cerrar ventanas de OpenCV
    #cv2.destroyAllWindows()

    return text

def scan_file(file):
    try:
        # Conectar al demonio ClamAV
        cd = pyclamd.ClamdUnixSocket()

        # Verificar si la conexión fue exitosa
        if not cd.ping():
            raise pyclamd.ConnectionError('No se pudo conectar al demonio ClamAV. ¿Está ejecutándose?')

        # Guardar el archivo temporalmente
        file_path = f'./tmp/{file.filename}'
        with open(file_path, 'wb') as f:
            f.write(file.file.read())

        # Escanear el archivo
        scan_result = cd.scan_file(file_path)

        # Verificar el resultado del escaneo
        if scan_result[file_path] == 'OK':
            print(f'El archivo {file.filename} está limpio, no se encontraron virus.')
        else:
            print(f'El archivo {file.filename} está infectado. Resultado del escaneo: {scan_result[file_path]}')

    except pyclamd.ConnectionError as e:
        print(f'Error de conexión: {e}')
    except Exception as e:
        print(f'Error inesperado: {e}')

def scan_file2(file):
    try:
        # Conectar al demonio ClamAV
        cd = pyclamd.ClamdUnixSocket()

        # Verificar si la conexión fue exitosa
        if not cd.ping():
            raise pyclamd.ConnectionError('No se pudo conectar al demonio ClamAV. ¿Está ejecutándose?')

        # Guardar el archivo temporalmente
        file_path = f'/tmp/{file.filename}'
        with open(file_path, 'wb') as f:
            f.write(file.file.read())

        # Escanear el archivo
        scan_result = cd.scan_file(file_path)

        # Verificar el resultado del escaneo
        if scan_result[file_path] == 'OK':
            print(f'El archivo {file.filename} está limpio, no se encontraron virus.')
        else:
            print(f'El archivo {file.filename} está infectado. Resultado del escaneo: {scan_result[file_path]}')

    except pyclamd.ConnectionError as e:
        print(f'Error de conexión: {e}')
    except Exception as e:
        print(f'Error inesperado: {e}')

def SaveImage(imagen,carpeta,nombre_archivo):
    # Ruta de la carpeta
    ruta_carpeta = os.path.abspath(f'./face_recognition/Images/{carpeta}')

    # Crear la carpeta
    os.makedirs(ruta_carpeta, exist_ok=True)

    # Cambiar permisos de la carpeta
    os.chmod(ruta_carpeta, 0o777)

    ruta_archivo = f"{ruta_carpeta}/{nombre_archivo}"

    with open(ruta_archivo, "wb") as archivo:
        archivo.write(imagen.file.read())
    #predict_moire(ruta_archivo)
    image = cv2.imread(ruta_archivo)
    #print(detect_moire_pattern(image))
    if es_blanco_negro(image):
        if nombre_archivo == "cedula_frente.jpg" or nombre_archivo == "selfie.jpg":
            faceClassif = cv2.CascadeClassifier(os.path.abspath('./config/haarcascade_frontalface_default.xml'))
            height,width = image.shape[:2]
            new_width = 800
            new_height = (new_width * height) / width
            resized_image = cv2.resize(image, (new_width, int(new_height)))
            #cv2.imshow("Imagen redimencionada",resized_image)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            faces = faceClassif.detectMultiScale(gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30,30),
                maxSize=(200,200))
            if len(faces) == 0:
                # Eliminar la carpeta y su contenido
                try:
                    shutil.rmtree(shutil.rmtree(os.path.abspath(ruta_carpeta)))
                except Exception as e:
                    print(e)
                raise HTTPException(status_code=400,detail=f"No se ha detectado ningún rostro en la imagen {imagen.filename}.")
        return ruta_archivo
    
def detector_texturas(imagen):
    # Convertir la imagen a escala de grises
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Aplicar un filtro de suavizado para eliminar el ruido
    imagen_suavizada = cv2.GaussianBlur(imagen_gris, (5, 5), 0)

    # Calcular el patrón de puntos de la imagen
    puntos = cv2.cornerHarris(imagen_suavizada, 2, 3, 0.04)

    # Contar el número de puntos
    numero_puntos = np.count_nonzero(puntos)
    # Si el número de puntos es mayor que un umbral, la imagen es una fotocopia
    if numero_puntos > 1000:
        raise HTTPException(status_code=406,detail="La imagen que intentas subir parece ser una copia del documento original, por favor toma una foto de la imagen sin manipular y subela de nuevo.")
    else:
        return True
    
def wavelet_decomposition(image):
    coeffs = wavedec2(image, 'haar', level=1)
    cA, (cH, cV, cD) = coeffs
    return cA

def detect_moire_pattern(image):
    model = load_model(os.path.abspath('./face_recognition/models_ia/moire.h5'))
    LL = wavelet_decomposition(image)
    LL = cv2.resize(LL, (64, 64))  # Redimensionar a (64, 64)
    LL = LL / 255.0
    LL = np.expand_dims(LL, axis=-1)  # Agregar dimensión para el canal
    prediction = model.predict(np.array([LL]))
    print(prediction[0][0])
    return prediction[0][0] > 0.5


def detectar_defectos(imagen, umbral=5600):
    # Convertir la imagen a escala de grises
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Aplicar un filtro de suavizado para eliminar el ruido
    imagen_suavizada = cv2.GaussianBlur(imagen_gris, (5, 5), 0)

    # Calcular la varianza de la imagen
    varianza = np.var(imagen_suavizada)
    #print(varianza)
    # Si la varianza es menor que el umbral, la imagen ha sido escaneada
    if varianza < umbral:
        raise HTTPException(status_code=406,detail="La imagen que intentas subir parece ser una copia del documento original, por favor toma una foto de la imagen sin manipular y subela de nuevo.")
    else:
        return True

def es_blanco_negro(imagen):
    for y in range(imagen.shape[0]):
        for x in range(imagen.shape[1]):
            b, g, r = imagen[y, x]
            if b != g or g != r:
                return True
    return False


def es_escaneada(imagen):
    # Cargar el modelo VGG16
    model = VGG16(weights='imagenet')

    # Redimensiona la imagen a 224x224 píxeles
    imagen_redimensionada = cv2.resize(imagen, (224, 224))

    # Preprocesar la imagen
    imagen_preprocesada = preprocess_input(image.img_to_array(imagen_redimensionada))

    # Predecir la clase de la imagen
    predicciones = model.predict(imagen_preprocesada)

    # Decodificar las predicciones
    decoded_predictions = decode_predictions(predicciones)

    # Si la clase con mayor probabilidad es "Documento", la imagen ha sido escaneada
    if decoded_predictions[0][0][1] == "Documento":
        raise HTTPException(status_code=406,detail="La imagen que intentas subir parece ser una copia del documento original, por favor toma una foto de la imagen sin manipular y subela de nuevo.")
    else:
        return True


def SaveImageOld(imagen,carpeta,nombre_archivo):
    
    # Ruta de la carpeta
    ruta_carpeta = os.path.abspath(f'./face_recognition/Images/{carpeta}')
    # Crear la carpeta
    os.makedirs(ruta_carpeta, exist_ok=True)

    # Cambiar permisos de la carpeta
    os.chmod(ruta_carpeta, 0o777)

    ruta_archivo = f"{ruta_carpeta}/{nombre_archivo}"

    # Convertir imagen a formato PIL
    image_original = Image.open(imagen.file)

    # Binarizar imagen
    image = image_original.convert("L")
    image = image.point(lambda x: 0 if x < 128 else 255)
    try:
        # Detectar orientación del texto
        data = pytesseract.image_to_osd(np.asarray(image),output_type=Output.DICT)
        #print(data)
        #print(data["orientation"])
        #print(data["rotate"])
        if  data["orientation"] != 180:
            image_original = image_original.rotate(data["orientation"],expand=True)
            # Redimensionar la imagen manteniendo la relación de aspecto

    except Exception as e:
        print(e)
    image_original.save(ruta_archivo)
    return ruta_archivo

def ExtraerContornos(path_image):

    # Carga la imagen
    imagen = cv2.imread(path_image)

    # Convertir la imagen a escala de grises
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Desvanecimiento gaussiano para eliminar ruido
    gris = cv2.GaussianBlur(gris, (5, 5), 0)

    # Otsu's thresholding para binarizar la imagen
    umbral = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Encontrar contornos
    contornos, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Encontrar el contorno más grande
    mayor_area = 0
    mejor_contorno = None
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        #print(area)
        if area > mayor_area:
            print(area)
            mayor_area = area
            mejor_contorno = contorno

    # Dibujar el contorno del documento
    cv2.drawContours(imagen, contornos, 41, (0, 255, 0), 3)
    #cv2.drawContours(imagen, [mejor_contorno], -1, (0, 255, 0), 3)

    # Extraer el texto del documento
    # (Aquí se necesitaría usar un OCR para obtener el texto)

    # Mostrar la imagen
    cv2.imwrite('../cedula_contorno.jpg', imagen)
    #cv2.imshow('Imagen con contorno', imagen)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

def leer_cedula(path_image):
    #codigo_barras = recortar_codigo_barras(path_image)
    #documento = Image.open(codigo_barras)
    documento = Image.open(path_image)
    try:
        decoder = PDF417Decoder(documento)
        print(decoder.decode())
    except Except as e:
        raise HTTPException(status_code=406,detail=e)
    if (decoder.decode() > 0):
        return decoder.barcode_data_index_to_string(0)
    else:
        return "No se pudo extraer los datos del documento."


# Función para recortar el código de barras de una imagen
def recortar_codigo_barras(path_image):
    imagen = cv2.imread(path_image)
    # Convertir la imagen a escala de grises
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Desvanecimiento gaussiano para eliminar ruido
    gris = cv2.GaussianBlur(gris, (5, 5), 0)

    # Otsu's thresholding para binarizar la imagen
    umbral = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Encontrar contornos
    contornos, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Encontrar el contorno más grande (asumiendo que es el código de barras)
    mayor_area = 0
    mejor_contorno = None
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area > mayor_area:
            mayor_area = area
            mejor_contorno = contorno

    # Recortar la imagen alrededor del contorno
    x, y, w, h = cv2.boundingRect(mejor_contorno)
    codigo_barras = imagen[y:y+h, x:x+w]
    cv2.imwrite('./codigo_barras.jpg', codigo_barras)
    return "./codigo_barras.jpg"

def is_photocopy(image):
    # Cargar la imagen
    #image = cv2.imread(image_path)
    
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplicar un filtro de desenfoque gaussiano
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Calcular la varianza de los valores de píxeles en la imagen desenfocada
    variance = np.var(blurred)
    
    # Las fotocopias suelen tener una varianza menor debido a la pérdida de detalle
    # Puedes ajustar el umbral según tus necesidades
    is_copy = variance < 20
    
    return is_copy

def analizar_imagen(imagen):
    # Cargar la imagen
    #imagen = cv2.imread(ruta_imagen)

    # Convertir la imagen a escala de grises
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Verificar si la imagen es una fotocopia
    # Aquí puedes implementar algoritmos más sofisticados para detectar patrones de ruido y artefactos
    if np.mean(imagen_gris) < 100:  # Umbral arbitrario para simular la detección de una fotocopia
        es_fotocopia = True
    else:
        es_fotocopia = False

    # Verificar si la imagen está en escala de grises o blanco y negro
    if len(imagen.shape) == 2:  # Solo un canal de color
        es_escala_grises = True
        es_blanco_negro = False
    else:
        es_escala_grises = False
        if imagen.shape[2] == 1:  # Solo un canal de color (blanco y negro)
            es_blanco_negro = True
        else:
            es_blanco_negro = False

    # Verificar si la imagen se parece a una copia de una máquina copiadora
    lineas_horizontales = cv2.HoughLinesP(imagen_gris, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    lineas_verticales = cv2.HoughLinesP(imagen_gris, rho=1, theta=np.pi/2, threshold=100, minLineLength=100, maxLineGap=10)
    if lineas_horizontales is not None and lineas_verticales is not None:
        parece_copia_maquina = True
    else:
        parece_copia_maquina = False

    return es_fotocopia, es_escala_grises, es_blanco_negro, parece_copia_maquina

def detectar_copia(imagen_cargada):
    imagen = Image.open(imagen_cargada)
    arreglo_imagen = np.array(imagen)
    
    # Calcular varianza y entropía
    varianza = np.var(arreglo_imagen)
    entropia = np.sum(arreglo_imagen * np.log2(arreglo_imagen + 1e-12)) / (-arreglo_imagen.size * np.log2(256))
    print(varianza)
    print(entropia)
    # Verificar si la imagen es binaria (blanco y negro)
    valores_unicos = np.unique(arreglo_imagen)
    if len(valores_unicos) <= 2:
        return True, "La imagen cargada es una copia (binaria)."
    
    # Verificar si la imagen tiene poca profundidad de color
    profundidad_color = arreglo_imagen.shape[2] if len(arreglo_imagen.shape) == 3 else 1
    if profundidad_color <= 8:
        return True, "La imagen cargada es una copia (poca profundidad de color)."
    
    # Verificar si la imagen tiene ruido de escaneo
    ruido_escaneo = np.mean(np.abs(arreglo_imagen - np.median(arreglo_imagen)))
    if ruido_escaneo > 10:
        return True, "La imagen cargada es una copia (ruido de escaneo)."
    
    # Umbrales para varianza y entropía
    umbral_varianza = 20
    umbral_entropía = 7
    
    if varianza < umbral_varianza and entropía < umbral_entropía:
        return True, "La imagen cargada es una copia (baja varianza y entropía)."
    else:
        return False, "La imagen cargada es una imagen original."