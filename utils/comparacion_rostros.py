from .face_segmentation import segmentar_rostro
from .faces_comparison import compare_faces
#from craft_detector import extract_text
from .ocr import extract_text_front, extract_text_back
import os

def VerificarDocumento(ruta_documento,ruta_documento_atras,ruta_selfie,nombre_carpeta):

    ruta_segmentacion_documento = os.path.abspath(f'./face_recognition/Images/{nombre_carpeta}/SegmentacionDocumento.jpeg')
    ruta_segmentacion_selfie = os.path.abspath(f'./face_recognition/Images/{nombre_carpeta}/SegmentacionSelfie.jpeg')
    
    # Segmentar el rostro del documento
    segmentar_rostro(ruta_documento, ruta_segmentacion_documento)

    # Segmentar el rostro Selfie
    segmentar_rostro(ruta_selfie, ruta_segmentacion_selfie)


    # Rutas a las imágenes
    rostro_documento = ruta_segmentacion_documento
    rostro_selfie = ruta_segmentacion_selfie

    # Comparar los rostros
    try:
        compare_face = compare_faces(rostro_documento, rostro_selfie, 0.6)
    except Exception as e:
        print(e)
    #print(compare_face)
    # Imprimir el resultado
    if compare_face["resultado"]:
        try:
            # Realizar el OCR con PyTesseract y easyocr sin CRAFT-Pythorch
            segments1 = extract_text_front(ruta_documento)
            segments2 = extract_text_back(ruta_documento_atras)
        except Exception as e:
            print(e)
        response = {
            "comparacion_facial":compare_face,
            "datos_frente_documento":segments1,
            "datos_reverso_documento":segments2
        }
        return response
    else:
        return {"comparacion_facial":compare_face}
        #Extraccion de datos con CRAFT-Pytorch y pytesseract
        """
        try:

            segments1 = extract_text.init_craft(
                ruta_documento,
                os.path.abspath("./face_recognition/models_ia/craft_mlt_25k.pth"),
                os.path.abspath("./face_recognition/models_ia/craft_refiner_CTW1500.pth"), 
                False,
            )
            segments2 = extract_text.init_craft(
                ruta_documento_atras,
                os.path.abspath("./face_recognition/models_ia/craft_mlt_25k.pth"),
                os.path.abspath("./face_recognition/models_ia/craft_refiner_CTW1500.pth"), 
                False,
            )
            response = {
                "comparacion_facial":compare_face,
                "datos_frente_documento":segments1,
                "datos_reverso_documento":segments2
            }
            return response
        except Exception as e:
            return e
        """