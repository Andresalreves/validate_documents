from fastapi import Depends, APIRouter, HTTPException, UploadFile, File
from utils.images import is_image, SaveImage, convert_to_opencv_extract_text
from utils.comparacion_rostros import VerificarDocumento
from middlewares.jwt_bearer import JWTBearer
from utils.detect_fake import perform_forensic_analysis
import uuid
import shutil
import os

ValidateDocuments_router = APIRouter()
#ValidateDocuments_router.debug = True
#dependencies=[Depends(JWTBearer())]
@ValidateDocuments_router.post("/DocumentoAdelante", tags=["/ValidarDocumentos"])
async def ValidateDocuments(
    cedula_frente: UploadFile = File(...)
):
    try:
        if(is_image(cedula_frente)):
            #scan_file2(cedula_frente)

            nombre_carpeta = uuid.uuid4()
            try:
                CedulaFrente = SaveImage(cedula_frente,nombre_carpeta,"cedula_frente.jpg")
            except Exception as e:
                print(e)
                
            #Analisis Metadatos    
            result_analysis = {
                "cedula_frente":perform_forensic_analysis(CedulaFrente, "exif")
            }

            # Ejemplos de otros tipo de analisis:
                #results = perform_forensic_analysis(CedulaFrente, "jpeg_ghost")
                #results = perform_forensic_analysis(CedulaFrente, "jpegghostm")
                #results = perform_forensic_analysis(CedulaFrente, "noise1")
                #results = perform_forensic_analysis(CedulaFrente, "noise2")
                #results = perform_forensic_analysis(CedulaFrente, "ela")
                #results = perform_forensic_analysis(CedulaFrente, "cfa")
                
            """
            try:
                # Eliminar la carpeta y su contenido
                shutil.rmtree(shutil.rmtree(os.path.abspath(f'./face_recognition/Images/{nombre_carpeta}')))
            except Exception as e:
                print(e)
            """
            return result_analysis
    
    except Exception as e:
        return e

@ValidateDocuments_router.post("/DocumentoAtras", tags=["/ValidarDocumentos"])
async def ValidateDocuments(
    cedula_atras: UploadFile = File(...)
):
    try:
        if(is_image(cedula_atras)):
            #scan_file(cedula_atras)
            
            nombre_carpeta = uuid.uuid4()
            CedulaAtras = SaveImage(cedula_atras,nombre_carpeta,"cedula_atras.jpg")
            
            #Analisis Metadatos    
            result_analysis = {
                "cedula_atras":perform_forensic_analysis(CedulaAtras, "exif")
            }

            # Ejemplos de otros tipo de analisis:
                #results = perform_forensic_analysis(CedulaFrente, "jpeg_ghost")
                #results = perform_forensic_analysis(CedulaFrente, "jpegghostm")
                #results = perform_forensic_analysis(CedulaFrente, "noise1")
                #results = perform_forensic_analysis(CedulaFrente, "noise2")
                #results = perform_forensic_analysis(CedulaFrente, "ela")
                #results = perform_forensic_analysis(CedulaFrente, "cfa")

            return result_analysis
    
    except Exception as e:
        return e

@ValidateDocuments_router.post("/Selfie", tags=["/ValidarDocumentos"])
async def ValidateDocuments(
    selfie: UploadFile = File(...)
):
    try:
        if is_image(is_image(selfie)):
            #scan_file(selfie)
            
            nombre_carpeta = uuid.uuid4()
            Selfie = SaveImage(selfie,nombre_carpeta,"selfie.jpg")

            #Analisis Metadatos    
            result_analysis = {
                "selfie":perform_forensic_analysis(Selfie, "exif")
            }

            # Ejemplos de otros tipo de analisis:
                #results = perform_forensic_analysis(CedulaFrente, "jpeg_ghost")
                #results = perform_forensic_analysis(CedulaFrente, "jpegghostm")
                #results = perform_forensic_analysis(CedulaFrente, "noise1")
                #results = perform_forensic_analysis(CedulaFrente, "noise2")
                #results = perform_forensic_analysis(CedulaFrente, "ela")
                #results = perform_forensic_analysis(CedulaFrente, "cfa")
                
            result = {
                "Verificacion_documento":VerificarDocumento(CedulaFrente,CedulaAtras,Selfie,nombre_carpeta),
                "Analisis_Metadatos":result_analysis
                }

            try:
                # Eliminar la carpeta y su contenido
                shutil.rmtree(shutil.rmtree(os.path.abspath(f'./face_recognition/Images/{nombre_carpeta}')))
            except Exception as e:
                print(e)

            return result
    
    except Exception as e:
        return e



"""
@ValidateDocuments_router.post("/ValidateDocuments", tags=["/ValidateDocuments"])
async def ValidateDocuments(
    cedula_frente: UploadFile = File(...),
    cedula_atras: UploadFile = File(...),
    selfie: UploadFile = File(...),
):
    try:
        if(is_image(cedula_frente) and is_image(cedula_atras) and is_image(selfie)):
            #scan_file2(cedula_frente)
            #scan_file(cedula_atras)
            #scan_file(selfie)
            
            nombre_carpeta = uuid.uuid4()
            CedulaFrente = SaveImage(cedula_frente,nombre_carpeta,"cedula_frente.jpg")
            CedulaAtras = SaveImage(cedula_atras,nombre_carpeta,"cedula_atras.jpg")
            Selfie = SaveImage(selfie,nombre_carpeta,"selfie.jpg")

            #Analisis Metadatos    
            result_analysis = {
                "cedula_frente":perform_forensic_analysis(CedulaFrente, "exif"),
                "cedula_atras":perform_forensic_analysis(CedulaAtras, "exif"),
                "selfie":perform_forensic_analysis(Selfie, "exif")
            }

            # Ejemplos de otros tipo de analisis:
                #results = perform_forensic_analysis(CedulaFrente, "jpeg_ghost")
                #results = perform_forensic_analysis(CedulaFrente, "jpegghostm")
                #results = perform_forensic_analysis(CedulaFrente, "noise1")
                #results = perform_forensic_analysis(CedulaFrente, "noise2")
                #results = perform_forensic_analysis(CedulaFrente, "ela")
                #results = perform_forensic_analysis(CedulaFrente, "cfa")
                
            result = {
                "Verificacion_documento":VerificarDocumento(CedulaFrente,CedulaAtras,Selfie,nombre_carpeta),
                "Analisis_Metadatos":result_analysis
                }

            try:
                # Eliminar la carpeta y su contenido
                shutil.rmtree(shutil.rmtree(os.path.abspath(f'./face_recognition/Images/{nombre_carpeta}')))
            except Exception as e:
                print(e)

            return result
    
    except Exception as e:
        return e
"""