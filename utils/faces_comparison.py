from .porcentaje_match import distancia_a_porcentaje
from fastapi import HTTPException
import face_recognition
import traceback
import os

def compare_faces(known_image_path, unknown_image_path,threshold):
  # Cargar las imagenes
  known_image = face_recognition.load_image_file(known_image_path)
  unknown_image = face_recognition.load_image_file(unknown_image_path)

  # Obtener los encodings de los rostros
  try:
    known_face_encoded = face_recognition.face_encodings(known_image)[0]
  except Exception as e:
    raise HTTPException(status_code=404,detail="La imagenen de tu cedula no pueden ser procesada, realiza la captura de nuevo, intenta con otro angulo, otras condiciones de iluminacion u otra camara de mayor resolucion.")
  try:  
    unknown_face_encoded = face_recognition.face_encodings(unknown_image)[0]
  except Exception as e:
    raise HTTPException(status_code=404,detail="Las imagenen de tu selfie no pueden ser procesada, realiza la captura de nuevo, intenta con otro angulo, otras condiciones de iluminacion u otra camara de mayor resolucion.")
  # Comparar los encodings de los rostros
  #results = face_recognition.compare_faces([known_face_encoded], unknown_face_encoded)

  distance = face_recognition.face_distance([known_face_encoded], unknown_face_encoded)
  print(distance[0])
  if(distance[0] < threshold):
    return {"resultado":True,"porcentaje":f"{distancia_a_porcentaje(distance[0],threshold)}%","message":"los rostros coinciden","distance":distance[0]}
  else:
    return {"resultado":False,"porcentaje":f"{distancia_a_porcentaje(distance[0],threshold)}%","message":"Los rostros no coinciden","distance":distance[0]}
