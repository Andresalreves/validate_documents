from PIL import Image
import face_recognition
from fastapi import HTTPException

def segmentar_rostro(ruta_imagen, ruta_segmentacion):

  # Cargar la imagen
  imagen = face_recognition.load_image_file(ruta_imagen)

  # Encontrar los rostros en la imagen
  try:
    ubicaciones_rostros = face_recognition.face_locations(imagen)
  except Exception as e:
    print(e)
    raise HTTPException(status_code=406,detail=f"Nose ha encontrado un rostro en esta imagen {ruta_imagen}")
  print(f"Se encontraron {len(ubicaciones_rostros)} rostros en la imagen.")

  for ubicacion_rostro in ubicaciones_rostros:
    top, right, bottom, left = ubicacion_rostro
    print(f"Rostro ubicado en: Top: {top}, Left: {left}, Bottom: {bottom}, Right: {right}")

    # Segmentar el rostro
    rostro_segmentado = imagen[top:bottom,left:right]
    Image.fromarray(rostro_segmentado).save(ruta_segmentacion)
    """
    # Guardar la imagen segmentada
    if ruta_imagen.find("selfie") != -1:
      new_width = 600
      with Image.open(ruta_segmentacion) as segmentacion:
        width, height = segmentacion.size
        new_height = int((new_width * height) / width)
        image_resize = segmentacion.resize((new_width, new_height))
        image_resize.save(ruta_segmentacion)
    """
    return ruta_segmentacion