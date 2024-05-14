import cv2
import numpy as np
from fastapi import HTTPException 

faceClassif = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

#image = cv2.imread('cedula_adelante_mama.jpg')


image = cv2.imread('cedula_atras_miguel.jpg')
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

if len(faces) > 0:
    return True
else:
    raise HTTPException(status_code=406,detail="No se ha detectado ningún rostro en la imagen.")

for (x,y,w,h) in faces:
	cv2.rectangle(resized_image,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow('image',resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()