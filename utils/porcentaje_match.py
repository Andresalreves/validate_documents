def distancia_a_porcentaje(distancia, umbral):
    if distancia < 0.6:
        porcentaje = 100 * (1-(int(distancia) / umbral))
    else:
        porcentaje = 100 * (1- distancia)
    return round(porcentaje,2)