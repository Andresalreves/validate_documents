
#This function(fwdHaarDWT2D) computes the 2D Wavelet Transform in the image. All the input images are passed through a Haar Wavelet Decomposition module, to get the LL, LH, HL and HHH component of the image

import numpy as np
import pywt

def splitFreqBands(img, levRows, levCols):
    halfRow = int(levRows/2)
    halfCol = int(levCols/2)
    LL = img[0:halfRow, 0:halfCol]
    LH = img[0:halfRow, halfCol:levCols]
    HL = img[halfRow:levRows, 0:halfCol]
    HH = img[halfRow:levRows, halfCol:levCols]
    
    return LL, LH, HL, HH
    
def haarDWT1D(data, length):
    avg0 = 0.5;
    avg1 = 0.5;
    dif0 = 0.5;
    dif1 = -0.5;
    temp = np.empty_like(data)
    temp = temp.astype(float)
    h = int(length/2)
    for i in range(h):
        k = i*2
        temp[i] = data[k] * avg0 + data[k + 1] * avg1;
        temp[i + h] = data[k] * dif0 + data[k + 1] * dif1;
    
    data[:] = temp

# computes the homography coefficients for PIL.Image.transform using point correspondences
def fwdHaarDWT2D(img):
    img = np.array(img)
    levRows = img.shape[0];
    levCols = img.shape[1];
    img = img.astype(float)
    for i in range(levRows):
        row = img[i,:]
        haarDWT1D(row, levCols)
        img[i,:] = row
    for j in range(levCols):
        col = img[:,j]
        haarDWT1D(col, levRows)
        img[:,j] = col
        
    return splitFreqBands(img, levRows, levCols)

def crop_to_min_shape(LL, LH, HL, HH):
    min_rows = min(LL.shape[0], LH.shape[0], HL.shape[0], HH.shape[0])
    min_cols = min(LL.shape[1], LH.shape[1], HL.shape[1], HH.shape[1])

    LL = LL[:min_rows, :min_cols]
    LH = LH[:min_rows, :min_cols]
    HL = HL[:min_rows, :min_cols]
    HH = HH[:min_rows, :min_cols]

    return LL, LH, HL, HH