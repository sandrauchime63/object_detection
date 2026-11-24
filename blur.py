import cv2 as cv
import numpy as np
from pathlib import Path


path=Path("License-Plate-Recognition-13")
train=path/'train'/'images'
train2=list(train.iterdir())
look=cv.imread(train2[0])


blurred=path/'blurred'
def create_motion_blurs(image, kernel_size=9):

    kernel_h = np.zeros((kernel_size, kernel_size))
    kernel_h[kernel_size // 2, :] = 1

    kernel_v = np.zeros((kernel_size, kernel_size))
    kernel_v[:, kernel_size // 2] = 1

    kernel_d = np.zeros((kernel_size, kernel_size))
    np.fill_diagonal(kernel_d, 1)

   
    kernel_h /= kernel_h.sum()
    kernel_v /= kernel_v.sum()
    kernel_d /= kernel_d.sum()

    
    horizontal = cv.filter2D(image, -1, kernel_h)
    vertical = cv.filter2D(image, -1, kernel_v)
    diagonal = cv.filter2D(image, -1, kernel_d)

    return {
        "horizontal": horizontal,
        "vertical": vertical,
        "diagonal": diagonal
    }
run=create_motion_blurs(look)
cv.imshow('run', run['diagonal'])
cv.waitKey(0)
cv.destroyAllWindows()  
def save_blur(image, blur_type, output_path):
    pass