import cv2
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
from matplotlib import pyplot as plt

def draw_circles(image,points,color):
    for i in xrange(0,points.shape[0]):
        x = int(points[i,0])
        y = int(points[i,1])
        cv2.circle(image,(x,y),3,color,-1)



def plot_prediction(image, x, y, y_pred, lookback=5):

    copy_image  = image.copy()

    if y is not None:
        points = np.reshape(y, (-1, 2))
        draw_circles(copy_image, points, (0, 0, 255))

    points = np.reshape(y_pred, (-1, 2))
    draw_circles(copy_image,points,(0,255,0))



    points = np.reshape(x,(lookback,-1))[:,0:2]
    draw_circles(copy_image, points, (255, 0, 0))

    return copy_image


def plot_ensemble(image,x, y, y_pred, lookback=5):
    ensemble_transparency = np.zeros(image.shape,dtype='uint8')
    for i in xrange(0,y_pred.shape[0]):
        points = np.reshape(y_pred[i,:],(-1,2))
        draw_circles(ensemble_transparency,points,(0,255,0))


    copy_image = image.copy()
    copy_image[ensemble_transparency>0] = (0.3*copy_image[ensemble_transparency>0] +
                                           0.7* ensemble_transparency[ensemble_transparency>0])

    copy_image = plot_prediction(copy_image,x,y,np.mean(y_pred,0),lookback)

    return copy_image


