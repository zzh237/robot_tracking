from datatools.bookkeeping import MultiInputDataHandler
from datatools.plotting import plot_ensemble
from custommodels.bootstrap import predict_ensemble
import numpy as np
import cv2

def write_prediction(y,output_file):
    y_shaped = np.reshape(y,(60,2))
    np.savetxt(output_file, y_shaped, '%i', ',')




file_list = ['./inputs/test01_with_velocity_and_video.txt',
             './inputs/test02_with_velocity_and_video.txt',
             './inputs/test03_with_velocity_and_video.txt',
             './inputs/test04_with_velocity_and_video.txt',
             './inputs/test05_with_velocity_and_video.txt',
             './inputs/test06_with_velocity_and_video.txt',
             './inputs/test07_with_velocity_and_video.txt',
             './inputs/test08_with_velocity_and_video.txt',
             './inputs/test09_with_velocity_and_video.txt',
             './inputs/test10_with_velocity_and_video.txt'
               ]

dataHanlder = MultiInputDataHandler(file_list,5,60)
x = dataHanlder.get_final_input()
y = predict_ensemble(x,50,'./ensemble',1,True)


background = cv2.imread('./figures/average.png')
for i in xrange(0,10):
    file_string = '%02i' % (i+1)
    write_prediction(np.mean(y,0)[i,:],'./predictions/test'+file_string +'.txt')
    cv2.imwrite('./figures/predictions/test' +file_string +'.png',plot_ensemble(background, x[i,:], None, y[:,i,:]))

