from custommodels.bootstrap import predict_ensemble
from datatools.bookkeeping import MultiInputDataHandler
from datatools.scoring import score_prediction
from datatools.plotting import plot_prediction, plot_ensemble

import numpy as np
import matplotlib
matplotlib.use('TKAgg')
from matplotlib import pyplot as plt
import cv2


file_list = ['./inputs/test01_with_velocity_and_video.txt',
             './inputs/test02_with_velocity_and_video.txt',
             './inputs/test03_with_velocity_and_video.txt',
             './inputs/test04_with_velocity_and_video.txt',
             './inputs/test05_with_velocity_and_video.txt',
             './inputs/test06_with_velocity_and_video.txt',
             './inputs/test07_with_velocity_and_video.txt',
             './inputs/test08_with_velocity_and_video.txt',
             './inputs/test09_with_velocity_and_video.txt',
             './inputs/test10_with_velocity_and_video.txt',
             './inputs/training_data_with_velocity_and_video.txt',
               ]

dataHandler = MultiInputDataHandler(file_list,5,60, test_fraction=0.9)
x_val, y_val = dataHandler.get_validation_data()



# Make a prediction using the ensemble
y_pred = predict_ensemble(x_val, 50, './ensemble',njobs=3,return_all=True)
mean_prediction = np.mean(y_pred,0)

# Print Ensembe Scores
score_prediction(mean_prediction,y_val)
score_prediction(mean_prediction,y_val, prediction_type='L2')

# Make Prediction Graphics
image = cv2.imread('./figures/average.png')

out_image = plot_ensemble(image,x_val[2586,:], y_val[2586,:],y_pred[:,2586,:])
cv2.imwrite('./figures/examples/0201.png',out_image)

out_image = plot_ensemble(image,x_val[2979,:], y_val[2979,:], y_pred[:,2979,:])
cv2.imwrite('./figures/examples/0500.png',out_image)

out_image = plot_ensemble(image,x_val[3332,:], y_val[3332,:], y_pred[:,3332,:])
cv2.imwrite('./figures/examples/0800.png',out_image)

out_image = plot_ensemble(image,x_val[941,:], y_val[941,:], y_pred[:,941,:])
cv2.imwrite('./figures/examples/1100.png',out_image)

out_image = plot_ensemble(image,x_val[4114,:], y_val[4114,:], y_pred[:,4114,:])
cv2.imwrite('./figures/examples/1401.png',out_image)

out_image = plot_ensemble(image,x_val[3644,:], y_val[3644,:], y_pred[:,3644,:])
cv2.imwrite('./figures/examples/1702.png',out_image)

# Make a histogram of scores
scores = np.sqrt(np.sum((mean_prediction-y_val)**2,1))

print float(np.sum(scores>1100)) / float(scores.shape[0])
plt.hist(scores,bins=30,range=[100,2000], normed=False)
plt.xlabel('L2 Score')
plt.ylabel('Count')
plt.title('Distribution of L2 Validation Scores')
plt.savefig('./figures/l2_distribution.png',bbox_inches='tight')
plt.close()


# Make a convergence graphic
l2_list = []
for i in xrange(1,50):
    prediction = np.mean(y_pred[0:i,:,:],0)
    scores = score_prediction(prediction, y_val,False, 'L2')
    l2_list.append(scores[1])

plt.plot(l2_list)
plt.xlabel('Number of Ensemble Members')
plt.ylabel('L2 Score')
plt.title('Bootstrap Aggregating Ensemble of Residual Networks')
plt.savefig('./figures/l2_ensemble_convergence.png',bbox_inches='tight')