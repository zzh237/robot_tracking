from sklearn.decomposition import PCA
from datatools.bookkeeping import DataHandler
from datatools.scoring import score_prediction
import numpy as np


dataHandler = DataHandler('./inputs/training_data.txt', 1, 60)

# Get the data
_, y = dataHandler.get_training_data()

for i in xrange(0,5):
    pca = PCA(i + 1)
    pca.fit(y)
    print '== Running PCA =='
    print str(i+1) + ' dimensions explain ' + str(np.sum(pca.explained_variance_ratio_)) + ' of the variance'
    temp = pca.transform(y)
    yprime = pca.inverse_transform(temp)
    score_prediction(y,yprime)
    print ' '
