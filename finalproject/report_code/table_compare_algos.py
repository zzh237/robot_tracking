from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from datatools import MultiInputDataHandler, DataHandler, score_prediction
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from custommodels import ResidualNetwork

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt


# check the mean_MSE for different points
data_file = './inputs/training_data.txt'
data = np.loadtxt(data_file, delimiter=",")
length = len(data)

def knn_points():
    i = 1
    x_axis = []
    y_axis = []

    while i <= 5:
        dataHandler = DataHandler('./inputs/training_data.txt', i, 60)

        # Train the model
        x, y = dataHandler.get_training_data()
        knn_regressor = KNeighborsRegressor(5, weights='distance')
        knn_regressor.fit(x,y)

        # Evaluate the model
        x_val, y_val = dataHandler.get_validation_data()
        scores = (score_prediction(knn_regressor.predict(x_val), y_val))

        x_pred = knn_regressor.predict(x_val)
        L2 = np.sqrt(np.sum((x_pred - y_val) ** 2, 1))
        mean_L2 = np.mean(L2)
        mean_MSE = scores[2]
        x_axis.append(i)
        y_axis.append(mean_MSE)
        i += 1

    return scores, x_axis, y_axis, mean_L2

scores, x_axis, y_axis, mean_L2 = knn_points()
plt.scatter(x_axis,y_axis,color='cyan', linewidth=3)
plt.show()


def knn():
    dataHandler = DataHandler('./inputs/training_data.txt', 5, 60)
    # Train the model
    x, y = dataHandler.get_training_data()
    knn_regressor = KNeighborsRegressor(5, weights='distance')
    knn_regressor.fit(x,y)
    # Evaluate the model
    x_val, y_val = dataHandler.get_validation_data()
    median_MSE, mean_MSE, max_MSE, min_MSE = score_prediction(knn_regressor.predict(x_val), y_val)
    x_pred = knn_regressor.predict(x_val)
    L2 = np.sqrt(np.sum((x_pred - y_val) ** 2, 1))
    mean_L2 = np.mean(L2)
    median_L2 = np.median(L2)
    return (mean_L2, median_L2, mean_MSE, median_MSE)


def gbr():
    dataHandler = DataHandler('./inputs/training_data.txt', 5, 60)

    # Train the model - note that we need to wrap the single output gradientBoostingRegressor with the
    # MultiOutputRegressor class to fit multiple output data
    x, y = dataHandler.get_training_data()
    boost_regressor = MultiOutputRegressor(GradientBoostingRegressor(learning_rate=0.1,n_estimators=100,verbose=0))
    boost_regressor.fit(x,y)

    # Evaluate the model
    x_val, y_val = dataHandler.get_validation_data()
    median_MSE, mean_MSE, max_MSE, min_MSE = score_prediction(boost_regressor.predict(x_val), y_val)
    x_pred = boost_regressor.predict(x_val)
    L2 = np.sqrt(np.sum((x_pred - y_val) ** 2, 1))
    mean_L2 = np.mean(L2)
    median_L2 = np.median(L2)
    return (mean_L2, median_L2, mean_MSE, median_MSE)

def hyperparameter():
    params = {'n_estimators': [5,10, 20],
              'max_depth': [None, 5, 10],
              'min_samples_split': [2,4,8]}

    dataHandler = DataHandler('./inputs/training_data.txt', 5, 60)

    # Get the data
    x, y = dataHandler.get_training_data()
    x_val, y_val = dataHandler.get_validation_data()

    grid_search = GridSearchCV(RandomForestRegressor(), params, verbose=3, n_jobs=-1)
    grid_search.fit(x,y)

    # Evaluate the model
    #
    median_MSE, mean_MSE, max_MSE, min_MSE = score_prediction(grid_search.predict(x_val), y_val)
    x_pred = grid_search.predict(x_val)
    L2 = np.sqrt(np.sum((x_pred - y_val) ** 2, 1))
    mean_L2 = np.mean(L2)
    median_L2 = np.median(L2)
    return (mean_L2, median_L2, mean_MSE, median_MSE)

def rn():
    file_list = ['./inputs/video_test01.txt',
                './inputs/video_test02.txt',
                './inputs/video_test03.txt',
                './inputs/video_test04.txt',
                './inputs/video_test05.txt',
                './inputs/video_test06.txt',
                './inputs/video_test07.txt',
                './inputs/video_test08.txt',
                './inputs/video_test09.txt',
                './inputs/video_test10.txt',
                './inputs/video_training_data.txt',
                   ]

    dataHandler = MultiInputDataHandler(file_list,5,60, test_fraction=0.9)

    # Train the model
    x, y = dataHandler.get_training_data()
    x_val, y_val = dataHandler.get_validation_data()
    i = 0
    while True:

        resnet = ResidualNetwork(dropout=0.2, num_steps=5, batch_size=128, patience=10, first_layer_size=8,
                                 layers_per_step=3, verbose=False)
        resnet.fit(x,y)

        scores = score_prediction(resnet.predict(x_val), y_val,verbose=False)
        median_MSE, mean_MSE, max_MSE, min_MSE = score_prediction(resnet.predict(x_val), y_val,verbose=False)
        x_pred = grid_search.predict(x_val)
        L2 = np.sqrt(np.sum((x_pred - y_val) ** 2, 1))
        mean_L2 = np.mean(L2)
        median_L2 = np.median(L2)
        return (mean_L2, median_L2, mean_MSE, median_MSE)
        print i, scores
    return scores
# Evaluate the model


columns = ['Mean L2', 'Median L2', 'Mean MSE', 'median MSE']
algos = dict()
algos['KNN'] = knn()
algos['Gradient Boosting Regressor'] = gbr()
algos['Hyperparameter on RandomForestRegressor'] = hyperparameter()
algos['Residual Network Repeat'] = rn()

def table_compare_glgos():
    data = pd.DataFrame(algos, columns=algos.keys())
    table = data.transpose()
    table.columns = columns
    table_output = table.astype(int)
    return table_output

table_output = table_compare_glgos()

print table_output






