from custommodels.residual import ResidualNetwork
from datatools.bookkeeping import MultiInputDataHandler
from datatools.scoring import score_prediction
from sklearn.model_selection import GridSearchCV
import sys


def find_hyperparameters(lookback):

    #Train the model

    params = {'num_steps': [5, 4, 3 ,2],
              'layers_per_step': [3, 2, 1],
              'first_layer_size': [16, 8],
              'dropout': [0.2, 0.4],
              'batch_size': [64, 128],
              'learning_rate': [0.5],
              'verbose': [False],
              'patience': [10]}

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

    print 'Running Lookback ' + str(lookback)

    dataHandler = MultiInputDataHandler(file_list, lookback,60, test_fraction=0.9)
    x, y = dataHandler.get_training_data()

    gridSearch = GridSearchCV(ResidualNetwork(), params, verbose=2, cv=5, n_jobs=32)
    gridSearch.fit(x,y)

    print gridSearch.best_params_

    x_val, y_val = dataHandler.get_validation_data()
    score_prediction(gridSearch.predict(x_val), y_val)
    score_prediction(gridSearch.predict(x_val), y_val, prediction_type='L2')

    print ' '


if __name__ == '__main__':
    lookback = int(sys.argv[1])
    find_hyperparameters(lookback)