
from datatools.bookkeeping import MultiInputDataHandler
from datatools.scoring import score_prediction
from custommodels.residual import ResidualNetwork
import numpy as np


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




def run_model_10x(lookback,dropout,num_steps,batch_size,patience,first_layer_size,layers_per_step,
                  learning_rate, loss, files):

    dataHandler = MultiInputDataHandler(files,lookback,60, test_fraction=0.9)

    # Train the model
    x, y = dataHandler.get_training_data()
    x_val, y_val = dataHandler.get_validation_data()
    i = 0
    for i in xrange(0,10):
        resnet = ResidualNetwork(dropout=dropout, num_steps=num_steps, batch_size=batch_size, patience=patience,
                                 first_layer_size=first_layer_size, learning_rate=learning_rate,
                                 layers_per_step=layers_per_step, loss = loss, verbose=False, sanity_check=True)
        resnet.fit(x,y)

        scores1 = score_prediction(resnet.predict(x_val), y_val,verbose=False, prediction_type='L2')
        scores2 = score_prediction(resnet.predict(x_val), y_val,verbose=False, prediction_type='mse')

        print i, scores1, scores2



if __name__ == '__main__':
    np.random.seed(12345)
    print 'Running L2 Trained Model'
    # Lookback 5
    run_model_10x(5, 0.4, 2, 64, 10, 16, 2, 0.5, 'L2', file_list)

    print 'Running MSE Trained Model'
    # Lookback 5
    run_model_10x(5, 0.4, 2, 64, 10, 16, 2, 0.5, 'mse', file_list)




