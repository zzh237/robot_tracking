import numpy as np


def score_prediction(y_prediction, y_actual, verbose=True, prediction_type = 'mse'):
    """
    Prints and returns information about how a prediction compares to expected results
    :param y_prediction:    The predicted output
    :param y_actual:        The actual output
    :param verbose:         If true, this function prints the results
    :param score_type:      Either "mse" or "L2"

    :return:                returns a tuple containing the (median, mean, max, min) error from the  predictions
    """

    if prediction_type == 'mse':
        scores = np.mean((y_prediction-y_actual)**2,1)
    elif prediction_type == 'L2':
        scores = np.sqrt(np.sum((y_prediction-y_actual)**2,1))

    min_score = np.min(scores)
    max_score = np.max(scores)
    mean_score = np.mean(scores)
    median_score = np.median(scores)
    if verbose:
        print 'Median Score:      %5.0f' % median_score
        print 'Mean Score  :      %5.0f' % mean_score
        print 'Max Score   :      %5.0f' % max_score
        print 'Min Score   :      %5.0f' % min_score

    return median_score, mean_score, max_score, min_score
