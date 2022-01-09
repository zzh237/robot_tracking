from .residual import ResidualNetwork, _l2_error
from keras.models import load_model
import os
import multiprocessing
import numpy as np
import itertools as iter


def fit_regressor(arguments):
    x, y, parameters, path, filename = arguments
    idx = np.random.choice(np.arange(0,x.shape[0]), x.shape[0], True)
    x_random = x[idx,:]
    y_random = y[idx,:]

    regressor = ResidualNetwork(**parameters)
    regressor.fit(x_random,y_random)
    regressor.model.save(os.path.join(path,filename))


def predict_model(args):
    x, path, filename, i = args
    print 'Processing ' + str(i)
    model = load_model(os.path.join(path,filename), custom_objects={'l2_error':_l2_error})
    return model.predict(x, x.shape[0])



def fit_ensemble(x, y, n, parameters, path, njobs = 1):
    x_iter = iter.repeat(x)
    y_iter = iter.repeat(y)
    parameters_iter = iter.repeat(parameters)
    path_iter = iter.repeat(path)

    filenames = []
    for i in xrange(0,n):
        filenames.append(str(i)+'.h5')


    pool = multiprocessing.Pool(njobs)
    pool.map(fit_regressor, zip(x_iter,y_iter,parameters_iter,path_iter,filenames))

def predict_ensemble(x, n, path, njobs=1, return_all = False):

    files = os.listdir(path)[0:n]
    i = range(0, len(files))
    x_iter = iter.repeat(x)
    path_iter = iter.repeat(path)

    pool = multiprocessing.Pool(njobs,maxtasksperchild=5)
    predictions = np.asarray(pool.map(predict_model, zip(x_iter, path_iter, files, i)))

    if not return_all:
        predictions = np.mean(predictions,0)

    return predictions
