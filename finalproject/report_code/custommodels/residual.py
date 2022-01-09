from keras.models import Model
from keras.layers import Dense, Activation, BatchNormalization, Input, Dropout
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping
from keras.optimizers import adam
from keras import backend as K
from sklearn.metrics import r2_score
import sys




class ResidualNetwork(object):

    def __init__(self, num_steps=4, layers_per_step=3, first_layer_size=8, dropout=0.2, learning_rate = 0.5,
                 batch_size = 64, patience=10, loss='mse', verbose = False, sanity_check = False):

        self.num_steps = num_steps
        self.layers_per_step = layers_per_step
        self.first_layer_size = first_layer_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.patience = patience
        self.loss = loss

        self.verbose = verbose
        self.sanity_check = sanity_check
        self.model = None
        self.estimator_type = "regressor"

    def get_params(self, deep=True):
        return {'num_steps': self.num_steps, 'layers_per_step': self.layers_per_step,
                'first_layer_size': self.first_layer_size, 'dropout': self.dropout, 'learning_rate': self.learning_rate,
                'batch_size': self.batch_size, 'patience': self.patience, 'verbose': self.verbose,
                'sanity_check': self.sanity_check}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self,parameter, value)
        return self

    def fit(self, X, y):

        sys.stdout.flush()

        # Build the model
        success = False
        while not success:
            # The input to the model is a vector whose length corresponds to the columns in X
            main_input = Input((X.shape[1],))

            # The first layer of the model is a fixed 128 node layer with batch normalization before the non-linear relu
            # activation
            intermediate = Dense(128)(main_input)
            intermediate = BatchNormalization()(intermediate)
            intermediate = Activation('relu')(intermediate)
            residual = Dropout(self.dropout)(intermediate)
            intermediate = residual
            rshape = 128
            # Add sets of layers
            for i in xrange(0, self.num_steps):
                growth = 2 ** i

                # Add individual layers
                for j in xrange(0, self.layers_per_step):
                    intermediate = Dense(self.first_layer_size*growth)(intermediate)
                    intermediate = BatchNormalization()(intermediate)
                    intermediate = Activation('relu')(intermediate)

                ishape = self.first_layer_size * growth
                new_residual = intermediate
                nr_shape = ishape
                # Feed forward the residual

                while rshape < ishape:
                    residual = concatenate([residual,residual])
                    rshape *=2

                while rshape > ishape:
                    intermediate = concatenate([intermediate,intermediate])
                    ishape *= 2

                intermediate = add([residual, intermediate])
                residual = new_residual
                rshape = nr_shape
            # The last layer is always a 120 node layer without dropout
            if len(y.shape)>1:
                intermediate = Dense(y.shape[1])(intermediate)
            else:
                intermediate = Dense(1)(intermediate)

            intermediate = BatchNormalization()(intermediate)
            output = Activation('relu')(intermediate)

            model = Model(main_input, output)

            optimizer = adam(lr= self.learning_rate)
            callbacks = [EarlyStopping(patience=self.patience)]

            if self.loss == 'L2':
                loss = _l2_error
            else:
                loss = self.loss

            model.compile(optimizer=optimizer, loss=loss)

            result = model.fit(X, y, epochs=1000, batch_size=self.batch_size, validation_split=0.1,
                                callbacks=callbacks, verbose=self.verbose)



            if result.history['val_loss'][-1] < 2 * result.history['loss'][-1] or not self.sanity_check:
                success = True

        self.model = model
        return self

    def predict(self,X):
        y = self.model.predict(X, self.batch_size, verbose=0)
        return y

    def score(self,X,y):
        return r2_score(y,self.predict(X),multioutput='uniform_average')


def _l2_error(y_true, y_pred):
    return K.mean(K.sqrt(K.sum(K.square(y_pred-y_true),axis=1)), axis=-1)
