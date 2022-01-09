import numpy as np

class DataHandler():

    def __init__(self,data_file, look_back=1, look_forward=60, skip=0, flatten=True,
                 input_features=None, output_features=None, test_fraction=0.8):
        """
        Initializes a dataHandler object.
        :param data_file:       Path to a text file containing robot data in the csv format
        :param look_back:       The number of previous frames to use when making a prediction
        :param look_forward:    The number of frames to predict (60 is 2 seconds)
        :param skip:            The number of frames to skip in between look_back and look_forward.
        :param flatten:         If true, flatten the output into a num_samples by num_features*num_frames array
                                If false, return the output as a three dimensional array
                                Most sklearn regressors want flatten=True, some neural networks want flatten=false
        :param input_features   A numpy boolean vector of features to use as inputs
        :param output_features  A numpy boolean vector of features to use as outputs
        :param test_fraction:   Fraction of the data to use for testing, the remaining data is used for validation
        :param val_fraction:    Fraction of the test data to reserve for final validation
        """

        self.data = np.loadtxt(data_file, delimiter=",")
        self.num_val = int(np.floor(float(self.data.shape[0])*(1-test_fraction)))
        self.look_back = look_back
        self.look_forward = look_forward
        self.skip = skip
        self.flatten = flatten

        # The input_features default is to use all of inputs
        if input_features is None:
            self.input_features = np.ones(self.data.shape[1],dtype='bool')
        else:
            self.input_features = input_features

        # The output_features default is to use only the first 2 outputs
        if output_features is None:
            self.output_features = np.zeros(self.data.shape[1],dtype='bool')
            self.output_features[0:2] = True
        else:
            self.output_features = output_features

    def get_training_data(self):
        """
        Returns an array of training data in the format specified when the dataHandler was constructed

        :return x:          These are the training data samples (X in scikit-learn notation)
        :return y:          These are the expected outputs (Y in scikit-learn notation)
        """
        x, y = self._get_data(self.num_val, self.data.shape[0])
        return x, y

    def get_validation_data(self):
        """
        Returns an array of validation data in the format specified when the dataHandler was constructed

        :return x:          These the validation data samples
        :return y:          These are the expected output
        """
        x, y = self._get_data(0, self.num_val)
        return x, y

    def get_final_input(self):
        x = self.data[-self.look_back:,self.input_features]
        if self.flatten:
            x = np.reshape(x, (1,self.look_back*np.sum(self.input_features)))
        return x

    def _get_data(self, start, end):

        look_back = self.look_back
        look_forward = self.look_forward
        skip = self.skip
        flatten = self.flatten

        num_cases = int(end - look_back - look_forward - skip - start+1)
        input = np.zeros((num_cases, look_back, np.sum(self.input_features)))
        output = np.zeros((num_cases, look_forward, np.sum(self.output_features)))
        for i in xrange(start, start+num_cases):
            input[i-start, :, :] = self.data[i:(i + look_back), self.input_features]
            output[i-start, :, :] = self.data[(i + look_back + skip):(i + look_back + skip + look_forward),
                                               self.output_features]


        if flatten:
            input = np.reshape(input, (num_cases, -1))
            output = np.reshape(output, (num_cases, -1))

        return input, output


class MultiInputDataHandler():

    def __init__(self, data_file_list, look_back=1, look_forward=60, skip=0, flatten=True,
                 input_features=None, output_features=None, test_fraction=0.8):
        """
        Initializes a dataHandler object.
        :param data_file:       Path to a text file containing robot data in the csv format
        :param look_back:       The number of previous frames to use when making a prediction
        :param look_forward:    The number of frames to predict (60 is 2 seconds)
        :param skip:            The number of frames to skip in between look_back and look_forward.
        :param flatten:         If true, flatten the output into a num_samples by num_features*num_frames array
                                If false, return the output as a three dimensional array
                                Most sklearn regressors want flatten=True, some neural networks want flatten=false
        :param input_features   A numpy boolean vector of features to use as inputs
        :param output_features  A numpy boolean vector of features to use as outputs
        :param test_fraction:   Fraction of the data to use for testing, the remaining data is used for validation
        """

        self.data_file_list = data_file_list
        self.test_fraction = test_fraction
        self.look_back = look_back
        self.look_forward = look_forward
        self.skip = skip
        self.flatten = flatten
        self.input_features = input_features
        self.output_feature = output_features

        self.data_handlers = []
        for data_file in data_file_list:
            self.data_handlers.append(DataHandler(data_file,look_back,look_forward,skip,flatten,input_features,
                                                  output_features,test_fraction))

    def get_training_data(self):
        """
        Returns an array of training data in the format specified when the dataHandler was constructed

        :return x:          These are the training data samples (X in scikit-learn notation)
        :return y:          These are the expected outputs (Y in scikit-learn notation)
        """
        x, y = self.data_handlers[0].get_training_data()

        for i in xrange(1,len(self.data_handlers)):
            x_new, y_new = self.data_handlers[i].get_training_data()
            x = np.row_stack((x,x_new))
            y = np.row_stack((y,y_new))

        return x, y

    def get_validation_data(self):
        """
        Returns an array of validation data in the format specified when the dataHandler was constructed

        :return x:          These the validation data samples
        :return y:          These are the expected output
        """
        x, y = self.data_handlers[0].get_validation_data()

        for i in xrange(1, len(self.data_handlers)):
            x_new, y_new = self.data_handlers[i].get_validation_data()
            x = np.row_stack((x, x_new))
            y = np.row_stack((y, y_new))

        return x,y

    def get_final_input(self):
        x = np.zeros((len(self.data_handlers),self.look_back*np.sum(self.data_handlers[0].input_features)))
        for i in xrange(0,len(self.data_handlers)):
            x[i,:] = self.data_handlers[i].get_final_input()

        return x