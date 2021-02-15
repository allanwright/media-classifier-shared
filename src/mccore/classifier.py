'''Defines a multi class logistic regression classifier.

'''

import numpy as np

from mccore import prediction
from mccore import preprocessing

class Classifier:
    '''Defines a multi class logistic regression classifier.

    '''

    def __init__(self, vectorizer, model, labels):
        '''Initializes a new instance of the Classifier object.

        Args:
            vectorizer (object): The trained vectorizer.
            model (object): the trained model.
            labels (dictionary): The label dictionary.
        '''
        self.vectorizer = vectorizer
        self.model = model
        self.labels = labels

    def predict(self, name):
        '''Predicts the class associated with the specified filename.

        Args:
            name (string): The name of the file to classify.

        Returns:
            label (string): The predicted label.
            estimate (float): The probability estimate.
        '''
        x = preprocessing.prepare_input(name)
        x = self.vectorizer.transform(np.array([x]))
        y = self.model.predict_proba(x)
        return prediction.get_class(y, self.labels)
