'''Defines a named entity recognizer.

'''

from titlecase import titlecase

from mccore import ner
from mccore import persistence
from mccore import preprocessing
from mccore import postprocessing

SEP = ' '
SID = 'season_id'
EID = 'episode_id'
TITLE = 'title'
EPNAME = 'episode_name'

class EntityRecognizer:
    '''Defines a named entity recognizer.

    '''

    def __init__(self, model):
        '''Initializes a new instance of the EntityRecognizer.

        Args:
            model (object): the trained model.
        '''
        self.model = model

    @classmethod
    def load_default(cls):
        '''Initializes a new instance of the Classifier object using the model
           included with the package.

        '''
        nlp, _ = ner.get_model()
        nlp.from_bytes(persistence.bin_res_to_obj('ner_mdl.pickle'))
        return cls(nlp)

    def predict(self, name):
        '''Recognizes entities contained within the specified filename.

        Args:
            name (string): The name of the file to recognize entities.

        Returns:
            list: The list of tuples containing the entity and associated value.
        '''

        x = preprocessing.prepare_input(name)
        x_out = postprocessing.prepare_output(name)
        x_out = x_out.split(SEP)
        y = self.model(x)
        y = [(e.label_, e.text, e.start) for e in y.ents]

        # Merge entities
        y_merged = {}
        for (label, _, start) in y:
            word = x_out[start]

            if label in y_merged:
                y_merged[label] = y_merged[label] + SEP + word
            else:
                y_merged[label] = word

        # Remove leading s and e from season and episode numbers
        if SID in y_merged:
            try:
                y_merged[SID] = int(y_merged[SID].lstrip('sS'))
            except ValueError:
                y_merged[SID] = y_merged[SID].lstrip('sS')

        if EID in y_merged:
            try:
                y_merged[EID] = int(y_merged[EID].lstrip('eE'))
            except ValueError:
                y_merged[EID] = y_merged[EID].lstrip('eE')

        # Title case title and episode names
        if TITLE in y_merged:
            y_merged[TITLE] = titlecase(y_merged[TITLE])

        if EPNAME in y_merged:
            y_merged[EPNAME] = titlecase(y_merged[EPNAME])

        return [(i, y_merged[i]) for i in y_merged]
