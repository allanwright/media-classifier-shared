# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring

import pytest

from mccore import persistence
from mccore.classifier import Classifier

@pytest.fixture(name='model')
def fixture_model():
    return Classifier(
        persistence.bin_to_obj('models/classifier_vec.pickle'),
        persistence.bin_to_obj('models/classifier_mdl.pickle'),
        persistence.json_to_obj('models/label_dictionary.json'))

@pytest.mark.parametrize(
    'name, expected',
    [('Billy Bonka & the Milk Factory (1971) (1080p RedRay x275 16bit Something).mkv',
      'movie'),
     ('Game.of.Drones.S01.E01-Summer.is.Here.mp4',
      'tv')])
def test_predict_produces_correct_title(model, name, expected):
    prediction = dict(model.predict(name))
    assert prediction['label']['name'] == expected
