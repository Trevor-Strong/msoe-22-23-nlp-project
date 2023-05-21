from sklearn_crfsuite import CRF
import pickle
from pathlib import Path
import logging as log
import train_model
import sys
from typing import Sequence
import util
import numpy as np
from collections import OrderedDict


def _train_and_save_model() -> CRF:
    model = train_model.train()
    log.info("Saving model to %s", util.FILEPATH)
    try:
        with open(util.FILEPATH, "wb") as f:
            pickle.dump(model, f)
    except OSError as e:
        log.error("Failed to save model", exc_info=e)
    return model


def load_model() -> CRF:
    log.info("Loading model")
    path = Path('model.pickle')
    if not path.exists():
        log.info("No model found, training new model...")
        model = _train_and_save_model()
    else:
        log.info("Loading model from %s", util.FILEPATH)
        try:
            with open(util.FILEPATH, "rb") as f:
                model = pickle.load(f)
        except OSError as e:
            log.error(
                "Failed to load model from file %s, training new model...",
                util.FILEPATH,
                exc_info=e
            )
            model = _train_and_save_model()

    return model


def predict_next_pos(transition_features, pos):
    keys = list(transition_features.keys())
    values = list(transition_features.values())
    sorted_value_index = np.argsort(values)
    sorted_dict = {keys[i]: values[i] for i in sorted_value_index}

    max_value = float('-inf')
    max_extract = None

    for extract in sorted_dict:
        if extract[0] == pos and sorted_dict[extract] > max_value:
            max_value = sorted_dict[extract]
            max_extract = extract

    if max_extract:
        return max_extract[1]

    return


def translate(pos):
    tagset = {
        'VERB': 'verb',
        'NOUN': 'noun',
        'PRON': 'pronoun',
        'ADJ': 'adjective',
        'ADV': 'adverb',
        'ADP': 'adposition',
        'CONJ': 'conjunction',
        'DET': 'determiner',
        'NUM': 'number',
        'PRT': 'particle',
        'X': 'foreign word, typo or an abbreviation',
        '.': 'punctuation mark'
    }

    return tagset.get(pos, 'Unknown part of speech')


def expand_output(model, tokens, prediction):
    output = ""

    i = 0
    j = 1
    for list in prediction:
        for pos in list:
            while i < (len(list) - 1):
                output += (tokens[0][j]['prev_word'] + ", is a " + translate(list[i])
                           + ". It is usually followed by a " +
                           translate(predict_next_pos(model.transition_features_, list[i])) + ".\n")

                i = i + 1
                j = j + 1

    return output


def main(args: Sequence[str] = (), printer=print, reader=input):
    util.nltk_download("punkt")

    import pprint
    log.basicConfig(format='[{levelname}]: {message}', level=log.INFO, style='{')
    model = load_model()

    print("Write text you want the model classify")
    while True:
        text = reader("Input:\n")
        tokens = util.tokenize_with_features(text)
        prediction = model.predict(tokens)
        printer("Prediction:")
        printer(prediction)
        printer(expand_output(model, tokens, prediction))


if __name__ == '__main__':
    try:
        main(sys.argv[1:])
    except KeyboardInterrupt:
        pass
