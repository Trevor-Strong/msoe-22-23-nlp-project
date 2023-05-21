from sklearn_crfsuite import CRF
import pickle
from pathlib import Path
import logging as log
import pos_model
import sys
from typing import Sequence
import util
import numpy as np


PartOfSpeach = str


def predict_next_pos(transition_features: dict[tuple[str, str], float], pos: str):
    keys = list(transition_features.keys())
    values = list(transition_features.values())
    sorted_value_index = np.argsort(values)
    sorted_dict = {keys[i]: values[i] for i in sorted_value_index}

    max_value = float('-inf')
    max_extract = None
    # return keys[sorted_value_index[-1]]
    for extract in sorted_dict:
        if extract[0] == pos and sorted_dict[extract] > max_value:
            max_value = sorted_dict[extract]
            max_extract = extract

    return max_extract[1] if max_extract is not None else None


def translate(pos: PartOfSpeach):
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


def expand_output(model: CRF,
                  tokens: list[list[util.Features]],
                  prediction: list[list[PartOfSpeach]]) -> str:
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
    util.init_logging()
    util.nltk_download("punkt")
    model = pos_model.load()

    print("Write text you want the model classify")
    while True:
        text = reader("Input:\n")
        tokens = util.tokenize_with_features(text)
        prediction: list[list[PartOfSpeach]] = model.predict(tokens)
        printer("Prediction:")
        printer(prediction)
        printer(expand_output(model, tokens, prediction))


if __name__ == '__main__':
    try:
        main(sys.argv[1:])
    except KeyboardInterrupt:
        pass
