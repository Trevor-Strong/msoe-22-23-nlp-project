import nltk
import pickle
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import CRF
import util
from typing import Sequence
import logging as log

Features = util.Features


TestData = list[list[str]]

MODEL_FILE = util.PROJ_ROOT / 'model.pickle'
TRAINING_DATA_FILE = util.PROJ_ROOT / 'test_data.pickle'


def load(*, auto_train: bool = False, save_retrained_model: bool = True) -> CRF:
    """
    Loads the model. If no model file is found, a model is trained
    :param auto_train: Automatically train the model if no model file is found
    :param save_retrained_model: Determines if a newly trained model should be saved or not
    :return: the model
    """
    log.info("Loading model")
    try:
        with util.MODEL_FILE.open(mode='rb') as f:
            return pickle.load(f)
    except (FileNotFoundError, IOError) as e:
        if not auto_train:
            reason = "No model found" if isinstance(e, FileNotFoundError) else \
                "Error loading model"

            ans = input(f"{reason}, train new model now? [y/n]: ").strip()
            if ans.startswith(('n', 'N')):
                raise KeyboardInterrupt()
            elif not ans.startswith(('y', 'Y')):
                print(f"Unknown input {ans!r}, exiting")
                exit(1)
        log.info('Training model...')
        return train(save=save_retrained_model)


def prepare_data(tagged_sentences: Sequence[Sequence[tuple[str, str]]]) -> tuple[list[list[Features]], list[list[str]]]:
    x: list[list[Features]] = []
    y: list[list[str]] = []
    for sentences in tagged_sentences:
        x_inner = []
        y_inner = []
        for prev, (word, tag), next in util.three_windowed(sentences):
            prev_word = '' if prev is None else prev[0]
            next_word = '' if next is None else next[0]
            x_inner.append(util.features(word, prev_word=prev_word, next_word=next_word))
            y_inner.append(tag)
        x.append(x_inner)
        y.append(y_inner)
    return x, y


def prep_input(usr_input: Sequence[str]) -> list[Features]:
    out = []
    for prev_word, word, next_word in util.three_windowed(usr_input, default=""):
        out.append(util.features(word, prev_word=prev_word, next_word=next_word))
    return out


def train(*, save: bool = False) -> CRF:
    util.nltk_download('treebank')
    util.nltk_download('universal_tagset')

    tagged_sentences: list[list[tuple[str, str]]] = nltk.corpus.treebank.tagged_sents(tagset='universal')

    train_set, test_set = train_test_split(tagged_sentences, test_size=0.2, random_state=1234)

    x_train, y_train = prepare_data(train_set)

    crf = CRF(
        algorithm='lbfgs',
        c1=0.01,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True,
    )
    crf.fit(x_train, y_train)

    if save:
        log.info("Saving model to %(filename)s...", {'filename': util.MODEL_FILE})
        try:
            with open(MODEL_FILE, mode='wb') as f:
                pickle.dump(crf, file=f)
        except IOError as e:
            log.error("Error saving model, bailing out of save", exc_info=e)
        else:
            log.info("Saved model")
            log.info("Saving training data to %(filename)s...", {'filename': util})
            try:
                with open(TRAINING_DATA_FILE, 'wb') as f:
                    pickle.dump(test_set, file=f)
            except IOError as e:
                log.error("Error saving training data", exc_info=e)
            else:
                log.info("Saved training data")

    return crf


def main():
    train(save=True)

    # VERB - verbs (all tenses and modes)
    # NOUN - nouns (common and proper)
    # PRON - pronouns
    # ADJ - adjectives
    # ADV - adverbs
    # ADP - adpositions (prepositions and postpositions)
    # CONJ - conjunctions
    # DET - determiners
    # NUM - cardinal numbers
    # PRT - particles or other function words
    # X - other: foreign words, typos, abbreviations
    # . - punctuation


if __name__ == '__main__':
    main()
