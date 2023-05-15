import nltk
import pickle
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import CRF
import util
from typing import Sequence


Features = util.Features


TestData = list[list[str]]


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


def train() -> CRF:
    util.nltk_download('treebank')
    util.nltk_download('universal_tagset')

    tagged_sentences: list[list[tuple[str, str]]] = nltk.corpus.treebank.tagged_sents(tagset='universal')

    train_set, test_set = train_test_split(tagged_sentences, test_size=0.2, random_state=1234)

    with open('test_data.pickle', 'wb') as f:
        pickle.dump(test_set, f)

    x_train, y_train = prepare_data(train_set)

    crf = CRF(
        algorithm='lbfgs',
        c1=0.01,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True,
    )
    crf.fit(x_train, y_train)
    return crf


def main():
    crf = train()

    with open(util.FILEPATH, "wb") as f:
        pickle.dump(crf, f)

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
