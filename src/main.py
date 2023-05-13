import nltk
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from itertools import chain
from sklearn_crfsuite import CRF, metrics

from typing import Sequence

nltk.download('treebank')
nltk.download('universal_tagset')

tagged_sentence: list[list[tuple[str, str]]] = nltk.corpus.treebank.tagged_sents(tagset='universal')

print(f"{tagged_sentence=}")

print("Number of Tagged Sentences ", len(tagged_sentence))
tagged_words: list[tuple[str, str]] = [tup for sent in tagged_sentence for tup in sent]
print("Total Number of Tagged words", len(tagged_words))
vocab = set([word for word, tag in tagged_words])
print("Vocabulary of the Corpus", len(vocab))
tags = set([tag for word, tag in tagged_words])
print("Number of Tags in the Corpus ", len(tags))

train_set, test_set = train_test_split(tagged_sentence, test_size=0.2, random_state=1234)
print("Number of Sentences in Training Data ", len(train_set))
print("Number of Sentences in Testing Data ", len(test_set))


from dataclasses import dataclass


@dataclass(eq=True)
class Features:

    is_first_capital: bool
    is_first_word: bool
    is_last_word: bool
    is_complete_capital: bool
    prev_word: str
    next_word: str
    is_numeric: bool
    is_alphanumeric: bool
    prefix_1: str
    prefix_2: str
    prefix_3: str
    prefix_4: str
    suffix_1: str
    suffix_2: str
    suffix_3: str
    suffix_4: str
    word_has_hyphen: bool


def features(sentence: Sequence[str], index: int):
    # sentence is of the form [w1,w2,w3,..], index is the position of the word in the sentence
    word = sentence[index]
    return {
        'is_first_capital': int(word[0].isupper()),
        'is_first_word': int(index == 0),
        'is_last_word': int(index == len(sentence) - 1),
        'is_complete_capital': int(word.isupper()),  # int(word.upper() == word),
        'prev_word': sentence[index - 1] if index else '',
        'next_word':'' if index==len(sentence)-1 else sentence[index+1],
        'is_numeric': int(sentence[index].isdigit()),
        'is_alphanumeric': sentence[index].isalnum(),  # int(bool((re.match('^(?=.*[0-9]$)(?=.*[a-zA-Z])', sentence[index])))),
        'prefix_1': sentence[index][0],
        'prefix_2': sentence[index][:2],
        'prefix_3': sentence[index][:3],
        'prefix_4': sentence[index][:4],
        'suffix_1': sentence[index][-1],
        'suffix_2': sentence[index][-2:],
        'suffix_3': sentence[index][-3:],
        'suffix_4': sentence[index][-4:],
        'word_has_hyphen': 1 if '-' in sentence[index] else 0

    }


def untag(sentence: Sequence[tuple[str, str]]) -> Sequence[str]:
    return [word for word, tag in sentence]


def prepare_data(tagged_sentences):
    x, y = [], []
    for sentences in tagged_sentences:
        x.append([features(untag(sentences), index) for index in range(len(sentences))])
        y.append([tag for word, tag in sentences])
    return x, y


x_train, y_train = prepare_data(train_set)
x_test, y_test = prepare_data(test_set)

crf = CRF(
    algorithm='lbfgs',
    c1=0.01,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
try:
    crf.fit(x_train, y_train)
except AttributeError:
    pass
y_pred = crf.predict(x_test)

print("F1 score on Test Data ")
print(metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=crf.classes_))
print("F score on Training Data ")
y_pred_train = crf.predict(x_train)
metrics.flat_f1_score(y_train, y_pred_train, average='weighted', labels=crf.classes_)

# Look at class wise score
print(classification_report(
    list(chain.from_iterable(y_test)), list(chain.from_iterable(y_pred)), labels=crf.classes_, digits=3
))
