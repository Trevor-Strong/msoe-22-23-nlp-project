import logging

import nltk
from typing import TypeVar, Sequence, Iterable, Optional, TypedDict, cast


BinInt = int


_T = TypeVar('_T')
_U = TypeVar('_U')


def unzip_unique(seq: Sequence[tuple[_T, _U]]) -> tuple[set[_T], set[_U]]:
    a, b = set(), set()
    for x, y in seq:
        a.add(x)
        b.add(y)
    return a, b


def three_windowed(source: Iterable[_T], /, *, default=None) -> Iterable[tuple[Optional[_T], _T, Optional[_T]]]:
    it = iter(source)
    prev: Optional[_T] = default
    try:
        curr: Optional[_T] = next(it)
    except StopIteration:
        return  # no elements in source

    for next_item in it:
        yield prev, curr, next_item
        prev = curr
        curr = next_item

    yield prev, curr, default


class Features(TypedDict):
    is_first_capital: BinInt
    is_first_word: BinInt
    is_last_word: BinInt
    is_complete_capital: BinInt
    is_alphanumeric: BinInt
    is_numeric: BinInt
    word_has_hyphen: BinInt
    prev_word: str
    next_word: str
    prefix_1: str
    prefix_2: str
    prefix_3: str
    prefix_4: str
    suffix_1: str
    suffix_2: str
    suffix_3: str
    suffix_4: str


def features(word: str, *, prev_word: str | None, next_word: str | None) -> Features:
    if prev_word is None:
        prev_word = ''

    if next_word is None:
        next_word = ''

    return Features(
        is_first_capital=int(word[0].isupper()),
        is_complete_capital=int(word.isupper()),
        is_first_word=int(prev_word == ''),
        is_last_word=int(next_word == ''),
        is_alphanumeric=int(word.isalpha()),
        is_numeric=int(word.isdigit()),
        word_has_hyphen=int('-' in word),
        prefix_1=word[0],
        prefix_2=word[:2],
        prefix_3=word[:3],
        prefix_4=word[:4],
        suffix_1=word[-1],
        suffix_2=word[-2:],
        suffix_3=word[-3:],
        suffix_4=word[-4:],
        next_word=next_word,
        prev_word=prev_word,
    )


def tokenize_text(text: str, /) -> list[list[str]]:
    """
    Splits the input in to tokens. Tokens are words

    :param text: input string to tokenize
    :return: a sequence containing sequences of tokens. Each inner sequence is sentence and each token is a word or
    punctuation element
    """
    sents = nltk.tokenize.sent_tokenize(text)

    val = [nltk.tokenize.word_tokenize(sent) for sent in sents]

    logging.info("tokenize_text %s", val)
    return val


def featurify(sentence: list[str], /) -> list[Features]:
    """
    Applies features to the tokens of a sentence
    :param sentence:
    :return: The same list object, with the tokens replaced with their features
    """
    last = len(sentence) - 1
    prev_tok = None
    out = cast(list[Features], sentence)
    for i, tok in enumerate(sentence):
        out[i] = features(tok, prev_word=prev_tok, next_word=None if i == last else sentence[i + 1])
        prev_tok = tok
    return out


def tokenize_with_features(text: str, /) -> list[list[Features]]:
    text_tokens = tokenize_text(text)
    for sentence in text_tokens:
        featurify(sentence)
    return cast(list[list[Features]], text_tokens)
