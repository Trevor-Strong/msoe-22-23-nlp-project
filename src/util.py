import enum
import logging
from pathlib import Path
from typing import TypeVar, Sequence, Iterable, Optional, TypedDict, cast, Generic

import nltk


PROJ_ROOT = Path(__file__).parents[1]
MODEL_FILE = PROJ_ROOT / 'model.pickle'

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


_logging_inited = False


def init_logging(*, level: int = logging.INFO, expect_first: bool = False):
    global _logging_inited
    if _logging_inited:
        if expect_first:
            logging.debug("Attempted to re-initialize logging with level %(level)s",
                          {'level': logging.getLevelName(level)})
        return
    _logging_inited = True
    logging.basicConfig(format='[%(levelname)s]: %(message)s', level=level)


_nltk_download_logger = None


def nltk_download(item: str, *, level: int = logging.INFO):
    global _nltk_download_logger

    if _nltk_download_logger is None:
        init_logging()

        class LoggerFile:
            level: int

            def __init__(self, level: int):
                self.level = level

            def write(self, data: str):
                if data.strip():  # only log if input is non-empty
                    for line in data.splitlines():
                        logging.log(self.level, line)
        _nltk_download_logger = LoggerFile(level=level)

    nltk.download(item, print_error_to=_nltk_download_logger)


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


def features(word: str, *, prev_word: Optional[str], next_word: Optional[str]) -> Features:
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

    logging.debug("tokenize_text %s", val)
    return val


def featurify(sentence: list[str], /) -> list[Features]:
    out = []
    for prev, curr, next in three_windowed(sentence):
        out.append(features(curr, prev_word=prev, next_word=next))
    return out


def tokenize_with_features(text: str, /) -> list[list[Features]]:
    text_tokens = tokenize_text(text)
    for sentence in text_tokens:
        featurify(sentence)
    return cast(list[list[Features]], text_tokens)


class POSTag(enum.Enum):
    VERB = 'Verb'
    NOUN = 'Noun'
    PRON = 'Pronoun'
    ADJ = 'Adjective'
    ADP = 'Adposition'
    CONJ = 'Conjunction'
    DET = 'Determiner'
    NUM = 'Number'
    PRT = 'Particles'
    X = 'Other'
    PUNCT = 'Punctuation'

    @classmethod
    def from_str(cls, pos_tag: str):
        if pos_tag == '.':
            return cls.PUNCT
        else:
            return cls[pos_tag]


class OutVar(Generic[_T]):
    value: Optional[_T]

    def __init__(self, default: Optional[_T] = None):
        self.value = default
