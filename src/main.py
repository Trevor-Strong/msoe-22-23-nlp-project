from typing import Callable, TextIO, Sequence, Optional

import numpy as np
from sklearn_crfsuite import CRF

import util
import pos_model

POS_TAGSET = {
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


def nameof(pos: str) -> str:
    return POS_TAGSET[pos]


def with_article(s: str) -> str:
    return '{} {}'.format('an' if s.startswith(tuple('aeiouAEIOU')) else 'a', s)


def guess_next_pos(transition_feats: dict[tuple[str, str], float], pos: str):
    poses = []
    probs = []
    for (the_pos, next_pos), prob in transition_feats.items():
        if the_pos == pos:
            poses.append(next_pos)
            probs.append(prob)
    i = np.argmax(probs)
    # print(f"{i!r}")
    return poses[int(i)]


def display(model: CRF, text: str, writer: Callable[[str], None]):
    tokens: list[list[str]] = util.tokenize_text(text)
    features: list[list[pos_model.Features]] = [util.featurify(sent) for sent in tokens]
    prediction: list[list[str]] = model.predict(features)
    writer("Prediction: ")

    for sents in zip(tokens, prediction):
        for i, (tok, pos) in enumerate(zip(*sents)):
            writer("{} is a {}. It is usually followed by {}".format(
                repr(tok),
                nameof(pos),
                with_article(nameof(guess_next_pos(model.transition_features_, pos)))
            ))


def _looping(writer: Callable[[str], None]):
    model = pos_model.load()
    print("Write out the sentence you want classified.")
    print("\nUse Ctl-C to exit")

    try:
        while True:
            text = input("> ")
            display(model, text, writer)

    except (KeyboardInterrupt, EOFError):
        pass


def _oneshot(reader: TextIO, writer: Callable[[str], None]):
    model = pos_model.load()
    with reader:
        text = " ".join(reader.readlines())
        display(model, text, writer)


def main(args: Sequence[str] = None) -> None:
    if args is None:
        import sys
        args = sys.argv[1:]

    if not args:
        return _looping(writer=print)

    import argparse

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--input",
                        action="store",
                        dest="reader",
                        default=None,
                        metavar="INPUT_FILE",
                        help="path to the input file to read from")
    parser.add_argument("--output",
                        action="store",
                        dest="writer",
                        default=None,
                        metavar="OUTPUT_FILE",
                        help="Path to the output file, to write results to")
    options = parser.parse_args(args)
    reader: Optional[str] = options.reader
    writer: Optional[str] = options.writer

    if reader is None:
        if writer is None:
            return _looping(writer=print)

        with open(writer, mode="wt", encoding="utf-8") as f:
            from functools import partial
            return _looping(writer=partial(print, file=f))

    reader: TextIO = open(reader, mode="rt", encoding="utf-8")

    if writer is None:
        _oneshot(writer=print, reader=reader)
    else:
        with open(writer, mode="wt", encoding="utf-8") as f:
            from functools import partial
            return _oneshot(reader=reader, writer=partial(print, file=f))


if __name__ == '__main__':
    main()
