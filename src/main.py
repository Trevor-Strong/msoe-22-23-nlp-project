from sklearn_crfsuite import CRF
import pickle
from pathlib import Path
import logging as log
import train_model
import sys
from typing import Sequence


def _train_and_save_model() -> CRF:
    model = train_model.train()
    log.info("Saving model to %(model_path)s", train_model.FILEPATH)
    try:
        with open(train_model.FILEPATH, "wb") as f:
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
        log.info("Loading model from %(model_path)s", train_model.FILEPATH)
        try:
            with open(train_model.FILEPATH, "rb") as f:
                model = pickle.load(f)
        except OSError as e:
            log.error(
                "Failed to load model from file %(model_path)s, training new model...",
                train_model.FILEPATH,
                exc_info=e
            )
            model = _train_and_save_model()

    return model


def main(args: Sequence[str] = ()):

    import pprint
    log.basicConfig(format='[{levelname}]: {message}', level=log.INFO, style='{')
    model = load_model()
    print("Write text you want the model classify")
    while True:
        text = input("Input:\n")
        tokens = tokenize_with_features(text)
        prediction = model.predict(tokens)
        print("Prediction:")
        pprint.pprint(prediction)


if __name__ == '__main__':
    main(sys.argv[1:])

