import json
import random
from datetime import datetime

from spacy.pipeline.ner import DEFAULT_NER_MODEL
from spacy.scorer import Scorer
from spacy.util import minibatch, compounding
from spacy.training import Example

import preprocess

import spacy
# spacy.require_gpu()


def train_ner_model(source_doc_bin, language, iterations, destination_model):
    config = {
        "moves": None,
        "update_with_oracle_cut_size": 100,
        "model": DEFAULT_NER_MODEL,
        "incorrect_spans_key": None,
        "scorer": {"@scorers": "spacy.ner_scorer.v1"},
    }

    nlp = spacy.blank(language)
    nlp.add_pipe("ner", config=config)

    en_train_examples = preprocess.examples_from_doc_bin(source_doc_bin, nlp)

    # create the model and start the training
    optimizer = nlp.begin_training()
    for i in range(iterations):
        print('{} Iteration #{}/{}'.format(datetime.now(), i + 1, iterations))
        now = datetime.now()
        # shuffle the training data
        random.shuffle(en_train_examples)
        losses = {}
        # create minibatches
        batches = minibatch(en_train_examples, size=compounding(4.0, 32.0, 1.001))
        # update the model for each minibatch
        for j, batch in enumerate(batches):
            # https://stackoverflow.com/questions/66342359/nlp-update-issue-with-spacy-3-0-typeerror-e978-the-language-update-method-ta
            for text, annotations in batch:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                nlp.update([example], sgd=optimizer, losses=losses)

        print('Losses {}'.format(losses))
        print('{} Iteration #{}/{} ended: {}'.format(datetime.now(), i + 1, iterations, datetime.now() - now))

    # save the trained model
    nlp.to_disk(destination_model)


def train_beam_ner_model(source_doc_bin, language, iterations, destination_model):
    config = {
        "moves": None,
        "update_with_oracle_cut_size": 100,
        "model": DEFAULT_NER_MODEL,
        "beam_density": 0.01,
        "beam_update_prob": 0.5,
        "beam_width": 32,
        "incorrect_spans_key": None,
        "scorer": None,
    }

    nlp = spacy.blank(language)
    nlp.add_pipe("beam_ner", config=config)

    en_train_examples = preprocess.examples_from_doc_bin(source_doc_bin, nlp)

    # create the model and start the training
    optimizer = nlp.begin_training()
    for i in range(iterations):
        print('{} Iteration #{}/{}'.format(datetime.now(), i + 1, iterations))
        now = datetime.now()
        # shuffle the training data
        random.shuffle(en_train_examples)
        losses = {}
        # create minibatches
        batches = minibatch(en_train_examples, size=compounding(4.0, 32.0, 1.001))
        # update the model for each minibatch
        for j, batch in enumerate(batches):
            # https://stackoverflow.com/questions/66342359/nlp-update-issue-with-spacy-3-0-typeerror-e978-the-language-update-method-ta
            for text, annotations in batch:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                nlp.update([example], sgd=optimizer, losses=losses)

        print('Losses {}'.format(losses))
        print('{} Iteration #{}/{} ended: {}'.format(datetime.now(), i + 1, iterations, datetime.now() - now))

    # save the trained model
    nlp.to_disk(destination_model)


def test_ner_model(source_model, source_doc_bin):
    nlp = spacy.load(source_model)
    en_dev_examples = preprocess.examples_from_doc_bin(source_doc_bin, nlp)

    examples = []
    scorer = Scorer()
    for text, annotations in en_dev_examples:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        example.predicted = nlp(str(example.predicted))
        examples.append(example)

    # https://github.com/explosion/spaCy/issues/4094 (How is the overall F Score of an NER Model calculated?)
    scores = scorer.score(examples)
    print(json.dumps(scores, indent=4))


def test_beam_ner_model(source_model, source_doc_bin):
    nlp = spacy.load(source_model)
    en_dev_examples = preprocess.examples_from_doc_bin(source_doc_bin, nlp)

    examples = []
    scorer = Scorer()
    for text, annotations in en_dev_examples:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        example.predicted = nlp(str(example.predicted))
        examples.append(example)

    # https://github.com/explosion/spaCy/issues/4094 (How is the overall F Score of an NER Model calculated?)
    scores = scorer.score(examples)
    print(json.dumps(scores, indent=4))

