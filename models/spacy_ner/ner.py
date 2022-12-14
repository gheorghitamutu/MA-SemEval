import os
import json
import random
from datetime import datetime

from spacy.pipeline.ner import DEFAULT_NER_MODEL
from spacy.scorer import Scorer
from spacy.util import minibatch, compounding
from spacy.training import Example

from tqdm.notebook import tqdm_notebook
from spacy_crfsuite import CRFExtractor
from spacy_crfsuite import read_file
from spacy_crfsuite.train import gold_example_to_crf_tokens
from spacy_crfsuite.tokenizer import SpacyTokenizer

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


def train_crf_ner_model(source_train_doc_bin, source_dev_doc_bin, language, destination_model):
    component_config = {
        "features": [
            [
                "low",
                "title",
                "upper",
                "pos",
                "pos2"
            ],
            [
                "low",
                "bias",
                "prefix5",
                "prefix2",
                "suffix5",
                "suffix3",
                "suffix2",
                "upper",
                "title",
                "digit",
                "pos",
                "pos2"
            ],
            [
                "low",
                "title",
                "upper",
                "pos",
                "pos2"
            ],
        ],
        "c1": 0.01,
        "c2": 0.22
    }

    crf_extractor = CRFExtractor(component_config=component_config)
    use_dense_features = crf_extractor.use_dense_features()

    if use_dense_features:
        nlp = spacy.load("{}_core_web_md".format(language))
    else:
        nlp = spacy.load("{}_core_web_sm".format(language))

    # https://github.com/TeamHG-Memex/sklearn-crfsuite/pull/69/files
    def read_examples(file, tokenizer, use_dense_features=False, limit=None):
        examples = []
        it = read_file(file)
        it = it[:limit] if limit else it
        for raw_example in tqdm_notebook(it, desc=file):
            crf_example = gold_example_to_crf_tokens(
                raw_example,
                tokenizer=tokenizer,
                use_dense_features=use_dense_features,
                bilou=False
            )
            examples.append(crf_example)
        return examples

    # Spacy tokenizer
    tokenizer = SpacyTokenizer(nlp)

    # OPTIONAL: fine-tune hyper-params
    # this is going to take a while, so you might need a coffee break ...
    # dev_examples = None
    dev_examples = read_examples(source_dev_doc_bin, tokenizer, use_dense_features=use_dense_features)

    if dev_examples:
        rs = crf_extractor.fine_tune(dev_examples, cv=5, n_iter=30, random_state=42)
        print("best params:", rs.best_params_, ", score:", rs.best_score_)
        crf_extractor.component_config.update(rs.best_params_)

    train_examples = read_examples(source_train_doc_bin, tokenizer=tokenizer, use_dense_features=use_dense_features)

    crf_extractor.train(train_examples, dev_samples=dev_examples)
    print(crf_extractor.explain())

    if not os.path.exists(destination_model):
        os.mkdir(destination_model)
    model_name = f"{destination_model}/{nlp._meta['lang']}_{nlp._meta['name']}.bz2"
    crf_extractor.to_disk(model_name)


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


def test_crf_ner_model(source_model, source_doc_bin):
    pass