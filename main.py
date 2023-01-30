import json
from pathlib import Path

from spacy.scorer import Scorer
from spacy.tokens import DocBin
from spacy_conll import ConllParser

import preprocess
import models.spacy_ner


def train_spacy_ner():
    data_dir = r'./MultiCoNER_2_train_dev'
    preprocess.remove_comments_from_conll_file(r'{}/train_dev/en/en-train.conll'.format(data_dir),
                                               r'{}/train_dev/en/en-train.conll.clean.conll'.format(data_dir))

    models.spacy_ner.train_ner_model('{}/train_dev_spacy/en-train.conll.clean.spacy'.format(data_dir),
                                     'en', 100, "./model_spacy_en_default_ner")


def test_spacy_ner():
    data_dir = r'./MultiCoNER_2_train_dev'
    preprocess.remove_comments_from_conll_file(r'{}/train_dev/en/en-dev.conll'.format(data_dir),
                                               r'{}/train_dev/en/en-dev.conll.clean.conll'.format(data_dir))

    models.spacy_ner.test_ner_model('./model_spacy_en_default_ner',
                                    '{}/train_dev_spacy/en-dev.conll.clean.spacy'.format(data_dir))


def train_spacy_beam_ner():
    data_dir = r'./MultiCoNER_2_train_dev'
    preprocess.remove_comments_from_conll_file(r'{}/train_dev/en/en-train.conll'.format(data_dir),
                                               r'{}/train_dev/en/en-train.conll.clean.conll'.format(data_dir))

    models.spacy_ner.train_beam_ner_model('{}/train_dev_spacy/en-train.conll.clean.spacy'.format(data_dir),
                                          'en', 100, "./model_spacy_en_beam_ner")


def test_spacy_beam_ner():
    data_dir = r'./MultiCoNER_2_train_dev'
    preprocess.remove_comments_from_conll_file(r'{}/train_dev/en/en-dev.conll'.format(data_dir),
                                               r'{}/train_dev/en/en-dev.conll.clean.conll'.format(data_dir))

    models.spacy_ner.test_beam_ner_model('./model_spacy_en_beam_ner',
                                         '{}/train_dev_spacy/en-dev.conll.clean.spacy'.format(data_dir))


def train_spacy_crf_ner():
    models.spacy_ner.train_crf_ner_model(
        r'Z:\Repositories\github\MA-SemEval\MultiCoNER_2_train_dev\train_dev\en\en-train.spacy',
        r'Z:\Repositories\github\MA-SemEval\MultiCoNER_2_train_dev\train_dev\en\en-dev.spacy',
        'en',
        "./model_spacy_en_crf_ner")


def test_spacy_crf_ner_model():
    models.spacy_ner.test_crf_ner_model(
        './model_spacy_en_crf_ner/en_core_web_sm.bz2',
        './model_spacy_en_default_ner',
        r'Z:\Repositories\github\MA-SemEval\public_dat\public_data\EN-English\en_dev.clean.spacy')


def test_spacy_model():
    results = \
        models.spacy_ner.test_spacy_model(
            r'Z:\Repositories\github\MA-SemEval\model_distilbert_base_uncased_finetuned_sst_2_english\model-best',
            r'Z:\Repositories\github\MA-SemEval\public_dat\public_data\EN-English\en_dev.spacy')
    print(results)


if __name__ == "__main__":
    # train_spacy_ner()
    # test_spacy_ner()
    # train_spacy_beam_ner()
    # test_spacy_beam_ner()
    train_spacy_crf_ner()
    test_spacy_crf_ner_model()
    # data_dir = r'./MultiCoNER_2_train_dev'
    # preprocess.remove_comments_from_conll_file('{}/train_dev/sv/sv-train.conll'.format(data_dir),
    #                                            '{}/train_dev/sv/sv-train.clean.conll'.format(data_dir))

    # preprocess.remove_comments_from_conll_file(
    #     r'Z:\Repositories\github\MA-SemEval\public_dat\public_data\EN-English\en_train.conll',
    #     r'Z:\Repositories\github\MA-SemEval\public_dat\public_data\EN-English\en_train.clean.conll')

    # test_spacy_model()

