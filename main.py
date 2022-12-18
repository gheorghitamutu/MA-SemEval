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
    data_dir = r'./MultiCoNER_2_train_dev'
    models.spacy_ner.train_crf_ner_model('{}/train_dev/en/en-train.conll'.format(data_dir),
                                         '{}/train_dev/en/en-dev.conll'.format(data_dir),
                                         'en', "./model_spacy_en_crf_ner")


def test_spacy_crf_ner_model():
    data_dir = r'./MultiCoNER_2_train_dev'
    models.spacy_ner.test_crf_ner_model('./model_spacy_en_crf_ner/en_core_web_sm.bz2',
                                        './model_spacy_en_default_ner',
                                        '{}/train_dev_spacy/en-dev.conll.clean.spacy'.format(data_dir))


if __name__ == "__main__":
    # train_spacy_ner()
    # test_spacy_ner()
    # train_spacy_beam_ner()
    # test_spacy_beam_ner()
    # train_spacy_crf_ner()
    test_spacy_crf_ner_model()


