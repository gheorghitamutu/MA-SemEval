import spacy
from spacy import Language
from spacy.tokens import DocBin
from spacy.training import biluo_to_iob


def get_iob_tags(doc_original, nlp):
    original_tokens = list()
    for token in doc_original:
        original_tokens.append(token)

    doc = nlp(doc_original.text)

    tokens = list()
    for token in doc:
        tokens.append(token)

    tokens_no = len(tokens)

    if tokens_no != len(original_tokens):
        while tokens_no != len(original_tokens):
            for i, (a, b) in enumerate(zip(original_tokens, doc)):
                if a.text != b.text:
                    if i + 1 < len(original_tokens):
                        next_a = original_tokens[i + 1]
                        offset = 0
                        for j in range(i + 1, tokens_no):
                            offset += 1
                            d_next = doc[j].text
                            if d_next == next_a.text or \
                                    (next_a.text.startswith(d_next) and j < tokens_no - 2):
                                break

                        if offset == 1:
                            offset += 1
                        if i + offset > tokens_no:
                            raise Exception('')

                        with doc.retokenize() as retokenizer:
                            retokenizer.merge(doc[i:i + offset])
                    else:
                        with doc.retokenize() as retokenizer:
                            retokenizer.merge(doc[i:tokens_no])

                    tmp_tokens_no = 0
                    for _ in doc:
                        tmp_tokens_no += 1

                    if tmp_tokens_no == tokens_no:
                        raise Exception('')
                    tokens_no = tmp_tokens_no

                    break

        tokens = list()
        for token in doc:
            tokens.append(token)

    biluo_tags = list()
    for token in tokens:
        if len(token.ent_type_) > 0:
            tag = f'{token.ent_iob_}-{token.ent_type_}'
        else:
            tag = token.ent_iob_
        # biluo_tags.append(f'({token} => {tag})')
        biluo_tags.append(tag)

    iob_tags = biluo_to_iob(biluo_tags)
    if iob_tags != biluo_tags:
        raise Exception('')

    return iob_tags


def generate_prediction_file(model_paths, source, target):
    nlps = list()
    for path in model_paths:
        nlps.append(spacy.load(path))

    # phrases = list()
    #
    # with open(source, 'r', encoding='utf-8') as f:
    #     phrase = list()
    #     for line in f.readlines():
    #         line = line.strip()
    #         if line.startswith('#'):
    #             continue
    #         if line == '':
    #             if len(phrase) != 0:
    #                 phrases.append(phrase)
    #                 phrase = list()
    #             continue
    #         pieces = line.split(' ')
    #         token = pieces[0]
    #         phrase.append(token)
    #
    # with open(target, 'w', encoding='utf-8') as f:

        # for phrase in phrases:
        #
        #     filtered_phrase = [token for token in phrase if len(token) > 2]
        #
        #     # https://github.com/explosion/spaCy/issues/2224#issuecomment-382548415
        #     tokens_in_vocab = list()
        #     for nlp in nlps:
        #         count = len([token for token in filtered_phrase if token in nlp.vocab])
        #         tokens_in_vocab.append(count)
        #
        #     max_count = max(tokens_in_vocab)
        #     if max_count == 0:
        #         # f.write('\n')  # USED ONLY FOR NON TEST
        #         f.write('\n')  # id d9b2af7f-1110-468e-a64c-75a3c7d07f6a
        #         for _ in range(len(phrase)):
        #             f.write(f'O\n')
        #         f.write('\n')  # empty line delimiting 2 phrases
        #
        #         continue
        #
        #     index_of_max_value = tokens_in_vocab.index(max_count)
        #
        #     best_nlp = nlps.index(index_of_max_value)
        #     for doc_original in best_nlp(token):
        #         pass

    doc_bin_1 = DocBin().from_disk(source)
    doc_bin_2 = DocBin().from_disk(source)
    doc_bin_3 = DocBin().from_disk(source)

    with open(target, 'w', encoding='utf-8') as f:

        for doc_1, doc_2, doc_3 in zip(
                doc_bin_1.get_docs(nlps[0].vocab),
                doc_bin_2.get_docs(nlps[1].vocab),
                doc_bin_3.get_docs(nlps[2].vocab)):

            docs = [doc_1, doc_2, doc_3]
            tags = list()
            for i, nlp in enumerate(nlps):
                tags.append(get_iob_tags(docs[i], nlp))

            final_tags = [set(elements) for elements in zip(*tags)]

            # f.write('\n')  # USED ONLY FOR NON TEST
            f.write('\n')  # id d9b2af7f-1110-468e-a64c-75a3c7d07f6a
            for label_set in final_tags:
                if len(label_set) > 1:
                    filtered_labels = [label for label in label_set if label != 'O']
                    if len(filtered_labels) > 0:
                        label = filtered_labels[0]
                    else:
                        label = 'O'
                else:
                    label = list(label_set)[0]

                f.write(f'{label}\n')
            f.write('\n')  # empty line delimiting 2 phrases
            f.write('\n')  # USED ONLY FOR NON TEST MULTI
            # f.flush()


generate_prediction_file([
        # r'Z:\Repositories\github\MA-SemEval\BEAM_NER_MODELS\en\distilbert-base-uncased-finetuned-sst-2-english-20230129T000320Z-001\distilbert-base-uncased-finetuned-sst-2-english\model-best',
        r'Z:\Repositories\github\MA-SemEval\BEAM_NER_MODELS\en\distilbert-base-uncased-finetuned-sst-2-english-20230129T000320Z-001\distilbert-base-uncased-finetuned-sst-2-english\model-best',
        r'Z:\Repositories\github\MA-SemEval\BEAM_NER_MODELS\es\bert-spanish-cased-finetuned-ner\model-best',
        r'Z:\Repositories\github\MA-SemEval\BEAM_NER_MODELS\sv\KB\bert-base-swedish-cased\model-best'
    ],
    # r'Z:\Repositories\github\MA-SemEval\public_da\public_data\MULTI_Multilingual\multi_dev.conll',
    r'Z:\Repositories\github\MA-SemEval\public_da\public_data\MULTI_Multilingual\multi_dev.spacy',
    r'Z:\Repositories\github\MA-SemEval\public_da\public_data\MULTI_Multilingual\multi_dev.pred.conll'
)

f = open(r'Z:\Repositories\github\MA-SemEval\public_da\public_data\MULTI_Multilingual\multi_dev.pred.conll', 'r', encoding='utf-8')
g = open(r'Z:\Repositories\github\MA-SemEval\public_da\public_data\MULTI_Multilingual\multi_dev.conll', 'r', encoding='utf-8')

f_lines = f.readlines()
g_lines = g.readlines()

lineNo = 0
for (i, fl), (j, gl) in zip(enumerate(f_lines), enumerate(g_lines)):
    fl = fl.strip()
    gl = gl.strip()

    print(f'[{lineNo}] => [{gl}] => [{fl}]')
    lineNo += 1

    if gl.strip().startswith('#') and len(fl) > 0:
        raise Exception('')
    if len(gl.strip()) == 0 and len(fl.strip()) != 0:
        raise Exception('')