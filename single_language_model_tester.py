import spacy
from spacy import Language
from spacy.tokens import DocBin
from spacy.training import biluo_to_iob


def generate_prediction_file(model_path, source, target):

    en_nlp = spacy.load(model_path)
    # disable=['tokenizer', 'parser'])
    # print(en_nlp.get_pipe('beam_ner').labels)


    @Language.component('retokenizer_en')
    def retokenizer_en(doc):
        merge = True
        while merge:
            for i, token in enumerate(doc):
                merge = False
                if token.text == '-':  # counter-revolutionary
                    with doc.retokenize() as retokenizer:
                        retokenizer.merge(doc[i-1:i + 2])
                    merge = True
                    break
                if token.text == '\'':  # guns n' roses
                    with doc.retokenize() as retokenizer:
                        retokenizer.merge(doc[i-1:i + 1])
                    merge = True
                    break
                if token.text == '.':  # st. louis => but [ ayodhyā . 300px ] is separated....

                    tokens_no = 0
                    for _ in doc:
                        tokens_no += 1

                    if i < tokens_no - 1:  # . would not denote the end of the sentence
                        with doc.retokenize() as retokenizer:
                            retokenizer.merge(doc[i - 1:i + 1])
                        merge = True
                        break

        return doc

    # en_nlp.add_pipe('retokenizer_en', first=True)

    doc_bin = DocBin().from_disk(source)

    with open(target, 'w', encoding='utf-8') as f:

        for doc_original in doc_bin.get_docs(en_nlp.vocab):

            original_tokens = list()
            for token in doc_original:
                original_tokens.append(token)

            doc = en_nlp(doc_original.text)

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

            f.write('\n')  # USED ONLY FOR NON TEST
            f.write('\n')  # id d9b2af7f-1110-468e-a64c-75a3c7d07f6a
            for label in iob_tags:
                f.write(f'{label}\n')
            f.write('\n')  # empty line delimiting 2 phrases


generate_prediction_file(
    # r'Z:\Repositories\github\MA-SemEval\BEAM_NER_MODELS\es\bert-spanish-cased-finetuned-ner\model-best',
    # r'Z:\Repositories\github\MA-SemEval\BEAM_NER_MODELS\en\distilbert-base-uncased-finetuned-sst-2-english-20230129T000320Z-001\distilbert-base-uncased-finetuned-sst-2-english\model-best',
    # r'Z:\Repositories\github\MA-SemEval\BEAM_NER_MODELS\en\distilbert-base-uncased-finetuned-sst-2-english\model-best',
    r'Z:\Repositories\github\MA-SemEval\BEAM_NER_MODELS\sv\KB\bert-base-swedish-cased\model-best',
    r'Z:\Repositories\github\MA-SemEval\public_da\public_data\SV-Swedish\sv_dev.spacy',
    r'Z:\Repositories\github\MA-SemEval\public_da\public_data\SV-Swedish\sv_dev.pred.conll')

f = open(r'Z:\Repositories\github\MA-SemEval\public_da\public_data\SV-Swedish\sv_dev.pred.conll', 'r', encoding='utf-8')
g = open(r'Z:\Repositories\github\MA-SemEval\public_da\public_data\SV-Swedish\sv_dev.conll', 'r', encoding='utf-8')

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
