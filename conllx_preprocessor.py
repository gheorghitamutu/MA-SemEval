import spacy
from alive_progress import alive_bar
from spacy.tokens import DocBin
from spacy.training import iob_to_biluo, biluo_tags_to_spans


TAXONOMY = {
    'Creative Works': [
        'VisualWork',
        'MusicalWork',
        'WrittenWork',
        'ArtWork',
        'Software',
        'OtherCW'
    ],
    'Group': [
        'MusicalGRP',
        'PublicCorp',
        'PrivateCorp',
        'OtherCORP',
        'AerospaceManufacturer',
        'SportsGRP',
        'CarManufacturer',
        'TechCORP',
        'ORG'
    ],
    'Person': [
        'Scientist',
        'Artist',
        'Athlete',
        'Politician',
        'Cleric',
        'SportsManager',
        'OtherPER'
    ],
    'Product': [
        'Clothing',
        'Vehicle',
        'Food',
        'Drink',
        'OtherPROD'
    ],
    'Medical': [
        'Medication/Vaccine',
        'MedicalProcedure',
        'AnatomicalStructure',
        'Symptom',
        'Disease',
    ],
    'Location': [
        'Facility',
        'OtherLOC',
        'HumanSettlement',
        'Station'
    ],
}

TAXONOMY_FLATENED = sorted({x for v in TAXONOMY.values() for x in v})
print(TAXONOMY_FLATENED)


def conllx_to_spacy(language, source, target, has_tags):
    print(f'Processing: {language} {source} {target} {has_tags}...')

    phrases = list()

    with open(source, 'r', encoding='utf-8') as f:
        phrase = list()
        for line in f.readlines():
            line = line.strip()
            if line.startswith('#'):
                continue
            if line == '':
                if len(phrase) != 0:
                    phrases.append(phrase)
                    phrase = list()
                continue
            pieces = line.split(' ')
            token, entity = pieces[0], pieces[-1]
            phrase.append((token, entity))

    if not language.lower().startswith('multi'):
        nlp = spacy.blank(language.split('-')[0].lower())
    else:
        nlp = spacy.blank('en')  # not good :(

    ner = nlp.add_pipe('ner', last=True)
    for label in TAXONOMY_FLATENED:
        ner.add_label(label)

    with alive_bar(len(phrases)) as bar:

        docs = list()
        for index, phrase in enumerate(phrases):
            bar()

            tokens = list()
            labels = list()

            for (token, label) in phrase:
                tokens.append(token)
                labels.append(label)

            sentence = ' '.join(tokens)
            doc = nlp.make_doc(sentence)

            tokens_no = 0
            for _ in doc:
                tokens_no += 1

            if tokens_no != len(tokens):
                while tokens_no != len(tokens):
                    for i, (a, b) in enumerate(zip(tokens, doc)):
                        if a != b.text:
                            if i + 1 < len(tokens):
                                next_a = tokens[i + 1]
                                offset = 0
                                for j in range(i + 1, tokens_no):
                                    offset += 1
                                    d_next = doc[j].text
                                    if d_next == next_a or \
                                            (next_a.startswith(d_next) and j < tokens_no - 2):
                                        break

                                if offset == 1:
                                    offset += 1
                                if i + offset > tokens_no:
                                    raise Exception('')
                                with doc.retokenize() as retokenizer:
                                    retokenizer.merge(doc[i:i+offset])
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

            if has_tags:
                iob_tags = list()
                for label in labels:
                    if label != 'O':
                        if '-' in label:
                            real_label = label.split('-')[-1]
                            if real_label not in TAXONOMY_FLATENED:
                                raise Exception(f'Missing tag: {real_label}')
                        else:
                            raise Exception(f'Missing tag: {label}')
                    iob_tags.append(label)

                biluo_tags = iob_to_biluo(iob_tags)
                doc.ents = biluo_tags_to_spans(doc, biluo_tags)

            docs.append(doc)

        doc_bin = DocBin(docs=docs)
        doc_bin.to_disk(target)

        # from spacy import displacy
        # displacy.serve(docs, style="ent")

        print('Done.')


root = r'Z:\Repositories\github\MA-SemEval\public_da\public_data'
types = ['dev', 'train', 'test']
languages = ['EN-English', 'BN-Bangla', 'DE-German', 'ES-Spanish', 'FA-Farsi', 'FR-French', 'HI-Hindi', 'IT-Italian',
             'MULTI_Multilingual', 'PT-Portuguese', 'SV-Swedish', 'UK-Ukrainian', 'ZH-Chinese']

for language in languages:
    for tp in types:
        if '-' in language:
            lg = language.split('-')[0].lower()
        elif '_' in language:
            lg = language.split('_')[0].lower()
        s = rf'{root}\{language}\{lg}_{tp}.conll'
        t = rf'{root}\{language}\{lg}_{tp}.spacy'
        conllx_to_spacy(language, s, t, tp != 'test')

