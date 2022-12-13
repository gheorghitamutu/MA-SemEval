from spacy.tokens import DocBin


def examples_from_doc_bin(source, nlp):
    doc_bin = DocBin().from_disk(source)
    return [(doc.text, {"entities": [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]})
            for doc in doc_bin.get_docs(nlp.vocab)]
