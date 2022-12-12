"""
Pre-trained model provided by the spaCy library in Python.
Here is an example of how to use this library to perform named entity recognition on a given input text.
"""
import random

# Import the necessary modules
import spacy
from spacy import displacy

# Load the pre-trained model
nlp = spacy.load("en_core_web_sm")

# Define a sample input text
input_text = "Apple is looking at buying U.K. startup for $1 billion"

# Run the named entity recognition on the input text
doc = nlp(input_text)

# Print the entities found in the text
for ent in doc.ents:
    print(ent.text, ent.label_)

# Visualize the entities in the input text
displacy.render(doc, style="ent")
displacy.render(nlp('Apple and Google are tech companies.'), style="ent")

# https://spacy.io/usage/visualizers
# displacy.serve(doc, style="ent")


"""
Here is an example of code that can be used to train a model 
for natural language processing (NLP) entity recognition.
"""

import spacy
from spacy.util import minibatch, compounding
from spacy.training import Example

# load the spacy model
nlp = spacy.blank('en')  # https://stackoverflow.com/questions/66367447/spacy-cant-find-tables-lexeme-norm-for-language-en-in-spacy-lookups-data

# define the training data
TRAIN_DATA = [
    ("Who is Shaka Khan?", {"entities": [(7, 17, "PERSON")]}),
    ("I like London and Berlin.", {"entities": [(7, 13, "LOC"), (18, 24, "LOC")]}),
]

# create the model and start the training
optimizer = nlp.begin_training()
for i in range(20):
    # shuffle the training data
    random.shuffle(TRAIN_DATA)
    # create minibatches
    batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
    # update the model for each minibatch
    for batch in batches:
        # https://stackoverflow.com/questions/66342359/nlp-update-issue-with-spacy-3-0-typeerror-e978-the-language-update-method-ta
        for text, annotations in batch:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update([example], sgd=optimizer)

# save the trained model
nlp.to_disk("./model")

# nlp = spacy.load("./model")

# input_text = "Apple is looking at buying U.K. startup for $1 billion"
# doc = nlp(input_text)
# displacy.serve(doc, style="ent")


import spacy
from spacy.pipeline.ner import DEFAULT_NER_MODEL
config = {
   "moves": None,
   "update_with_oracle_cut_size": 100,
   "model": DEFAULT_NER_MODEL,
   "incorrect_spans_key": "incorrect_spans",
}

# Load the spaCy English language model
nlp = spacy.blank('en')

# Create a new EntityRecognizer and add it to the pipeline
# ner = EntityRecognizer(nlp.vocab)
# nlp.add_pipe(ner)
nlp.add_pipe("ner", config=config)

# Load the training data, which should be a list of text examples and their corresponding
# entities (the entities should be a list of tuples with the start and end character offsets
# for each entity in the text)
train_data = [('Apple is a fruit.', {"entities": [(0, 5, 'ORG')]}),
              ('I like to eat apples.', {"entities": [(14, 20, 'ORG')]}),
              ('He works for Google.', {"entities": [(13, 19, 'ORG')]})]

# Train the EntityRecognizer on the training data
optimizer = nlp.begin_training()
for i in range(20):
    # shuffle the training data
    random.shuffle(train_data)
    # create minibatches
    batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
    # update the model for each minibatch
    for batch in batches:
        # https://stackoverflow.com/questions/66342359/nlp-update-issue-with-spacy-3-0-typeerror-e978-the-language-update-method-ta
        for text, annotations in batch:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update([example], sgd=optimizer)

# Save the trained model
nlp.to_disk('./model2')

# Test the trained model on some new text
nlp = spacy.load('./model2')
text = 'Apple and Google are tech companies.'
doc = nlp(text)
for ent in doc.ents:
    print(ent.text, ent.label_)
displacy.serve(doc, style="ent")
