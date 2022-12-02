# MA-SemEval
Machine Translation Final Project - SemEval

# State of Art & Previous Work

## What is SemEval?
> SemEval is a series of international natural language processing (NLP) research workshops whose mission is to advance the current state of the art in semantic analysis and to help create high-quality annotated datasets in a range of increasingly challenging problems in natural language semantics. Each year's workshop features a collection of shared tasks in which computational semantic analysis systems designed by different teams are presented and compared.

https://semeval.github.io

## SemEval 2023 Multilingual Tasks
https://semeval.github.io/SemEval2023

| Type                        | Task                                                                                                                                                                    |
| ----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Semantic Structure          | [Task 2: Multilingual Complex Named Entity Recognition (MultiCoNER 2)](http://multiconer.github.io)                                                                     |
| Discourse and Argumentation | [Task 3: Detecting the Category, the Framing, and the Persuasion Techniques in Online News in a Multi-lingual Setup](https://propaganda.math.unipd.it/semeval2023task3) |
| Social Attitudes            | [Task 9: Multilingual Tweet Intimacy Analysis](https://semeval.github.io/SemEval2023/tasks#:~:text=Task%209%3A%20Multilingual%20Tweet%20Intimacy%20Analysis)            |

## Previous Work
FII had a previous entry at SemEval in 2021 for [Task 2: Multilingual and Cross-lingual Word-in-Context Disambiguation](https://aclanthology.org/2021.semeval-1.104).

The paper presents:
> a word-in-context disambiguation system. The task focuses on capturing the polysemous nature of words in a multilingual and cross-lingual 
setting, without considering a strict inventory of word meanings. The system applies Natural Language Processing algorithms on datasets from SemEval 2021 
Task 2, being able to identify the meaning of words for the languages Arabic, Chinese, English, French and Russian, without making use of any additional mono- or 
multilingual resources.

Models used in that task were, as stated [here](https://aclanthology.org/2021.semeval-1.104.pdf): 
> Sent2Vec presents a simple but efficient unsupervised method to train distributed representations of sentences.

> As baseline, we used the Lesk algorithm.

> Our final approach was to combine the Lesk algorithm, along with Sent2Vec and vector cosine (Bojanowski et al., 2017). The cosine similarity is 
computed for each pair of sentences in our input. The pipeline used cross-lingually aligned versions of fasttext word vectors.
 
