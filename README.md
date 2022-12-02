# MA-SemEval
Machine Translation Final Project - SemEval

- [MA-SemEval](#ma-semeval)
- [State of Art \& Previous Work](#state-of-art--previous-work)
  - [What is SemEval?](#what-is-semeval)
  - [Previous Work (FII Team)](#previous-work-fii-team)
  - [SemEval 2023 Multilingual Tasks](#semeval-2023-multilingual-tasks)
  - [SemEval 2022 Multilingual Tasks](#semeval-2022-multilingual-tasks)
  - [SemEval 2021 Multilingual Tasks](#semeval-2021-multilingual-tasks)
  - [SemEval 2020 Multilingual Tasks](#semeval-2020-multilingual-tasks)


# State of Art & Previous Work

## What is SemEval?
> SemEval is a series of international natural language processing (NLP) research workshops whose mission is to advance the current state of the art in semantic analysis and to help create high-quality annotated datasets in a range of increasingly challenging problems in natural language semantics. Each year's workshop features a collection of shared tasks in which computational semantic analysis systems designed by different teams are presented and compared.

https://semeval.github.io

## Previous Work (FII Team)
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
 
 The structure of this entry is:

 Section 1 - Introduction
 > section 2 describes the literature related to sense disambiguation, section 3 presents the dataset and 
method of this study, section 4 resumes the results of the conducted experiments, followed by section 
5 with the conclusions and discussions about how to increase the accuracy.

## SemEval 2023 Multilingual Tasks
https://semeval.github.io/SemEval2023

| Type                        | Task                                                                                                                                                                    |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Semantic Structure          | [Task 2: Multilingual Complex Named Entity Recognition (MultiCoNER 2)](http://multiconer.github.io)                                                                     |
| Discourse and Argumentation | [Task 3: Detecting the Category, the Framing, and the Persuasion Techniques in Online News in a Multi-lingual Setup](https://propaganda.math.unipd.it/semeval2023task3) |
| Social Attitudes            | [Task 9: Multilingual Tweet Intimacy Analysis](https://semeval.github.io/SemEval2023/tasks#:~:text=Task%209%3A%20Multilingual%20Tweet%20Intimacy%20Analysis)            |

| Task | Papers                                                                                                                                                                                                      |
| ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2    | [SemEval-2022 Task 2: Multilingual Idiomaticity Detection and Sentence Embedding](https://aclanthology.org/2022.semeval-1.13)                                                                               |
| 2    | [Helsinki-NLP at SemEval-2022 Task 2: A Feature-Based Approach to Multilingual Idiomaticity Detection](https://aclanthology.org/2022.semeval-1.14)                                                          |
| 2    | [Hitachi at SemEval-2022 Task 2: On the Effectiveness of Span-based Classification Approaches for Multilingual Idiomaticity Detection](https://aclanthology.org/2022.semeval-1.15)                          |
| 2    | [UAlberta at SemEval 2022 Task 2: Leveraging Glosses and Translations for Multilingual Idiomaticity Detection](https://aclanthology.org/2022.semeval-1.16)                                                  |
| 2    | [HYU at SemEval-2022 Task 2: Effective Idiomaticity Detection with Consideration at Different Levels of Contextualization](https://aclanthology.org/2022.semeval-1.17)                                      |
| 2    | [drsphelps at SemEval-2022 Task 2: Learning idiom representations using BERTRAM](https://aclanthology.org/2022.semeval-1.18)                                                                                |
| 2    | [JARVix at SemEval-2022 Task 2: It Takes One to Know One? Idiomaticity Detection using Zero and One-Shot Learning](https://aclanthology.org/2022.semeval-1.19)                                              |
| 2    | [CardiffNLP-Metaphor at SemEval-2022 Task 2: Targeted Fine-tuning of Transformer-based Language Models for Idiomaticity Detection](https://aclanthology.org/2022.semeval-1.20)                              |
| 2    | [kpfriends at SemEval-2022 Task 2: NEAMER - Named Entity Augmented Multi-word Expression Recognizer](https://aclanthology.org/2022.semeval-1.21)                                                            | 
| 2    | [daminglu123 at SemEval-2022 Task 2: Using BERT and LSTM to Do Text Classification](https://aclanthology.org/2022.semeval-1.22)                                                                             |
| 2    | [HiJoNLP at SemEval-2022 Task 2: Detecting Idiomaticity of Multiword Expressions using Multilingual Pretrained Language Models](https://aclanthology.org/2022.semeval-1.23)                                 |
| 2    | [ZhichunRoad at SemEval-2022 Task 2: Adversarial Training and Contrastive Learning for Multiword Representations](https://aclanthology.org/2022.semeval-1.24)                                               |
| 2    | [NER4ID at SemEval-2022 Task 2: Named Entity Recognition for Idiomaticity Detection](https://aclanthology.org/2022.semeval-1.25)                                                                            |
| 2    | [YNU-HPCC at SemEval-2022 Task 2: Representing Multilingual Idiomaticity based on Contrastive Learning](https://aclanthology.org/2022.semeval-1.26)                                                         |
| 2    | [OCHADAI at SemEval-2022 Task 2: Adversarial Training for Multilingual Idiomaticity Detection](https://aclanthology.org/2022.semeval-1.27)                                                                  |
| 2    | [HIT at SemEval-2022 Task 2: Pre-trained Language Model for Idioms Detection](https://aclanthology.org/2022.semeval-1.28)                                                                                   |
| 3    | [SemEval-2022 Task 8: Multilingual news article similarity](https://aclanthology.org/2022.semeval-1.155)                                                                                                    |
| 3    | [EMBEDDIA at SemEval-2022 Task 8: Investigating Sentence, Image, and Knowledge Graph Representations for Multilingual News Article Similarity](https://aclanthology.org/2022.semeval-1.156)                 |
| 3    | [HFL at SemEval-2022 Task 8: A Linguistics-inspired Regression Model with Data Augmentation for Multilingual News Similarity](https://aclanthology.org/2022.semeval-1.157)                                  |
| 3    | [GateNLP-UShef at SemEval-2022 Task 8: Entity-Enriched Siamese Transformer for Multilingual News Article Similarity](https://aclanthology.org/2022.semeval-1.158)                                           |
| 3    | [SemEval-2022 Task 8: Multi-lingual News Article Similarity](https://aclanthology.org/2022.semeval-1.159)                                                                                                   |
| 3    | [SkoltechNLP at SemEval-2022 Task 8: Multilingual News Article Similarity via Exploration of News Texts to Vector Representations](https://aclanthology.org/2022.semeval-1.160)                             |
| 3    | [IIIT-MLNS at SemEval-2022 Task 8: Siamese Architecture for Modeling Multilingual News Similarity](https://aclanthology.org/2022.semeval-1.161)                                                             |
| 3    | [HuaAMS at SemEval-2022 Task 8: Combining Translation and Domain Pre-training for Cross-lingual News Article Similarity](https://aclanthology.org/2022.semeval-1.162)                                       |
| 3    | [DartmouthCS at SemEval-2022 Task 8: Predicting Multilingual News Article Similarity with Meta-Information and Translation](https://aclanthology.org/2022.semeval-1.163)                                    |
| 3    | [Team Innovators at SemEval-2022 for Task 8: Multi-Task Training with Hyperpartisan and Semantic Relation for Multi-Lingual News Article Similarity](https://aclanthology.org/2022.semeval-1.164)           |
| 3    | [OversampledML at SemEval-2022 Task 8: When multilingual news similarity met Zero-shot approaches](https://aclanthology.org/2022.semeval-1.165)                                                             |
| 3    | [Team TMA at SemEval-2022 Task 8: Lightweight and Language-Agnostic News Similarity Classifier](https://aclanthology.org/2022.semeval-1.166)                                                                |
| 3    | [ITNLP2022 at SemEval-2022 Task 8: Pre-trained Model with Data Augmentation and Voting for Multilingual News Similarity](https://aclanthology.org/2022.semeval-1.167)                                       |
| 3    | [LSX_team5 at SemEval-2022 Task 8: Multilingual News Article Similarity Assessment based on Word- and Sentence Mover’s Distance](https://aclanthology.org/2022.semeval-1.168)                               |
| 3    | [Team dina at SemEval-2022 Task 8: Pre-trained Language Models as Baselines for Semantic Similarity](https://aclanthology.org/2022.semeval-1.169)                                                           |
| 3    | [TCU at SemEval-2022 Task 8: A Stacking Ensemble Transformer Model for Multilingual News Article Similarity](https://aclanthology.org/2022.semeval-1.170)                                                   |
| 3    | [Nikkei at SemEval-2022 Task 8: Exploring BERT-based Bi-Encoder Approach for Pairwise Multilingual News Article Similarity](https://aclanthology.org/2022.semeval-1.171)                                    |
| 3    | [YNU-HPCC at SemEval-2022 Task 8: Transformer-based Ensemble Model for Multilingual News Article Similarity](https://aclanthology.org/2022.semeval-1.172)                                                   |
| 3    | [BL.Research at SemEval-2022 Task 8: Using various Semantic Information to evaluate document-level Semantic Textual Similarity](https://aclanthology.org/2022.semeval-1.173)                                |
| 3    | [DataScience-Polimi at SemEval-2022 Task 8: Stacking Language Models to Predict News Article Similarity](https://aclanthology.org/2022.semeval-1.174)                                                       |
| 3    | [WueDevils at SemEval-2022 Task 8: Multilingual News Article Similarity via Pair-Wise Sentence Similarity Matrices](https://aclanthology.org/2022.semeval-1.175)                                            |
| 11   | [SemEval-2022 Task 11: Multilingual Complex Named Entity Recognition (MultiCoNER)](https://aclanthology.org/2022.semeval-1.196)                                                                             |
| 11   | [LMN at SemEval-2022 Task 11: A Transformer-based System for English Named Entity Recognition](https://aclanthology.org/2022.semeval-1.197)                                                                 |
| 11   | [PA Ph&Tech at SemEval-2022 Task 11: NER Task with Ensemble Embedding from Reinforcement Learning](https://aclanthology.org/2022.semeval-1.198)                                                             |
| 11   | [UC3M-PUCPR at SemEval-2022 Task 11: An Ensemble Method of Transformer-based Models for Complex Named Entity Recognition](https://aclanthology.org/2022.semeval-1.199)                                      |
| 11   | [DAMO-NLP at SemEval-2022 Task 11: A Knowledge-based System for Multilingual Named Entity Recognition](https://aclanthology.org/2022.semeval-1.200)                                                         |
| 11   | [Multilinguals at SemEval-2022 Task 11: Complex NER in Semantically Ambiguous Settings for Low Resource Languages](https://aclanthology.org/2022.semeval-1.201)                                             |
| 11   | [AaltoNLP at SemEval-2022 Task 11: Ensembling Task-adaptive Pretrained Transformers for Multilingual Complex NER](https://aclanthology.org/2022.semeval-1.202)                                              |
| 11   | [DANGNT-SGU at SemEval-2022 Task 11: Using Pre-trained Language Model for Complex Named Entity Recognition](https://aclanthology.org/2022.semeval-1.203)                                                    |
| 11   | [OPDAI at SemEval-2022 Task 11: A hybrid approach for Chinese NER using outside Wikipedia knowledge](https://aclanthology.org/2022.semeval-1.204)                                                           |
| 11   | [Sliced at SemEval-2022 Task 11: Bigger, Better? Massively Multilingual LMs for Multilingual Complex NER on an Academic GPU Budget](https://aclanthology.org/2022.semeval-1.205)                            |
| 11   | [Infrrd.ai at SemEval-2022 Task 11: A system for named entity recognition using data augmentation, transformer-based sequence labeling model, and EnsembleCRF](https://aclanthology.org/2022.semeval-1.205) | 
| 11   | [UM6P-CS at SemEval-2022 Task 11: Enhancing Multilingual and Code-Mixed Complex Named Entity Recognition via Pseudo Labels using Multilingual Transformer](https://aclanthology.org/2022.semeval-1.207)     |
| 11   | [CASIA at SemEval-2022 Task 11: Chinese Named Entity Recognition for Complex and Ambiguous Entities](https://aclanthology.org/2022.semeval-1.208)                                                           |
| 11   | [TEAM-Atreides at SemEval-2022 Task 11: On leveraging data augmentation and ensemble to recognize complex Named Entities in Bangla](https://aclanthology.org/2022.semeval-1.209)                            |
| 11   | [KDDIE at SemEval-2022 Task 11: Using DeBERTa for Named Entity Recognition](https://aclanthology.org/2022.semeval-1.210)                                                                                    |
| 11   | [silpa_nlp at SemEval-2022 Tasks 11: Transformer based NER models for Hindi and Bangla languages](https://aclanthology.org/2022.semeval-1.211)                                                              |
| 11   | [DS4DH at SemEval-2022 Task 11: Multilingual Named Entity Recognition Using an Ensemble of Transformer-based Language Models](https://aclanthology.org/2022.semeval-1.212)                                  |
| 11   | [CSECU-DSG at SemEval-2022 Task 11: Identifying the Multilingual Complex Named Entity in Text Using Stacked Embeddings and Transformer based Approach](https://aclanthology.org/2022.semeval-1.213)         |
| 11   | [CMNEROne at SemEval-2022 Task 11: Code-Mixed Named Entity Recognition by leveraging multilingual data](https://aclanthology.org/2022.semeval-1.214)                                                        |
| 11   | [RACAI at SemEval-2022 Task 11: Complex named entity recognition using a lateral inhibition mechanism](https://aclanthology.org/2022.semeval-1.215)                                                         |
| 11   | [NamedEntityRangers at SemEval-2022 Task 11: Transformer-based Approaches for Multilingual Complex Named Entity Recognition](https://aclanthology.org/2022.semeval-1.216)                                   |
| 11   | [Raccoons at SemEval-2022 Task 11: Leveraging Concatenated Word Embeddings for Named Entity Recognition](https://aclanthology.org/2022.semeval-1.217)                                                       |
| 11   | [SeqL at SemEval-2022 Task 11: An Ensemble of Transformer Based Models for Complex Named Entity Recognition Task](https://aclanthology.org/2022.semeval-1.218)                                              |
| 11   | [SFE-AI at SemEval-2022 Task 11: Low-Resource Named Entity Recognition using Large Pre-trained Language Models](https://aclanthology.org/2022.semeval-1.219)                                                |
| 11   | [NCUEE-NLP at SemEval-2022 Task 11: Chinese Named Entity Recognition Using the BERT-BiLSTM-CRF Model](https://aclanthology.org/2022.semeval-1.220)                                                          |
| 11   | [CMB AI Lab at SemEval-2022 Task 11: A Two-Stage Approach for Complex Named Entity Recognition via Span Boundary Detection and Span Classification](https://aclanthology.org/2022.semeval-1.221)            |
| 11   | [UA-KO at SemEval-2022 Task 11: Data Augmentation and Ensembles for Korean Named Entity Recognition](https://aclanthology.org/2022.semeval-1.222)                                                           |
| 11   | [USTC-NELSLIP at SemEval-2022 Task 11: Gazetteer-Adapted Integration Network for Multilingual Complex Named Entity Recognition](https://aclanthology.org/2022.semeval-1.223)                                |
| 11   | [Multilinguals at SemEval-2022 Task 11: Transformer Based Architecture for Complex NER](https://aclanthology.org/2022.semeval-1.224)                                                                        |
| 11   | [L3i at SemEval-2022 Task 11: Straightforward Additional Context for Multilingual Named Entity Recognition](https://aclanthology.org/2022.semeval-1.225)                                                    |
| 11   | [MarSan at SemEval-2022 Task 11: Multilingual complex named entity recognition using T5 and transformer encoder](https://aclanthology.org/2022.semeval-1.226)                                               |
| 11   | [SU-NLP at SemEval-2022 Task 11: Complex Named Entity Recognition with Entity Linking](https://aclanthology.org/2022.semeval-1.227)                                                                         |
| 11   | [Qtrade AI at SemEval-2022 Task 11: An Unified Framework for Multilingual NER Task](https://aclanthology.org/2022.semeval-1.228)                                                                            |
| 11   | [PAI at SemEval-2022 Task 11: Name Entity Recognition with Contextualized Entity Representations and Robust Loss Functions](https://aclanthology.org/2022.semeval-1.229)                                    |

## SemEval 2022 Multilingual Tasks
https://semeval.github.io/SemEval2022

| Type                                    | Task                                                                                                                              |
| ----------------------------------------|---------------------------------------------------------------------------------------------------------------------------------- |
| Lexical semantics                       | [Task 2: Multilingual Idiomaticity Detection and Sentence Embedding](https://sites.google.com/view/semeval2022task2-idiomaticity) |
| Discourse, documents, and multimodality | [Task 8: Multilingual news article similarity](http://euagendas.org/semeval2022)                                                  |
| Information extraction                  | [Task 11: MultiCoNER - Multilingual Complex Named Entity Recognition](https://multiconer.github.io/)                              |

## SemEval 2021 Multilingual Tasks
https://semeval.github.io/SemEval2021

| Type                                    | Task                                                                                                                         |
| ----------------------------------------|----------------------------------------------------------------------------------------------------------------------------- |
| Lexical semantics                       | [Task 2: Multilingual and Cross-lingual Word-in-Context Disambiguation](https://competitions.codalab.org/competitions/27054) |
| Discourse, documents, and multimodality | Task 3: Span- and Dependency-based Multilingual and Cross-lingual Semantic Role Labeling (no entries, link not available)    |

## SemEval 2020 Multilingual Tasks
https://alt.qcri.org/semeval2020

| Type                                    | Task                                                                                                                                         |
| ----------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------- |
| Lexical semantics                       | [Task 2: Predicting Multilingual and Cross-Lingual (Graded) Lexical Entailment](https://competitions.codalab.org/competitions/20865)         |
| Societal Applications of NLP            | [Task 12: OffensEval 2: Multilingual Offensive Language Identification in Social Media](https://sites.google.com/site/offensevalsharedtask/) |