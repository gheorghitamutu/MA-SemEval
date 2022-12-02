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

## SemEval 2022 Multilingual Tasks
https://semeval.github.io/SemEval2022

| Type                                    | Task                                                                                                                              |
| ----------------------------------------|---------------------------------------------------------------------------------------------------------------------------------- |
| Lexical semantics                       | [Task 2: Multilingual Idiomaticity Detection and Sentence Embedding](https://sites.google.com/view/semeval2022task2-idiomaticity) |
| Discourse, documents, and multimodality | [Task 8: Multilingual news article similarity](http://euagendas.org/semeval2022)                                                  |
| Information extraction                  | [Task 11: MultiCoNER - Multilingual Complex Named Entity Recognition](https://multiconer.github.io/)                              |


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

## SemEval 2021 Multilingual Tasks
https://semeval.github.io/SemEval2021

| Type                                    | Task                                                                                                                         |
| ----------------------------------------|----------------------------------------------------------------------------------------------------------------------------- |
| Lexical semantics                       | [Task 2: Multilingual and Cross-lingual Word-in-Context Disambiguation](https://competitions.codalab.org/competitions/27054) |
| Discourse, documents, and multimodality | Task 3: Span- and Dependency-based Multilingual and Cross-lingual Semantic Role Labeling (no entries, link not available)    |


| Task | Papers                                                                                                                                                                                                              |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2    | [SemEval-2021 Task 2: Multilingual and Cross-lingual Word-in-Context Disambiguation (MCL-WiC)](https://aclanthology.org/2021.semeval-1.3  )                                                                         |
| 2    | [Uppsala NLP at SemEval-2021 Task 2: Multilingual Language Models for Fine-tuning and Feature Extraction in Word-in-Context Disambiguation](https://aclanthology.org/2021.semeval-1.15 )                            |
| 2    | [SkoltechNLP at SemEval-2021 Task 2: Generating Cross-Lingual Training Data for the Word-in-Context Task](https://aclanthology.org/2021.semeval-1.16 )                                                              |
| 2    | [Zhestyatsky at SemEval-2021 Task 2: ReLU over Cosine Similarity for BERT Fine-tuning](https://aclanthology.org/2021.semeval-1.17 )                                                                                 |
| 2    | [SzegedAI at SemEval-2021 Task 2: Zero-shot Approach for Multilingual and Cross-lingual Word-in-Context Disambiguation](https://aclanthology.org/2021.semeval-1.18 )                                                |
| 2    | [MCL@IITK at SemEval-2021 Task 2: Multilingual and Cross-lingual Word-in-Context Disambiguation using Augmented Data, Signals, and Transformers](https://aclanthology.org/2021.semeval-1.62 )                       |
| 2    | [GX at SemEval-2021 Task 2: BERT with Lemma Information for MCL-WiC Task](https://aclanthology.org/2021.semeval-1.92 )                                                                                              |
| 2    | [PALI at SemEval-2021 Task 2: Fine-Tune XLM-RoBERTa for Word in Context Disambiguation](https://aclanthology.org/2021.semeval-1.93 )                                                                                |
| 2    | [hub at SemEval-2021 Task 2: Word Meaning Similarity Prediction Model Based on RoBERTa and Word Frequency](https://aclanthology.org/2021.semeval-1.94 )                                                             |
| 2    | [Lotus at SemEval-2021 Task 2: Combination of BERT and Paraphrasing for English Word Sense Disambiguation](https://aclanthology.org/2021.semeval-1.95 )                                                             |
| 2    | [Cambridge at SemEval-2021 Task 2: Neural WiC-Model with Data Augmentation and Exploration of Representation](https://aclanthology.org/2021.semeval-1.96 )                                                          |
| 2    | [UoB_UK at SemEval 2021 Task 2: Zero-Shot and Few-Shot Learning for Multi-lingual and Cross-lingual Word Sense Disambiguation.](https://aclanthology.org/2021.semeval-1.97 )                                        |
| 2    | [PAW at SemEval-2021 Task 2: Multilingual and Cross-lingual Word-in-Context Disambiguation : Exploring Cross Lingual Transfer, Augmentations and Adversarial Training](https://aclanthology.org/2021.semeval-1.98 ) |
| 2    | [LU-BZU at SemEval-2021 Task 2: Word2Vec and Lemma2Vec performance in Arabic Word-in-Context disambiguation](https://aclanthology.org/2021.semeval-1.99 )                                                           |
| 2    | [GlossReader at SemEval-2021 Task 2: Reading Definitions Improves Contextualized Word Embeddings](https://aclanthology.org/2021.semeval-1.100)                                                                      |
| 2    | [UAlberta at SemEval-2021 Task 2: Determining Sense Synonymy via Translations](https://aclanthology.org/2021.semeval-1.101)                                                                                         |
| 2    | [TransWiC at SemEval-2021 Task 2: Transformer-based Multilingual and Cross-lingual Word-in-Context Disambiguation](https://aclanthology.org/2021.semeval-1.102)                                                     |
| 2    | [LIORI at SemEval-2021 Task 2: Span Prediction and Binary Classification approaches to Word-in-Context Disambiguation](https://aclanthology.org/2021.semeval-1.103)                                                 |
| 2    | [FII_CROSS at SemEval-2021 Task 2: Multilingual and Cross-lingual Word-in-Context Disambiguation](https://aclanthology.org/2021.semeval-1.104)                                                                      |
| 3    | Task 3: Span- and Dependency-based Multilingual and Cross-lingual Semantic Role Labeling (no entries, link not available)                                                                                           |

## SemEval 2020 Multilingual Tasks
https://alt.qcri.org/semeval2020

| Type                                    | Task                                                                                                                                        |
| ----------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------- |
| Lexical semantics                       | [Task 2: Predicting Multilingual and Cross-Lingual (Graded) Lexical Entailment](https://competitions.codalab.org/competitions/20865)        |
| Societal Applications of NLP            | [Task 12: OffensEval 2: Multilingual Offensive Language Identification in Social Media](https://sites.google.com/site/offensevalsharedtask) |


| Task | Papers                                                                                                                                                                                                              |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2    | [SemEval-2020 Task 2: Predicting Multilingual and Cross-Lingual (Graded) Lexical Entailment](https://aclanthology.org/2020.semeval-1.2 )                                                                            |
| 2    | [BMEAUT at SemEval-2020 Task 2: Lexical Entailment with Semantic Graphs](https://aclanthology.org/2020.semeval-1.15)                                                                                                |
| 2    | [SHIKEBLCU at SemEval-2020 Task 2: An External Knowledge-enhanced Matrix for Multilingual and Cross-Lingual Lexical Entailment](https://aclanthology.org/2020.semeval-1.31)                                         |
| 2    | [UAlberta at SemEval-2020 Task 2: Using Translations to Predict Cross-Lingual Entailment](https://aclanthology.org/2020.semeval-1.32)                                                                               |
| 12   | [SemEval-2020 Task 12: Multilingual Offensive Language Identification in Social Media (OffensEval 2020)](https://aclanthology.org/2020.semeval-1.188)                                                               |
| 12   | [Galileo at SemEval-2020 Task 12: Multi-lingual Learning for Offensive Language Identification Using Pre-trained Language Models](https://aclanthology.org/2020.semeval-1.189)                                      |
| 12   | [AdelaideCyC at SemEval-2020 Task 12: Ensemble of Classifiers for Offensive Language Detection in Social Media](https://aclanthology.org/2020.semeval-1.198)                                                        |
| 12   | [ANDES at SemEval-2020 Task 12: A Jointly-trained BERT Multilingual Model for Offensive Language Detection](https://aclanthology.org/2020.semeval-1.199)                                                            |
| 12   | [BhamNLP at SemEval-2020 Task 12: An Ensemble of Different Word Embeddings and Emotion Transfer Learning for Arabic Offensive Language Identification in Social Media](https://aclanthology.org/2020.semeval-1.200) |
| 12   | [FBK-DH at SemEval-2020 Task 12: Using Multi-channel BERT for Multilingual Offensive Language Detection](https://aclanthology.org/2020.semeval-1.201)                                                               |
| 12   | [GruPaTo at SemEval-2020 Task 12: Retraining mBERT on Social Media and Fine-tuned Offensive Language Models](https://aclanthology.org/2020.semeval-1.202)                                                           |
| 12   | [GUIR at SemEval-2020 Task 12: Domain-Tuned Contextualized Models for Offensive Language Detection](https://aclanthology.org/2020.semeval-1.203)                                                                    |
| 12   | [IIITG-ADBU at SemEval-2020 Task 12: Comparison of BERT and BiLSTM in Detecting Offensive Language](https://aclanthology.org/2020.semeval-1.204)                                                                    |
| 12   | [LT@Helsinki at SemEval-2020 Task 12: Multilingual or Language-specific BERT?](https://aclanthology.org/2020.semeval-1.205)                                                                                         |
| 12   | [NLPDove at SemEval-2020 Task 12: Improving Offensive Language Detection with Cross-lingual Transfer](https://aclanthology.org/2020.semeval-1.206)                                                                  |
| 12   | [Nova-Wang at SemEval-2020 Task 12: OffensEmblert: An Ensemble ofOffensive Language Classifiers](https://aclanthology.org/2020.semeval-1.207)                                                                       |
| 12   | [NUIG at SemEval-2020 Task 12: Pseudo Labelling for Offensive Content Classification](https://aclanthology.org/2020.semeval-1.208)                                                                                  |
| 12   | [PRHLT-UPV at SemEval-2020 Task 12: BERT for Multilingual Offensive Language Detection](https://aclanthology.org/2020.semeval-1.209)                                                                                |
| 12   | [PUM at SemEval-2020 Task 12: Aggregation of Transformer-based Models’ Features for Offensive Language Recognition](https://aclanthology.org/2020.semeval-1.210)                                                    |
| 12   | [SINAI at SemEval-2020 Task 12: Offensive Language Identification Exploring Transfer Learning Models](https://aclanthology.org/2020.semeval-1.211)                                                                  |
| 12   | [Team Oulu at SemEval-2020 Task 12: Multilingual Identification of Offensive Language, Type and Target of Twitter Post Using Translated Datasets](https://aclanthology.org/2020.semeval-1.212)                      |
| 12   | [UHH-LT at SemEval-2020 Task 12: Fine-Tuning of Pre-Trained Transformer Networks for Offensive Language Detection](https://aclanthology.org/2020.semeval-1.213)                                                     |
| 12   | [AlexU-BackTranslation-TL at SemEval-2020 Task 12: Improving Offensive Language Detection Using Data Augmentation and Transfer Learning](https://aclanthology.org/2020.semeval-1.248)                               |
| 12   | [ALT at SemEval-2020 Task 12: Arabic and English Offensive Language Identification in Social Media](https://aclanthology.org/2020.semeval-1.249)                                                                    |
| 12   | [Amsqr at SemEval-2020 Task 12: Offensive Language Detection Using Neural Networks and Anti-adversarial Features](https://aclanthology.org/2020.semeval-1.250)                                                      |
| 12   | [BRUMS at SemEval-2020 Task 12: Transformer Based Multilingual Offensive Language Identification in Social Media](https://aclanthology.org/2020.semeval-1.251)                                                      |
| 12   | [CoLi at UdS at SemEval-2020 Task 12: Offensive Tweet Detection with Ensembling](https://aclanthology.org/2020.semeval-1.252)                                                                                       |
| 12   | [CyberTronics at SemEval-2020 Task 12: Multilingual Offensive Language Identification over Social Media](https://aclanthology.org/2020.semeval-1.253)                                                               |
| 12   | [DoTheMath at SemEval-2020 Task 12 : Deep Neural Networks with Self Attention for Arabic Offensive Language Detection](https://aclanthology.org/2020.semeval-1.254)                                                 |
| 12   | [Duluth at SemEval-2020 Task 12: Offensive Tweet Identification in English with Logistic Regression](https://aclanthology.org/2020.semeval-1.255)                                                                   |
| 12   | [Ferryman at SemEval-2020 Task 12: BERT-Based Model with Advanced Improvement Methods for Multilingual Offensive Language Identification](https://aclanthology.org/2020.semeval-1.256)                              |
| 12   | [Garain at SemEval-2020 Task 12: Sequence Based Deep Learning for Categorizing Offensive Language in Social Media](https://aclanthology.org/2020.semeval-1.257)                                                     |
| 12   | [Hitachi at SemEval-2020 Task 12: Offensive Language Identification with Noisy Labels Using Statistical Sampling and Post-Processing](https://aclanthology.org/2020.semeval-1.258)                                  |
| 12   | [I2C at SemEval-2020 Task 12: Simple but Effective Approaches to Offensive Speech Detection in Twitter](https://aclanthology.org/2020.semeval-1.259)                                                                |
| 12   | [iCompass at SemEval-2020 Task 12: From a Syntax-ignorant N-gram Embeddings Model to a Deep Bidirectional Language Model](https://aclanthology.org/2020.semeval-1.260)                                              |
| 12   | [IITP-AINLPML at SemEval-2020 Task 12: Offensive Tweet Identification and Target Categorization in a Multitask Environment](https://aclanthology.org/2020.semeval-1.261)                                            |
| 12   | [INGEOTEC at SemEval-2020 Task 12: Multilingual Classification of Offensive Text](https://aclanthology.org/2020.semeval-1.262)                                                                                      |
| 12   | [IR3218-UI at SemEval-2020 Task 12: Emoji Effects on Offensive Language IdentifiCation](https://aclanthology.org/2020.semeval-1.263)                                                                                |
| 12   | [IRLab_DAIICT at SemEval-2020 Task 12: Machine Learning and Deep Learning Methods for Offensive Language Identification](https://aclanthology.org/2020.semeval-1.264)                                               |
| 12   | [IRlab@IITV at SemEval-2020 Task 12: Multilingual Offensive Language Identification in Social Media Using SVM](https://aclanthology.org/2020.semeval-1.265)                                                         |
| 12   | [JCT at SemEval-2020 Task 12: Offensive Language Detection in Tweets Using Preprocessing Methods, Character and Word N-grams](https://aclanthology.org/2020.semeval-1.266)                                          |
| 12   | [KAFK at SemEval-2020 Task 12: Checkpoint Ensemble of Transformers for Hate Speech Classification](https://aclanthology.org/2020.semeval-1.267)                                                                     |
| 12   | [KDELAB at SemEval-2020 Task 12: A System for Estimating Aggression of Tweets Using Two Layers of BERT Features](https://aclanthology.org/2020.semeval-1.268)                                                       |
| 12   | [KEIS@JUST at SemEval-2020 Task 12: Identifying Multilingual Offensive Tweets Using Weighted Ensemble and Fine-Tuned BERT](https://aclanthology.org/2020.semeval-1.270)                                             |
| 12   | [KS@LTH at SemEval-2020 Task 12: Fine-tuning Multi- and Monolingual Transformer Models for Offensive Language Detection](https://aclanthology.org/2020.semeval-1.270)                                               |
| 12   | [KUISAIL at SemEval-2020 Task 12: BERT-CNN for Offensive Speech Identification in Social Media](https://aclanthology.org/2020.semeval-1.271)                                                                        |
| 12   | [Kungfupanda at SemEval-2020 Task 12: BERT-Based Multi-TaskLearning for Offensive Language Detection](https://aclanthology.org/2020.semeval-1.272)                                                                  |
| 12   | [Lee at SemEval-2020 Task 12: A BERT Model Based on the Maximum Self-ensemble Strategy for Identifying Offensive Language](https://aclanthology.org/2020.semeval-1.273)                                             |
| 12   | [LIIR at SemEval-2020 Task 12: A Cross-Lingual Augmentation Approach for Multilingual Offensive Language Identification](https://aclanthology.org/2020.semeval-1.274)                                               |
| 12   | [LISAC FSDM-USMBA Team at SemEval-2020 Task 12: Overcoming AraBERT’s pretrain-finetune discrepancy for Arabic offensive language identification](https://aclanthology.org/2020.semeval-1.275)                       |
| 12   | [NAYEL at SemEval-2020 Task 12: TF/IDF-Based Approach for Automatic Offensive Language Detection in Arabic Tweets](https://aclanthology.org/2020.semeval-1.276)                                                     |
| 12   | [NLP_Passau at SemEval-2020 Task 12: Multilingual Neural Network for Offensive Language Detection in English, Danish and Turkish](https://aclanthology.org/2020.semeval-1.277)                                      |
| 12   | [nlpUP at SemEval-2020 Task 12 : A Blazing Fast System for Offensive Language Detection](https://aclanthology.org/2020.semeval-1.278)                                                                               |
| 12   | [NTU_NLP at SemEval-2020 Task 12: Identifying Offensive Tweets Using Hierarchical Multi-Task Learning Approach](https://aclanthology.org/2020.semeval-1.279)                                                        |
| 12   | [PGSG at SemEval-2020 Task 12: BERT-LSTM with Tweets’ Pretrained Model and Noisy Student Training Method](https://aclanthology.org/2020.semeval-1.280)                                                              |
| 12   | [Pin_cod_ at SemEval-2020 Task 12: Injecting Lexicons into Bidirectional Long Short-Term Memory Networks to Detect Turkish Offensive Tweets](https://aclanthology.org/2020.semeval-1.281)                           |
| 12   | [problemConquero at SemEval-2020 Task 12: Transformer and Soft Label-based Approaches](https://aclanthology.org/2020.semeval-1.282)                                                                                 |
| 12   | [SalamNET at SemEval-2020 Task 12: Deep Learning Approach for Arabic Offensive Language Detection](https://aclanthology.org/2020.semeval-1.283)                                                                     |
| 12   | [Smatgrisene at SemEval-2020 Task 12: Offense Detection by AI - with a Pinch of Real I](https://aclanthology.org/2020.semeval-1.284)                                                                                |
| 12   | [Sonal.kumari at SemEval-2020 Task 12: Social Media Multilingual Offensive Text Identification and Categorization Using Neural Network Models](https://aclanthology.org/2020.semeval-1.285)                         |
| 12   | [Ssn_nlp at SemEval 2020 Task 12: Offense Target Identification in Social Media Using Traditional and Deep Machine Learning Approaches](https://aclanthology.org/2020.semeval-1.286)                                |
| 12   | [SSN_NLP_MLRG at SemEval-2020 Task 12: Offensive Language Identification in English, Danish, Greek Using BERT and Machine Learning Approach](https://aclanthology.org/2020.semeval-1.287)                           |
| 12   | [SU-NLP at SemEval-2020 Task 12: Offensive Language IdentifiCation in Turkish Tweets](https://aclanthology.org/2020.semeval-1.288)                                                                                  |
| 12   | [TAC at SemEval-2020 Task 12: Ensembling Approach for Multilingual Offensive Language Identification in Social Media](https://aclanthology.org/2020.semeval-1.289)                                                  |
| 12   | [Team Rouges at SemEval-2020 Task 12: Cross-lingual Inductive Transfer to Detect Offensive Language](https://aclanthology.org/2020.semeval-1.290)                                                                   |
| 12   | [TECHSSN at SemEval-2020 Task 12: Offensive Language Detection Using BERT Embeddings](https://aclanthology.org/2020.semeval-1.291)                                                                                  |
| 12   | [TheNorth at SemEval-2020 Task 12: Hate Speech Detection Using RoBERTa](https://aclanthology.org/2020.semeval-1.292)                                                                                                |
| 12   | [UJNLP at SemEval-2020 Task 12: Detecting Offensive Language Using Bidirectional Transformers](https://aclanthology.org/2020.semeval-1.293)                                                                         |
| 12   | [UNT Linguistics at SemEval-2020 Task 12: Linear SVC with Pre-trained Word Embeddings as Document Vectors and Targeted Linguistic Features](https://aclanthology.org/2020.semeval-1.294)                            |
| 12   | [UoB at SemEval-2020 Task 12: Boosting BERT with Corpus Level Information](https://aclanthology.org/2020.semeval-1.295)                                                                                             |
| 12   | [UPB at SemEval-2020 Task 12: Multilingual Offensive Language Detection on Social Media by Fine-tuning a Variety of BERT-based Models](https://aclanthology.org/2020.semeval-1.296)                                 |
| 12   | [UTFPR at SemEval 2020 Task 12: Identifying Offensive Tweets with Lightweight Ensembles](https://aclanthology.org/2020.semeval-1.297)                                                                               |
| 12   | [WOLI at SemEval-2020 Task 12: Arabic Offensive Language Identification on Different Twitter Datasets](https://aclanthology.org/2020.semeval-1.298)                                                                 |
| 12   | [XD at SemEval-2020 Task 12: Ensemble Approach to Offensive Language Identification in Social Media Using Transformer Encoders](https://aclanthology.org/2020.semeval-1.299)                                        |
| 12   | [YNU_oxz at SemEval-2020 Task 12: Bidirectional GRU with Capsule for Identifying Multilingual Offensive Language](https://aclanthology.org/2020.semeval-1.300)                                                      |