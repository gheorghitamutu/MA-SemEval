# MA-SemEval
Machine Translation Final Project - SemEval

- [MA-SemEval](#ma-semeval)
- [State of Art](#state-of-art)
  - [What is SemEval?](#what-is-semeval)
  - [SemEval 2023 Multilingual Tasks](#semeval-2023-multilingual-tasks)
  - [SemEval 2022 Multilingual Tasks](#semeval-2022-multilingual-tasks)
  - [SemEval 2021 Multilingual Tasks](#semeval-2021-multilingual-tasks)
  - [SemEval 2020 Multilingual Tasks](#semeval-2020-multilingual-tasks)
  - [Previous Work](#previous-work)
    - [FII](#fii)
    - [MultiCoNER](#multiconer)
- [Options from SemEVAL 2023 tasks](#options-from-semeval-2023-tasks)
    - [Task 2: Multilingual Complex Named Entity Recognition (MultiCoNER 2)](#task-2-multilingual-complex-named-entity-recognition-multiconer-2)
    - [Task 3: Detecting the Category, the Framing, and the Persuasion Techniques in Online News in a Multi-lingual Setup](#task-3-detecting-the-category-the-framing-and-the-persuasion-techniques-in-online-news-in-a-multi-lingual-setup)
    - [Task 9: Multilingual Tweet Intimacy Analysis](#task-9-multilingual-tweet-intimacy-analysis)
- [Partial Conclusion before choosing the project](#partial-conclusion-before-choosing-the-project)
- [Top Models for Task 2](#top-models-for-task-2)
- [Task 2 Deadlines](#task-2-deadlines)
- [Frameworks, tools \& data for Task 2 (MultiCoNER 2)](#frameworks-tools--data-for-task-2-multiconer-2)
- [Models compared](#models-compared)
  - [Legend](#legend)
  - [SpaCy Default NER](#spacy-default-ner)
  - [SpaCy BEAM NER](#spacy-beam-ner)
  - [SpaCy CRF NER](#spacy-crf-ner)
  - [SpaCy Pipeline (transformer + NER)](#spacy-pipeline-transformer--ner)
    - [roberta\_base\_en + NER](#roberta_base_en--ner)
    - [xlm\_roberta\_base\_en + NER](#xlm_roberta_base_en--ner)
    - [distilbert-base-uncased-finetuned-sst-2-english + NER](#distilbert-base-uncased-finetuned-sst-2-english--ner)
    - [roberta\_base\_en + BEAM NER](#roberta_base_en--beam-ner)
    - [distilbert-base-uncased-finetuned-sst-2-english + BEAM NER](#distilbert-base-uncased-finetuned-sst-2-english--beam-ner)
    - [Model comparison](#model-comparison)
- [Bibliography](#bibliography)
  - [analyticsvidhya](#analyticsvidhya)


# State of Art

## What is SemEval?
> SemEval is a series of international natural language processing (NLP) research workshops whose mission is to advance the current state of the art in semantic analysis and to help create high-quality annotated datasets in a range of increasingly challenging problems in natural language semantics. Each year's workshop features a collection of shared tasks in which computational semantic analysis systems designed by different teams are presented and compared.

https://semeval.github.io

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
| Information extraction                  | [Task 11: MultiCoNER - Multilingual Complex Named Entity Recognition](https://multiconer.github.io)                               |


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

 ## Previous Work

 ### FII
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

### MultiCoNER

MultiCoNER purpose:
> Complex named entities (NE), like the titles of creative works, are not simple nouns and pose challenges for NER systems (Ashwini and Choi, 2014).
>  
> They can take the form of any linguistic constituent, like an imperative clause (“Dial M for Murder”), and do not look like traditional NEs (Persons, Locations, etc.). 
> 
> This syntactic ambiguity makes it challenging to recognize them based on context.
> 
MultiCoNER had a task in 2022 at category [Information extraction - Task 11: MultiCoNER - Multilingual Complex Named Entity Recognition](https://multiconer.github.io).

Its findings are presented [here](https://aclanthology.org/2022.semeval-1.196.pdf).

> Divided into 13 tracks, the task focused on methods to identify complex named entities (like media titles, products, and groups) in 11 languages in both monolingual and multi-lingual scenarios.
> 
> Eleven tracks were for building monolingual NER models for individual languages, one track focused on multilingual models able to work on all languages, and the last track featured 
> 
> code-mixed texts within any of these languages.

Subsets:
> Monolingual Subsets: Each of the 11 languages has its own subset, which includes data from all three domains.
> 
> Multilingual Subset: This contains randomly sampled data from all the languages mixed into a single subset. 
> 
> This subset is designed for evaluating multilingual models, and should ideally be used under the assumption that the language for each sentence is unknown.
> 
> Code-mixed Subset: This subset contains codemixed instances, where the entity is from one language and the rest of the text is written in another language. 
> 
> Like the multilingual subset, this subset should also be used under the assumption that the languages present in an instance are unknown.

Participating Systems and Results:
> received submissions from 55 different teams. Among the monolingual tracks, we have observed the highest participation of 30 teams in the English track.
> 
> Ordered by the number of participating teams, the other monolingual tracks are Chinese (21), Bangla (18), Spanish (18), Hindi (17), Korean (17), German (16), Dutch (15), Farsi (15), Turkish (15), and Russian (14). 
> 
> The number of participating teams for the Multilingual and Code-mixed tracks are 25 and 21, respectively.
> 
> Most of the top-performing teams aimed at building their system targeting the multilingual track, and then retrained it for the other tracks separately and made submissions to all the 13 tracks.

Top Multilingual Systems
From the paper, best systems were DAMO-NLP & USTC-NELSLIP followed by QTrade AI.

> DAMO-NLP (Wang et al., 2022) ranked 1st in the multilingual (MULTI) track and all the monolingual tracks except BN (2nd) and ZH (4th). 
> 
> Given a text, they used a knowledge retrieval module to retrieve K most relevant paragraphs from a knowledge base (i.e. Wikipedia). 
> 
> Paragraphs were concatenated together with the input, and token representations were passed through a CRF to predict the labels. 
> 
> They employed multiple such XLMRoBERTa models with random seeds and then used a voting strategy to make the final prediction.
> 

> USTC-NELSLIP (Chen et al., 2022a) ranked 1st in three tracks (MIX, ZH, BN), and 2nd for all the other tracks. 
> 
> The average performance gap between USTC-NELSLIP and DAMO-NLP is ≈3% for all the 13 tracks. 
> 
> USTC-NELSLIP aimed at fine-tuning a Gazetteer enhanced BiLSTM network in such a way that the representation produced for an entity has similarity with the representation produced by a pre-trained language model (LM).
> 
> They developed a two-step process with two parallel networks, where a Gazetteer-BiLSTM uses a Gazetteer search to produce one-hot labels for each token in a given text and a BiLSTM produces 
> 
> a dense vector representation for each token. Another network uses a frozen XLM-RoBERTa to produce an embedding vector for each token. 
> 
> A KL divergence loss is applied to make the Gazetteer network’s output similar to the LM. These two networks are jointly trained together again and their outputs are fused together for the final prediction.

> QTrade AI (Gan et al., 2022) ranked 3rd in MULTI, 4th in MIX, and 8th in ZH. 
> 
> They used an XLM-RoBERTa encoder and applied sample mixing for data augmentation, along with adversarial training through data noising. 
> 
> For the multilingual track, they leveraged an architecture with shared and per-language representations. 
> 
> Finally, they created an ensemble of models trained with different approaches.

The entire list also contains: SeqL & CMB AI Lab. Other noteworthy systems were RACAI, Sliced, MaChAmp, OPDAI, CASIA, PAI, SU-NLP, Infrrd.ai, UM6P-CS, Multilinguals, L3i, MarSan, TEAM-Atreides, UA-KO
CSECU-DSG, PA Ph&Tech, Raccoons, AaltoNLP, LMN, UC3M-PUCPR, NamedEntityRangers, CMNEROne, KDDIE, DS4DH & NCUEE-NLP. All of them are presented in the paper.

Conclusion
> Most of the top-performing teams in MULTICONER utilized external knowledge bases like Wikipedia and Gazetteer. They also tend to use XLM-RoBERTa as the pre-trained language model. 
> 
> In terms of modeling approaches, ensemble strategies helped the systems to achieve strong performance. 
> 
> Results from the top teams indicate that identifying complex entities like creative works is still difficult among all the classes even with the usage of external data.

# Options from SemEVAL 2023 tasks

From the 3 multilingual tasks available for this edition:

### [Task 2: Multilingual Complex Named Entity Recognition (MultiCoNER 2)](http://multiconer.github.io) 
> Complex named entities (NE), like the titles of creative works, are not simple nouns and pose challenges for NER systems (Ashwini and Choi, 2014). 
> 
> They can take the form of any linguistic constituent, like an imperative clause (“Dial M for Murder”), and do not look like traditional NEs (Persons, Locations, etc.). 
> 
> This syntactic ambiguity makes it challenging to recognize them based on context. 
> 
> We organized the MultiCoNER task (Malmasi et al., 2022) at SemEval-2022 to address these challenges in 11 languages, receiving a very positive community response with 34 system papers. 
> 
> Results confirmed the challenges of processing complex and long-tail NEs: even the largest pre-trained Transformers did not achieve top performance without external knowledge. 
> 
> The top systems infused transformers with knowledge bases and gazetteers. 
> 
> However, such solutions are brittle against out of knowledge-base entities and noisy scenarios like the presence of spelling mistakes and typos. 
> 
> We propose MultiCoNER II which represents novel challenges through new tasks that emphasize the shortcomings of the current top models.

There are examples from the past year. It may be the *easiest* to reproduce and then maybe tweaking the systems for better results. We already know what worked and what not.

### [Task 3: Detecting the Category, the Framing, and the Persuasion Techniques in Online News in a Multi-lingual Setup](https://propaganda.math.unipd.it/semeval2023task3)
> Subtask 1: News Genre Categorisation
> 
> Definition: given a news article, determine whether it is an opinion piece, aims at objective news reporting, or is a satire piece. This is a multi-class (single-label) task at article-level.

> Subtask 2: Framing Detection
> 
> Definition: given a news article, identify the frames used in the article. This is a multi-label task at article-level.

> Subtask 3: Persuasion Techniques Detection
> 
> Definition 1: given a news article, identify the persuasion techniques in each paragraph. This is a multi-label task at paragraph level.

From a personal viewpoint, I didn't find much data on SemEval about this type of task. It may require the most research to be done comparing with Task 2 & Task 9.

### [Task 9: Multilingual Tweet Intimacy Analysis](https://semeval.github.io/SemEval2023/tasks#:~:text=Task%209%3A%20Multilingual%20Tweet%20Intimacy%20Analysis) 

> The goal of this task is to predict the intimacy of tweets in 10 languages. 
> 
> You are given a set of tweets in six languages (English, Spanish, Italian, Portuguese, French, and Chinese) annotated with intimacy scores ranging from 1-5 to train your model.
> 
> You are encouraged (but not required) to also use the question intimacy dataset (Pei and Jurgens, 2020) which contains 2247 English questions from Reddit as well as another 150 questions from Books, Movies, and Twitter. 
> 
> Please note that the intimacy scores in this dataset range from -1 to 1 so you might need to consider data augmentation methods or other methods mapping the intimacy scores to the 1-5 range in the current task. 
> 
> Please check out the paper for more details about this question intimacy dataset.
> 
> The model performance will be evaluated on the test set in the given 6 languages as well as an external test set with 4 languages not in the training data (Hindi, Arabic, Dutch and Korean).
> 
> We will use Pearson's r as the evaluation metric. 

No previous multilingual Twitter Intimacy Analysis were given at SemEval but there are several papers quantifying this:
- [Self-disclosure topic model for classifying and analyzing Twitter conversations](https://aclanthology.org/D14-1213.pdf)
- [Quantifying Intimacy in Language](https://aclanthology.org/2020.emnlp-main.428.pdf)

Probably the hardest part is using a model targeting the multilingual viewpoint.

# Partial Conclusion before choosing the project
I'd probably go with [Task 2: Multilingual Complex Named Entity Recognition (MultiCoNER 2)](http://multiconer.github.io) and see what I can get from the best models available.
I assume tweaking the models will come after researching each model involved in top 3 systems from last year. I'll continue with a list and their implementation in Python (packages, usage, etc).

# Top Models for Task 2


[DAMO-NLP at SemEval-2022 Task 11: A Knowledge-based System for Multilingual Named Entity Recognition](https://aclanthology.org/2022.semeval-1.200)
> *knowledge retrieval module to retrieve K most relevant paragraphs* from a knowledge base
> 
> token representations were passed through a *CRF* to predict the labels
> 
> multiple such *XLM-RoBERTa* models with random seeds and then used a voting strategy to make the final prediction

[USTC-NELSLIP at SemEval-2022 Task 11: Gazetteer-Adapted Integration Network for Multilingual Complex Named Entity Recognition](https://aclanthology.org/2022.semeval-1.223)
> fine-tuning a *Gazetteer* enhanced *BiLSTM network*
> 
> a two-step process with two parallel networks, where a Gazetteer-BiLSTM uses a *Gazetteer* search to produce *one-hot labels* for each token in a given text and a *BiLSTM* produces a *dense vector representation* for each token
> 
> uses a *frozen XLM-RoBERTa* to produce an embedding vector for each token
> 
> *KL divergence* loss is applied to make the Gazetteer network’s output similar to the LM
> 
> These two networks are jointly trained together again and *their outputs are fused together* for the final prediction

[Qtrade AI at SemEval-2022 Task 11: An Unified Framework for Multilingual NER Task](https://aclanthology.org/2022.semeval-1.228)
> *XLM-RoBERTa encoder* and applied sample mixing for data augmentation, along with adversarial training through data noising
> 
> leveraged an *architecture with shared and per-language* representations
> 
> created an *ensemble of models* trained with different approaches

# Task 2 Deadlines
| Event                                   | Date               |
| --------------------------------------- | ------------------ |
| Trial Data Ready                        | Jul 15 (Fri), 2022 |
| Training Data Ready                     | Sep 30 (Fri), 2022 |
| Evaluation Start                        | Jan 10 (Tue), 2023 |
| Evaluation End                          | Jan 31 (Tue), 2023 |
| System Description Paper Submission Due | Feb 01 (Wed), 2023 |
| Notification to Authors                 | Mar 01 (Wed), 2023 |
| Camera-ready Due                        | Apr 01 (Sat), 2023 |
| Workshop                                |                TBD |

# Frameworks, tools & data for Task 2 (MultiCoNER 2)
| Tool                                                            | Description                                                              |
| --------------------------------------------------------------- | ------------------------------------------------------------------------ |
| [Python](https://www.python.org)                                | Programming language                                                     |
| [spaCy](https://spacy.io)                                       | Industrial-Strength Natural Language Processing (in Python)              |
| [Codalab](http://codalab.org)                                   | Accelerating reproducible computational research                         |
| [AWS MultiCONER data](https://registry.opendata.aws/multiconer) | A large multilingual dataset (11 languages) for Named Entity Recognition |

# Models compared

## Legend 

| Notation | Name      | Description / Definition                                               |
| -------- | --------- | ---------------------------------------------------------------------- |
| NER P    | Precision | Percentage value indicating how many of those results are correct.     |
| NER R    | Recall    | Percentage value indicating how many of the correct results are found. |
| NER F    | F-score   | Harmonic mean of Precision and Recall values of a system => it answers to the following formula: 2 x [(Precision x Recall) / (Precision + Recall)]. |

## SpaCy Default NER
[SpaCy Default NER](https://spacy.io/api/architectures#parser)

Used 100 iterations in order to train the model on MultiCONER en train dataset.
The dataset was preprocessed to remove comments and transform it into examples in order to feed them to spaCy model.

Results:
F-Score: 0.39276257722859664

## SpaCy BEAM NER

Used 100 iterations in order to train the model on MultiCONER en train dataset.
The dataset was preprocessed to remove comments and transform it into examples in order to feed them to spaCy model.

Results:
F-Score: 0.38953488372093026

## SpaCy CRF NER

Results:

Fine Tuning: C1: 0.08481243523117314 C2: 0.008479814595730259 score: 0.123631650055735

Most likely transitions:

B-Artist   -> I-Artist   13.407984

B-Politician -> I-Politician 13.101479

B-Athlete  -> I-Athlete  12.794293

B-OtherPER -> I-OtherPER 12.693526

B-Scientist -> I-Scientist 12.550463

B-Facility -> I-Facility 12.447485

B-MusicalGRP -> I-MusicalGRP 12.236781

B-SportsManager -> I-SportsManager 12.097365

B-VisualWork -> I-VisualWork 12.080856

B-Cleric   -> I-Cleric   11.828867


Positive features:

13.212218 B-Vehicle  -1:low:streamliner

11.960911 B-MusicalWork -1:low:album

10.204279 B-VisualWork -1:low:film

10.153886 B-VisualWork -1:low:sitcom

9.657351 B-Vehicle  0:prefix2:u-

9.300112 I-HumanSettlement -1:low:bosnia

9.247446 B-Facility 1:low:dwelling

8.806462 B-MusicalGRP -1:low:band

8.337420 B-MusicalWork -1:low:feeling

8.294854 B-Artist   -1:low:rapper

But I was unable to connect it to a pipeline for along with a transformer.


## SpaCy Pipeline (transformer + NER)

### roberta_base_en + NER

============================= Training pipeline =============================

| E |    #     | LOSS TRANS...  | LOSS NER   | ENTS_F  | ENTS_P | ENTS_R | SCORE  |
| - |  ------- | -------------- | ---------- | ------- | ------ | ------ | ------ |
|  0 |       0 |       15587.65 |    895.39  |   0.09  |   0.06  |  0.23  |  0.00 | 
|  1 |     200 |     1175642.22 |  128620.26 |   18.76 |   22.56 |  16.05 |  0.19 | 
|  2 |     400 |      607301.92 |  75229.26  |  21.46  |  27.68  | 17.52  |  0.21 | 
|  4 |     600 |      276440.03 |  63096.11  |  35.95  |  37.71  | 34.34  |  0.36 | 
|  5 |     800 |      215705.56 |  48943.47  |  55.76  |  56.86  | 54.71  |  0.56 | 
|  7 |    1000 |       40475.88 |  32492.90  |  60.27  |  58.39  | 62.27  |  0.60 | 
|  8 |    1200 |       26682.23 |  28128.98  |  65.59  |  63.84  | 67.44  |  0.66 | 
| 10 |    1400 |       24873.72 |  25686.51  |  66.46  |  65.30  | 67.67  |  0.66 | 
| 11 |    1600 |       13519.17 |  20718.79  |  64.41  |  62.85  | 66.05  |  0.64 | 
| 13 |    1800 |       14386.91 |  19809.12  |  68.43  |  66.94  | 69.98  |  0.68 | 
| 14 |    2000 |       10437.40 |  17163.58  |  68.68  |  68.08  | 69.29  |  0.69 | 
| 16 |    2200 |       10704.12 |  15873.18  |  67.52  |  66.27  | 68.83  |  0.68 | 
| 17 |    2400 |       19513.82 |  15068.64  |  68.17  |  66.59  | 69.83  |  0.68 | 
| 18 |    2600 |        9033.66 |  14292.59  |  69.48  |  67.34  | 71.76  |  0.69 | 
| 20 |    2800 |        8927.03 |  13346.65  |  70.53  |  68.36  | 72.84  |  0.71 | 
| 21 |    3000 |       22467.14 |  12312.48  |  69.78  |  67.09  | 72.69  |  0.70 | 

================================== Results ==================================

| TOK   | -     |
| ----- | ----- |
| NER P | 68.36 |
| NER R | 72.84 |
| NER F | 70.53 |
| SPEED | 6429  |


=============================== NER (per type) ===============================

|                       |     P |     R |     F |
| --------------------- | ----- | ----- | ----- |
| OtherPER              | 69.15 | 71.43 | 70.27 |
| PublicCorp            | 55.26 | 75.00 | 63.64 |
| OtherPROD             | 61.90 | 53.06 | 57.14 |
| HumanSettlement       | 87.39 | 88.99 | 88.18 |
| Artist                | 55.24 | 82.08 | 66.03 |
| Scientist             | 50.00 | 60.00 | 54.55 |
| ORG                   | 63.29 | 64.10 | 63.69 |
| Politician            | 57.69 | 56.60 | 57.14 |
| Athlete               | 86.59 | 89.87 | 88.20 |
| Facility              | 75.00 | 75.00 | 75.00 |
| SportsGRP             | 85.00 | 82.93 | 83.95 |
| CarManufacturer       | 75.00 | 69.23 | 72.00 |
| MusicalWork           | 71.43 | 65.57 | 68.38 |
| MusicalGRP            | 82.05 | 86.49 | 84.21 |
| WrittenWork           | 76.00 | 70.37 | 73.08 |
| Cleric                | 80.00 | 53.33 | 64.00 |
| VisualWork            | 77.55 | 62.30 | 69.09 |
| Software              | 75.00 | 80.77 | 77.78 |
| MedicalProcedure      | 64.29 | 69.23 | 66.67 |
| SportsManager         | 66.67 | 87.50 | 75.68 |
| OtherLOC              | 73.33 | 68.75 | 70.97 |
| Medication/Vaccine    | 75.00 | 83.33 | 78.95 |
| Disease               | 34.78 | 44.44 | 39.02 |
| Symptom               | 40.00 | 20.00 | 26.67 |
| Vehicle               | 77.78 | 70.00 | 73.68 |
| Station               | 88.89 | 80.00 | 84.21 |
| Food                  | 72.22 | 68.42 | 70.27 |
| AnatomicalStructure   | 63.16 | 70.59 | 66.67 |
| Clothing              | 62.50 | 50.00 | 55.56 |
| PrivateCorp           |  0.00 |  0.00 |  0.00 |
| Drink                 | 88.89 | 72.73 | 80.00 |
| AerospaceManufacturer | 64.29 | 90.00 | 75.00 |
| ArtWork               | 50.00 | 46.15 | 48.00 |

### xlm_roberta_base_en + NER

| E  | #       | LOSS TRANS... | LOSS NER  | ENTS_F  | ENTS_P  | ENTS_R  | SCORE  | 
|----| ------- | ------------- | --------  | ------- | ------- | ------- | ------ | 
|  0 |       0 |       8980.56 |   855.35  |   0.00  |   0.00  |   0.00  |   0.00 | 
|  1 |     200 |    1611143.81 | 119900.28 |   13.01 |   19.36 |    9.80 |   0.13 |
|  2 |     400 |     361167.96 | 68349.85  |  12.50  |  17.26  |   9.80  |   0.12 | 
|  4 |     600 |     277506.94 | 61741.50  |  25.92  |  32.74  |  21.45  |   0.26 | 
|  5 |     800 |     340300.63 | 60447.22  |  23.71  |  29.66  |  19.75  |   0.24 | 
|  7 |    1000 |     286867.06 | 56465.53  |  42.15  |  45.08  |  39.58  |   0.42 | 
|  8 |    1200 |      89696.04 | 39771.75  |  51.01  |  51.25  |  50.77  |   0.51 | 
| 10 |    1400 |      83717.92 | 32445.34  |  60.25  |  62.74  |  57.95  |   0.60 | 
| 11 |    1600 |      32794.76 | 25870.37  |  60.22  |  62.76  |  57.87  |   0.60 | 
| 13 |    1800 |      24029.22 | 22919.17  |  66.27  |  68.92  |  63.81  |   0.66 | 
| 14 |    2000 |      19856.78 | 20191.92  |  66.64  |  68.66  |  64.74  |   0.67 | 
| 16 |    2200 |      14429.42 | 17756.24  |  65.77  |  67.59  |  64.04  |   0.66 | 
| 17 |    2400 |      12017.81 | 15216.74  |  66.33  |  67.25  |  65.43  |   0.66 | 
| 18 |    2600 |      12936.71 | 13860.87  |  67.20  |  66.97  |  67.44  |   0.67 | 
| 20 |    2800 |       8309.24 | 11845.72  |  68.88  |  69.72  |  68.06  |   0.69 | 
| 21 |    3000 |       8183.95 | 11458.74  |  71.58  |  72.03  |  71.14  |   0.72 | 
| 23 |    3200 |       6593.13 |  9795.43  |  70.23  |  69.93  |  70.52  |   0.70 | 
| 24 |    3400 |       7169.88 |  9480.61  |  68.98  |  69.77  |  68.21  |   0.69 | 
| 26 |    3600 |       7082.02 |  9704.53  |  70.90  |  72.09  |  69.75  |   0.71 | 
| 27 |    3800 |       5662.96 |  8227.89  |  70.38  |  71.01  |  69.75  |   0.70 | 
| 29 |    4000 |       5273.54 |  7422.72  |  71.45  |  72.24  |  70.68  |   0.71 | 
| 30 |    4200 |       5134.92 |  6937.31  |  69.47  |  68.31  |  70.68  |   0.69 | 
| 32 |    4400 |       4149.54 |  5970.14  |  72.53  |  72.84  |  72.22  |   0.73 | 
| 33 |    4600 |       4185.03 |  5892.15  |  71.42  |  71.78  |  71.06  |   0.71 | 
| 35 |    4800 |       3761.30 |  5523.76  |  70.35  |  71.44  |  69.29  |   0.70 | 
| 36 |    5000 |       3627.16 |  5172.78  |  69.74  |  70.68  |  68.83  |   0.70 | 
| 37 |    5200 |       3675.49 |  5154.36  |  71.76  |  72.97  |  70.60  |   0.72 | 
| 39 |    5400 |       3308.64 |  4645.49  |  70.80  |  71.95  |  69.68  |   0.71 | 
| 40 |    5600 |       3169.18 |  4566.04  |  71.83  |  73.94  |  69.83  |   0.72 | 
| 42 |    5800 |       3071.08 |  4388.44  |  70.78  |  70.27  |  71.30  |   0.71 | 
| 43 |    6000 |       2916.21 |  3959.14  |  70.95  |  72.02  |  69.91  |   0.71 | 

================================== Results ==================================

| TOK   | -     |
| ----- | ----- |
| NER P | 72.84 |
| NER R | 72.22 |
| NER F | 72.53 |
| SPEED | 2471  |


=============================== NER (per type) ===============================

|                       |      P |     R |     F |
| --------------------- | ------ | ----- | ----- |
| OtherPER              |  63.06 | 76.92 | 69.31 |
| HumanSettlement       |  87.96 | 87.16 | 87.56 |
| Scientist             |  61.90 | 86.67 | 72.22 |
| OtherPROD             |  54.76 | 46.94 | 50.55 |
| PublicCorp            |  77.78 | 50.00 | 60.87 |
| Artist                |  79.28 | 83.02 | 81.11 |
| Politician            |  85.29 | 54.72 | 66.67 |
| ORG                   |  67.11 | 65.38 | 66.23 |
| Athlete               |  86.42 | 88.61 | 87.50 |
| Facility              |  86.67 | 75.00 | 80.41 |
| SportsGRP             |  83.33 | 85.37 | 84.34 |
| CarManufacturer       |  64.29 | 69.23 | 66.67 |
| MusicalWork           |  70.37 | 62.30 | 66.09 |
| MusicalGRP            |  60.87 | 75.68 | 67.47 |
| WrittenWork           |  77.55 | 70.37 | 73.79 |
| Cleric                |  64.71 | 73.33 | 68.75 |
| VisualWork            |  60.71 | 55.74 | 58.12 |
| Software              |  60.71 | 65.38 | 62.96 |
| SportsManager         | 100.00 | 68.75 | 81.48 |
| PrivateCorp           |  43.75 | 63.64 | 51.85 |
| Medication/Vaccine    |  60.87 | 77.78 | 68.29 |
| Disease               |  58.33 | 38.89 | 46.67 |
| AnatomicalStructure   |  52.17 | 70.59 | 60.00 |
| MedicalProcedure      |  66.67 | 46.15 | 54.55 |
| OtherLOC              |  66.67 | 75.00 | 70.59 |
| Vehicle               |  70.59 | 60.00 | 64.86 |
| Station               |  88.89 | 80.00 | 84.21 |
| Food                  |  73.68 | 73.68 | 73.68 |
| Clothing              |  66.67 | 60.00 | 63.16 |
| Symptom               |  64.29 | 90.00 | 75.00 |
| Drink                 |  80.00 | 72.73 | 76.19 |
| AerospaceManufacturer |  50.00 | 60.00 | 54.55 |
| ArtWork               |  60.00 | 46.15 | 52.17 |

### distilbert-base-uncased-finetuned-sst-2-english + NER

============================= Training pipeline =============================

| E   | #      | LOSS TRANS... | LOSS NER  | ENTS_F | ENTS_P | ENTS_R | SCORE  | 
| --- | ------ | ------------- | --------  | ------ | ------ | ------ | ------ | 
|   0 |      0 |       2484.16 |  1127.79  |   0.15 |   0.09 |   0.77 |   0.00 | 
|   1 |    200 |     740634.88 | 147972.35 |   6.49 |  11.81 |   4.48 |   0.06 | 
|   2 |    400 |     612423.57 | 68010.06  |  31.44 |  38.23 |  26.70 |   0.31 | 
|   4 |    600 |     425195.81 | 54411.39  |  41.06 |  47.99 |  35.88 |   0.41 | 
|   5 |    800 |     333951.87 | 51336.60  |  46.77 |  53.42 |  41.59 |   0.47 | 
|   7 |   1000 |     145869.27 | 36149.23  |  57.13 |  59.60 |  54.86 |   0.57 | 
|   8 |   1200 |      95389.10 | 29816.81  |  63.32 |  66.53 |  60.42 |   0.63 | 
|  10 |   1400 |      51251.20 | 24595.82  |  67.15 |  70.67 |  63.97 |   0.67 | 
|  11 |   1600 |      24554.33 | 21142.78  |  66.85 |  69.65 |  64.27 |   0.67 | 
|  13 |   1800 |      21966.18 | 18779.06  |  67.68 |  69.90 |  65.59 |   0.68 | 
|  14 |   2000 |      16754.66 | 16310.77  |  69.20 |  71.06 |  67.44 |   0.69 | 
|  16 |   2200 |      18446.55 | 15261.84  |  70.40 |  73.19 |  67.82 |   0.70 | 

================================== Results ==================================

| TOK   | -     |
| ----- | ----- |
| NER P | 73.19 |
| NER R | 67.82 |
| NER F | 70.40 |
| SPEED | 862   |


=============================== NER (per type) ===============================

|                       |      P |     R |     F |
| --------------------- | ------ | ----- | ----- |
| OtherPER              |  78.79 | 57.14 | 66.24 |
| HumanSettlement       |  85.19 | 84.40 | 84.79 |
| PublicCorp            |  76.92 | 71.43 | 74.07 |
| Scientist             |  57.14 | 26.67 | 36.36 |
| OtherPROD             |  55.56 | 40.82 | 47.06 |
| Artist                |  73.41 | 87.26 | 79.74 |
| ORG                   |  77.97 | 58.97 | 67.15 |
| Politician            |  73.81 | 58.49 | 65.26 |
| Athlete               |  83.33 | 88.61 | 85.89 |
| Facility              |  75.56 | 65.38 | 70.10 |
| SportsGRP             |  89.47 | 82.93 | 86.08 |
| CarManufacturer       |  66.67 | 61.54 | 64.00 |
| MusicalWork           |  70.00 | 57.38 | 63.06 |
| MusicalGRP            |  68.89 | 83.78 | 75.61 |
| WrittenWork           |  71.70 | 70.37 | 71.03 |
| Cleric                |  85.71 | 80.00 | 82.76 |
| VisualWork            |  64.81 | 57.38 | 60.87 |
| Software              |  60.00 | 57.69 | 58.82 |
| SportsManager         |  76.47 | 81.25 | 78.79 |
| Medication/Vaccine    |  65.00 | 72.22 | 68.42 |
| Disease               |  37.50 | 33.33 | 35.29 |
| MedicalProcedure      | 100.00 | 53.85 | 70.00 |
| Symptom               |  33.33 | 10.00 | 15.38 |
| OtherLOC              |  70.00 | 43.75 | 53.85 |
| Vehicle               |  56.25 | 45.00 | 50.00 |
| Station               |  94.44 | 85.00 | 89.47 |
| Food                  |  50.00 | 47.37 | 48.65 |
| Clothing              |  62.50 | 50.00 | 55.56 |
| AnatomicalStructure   |  42.11 | 47.06 | 44.44 |
| PrivateCorp           |  83.33 | 45.45 | 58.82 |
| Drink                 |  66.67 | 54.55 | 60.00 |
| AerospaceManufacturer |  58.33 | 70.00 | 63.64 |
| ArtWork               |  66.67 | 30.77 | 42.11 |


### roberta_base_en + BEAM NER

============================= Training pipeline =============================

| E   | #      | LOSS TRANS... | LOSS BEAM_NER |  ENTS_F |  ENTS_P |  ENTS_R |  SCORE | 
| --- | ------ | ------------- | ------------- | ------- | ------- | ------- | ------ | 
|  0  |     0  |     15587.65  |       895.39  |  0.26   |  0.14   |  1.47   |  0.00  |   
|  1  |   200  | 1257972705.50 |     224178.17 |  14.92  |  14.01  |  15.97  |  0.15  |
|  2  |   400  |  39449882.52  |     78984.63  | 37.95   | 40.09   | 36.03   |  0.38  |
|  4  |   600  |  10279088.87  |     50854.24  | 53.72   | 54.53   | 52.93   |  0.54  |
|  5  |   800  |   5868562.60  |     36762.72  | 55.90   | 60.12   | 52.24   |  0.56  |
|  7  |  1000  |   7053234.66  |     28382.57  | 63.01   | 62.67   | 63.35   |  0.63  |
|  8  |  1200  |   6371151.00  |     23248.58  | 66.97   | 65.00   | 69.06   |  0.67  |
| 10  |  1400  |  12697538.69  |     18529.36  | 68.02   | 67.37   | 68.67   |  0.68  |
| 11  |  1600  |   1123324.32  |     16587.10  | 66.47   | 63.48   | 69.75   |  0.66  |
| 13  |  1800  |    671107.86  |     14238.98  | 68.33   | 65.28   | 71.68   |  0.68  |
| 14  |  2000  |    423241.37  |     12337.46  | 68.80   | 67.87   | 69.75   |  0.69  |
| 16  |  2200  |    972581.84  |     11298.61  | 70.61   | 68.58   | 72.76   |  0.71  |
| 17  |  2400  |    224751.48  |      9217.54  | 68.60   | 66.16   | 71.22   |  0.69  |
| 18  |  2600  |    174965.92  |      8157.46  | 69.35   | 66.50   | 72.45   |  0.69  |
| 20  |  2800  |    127303.03  |      7394.08  | 69.28   | 66.43   | 72.38   |  0.69  |
| 21  |  3000  |    111412.02  |      6430.25  | 70.05   | 67.40   | 72.92   |  0.70  |
| 23  |  3200  |     80211.65  |      6143.11  | 70.74   | 69.52   | 71.99   |  0.71  |
| 24  |  3400  |     68730.80  |      5364.46  | 70.16   | 67.60   | 72.92   |  0.70  |
| 26  |  3600  |     56046.55  |      5029.75  | 69.39   | 66.83   | 72.15   |  0.69  |
| 27  |  3800  |     35294.02  |      4594.10  | 68.81   | 66.69   | 71.06   |  0.69  |
| 29  |  4000  |     43469.90  |      4289.72  | 70.38   | 68.29   | 72.61   |  0.70  |
| 30  |  4200  |     23209.12  |      3771.54  | 72.19   | 70.19   | 74.31   |  0.72  |
| 32  |  4400  |     17949.68  |      3266.47  | 71.65   | 69.18   | 74.31   |  0.72  |
| 33  |  4600  |     17712.73  |      3445.55  | 71.85   | 69.42   | 74.46   |  0.72  |
| 35  |  4800  |     16479.23  |      3058.39  | 72.31   | 70.28   | 74.46   |  0.72  |
| 36  |  5000  |     10446.08  |      2723.95  | 71.59   | 69.01   | 74.38   |  0.72  |
| 37  |  5200  |     24989.66  |      2733.86  | 71.34   | 69.42   | 73.38   |  0.71  |
| 39  |  5400  |     11794.38  |      2522.46  | 71.46   | 70.13   | 72.84   |  0.71  |
| 40  |  5600  |     16145.77  |      4119.26  | 72.10   | 69.96   | 74.38   |  0.72  |
| 42  |  5800  |     22374.62  |      3798.36  | 71.23   | 69.91   | 72.61   |  0.71  |
| 43  |  6000  |     20812.43  |      3493.53  | 71.76   | 69.12   | 74.61   |  0.72  |
| 45  |  6200  |     16218.19  |      3000.15  | 70.33   | 69.18   | 71.53   |  0.70  |
| 46  |  6400  |     13569.74  |      2931.55  | 71.10   | 69.30   | 72.99   |  0.71  |

================================== Results ==================================

| TOK   | -     |
| ----- | ----- |
| NER P | 70.72 |
| NER R | 74.54 |
| NER F | 72.58 |
| SPEED | 3110  |


=============================== NER (per type) ===============================

|                       |       P  |      R |     F | 
| --------------------- | -------- | ------ | ----- |
| OtherPER              |   77.78  |  61.54 | 68.71 |
| ORG                   |   65.00  |  66.67 | 65.82 |
| HumanSettlement       |   85.59  |  87.16 | 86.36 |
| Artist                |   55.22  |  87.26 | 67.64 |
| Politician            |   70.21  |  62.26 | 66.00 |
| Scientist             |   76.92  |  66.67 | 71.43 |
| PublicCorp            |   81.82  |  64.29 | 72.00 |
| OtherPROD             |   61.54  |  48.98 | 54.55 |
| Athlete               |   78.41  |  87.34 | 82.63 |
| Facility              |   86.27  |  84.62 | 85.44 |
| SportsGRP             |   85.37  |  85.37 | 85.37 |
| CarManufacturer       |   75.00  |  69.23 | 72.00 |
| MusicalWork           |   66.67  |  68.85 | 67.74 |
| MusicalGRP            |   78.38  |  78.38 | 78.38 |
| WrittenWork           |   82.35  |  77.78 | 80.00 |
| Cleric                |   80.00  |  53.33 | 64.00 |
| VisualWork            |   75.47  |  65.57 | 70.18 |
| Software              |   76.92  |  76.92 | 76.92 |
| MedicalProcedure      |   64.29  |  69.23 | 66.67 |
| SportsManager         |   82.35  |  87.50 | 84.85 |
| Medication/Vaccine    |   83.33  |  83.33 | 83.33 |
| Disease               |   53.33  |  44.44 | 48.48 |
| Symptom               |   83.33  | 100.00 | 90.91 |
| OtherLOC              |   72.22  |  81.25 | 76.47 |
| Vehicle               |   73.68  |  70.00 | 71.79 |
| Station               |  100.00  |  75.00 | 85.71 |
| Food                  |   51.85  |  73.68 | 60.87 |
| AnatomicalStructure   |   65.00  |  76.47 | 70.27 |
| Clothing              |   87.50  |  70.00 | 77.78 |
| PrivateCorp           |  100.00  |  18.18 | 30.77 |
| Drink                 |   70.00  |  63.64 | 66.67 |
| AerospaceManufacturer |   72.73  |  80.00 | 76.19 |
| ArtWork               |   66.67  |  46.15 | 54.55 |


### distilbert-base-uncased-finetuned-sst-2-english + BEAM NER

============================= Training pipeline =============================

| E   | #      | LOSS TRANS... | LOSS BEAM_NER | ENTS_F | ENTS_P | ENTS_R | SCORE  |   
| --- | ------ | ------------- | ------------- | ------ | ------ | ------ | ------ |
|   0 |      0 |       2484.16 |       1127.79 |   0.18 |   0.10 |   1.00 |   0.00 |
|   1 |    200 | 1292369920.39 |     253353.31 |  12.50 |  10.43 |  15.59 |   0.12 |
|   2 |    400 |   61876395.12 |      86622.96 |  41.50 |  55.19 |  33.26 |   0.42 |
|   4 |    600 |   13574742.91 |      57701.56 |  53.97 |  54.72 |  53.24 |   0.54 |
|   5 |    800 |   17634889.84 |      42600.85 |  63.20 |  66.64 |  60.11 |   0.63 |
|   7 |   1000 |    6546030.99 |      30392.23 |  66.33 |  67.09 |  65.59 |   0.66 |
|   8 |   1200 |    1355454.73 |      23542.35 |  68.46 |  70.09 |  66.90 |   0.68 |
|  10 |   1400 |     883533.74 |      15967.77 |  66.77 |  66.95 |  66.59 |   0.67 |
|  11 |   1600 |     497234.42 |      12501.32 |  72.29 |  74.33 |  70.37 |   0.72 |
|  13 |   1800 |    9161565.39 |       9620.60 |  70.68 |  71.63 |  69.75 |   0.71 |
|  14 |   2000 |     105568.89 |       6396.00 |  71.61 |  73.32 |  69.98 |   0.72 |
|  16 |   2200 |      73057.90 |       5060.82 |  71.99 |  73.11 |  70.91 |   0.72 |
|  17 |   2400 |      49050.42 |       3726.66 |  73.94 |  74.67 |  73.23 |   0.74 |
|  18 |   2600 |      24709.75 |       2777.18 |  71.88 |  72.16 |  71.60 |   0.72 |
|  20 |   2800 |      13902.56 |       2256.34 |  73.16 |  73.33 |  72.99 |   0.73 |
|  21 |   3000 |      11249.16 |       1742.92 |  70.73 |  71.56 |  69.91 |   0.71 |
|  23 |   3200 |      11542.59 |       1355.94 |  73.45 |  75.39 |  71.60 |   0.73 |
|  24 |   3400 |       5271.25 |       1050.57 |  73.80 |  74.47 |  73.15 |   0.74 |
|  26 |   3600 |       5648.90 |        888.42 |  72.61 |  72.70 |  72.53 |   0.73 |
|  27 |   3800 |       2968.32 |        770.45 |  73.24 |  73.73 |  72.76 |   0.73 |
|  29 |   4000 |       4782.15 |        625.39 |  72.47 |  72.64 |  72.30 |   0.72 |

================================== Results ====================================

| TOK   |  -     |
| ----- | ------ |
| NER P |  74.67 |
| NER R |  73.23 |
| NER F |  73.94 |
| SPEED |  850   |


=============================== NER (per type) ===============================

|                       |       P  |     R |      F |
| --------------------- | -------- | ----- | ------ |
| OtherPER              |   70.37  | 62.64 |  66.28 |
| ORG                   |   73.68  | 71.79 |  72.73 |
| HumanSettlement       |   85.59  | 87.16 |  86.36 |
| Artist                |   79.91  | 84.43 |  82.11 |
| Scientist             |  100.00  |  6.67 |  12.50 |
| PublicCorp            |   65.38  | 60.71 |  62.96 |
| OtherPROD             |   62.16  | 46.94 |  53.49 |
| Politician            |   64.81  | 66.04 |  65.42 |
| WrittenWork           |   72.88  | 79.63 |  76.11 |
| Athlete               |   76.60  | 91.14 |  83.24 |
| Facility              |   71.19  | 80.77 |  75.68 |
| SportsGRP             |   92.31  | 87.80 |  90.00 |
| CarManufacturer       |   66.67  | 76.92 |  71.43 |
| MusicalWork           |   75.44  | 70.49 |  72.88 |
| MusicalGRP            |   75.68  | 75.68 |  75.68 |
| Cleric                |   86.67  | 86.67 |  86.67 |
| VisualWork            |   76.79  | 70.49 |  73.50 |
| Software              |   80.95  | 65.38 |  72.34 |
| SportsManager         |   76.47  | 81.25 |  78.79 |
| Medication/Vaccine    |   76.47  | 72.22 |  74.29 |
| Disease               |   69.23  | 50.00 |  58.06 |
| MedicalProcedure      |   66.67  | 61.54 |  64.00 |
| Symptom               |   69.23  | 90.00 |  78.26 |
| OtherLOC              |   62.50  | 62.50 |  62.50 |
| Vehicle               |   68.42  | 65.00 |  66.67 |
| AerospaceManufacturer |   72.73  | 80.00 |  76.19 |
| Station               |   89.47  | 85.00 |  87.18 |
| Drink                 |   66.67  | 72.73 |  69.57 |
| Clothing              |   28.57  | 20.00 |  23.53 |
| Food                  |   50.00  | 47.37 |  48.65 |
| AnatomicalStructure   |   47.62  | 58.82 |  52.63 |
| PrivateCorp           |   83.33  | 45.45 |  58.82 |
| ArtWork               |   62.50  | 38.46 |  47.62 |

### Model comparison

| Model                                                      | NER P | NER R | NER F | SPEED |
| ---------------------------------------------------------- | ----- | ----- | ----- | ----- |
| roberta_base_en + BEAM NER                                 | 70.72 | 74.54 | 72.58 | 3110  |
| distilbert-base-uncased-finetuned-sst-2-english + NER      | 73.19 | 67.82 | 70.40 | 862   |
| xlm_roberta_base_en + NER                                  | 72.84 | 72.22 | 72.53 | 2471  |
| roberta_base_en + NER                                      | 68.36 | 72.84 | 70.53 | 6429  |
| distilbert-base-uncased-finetuned-sst-2-english + BEAM NER | 74.67 | 73.23 | 73.94 | 850   |

# Bibliography
## analyticsvidhya
1. [NLP 01 - Introduction](https://www.analyticsvidhya.com/blog/2021/06/part-1-step-by-step-guide-to-master-natural-language-processing-nlp-in-python)
2. [NLP 02 - Knowledge Required to Learn NLP](https://www.analyticsvidhya.com/blog/2021/06/part-2-step-by-step-guide-to-master-natural-language-processing-nlp-in-python)
3. [NLP 03 - Text Cleaning and Preprocessing](https://www.analyticsvidhya.com/blog/2021/06/part-3-step-by-step-guide-to-nlp-text-cleaning-and-preprocessing)
4. [NLP 04 - Text Cleaning Techniques](https://www.analyticsvidhya.com/blog/2021/06/part-4-step-by-step-guide-to-master-natural-language-processing-in-python)
5. [NLP 05 - Word Embedding and Text Vectorization](https://www.analyticsvidhya.com/blog/2021/06/part-5-step-by-step-guide-to-master-nlp-text-vectorization-approaches)
6. [NLP 06 - Word2Vec](https://www.analyticsvidhya.com/blog/2021/06/part-6-step-by-step-guide-to-master-nlp-word2vec)
7. [NLP 07 - Word Embedding in Detail](https://www.analyticsvidhya.com/blog/2021/06/part-7-step-by-step-guide-to-master-nlp-word-embedding)
8. [NLP 08 - Useful Natural Language Processing Tasks](https://www.analyticsvidhya.com/blog/2021/06/part-8-step-by-step-guide-to-master-nlp-useful-natural-language-processing-tasks)
9. [NLP 09 - Semantic Analysis](https://www.analyticsvidhya.com/blog/2021/06/part-9-step-by-step-guide-to-master-nlp-semantic-analysis)
10. [NLP 10 - Named Entity Recognition](https://www.analyticsvidhya.com/blog/2021/06/part-10-step-by-step-guide-to-master-nlp-named-entity-recognition)
11. [NLP 11 - Syntactic Analysis](https://www.analyticsvidhya.com/blog/2021/06/part-11-step-by-step-guide-to-master-nlp-syntactic-analysis)
12. [NLP 12 - Grammar in NLP](https://www.analyticsvidhya.com/blog/2021/06/part-12-step-by-step-guide-to-master-nlp-grammar-in-nlp)
13. [NLP 13 - Regular Expressions](https://www.analyticsvidhya.com/blog/2021/06/part-13-step-by-step-guide-to-master-nlp-regular-expressions)
14. [NLP 14 - Basics of Topic Modelling](https://www.analyticsvidhya.com/blog/2021/06/part-14-step-by-step-guide-to-master-nlp-basics-of-topic-modelling)
15. [NLP 15 - Topic Modelling using NMF](https://www.analyticsvidhya.com/blog/2021/06/part-15-step-by-step-guide-to-master-nlp-topic-modelling-using-nmf)
16. [NLP 16 - Topic Modelling using LSA](https://www.analyticsvidhya.com/blog/2021/06/part-16-step-by-step-guide-to-master-nlp-topic-modelling-using-lsa)
17. [NLP 17 - Topic Modelling using pLSA](https://www.analyticsvidhya.com/blog/2021/06/part-17-step-by-step-guide-to-master-nlp-topic-modelling-using-plsa)
18. [NLP 18 - Topic Modelling using LDA (Probabilistic Approach)](https://www.analyticsvidhya.com/blog/2021/06/part-18-step-by-step-guide-to-master-nlp-topic-modelling-using-lda-probabilistic-approach)
19. [NLP 19 - Topic Modelling using LDA (Matrix Factorization Approach)](https://www.analyticsvidhya.com/blog/2021/06/part-19-step-by-step-guide-to-master-nlp-topic-modelling-using-lda-matrix-factorization-approach)
20. [NLP 20 - Information Retrieval](https://www.analyticsvidhya.com/blog/2021/06/part-20-step-by-step-guide-to-master-nlp-information-retrieval)
