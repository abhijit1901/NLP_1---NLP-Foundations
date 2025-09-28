# NLP Fundamentals with NLTK and Scikit-learn

This repository is a collection of Jupyter notebooks designed to provide a hands-on introduction to fundamental Natural Language Processing (NLP) techniques. Starting from the basics of text tokenization, we move through normalization methods like stemming and lemmatization, and finally into feature extraction techniques like Bag of Words and TF-IDF.

Each notebook is self-contained and focuses on a specific concept, using the **NLTK** and **Scikit-learn** libraries.

-----

## 📋 Table of Contents

- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Notebook Descriptions](#-notebook-descriptions)
  - [1. Tokenization](#1-tokenization-1_tokenizationipynb)
  - [2. Stemming](#2-stemming-2_stemmingipynb)
  - [3. Lemmatization](#3-lemmatization-3_lemmatizationipynb)
  - [4. Text Preprocessing Workflow](#4-text-preprocessing-workflow-4_text_preprocessing_stopwords_with_nltkipynb)
  - [5. Part-of-Speech (POS) Tagging](#5-part-of-speech-pos-tagging-5_pos_taggingipynb)
  - [6. Named Entity Recognition (NER)](#6-named-entity-recognition-ner-6_named_entity_recognitionipynb)
  - [7. Bag of Words (BoW)](#7-bag-of-words-bow-7_bag_of_wordsipynb)
  - [8. TF-IDF](#8-tf-idf-8_tf-idfipynb)
 
## direct file access
  - [1. Tokenization](./1_tokenization.ipynb)
  - [2. Stemming](./2_stemming.ipynb)
  - [3. Lemmatization](./3_lemmatization.ipynb)
  - [4. Text Preprocessing Workflow](./4_Text_Preprocessing_Stopwords_With_NLTK.ipynb)
  - [5. Part-of-Speech (POS) Tagging](./5_POS_tagging.ipynb)
  - [6. Named Entity Recognition (NER)](./6_Named_Entity_Recognition.ipynb)
  - [7. Bag of Words (BoW)](./7_Bag_of_words.ipynb)
  - [8. TF-IDF](./8_tf-idf.ipynb)


-----

## 🚀 Getting Started

Follow these instructions to set up your environment and run the notebooks.

### Prerequisites

  * Python 3.x
  * Jupyter Notebook or JupyterLab

### Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

2.  **Install the required Python libraries:**

    ```bash
    pip install pandas nltk scikit-learn
    ```

3.  **Download NLTK Data:**
    Run the following commands in a Python interpreter or a Jupyter cell to download the necessary NLTK models and corpora.

    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
    ```

-----

## 📚 Notebook Descriptions

### 1\. Tokenization (`1_tokenization.ipynb`)

This notebook introduces tokenization, the fundamental process of breaking down a text corpus into smaller units like sentences or words.

**Concepts Covered:**

  * **Sentence Tokenization**: Splitting a paragraph into individual sentences. This is useful for analyzing sentences one by one. It's achieved using `nltk.sent_tokenize()`.
  * **Word Tokenization**: Splitting a sentence into individual words (or tokens), including punctuation. This is a crucial first step for most NLP tasks. It's achieved using `nltk.word_tokenize()`.

**Key Takeaways:**

  * Understand the difference between a corpus, a sentence, and a token.
  * Learn how to use NLTK to perform both sentence and word tokenization.

### 2\. Stemming (`2_stemming.ipynb`)

This notebook explores stemming, a text normalization technique that reduces words to their root or base form (known as a "stem").

**Concepts Covered:**

  * **What is Stemming?**: The process of chopping off prefixes and suffixes to get to the root form of a word. The resulting stem may not always be a valid dictionary word (e.g., "history" -\> "histori").
  * **PorterStemmer**: One of the most common and gentle stemming algorithms.
  * **RegexpStemmer**: A stemmer that uses regular expressions to define custom rules for stripping prefixes or suffixes.
  * **SnowballStemmer**: An improvement over the Porter Stemmer, also known as the "Porter2" stemmer. It is more aggressive and supports multiple languages.

**Key Takeaways:**

  * Understand the purpose of stemming in reducing the vocabulary size.
  * Learn to implement and compare different stemmers available in NLTK.
  * Recognize the limitations of stemming, such as producing non-words.

### 3\. Lemmatization (`3_lemmatization.ipynb`)

This notebook covers lemmatization, a more advanced normalization technique that reduces words to their meaningful base form (known as a "lemma").

**Concepts Covered:**

  * **Stemming vs. Lemmatization**: While stemming simply chops off ends, lemmatization uses vocabulary and morphological analysis to return a valid dictionary word. For example, the lemma of "better" is "good".
  * **WordNetLemmatizer**: NLTK's lemmatizer which uses the WordNet database.
  * **Part-of-Speech (POS) Tagging**: Lemmatization is highly dependent on the word's part of speech. The notebook demonstrates how providing a POS tag (e.g., `'v'` for verb) to the lemmatizer (`lemmatizer.lemmatize(word, pos='v')`) yields more accurate results.

**Key Takeaways:**

  * Understand the difference between a stem and a lemma.
  * Learn how to use `WordNetLemmatizer` for accurate text normalization.
  * Appreciate the critical role of POS tags in achieving correct lemmatization.

### 4\. Text Preprocessing Workflow (`4_Text_Preprocessing_Stopwords_With_NLTK.ipynb`)

This notebook combines the concepts from the previous files into a complete text preprocessing pipeline, using a speech by Dr. APJ Abdul Kalam as the sample corpus.

**Concepts Covered:**

  * **Stopword Removal**: Removing common words (like "the", "a", "is") that add little semantic value to the text. NLTK provides a standard list of stopwords for various languages via `stopwords.words('english')`.
  * **Integrated Pipeline**: The notebook shows a complete workflow:
    1.  Sentence Tokenization.
    2.  Word Tokenization.
    3.  Stopword Removal.
    4.  Stemming / Lemmatization.

**Key Takeaways:**

  * Learn how to build a practical text cleaning pipeline.
  * See the effect of stopword removal on the text.
  * Compare the outputs of stemming and lemmatization on a real-world text.

### 5\. Part-of-Speech (POS) Tagging (`5_POS_tagging.ipynb`)

This notebook focuses on Part-of-Speech (POS) Tagging, the process of assigning a grammatical category (like noun, verb, adjective) to each word in a text.

**Concepts Covered:**

  * **What is POS Tagging?**: Understanding the importance of identifying the grammatical components of a sentence.
  * **NLTK's `pos_tag` Function**: The notebook uses `nltk.pos_tag()` to tag tokenized words.
  * **POS Tag Set**: It introduces common tags like `NNP` (Proper Noun, singular), `VBD` (Verb, past tense), and `JJ` (Adjective).

**Key Takeaways:**

  * Learn how to perform POS tagging on a text.
  * Understand how POS information can be used to understand the grammatical structure of a sentence, which is essential for tasks like lemmatization and NER.

### 6\. Named Entity Recognition (NER) (`6_Named_Entity_Recognition.ipynb`)

This notebook introduces Named Entity Recognition (NER), a task focused on identifying and classifying named entities in text into predefined categories.

**Concepts Covered:**

  * **What are Named Entities?**: These are real-world objects, such as persons, locations, organizations, dates, etc.
  * **NER with NLTK**: The notebook uses a three-step process:
    1.  Word Tokenization.
    2.  POS Tagging.
    3.  Chunking with `nltk.ne_chunk()`.
  * **Entity Categories**: The output identifies entities like `PERSON`, `ORGANIZATION`, and `GPE` (Geopolitical Entity).

**Key Takeaways:**

  * Understand the goal of NER in information extraction.
  * Learn how to use NLTK's pre-trained NER chunker to identify entities in a sentence.

### 7\. Bag of Words (BoW) (`7_Bag_of_words.ipynb`)

This notebook transitions from text preprocessing to feature extraction, demonstrating how to convert text into numerical vectors using the Bag of Words (BoW) model.

**Concepts Covered:**

  * **Bag of Words Intuition**: A model that represents text by counting the occurrence of each word, disregarding grammar and word order.
  * **`CountVectorizer`**: A powerful tool from Scikit-learn used to create a document-term matrix from a text corpus.
  * **N-Grams**: The notebook explains how n-grams (sequences of n consecutive words) can capture context that single words (unigrams) miss. It shows how to generate bigrams (`ngram_range=(2,2)`) to find phrases like "free entry".
  * **Full Pipeline**: It applies a full preprocessing pipeline to the SMS Spam Dataset and then converts the cleaned text into BoW vectors.

**Key Takeaways:**

  * Understand how to convert raw text into a numerical format suitable for machine learning models.
  * Learn to use `CountVectorizer` to build a BoW matrix.
  * Grasp the concept of n-grams and their importance in capturing local word context.

### 8\. TF-IDF (`8_tf-idf.ipynb`)

This notebook covers Term Frequency-Inverse Document Frequency (TF-IDF), a more advanced feature extraction technique that weighs words based on their importance in a corpus.

**Concepts Covered:**

  * **TF-IDF Intuition**:
      * **Term Frequency (TF)**: Measures how frequently a term appears in a document.
      * **Inverse Document Frequency (IDF)**: Measures how important a term is by down-weighting terms that are very common across all documents.
  * **TF-IDF Score**: The product of TF and IDF. Words that are frequent in one document but rare across the entire corpus receive a high score.
  * **`TfidfVectorizer`**: Scikit-learn's tool for converting a text corpus directly into a TF-IDF matrix.
  * **Application**: The notebook applies TF-IDF to the preprocessed SMS Spam Dataset and also demonstrates its use with n-grams.

**Key Takeaways:**

  * Understand the limitations of simple word counts (BoW) and how TF-IDF provides a more nuanced word-weighting scheme.
  * Learn how to use `TfidfVectorizer` to create TF-IDF feature vectors.
  * Compare the BoW and TF-IDF approaches for text representation.
