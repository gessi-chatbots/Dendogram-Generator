import stanza
import unicodedata
import contractions
import re
import string
import spacy
from .Context import Context
from . import Affinity_strategy

def generate_dendogram(preprocessing, embedding, features):

    if preprocessing:
        features = preprocess_features(features)

    model_file_name = None

    if embedding == 'tf-idf-cosine' or embedding == 'all':
        context = Context(Affinity_strategy.TfIdfCosineAffinity())
        model_file_name = context.use_affinity_algorithm(features)

    if embedding == 'tf-idf-euclidean' or embedding == 'all':
        context = Context(Affinity_strategy.TfIdfEuclideanAffinity())
        model_file_name = context.use_affinity_algorithm(features)

    if embedding == 'bert-embedding-euclidean' or embedding == 'all':
        context = Context(Affinity_strategy.BERTEuclideanEmbeddingAffinity())
        model_file_name = context.use_affinity_algorithm(features)

    if embedding == 'bert-embedding-cosine' or embedding == 'all':
        context = Context(Affinity_strategy.BERTCosineEmbeddingAffinity())
        model_file_name = context.use_affinity_algorithm(features)

    return model_file_name

def preprocess_features(features):
    stanza.download('en')
    preprocessed_features = []
    for feature in features:
        preprocessed_features.append(preprocess_feature(feature))
    return preprocessed_features

def preprocess_feature(feature):
    feature = remove_mentions_and_tags(feature)
    feature = remove_special_characters(feature)
    feature = remove_numbers(feature)
    feature = remove_punctuation(feature)
    feature = camel_case_to_words(feature)
    feature = expand_contractions(feature)
    feature = standarize_accents(feature)
    feature = lemmatize_spacy(feature)
    # feature = lemmatize_stanza(feature)

def expand_contractions(feature):
    expanded_words = []
    for word in feature.split():
        expanded_words.append(contractions.fix(word))
    return ' '.join(expanded_words)

def standarize_accents(feature): 
    return unicodedata.normalize('NFKD', feature).encode('ascii', 'ignore').decode('utf-8', 'ignore')

def remove_mentions_and_tags(text):
    text = re.sub(r'@\S*', '', text)
    return re.sub(r'#\S*', '', text)

def remove_special_characters(text):
    pat = r'[^a-zA-z0-9.,!?/:;\"\'\s]' 
    return re.sub(pat, '', text)

def remove_numbers(text):
    pattern = r'[^a-zA-z.,!?/:;\"\'\s]' 
    return re.sub(pattern, '', text)

def remove_punctuation(text):
    return ''.join([c for c in text if c not in string.punctuation])
    
def camel_case_to_words(camel_case_str):
    words = re.sub('([a-z])([A-Z])', r'\1 \2', camel_case_str)
    return words

def lemmatize_spacy(feature):
    nlp = spacy.load('en', disable = ['parser','ner']) 
    doc = nlp(feature)
    return " ".join([token.lemma_ for token in doc])

def lemmatize_stanza(feature):
    nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma')
    doc = nlp(feature)
    lemmatized_feature = ''
    for word in doc.sentences[0].words:
        lemmatized_feature = lemmatized_feature + word.lemma + ' '
        lemmatized_feature = lemmatized_feature[:len(lemmatized_feature) - 1]
    return lemmatized_feature
