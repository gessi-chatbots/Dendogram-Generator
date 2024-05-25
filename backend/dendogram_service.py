import stanza
from . import Context
from . import Affinity_strategy
def generate_dendogram(preprocessing, embedding, features):
    
    # TODO throw custom exception if embedding is none
    # TODO throw custom exception if preprocessing is none
    # TODO throw custom exception if features is none
    if preprocessing: 
        features = preprocess_features(features)
    context = None
    if embedding == 'tf-idf-cosine' or embedding == 'all':
        context = Context(Affinity_strategy.TfIdfCosineAffinity())
        context.use_affinity_algorithm(features)
    
def preprocess_features(features):
    nlp_pipeline = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma')
    preprocessed_features = []
    for feature in features:
        doc = nlp_pipeline(feature)
        lemmatized_feature = ""
        for word in doc.sentences[0].words:
            lemmatized_feature = lemmatized_feature + word.lemma + " "
        preprocessed_features.append(lemmatized_feature[:len(lemmatized_feature) - 1])

