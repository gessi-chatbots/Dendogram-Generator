# preprocessing_service.py

from flask import Flask, request, jsonify
import os
import json
from spellchecker import SpellChecker
import stanza
import unicodedata
import contractions
import re
import string
import spacy

app = Flask(__name__)

@app.route('/preprocess', methods=['POST'])
def preprocess():
    try:
        request_data = request.get_json()
        app_name = request_data['app_name']
        features = request_data['features']

        if 'preprocessing' in request_data and request_data['preprocessing']:
            # Preprocess features
            features = preprocess_features(features)
            save_preprocessed_features(features, app_name)
        else:
            features = load_saved_preprocessed_features(app_name)

        return jsonify({"preprocessed_features": features}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def preprocessed_app(app_name):
    file_path = f"static/preprocessed_jsons/{app_name}Features.json"
    return os.path.exists(file_path) and os.path.getsize(file_path) > 0

def save_preprocessed_features(features, app_name):
    file_path = f"static/preprocessed_jsons/{app_name}Features.json"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as json_file:
        json.dump(features, json_file)

def load_saved_preprocessed_features(app_name):
    file_path = f"static/preprocessed_jsons/{app_name}Features.json"
    if os.path.exists(file_path):
        with open(file_path, "r") as json_file:
            return json.load(json_file)
    return None

def preprocess_features(features):
    preprocessed_features = set()

    for feature in features:
        if not is_emoji_only(feature) and not contains_weird_characters(feature):
            preprocessed_feature = preprocess_feature(feature)
            if is_english(preprocessed_feature):
                preprocessed_features.add(preprocessed_feature)

    return list(preprocessed_features)

def preprocess_feature(feature):
    feature = feature.replace('_', ' ')
    feature = remove_mentions_and_tags(feature)
    feature = camel_case_to_words(feature)
    feature = expand_contractions(feature)
    feature = remove_special_characters(feature)
    feature = remove_punctuation(feature)
    feature = standarize_accents(feature)
    feature = lemmatize_spacy(feature)
    feature = feature.lower()
    return feature

# Utility functions (these are the same as from your original code)
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

def remove_punctuation(text):
    return ''.join([c for c in text if c not in string.punctuation])

def camel_case_to_words(camel_case_str):
    words = re.sub('([a-z])([A-Z])', r'\1 \2', camel_case_str)
    return words

def lemmatize_spacy(feature):
    spacy.prefer_gpu()
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    doc = nlp(feature)
    return " ".join([token.lemma_ for token in doc])

def is_english(text):
    pattern = re.compile(r'^[a-zA-Z0-9\s.,?!\'"-]+$')
    return bool(pattern.match(text))

def is_emoji_only(text):
    emoji_pattern = re.compile("[\U00010000-\U0010FFFF]+", flags=re.UNICODE)
    return bool(emoji_pattern.fullmatch(text))

def contains_weird_characters(text):
    weird_characters_pattern = re.compile(r'[^a-zA-Z0-9\s.,?!\'"_-]')
    return bool(weird_characters_pattern.search(text))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
