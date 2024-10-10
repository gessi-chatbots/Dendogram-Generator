import requests
import json
import os
from .Context import Context
from . import Affinity_strategy
from dotenv import load_dotenv
from .preprocessing_service import preprocess_features

load_dotenv()
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
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r") as json_file:
        return json.load(json_file)
    return None

def generate_dendogram(preprocessing,
                       embedding,
                       metric,
                       linkage,
                       distance_threshold,
                       object_weight,
                       verb_weight,
                       request_content):
    app_name = request_content['app_name']
    features = request_content['features']

    # Preprocessing step
    if preprocessing and not preprocessed_app(app_name):
        features = preprocess_features(features)
        save_preprocessed_features(features, app_name)
    elif preprocessing and preprocessed_app(app_name):
        features = load_saved_preprocessed_features(app_name)

    if embedding == 'bert':
        context = Context(Affinity_strategy.BertEmbeddingAffinity())
    elif embedding == 'paraphrase':
        context = Context(Affinity_strategy.ParaphraseEmbeddingAffinity())
    elif embedding == 'tf-idf':
        context = Context(Affinity_strategy.TfidfEmbeddingAffinity())
    else:
        raise ValueError(f"Unsupported embedding method: {embedding}")

    return context.use_affinity_algorithm(application_name=app_name,
                                          data=features,
                                          linkage=linkage,
                                          object_weight=object_weight,
                                          verb_weight=verb_weight,
                                          distance_threshold=distance_threshold,
                                          metric=metric)



def call_preprocessing_service(features):
    url = os.getenv("DG_SERVICE_URL")
    port = os.getenv("DG_SERVICE_PORT")

    if not url or not port:
        raise Exception("Preprocessing service URL or port not found in environment variables.")

    full_url = f"{url}:{port}/preprocess"

    data = {
        "features": features
    }

    try:
        response = requests.post(full_url, json=data)
        if response.status_code == 200:
            return response.json()['preprocessed_features']
        else:
            raise Exception(
                f"Failed to preprocess features. Status code: {response.status_code}, Response: {response.text}")
    except Exception as e:
        raise Exception(f"Error occurred while calling preprocessing service: {str(e)}")