import os

import joblib
import pandas as pd
import torch
import numpy as np

MODEL_DIRECTORY_PATH = 'static' + os.path.sep + 'pkls'
MODEL_DIRECTORY_CSV_PATH = 'static' + os.path.sep + 'csv'
MODEL_DIRECTORY_CSV_EMBEDDIGNS_PATH = 'static' + os.path.sep + 'csv' + os.path.sep + 'embeddings'

class Utils:
    @staticmethod
    def process_batch(self, batch_data, batch_index, data_size):
        print(f"Processing batch {batch_index + 1}/{(data_size + len(batch_data) - 1) // len(batch_data)}...")

        inputs = self.tokenizer(batch_data, return_tensors='pt', padding=True, truncation=True)

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)

        self.ponderate_embeddings(
            batch_index=batch_index,
            batch_data=batch_data,
            embeddings=embeddings,
            tokenizer=self.tokenizer,
            model=self.model,
            nlp=self.nlp,
            verb_weight=self.verb_weight,
            object_weight=self.object_weight
        )

        return embeddings

    @staticmethod
    def ponderate_embeddings(batch_index, batch_data, embeddings, tokenizer, model, nlp, verb_weight, object_weight):
        print(f"Applying verb and object weights for batch {batch_index + 1}...")

        tagged_data = [nlp(sent) for sent in batch_data]

        for i, doc in enumerate(tagged_data):
            verb_weights = []
            obj_weights = []
            token_embeddings = []
            tokens = []

            for token in doc:
                if token.pos_ == 'VERB' and verb_weight != 0:
                    verb_weights.append(verb_weight)
                    tokens.append(token.text)
                elif token.dep_ in ('dobj', 'nsubj', 'attr', 'prep', 'pobj') and object_weight != 0:
                    obj_weights.append(object_weight)
                    tokens.append(token.text)

            if not tokens:
                continue

            for token in tokens:
                inputs = tokenizer(token, return_tensors='pt')
                with torch.no_grad():
                    outputs = model(**inputs)
                    token_embedding = outputs.last_hidden_state.mean(dim=1).numpy()
                    token_embeddings.append(token_embedding[0])

            weights = np.array(verb_weights + obj_weights)

            token_embeddings = np.array(token_embeddings)
            weighted_embeddings = token_embeddings * weights[:, np.newaxis]

            sentence_embedding = np.mean(weighted_embeddings, axis=0)
            embeddings[i] = torch.tensor(sentence_embedding)
    @staticmethod
    def save_to_csv(dataframe: pd.DataFrame, csv_filename: str):
        if not os.path.exists(MODEL_DIRECTORY_CSV_PATH):
            os.makedirs(MODEL_DIRECTORY_CSV_PATH)
        csv_file_path = os.path.join(os.getcwd(), MODEL_DIRECTORY_CSV_PATH, csv_filename)
        print(f"Saving CSV to {csv_file_path}...")
        dataframe.to_csv(csv_file_path, index=False)
        return csv_file_path

    @staticmethod
    def save_to_pkl(model_info: dict, pkl_filename: str):
        if not os.path.exists(MODEL_DIRECTORY_PATH):
            os.makedirs(MODEL_DIRECTORY_PATH)
        pkl_file_path = os.path.join(os.getcwd(), MODEL_DIRECTORY_PATH, pkl_filename)
        print(f"Saving model to {pkl_file_path}...")
        joblib.dump(model_info, pkl_file_path)
        return pkl_file_path

    @staticmethod
    def generate_pkl(application_name, clustering_model, model_name, dense_data_array, labels, distance_threshold, linkage, metric, verb_weight, object_weight):
        print("Saving clustering metadata for plotting...")
        model_info = {
            'affinity': f'{model_name} {metric} {linkage}',
            'labels': labels,
            'model_name': model_name,
            'model': clustering_model,
            'data_points': dense_data_array,
            'application_name': application_name,
            'distance_threshold': distance_threshold,
            'verb_weight': verb_weight,
            'object_weight': object_weight
        }

        if hasattr(clustering_model, 'cluster_centers_'):
            model_info['cluster_centers'] = clustering_model.cluster_centers_

        pkl_file_name = f"{application_name}_{model_name.lower()}_{metric}_{linkage}_thr-{distance_threshold}_vw-{verb_weight}_ow-{object_weight}.pkl"
        return Utils.save_to_pkl(model_info, pkl_file_name)
