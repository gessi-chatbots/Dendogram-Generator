from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
from backend.utils import Utils
from backend.Affinity_strategy import AffinityStrategy
import torch
import pandas as pd

BATCH_SIZE = 32

class MiniLMEmbeddingService(AffinityStrategy):
    def __init__(self, verb_weight=1.0, object_weight=1.0):
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.verb_weight = verb_weight
        self.object_weight = object_weight

    def compute_affinity(self,
                         application_name,
                         labels,
                         linkage,
                         object_weight,
                         verb_weight,
                         distance_threshold,
                         metric):

        self.verb_weight = verb_weight
        self.object_weight = object_weight

        all_embeddings = []

        print(f"Processing data in batches of size {BATCH_SIZE}...")
        for i in range(0, len(labels), BATCH_SIZE):
            batch_data = labels[i:i + BATCH_SIZE]
            batch_index = i // BATCH_SIZE
            print(f"Processing batch {batch_index}...")
            batch_embeddings = self.model.encode(batch_data, convert_to_tensor=True)
            all_embeddings.append(batch_embeddings)

        print("Concatenating all batch embeddings...")
        all_embeddings = torch.cat(all_embeddings, dim=0)
        dense_data_array = all_embeddings.cpu().numpy()

        print("Saving embeddings to CSV...")
        embedding_df = pd.DataFrame(dense_data_array)
        embedding_df['Sentence'] = labels
        csv_filename = f"{application_name}_minilm_{metric}_{linkage}_embeddings.csv"
        Utils.save_to_csv(embedding_df, csv_filename)

        print("Performing Agglomerative Clustering...")
        clustering_model = AgglomerativeClustering(n_clusters=None,
                                                   linkage=linkage,
                                                   distance_threshold=distance_threshold,
                                                   metric=metric)
        clustering_model.fit(dense_data_array)

        return Utils.generate_pkl(application_name,
                                  clustering_model,
                                  'MiniLMEmbedding',
                                  dense_data_array,
                                  labels,
                                  distance_threshold,
                                  linkage,
                                  metric,
                                  verb_weight,
                                  object_weight)
