from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from backend.utils import Utils
from backend.Affinity_strategy import AffinityStrategy
import pandas as pd

BATCH_SIZE = 32


class TfidfEmbeddingService(AffinityStrategy):
    def __init__(self, verb_weight=1.0, object_weight=1.0):
        self.vectorizer = TfidfVectorizer()
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

        print("Fitting TF-IDF vectorizer...")
        all_embeddings = self.vectorizer.fit_transform(labels)
        dense_data_array = all_embeddings.toarray()

        print("Saving embeddings to CSV...")
        embedding_df = pd.DataFrame(dense_data_array)
        embedding_df['Sentence'] = labels
        csv_filename = f"{application_name}_tfidf_{metric}_{linkage}_embeddings.csv"
        Utils.save_to_csv(embedding_df, csv_filename)

        print("Performing Agglomerative Clustering...")
        clustering_model = AgglomerativeClustering(n_clusters=None,
                                                   linkage=linkage,
                                                   distance_threshold=distance_threshold,
                                                   metric=metric)
        clustering_model.fit(dense_data_array)

        return Utils.generate_pkl(application_name,
                                  clustering_model,
                                  'TfidfEmbedding',
                                  dense_data_array,
                                  labels,
                                  distance_threshold,
                                  linkage,
                                  metric,
                                  verb_weight,
                                  object_weight)
