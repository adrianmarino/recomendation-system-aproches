from sklearn.neighbors import NearestNeighbors


class SimilarityModel:
    def __init__(self, embeddings_matrix, neighbors_count=10, metric='cosine'):
        self.model = NearestNeighbors(n_neighbors=neighbors_count, metric=metric)
        self.model.fit(embeddings_matrix)
        self.embeddings_matrix = embeddings_matrix

    def similar(self, emb_vector_idx):
        emb_vector = self.embeddings_matrix[emb_vector_idx]
        neighbors = self.model.kneighbors([emb_vector])
        return neighbors[1][0]
