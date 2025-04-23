import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.manifold import trustworthiness

def calc_noiseless_silhouette(
        embeddings,
        embeddings_reduced,
        labels,
        n_neighbours=100,
        lambda_tw=0.1,
        lambda_s=0.5, # Если много выбросов изменить на 0.45
        lambda_nf=0.3 # Если много выбросов изменить на 0.45
):
    mask = labels != -1
    if mask.sum() < 10 or len(np.unique(labels[mask])) < 2 or np.bincount(labels[mask]).min() < 5:
        return -1.0

    sil = silhouette_score(embeddings[mask], labels[mask])
    sil_norm = (sil + 1) / 2

    N = embeddings.shape[0]
    noise_frac = (labels == -1).sum() / N
    nf_term = 1 - noise_frac

    tw = trustworthiness(embeddings, embeddings_reduced, n_neighbors=n_neighbours)

    metric = lambda_tw * tw + lambda_s * sil_norm + lambda_nf * nf_term
    return metric