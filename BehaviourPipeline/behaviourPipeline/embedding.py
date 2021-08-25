import umap
from sklearn.preprocessing import StandardScaler
def embed_data(feats, **umap_params):
    feats = StandardScaler().fit_transform(feats)
    embedding = umap.UMAP(**umap_params).fit_transform(feats)
    return embedding
    