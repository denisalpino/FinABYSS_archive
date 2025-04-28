## Hypreparameters opimization was complited with Bayesian Optimization HyperBand with Ray Tune

### Clustering results:
* clusters = 126;
* noise    = 2784 observations.

### The best Noiseless Silhouette score: 0.5858

### The best hyperparameters
* UMAP (cuML):
  * n_components = 38;
  * n_neighbors = 17;
  * min_dist = 0.100331082195;
  * metric = "cosine".
* HDBSCAN (cuML):
  * min_cluster_size = 10;
  * min_samples = 6;
  * cluster_selection_epsilon = 0.1907138159624;
  * cluster_selection_method = "leaf".
* PaCMAP:
  * n_components=2;
  * random_state=42;
  * distance="angular";
  * MN_ratio=18;
  * FP_ratio=4.