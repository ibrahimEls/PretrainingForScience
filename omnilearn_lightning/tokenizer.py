from fast_pytorch_kmeans import KMeans


class KMeansTokenizer:
    def __init__(self, n_clusters: int, mode: str = "euclidean", **kwargs):
        self.n_clusters = n_clusters
        self.mode = mode
        self.kmeans_kwargs = kwargs
        self.kmeans = KMeans(n_clusters=n_clusters, mode=mode, **kwargs)

    def fit(self, x, mask=None):
        """Fit the KMeans model to the data.

        Parameters
        ----------
        x: torch.Tensor
            Input data of shape (num_samples, num_features) or
            (batch_size, num_points, num_features).
        mask: torch.Tensor, optional
            Boolean mask of shape (batch_size, num_points) indicating which points to
            include in the fitting process. If provided, only the points where mask is
            True will be used for fitting. Default is None.
        """
        # if mask is provided: apply it to x (then x is (batch_size, num_points, num_features))
        # if not, x is (num_samples, num_features)
        if mask is not None:
            x = x[mask]
        self.kmeans.fit(x)

    def predict(self, x):
        """Predict the closest cluster each sample in x belongs to.

        Parameters
        -----------
        x: torch.Tensor
            Input data of shape (batch_size, num_points, num_features).

        Returns
        -------
        labels: torch.Tensor
            Index of the cluster each sample belongs to, of shape (batch_size, num_points).
        """
        batch_size, num_points, num_features = x.shape
        x_reshaped = x.reshape(-1, num_features)
        labels = self.kmeans.predict(x_reshaped)
        # reshape back to (batch_size, num_points)
        labels = labels.reshape(batch_size, num_points)
        return labels

    def transform(self, x):
        """Transform the data to the nearest cluster centroids.

        Parameters
        -----------
        x: torch.Tensor
            Input data of shape (batch_size, num_points, num_features).

        Returns
        -------
        tokenized: torch.Tensor
            Tokenized data of shape (batch_size, num_points, num_features).
        """
        batch_size, num_points, num_features = x.shape
        x_reshaped = x.reshape(-1, num_features)
        centroids = self.kmeans.centroids
        # get the centroid for each point
        tokenized = centroids[self.kmeans.predict(x_reshaped)]
        # reshape back to (batch_size, num_points, num_features)
        tokenized = tokenized.reshape(batch_size, num_points, num_features)
        return tokenized
