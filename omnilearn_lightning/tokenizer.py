import torch
from fast_pytorch_kmeans import KMeans

from .array_utils import preprocess_tensor


class KMeansTokenizer:
    def __init__(
        self,
        n_clusters: int,
        scale_factors: torch.Tensor,
        mode: str = "euclidean",
        **kwargs,
    ):
        """KMeans-based tokenizer for particle features.
        Parameters
        ----------
        n_clusters: int
            Number of clusters (codebook size).
        scale_factors: torch.Tensor
            Scale factors for each feature dimension.
        mode: str, optional
            Distance metric to use. Default is "euclidean".
        kwargs:
            Additional keyword arguments for the KMeans model.
        """
        self.n_clusters = n_clusters
        self.mode = mode
        self.kmeans_kwargs = kwargs
        self.kmeans = KMeans(n_clusters=n_clusters, mode=mode, **kwargs)
        self.scale_factors = scale_factors

    def fit(self, x: torch.Tensor, mask: torch.Tensor = None):
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
        x = preprocess_tensor(
            x,
            index_PID=4,
            scale_factors=self.scale_factors.to(x.device),
        )
        if mask is not None:
            x = x[mask]
        self.kmeans.fit(x)

    def predict(self, x: torch.Tensor):
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
        x = preprocess_tensor(
            x,
            index_PID=4,
            scale_factors=self.scale_factors.to(x.device),
        )
        # move centroids to the same device as x
        self.kmeans.centroids = self.kmeans.centroids.to(x.device)
        batch_size, num_points, num_features = x.shape
        x_reshaped = x.reshape(-1, num_features)
        labels = self.kmeans.predict(x_reshaped)
        labels = labels.reshape(batch_size, num_points)
        return labels

    def transform(self, x: torch.Tensor):
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
        # get the PID values
        pid_values = x[:, :, 4]
        labels = self.predict(x)
        # get the centroid for each point
        x_tokenized = self.kmeans.centroids.to(x.device)[labels]
        # reshape back to (batch_size, num_points, num_features)
        x_tokenized = x_tokenized.reshape(batch_size, num_points, num_features - 1)
        # invert the preprocessing
        x_tokenized = preprocess_tensor(
            x_tokenized,
            index_PID=4,
            scale_factors=self.scale_factors.to(x.device),
            pid_values=pid_values,
            inverse=True,
        )
        return x_tokenized

    def reconstruct(self, labels: torch.Tensor, pid_values: torch.Tensor):
        """Reconstruct the data from the cluster labels.

        Parameters
        -----------
        labels: torch.Tensor
            Cluster labels of shape (batch_size, num_points).
        pid_values: torch.Tensor
            PID values of shape (batch_size, num_points).

        Returns
        -------
        reconstructed: torch.Tensor
            Reconstructed data of shape (batch_size, num_points, num_features).
        """
        batch_size, num_points = labels.shape
        x_reconstructed = self.kmeans.centroids.to(labels.device)[labels]
        x_reconstructed = x_reconstructed.reshape(batch_size, num_points, -1)
        x_reconstructed = preprocess_tensor(
            x_reconstructed,
            index_PID=4,
            scale_factors=self.scale_factors.to(labels.device),
            pid_values=pid_values,
            inverse=True,
        )
        return x_reconstructed
