import torch
from fast_pytorch_kmeans import KMeans

from .array_utils import preprocess_tensor


class KMeansTokenizer:
    def __init__(
        self,
        n_clusters: int,
        scale_factors_x: torch.Tensor,
        scale_factors_add_info: torch.Tensor,
        mode: str = "euclidean",
        **kwargs,
    ):
        """KMeans-based tokenizer for particle features.

        Parameters
        ----------
        n_clusters: int
            Number of clusters (codebook size).
        scale_factors_x: torch.Tensor
            Scale factors for x features.
        scale_factors_add_info: torch.Tensor
            Scale factors for add_info features.
        mode: str, optional
            Distance metric to use. Default is "euclidean".
        kwargs:
            Additional keyword arguments for the KMeans model.
        """
        self.n_clusters = n_clusters
        self.mode = mode
        self.kmeans_kwargs = kwargs
        self.kmeans = KMeans(n_clusters=n_clusters, mode=mode, **kwargs)
        self.scale_factors_x = scale_factors_x
        self.scale_factors_add_info = scale_factors_add_info

    def fit(self, x: torch.Tensor, add_info: torch.Tensor, mask: torch.Tensor = None):
        """Fit the KMeans model to the data.

        Parameters
        ----------
        x: torch.Tensor
            Input x data of shape (batch_size, num_points, num_features_x).
        add_info: torch.Tensor
            Input add_info data of shape (batch_size, num_points, num_features_add_info).
        mask: torch.Tensor, optional
            Boolean mask of shape (batch_size, num_points) indicating which points to
            include in the fitting process. If provided, only the points where mask is
            True will be used for fitting. Default is None.
        """
        # Normalize and concatenate x and add_info
        combined = preprocess_tensor(
            x,
            add_info,
            inverse=False,
            scale_factors_x=self.scale_factors_x.to(x.device),
            scale_factors_add_info=self.scale_factors_add_info.to(add_info.device),
        )

        # Handle 3D input (batch_size, num_points, num_features)
        if combined.dim() == 3:
            batch_size, num_points, num_features = combined.shape
            combined = combined.reshape(-1, num_features)
            if mask is not None:
                mask_flat = mask.reshape(-1)
                combined = combined[mask_flat]
        elif mask is not None:
            combined = combined[mask]

        self.kmeans.fit(combined)

    def predict(self, x: torch.Tensor, add_info: torch.Tensor):
        """Predict the closest cluster each sample belongs to.

        Parameters
        -----------
        x: torch.Tensor
            Input x data of shape (batch_size, num_points, num_features_x).
        add_info: torch.Tensor
            Input add_info data of shape (batch_size, num_points, num_features_add_info).

        Returns
        -------
        labels: torch.Tensor
            Index of the cluster each sample belongs to, of shape (batch_size, num_points).
        """
        # Normalize and concatenate
        combined = preprocess_tensor(
            x,
            add_info,
            inverse=False,
            scale_factors_x=self.scale_factors_x.to(x.device),
            scale_factors_add_info=self.scale_factors_add_info.to(add_info.device),
        )

        # move centroids to the same device as x
        self.kmeans.centroids = self.kmeans.centroids.to(x.device)
        batch_size, num_points, num_features = combined.shape
        combined_reshaped = combined.reshape(-1, num_features)
        labels = self.kmeans.predict(combined_reshaped)
        labels = labels.reshape(batch_size, num_points)
        return labels

    def transform(self, x: torch.Tensor, add_info: torch.Tensor):
        """Transform the data to the nearest cluster centroids.

        Parameters
        -----------
        x: torch.Tensor
            Input x data of shape (batch_size, num_points, num_features_x).
        add_info: torch.Tensor
            Input add_info data of shape (batch_size, num_points, num_features_add_info).

        Returns
        -------
        tokenized_x: torch.Tensor
            Tokenized x data of shape (batch_size, num_points, num_features_x).
        tokenized_add_info: torch.Tensor
            Tokenized add_info data of shape (batch_size, num_points, num_features_add_info).
        """
        batch_size, num_points, _ = x.shape

        # Get cluster labels
        labels = self.predict(x, add_info)

        # Get the centroid for each point (in normalized space)
        centroids_normalized = self.kmeans.centroids.to(x.device)[labels]
        centroids_normalized = centroids_normalized.reshape(batch_size, num_points, -1)

        # Split centroids back into x and add_info parts
        num_features_x = x.shape[-1]
        centroids_x = centroids_normalized[..., :num_features_x]
        centroids_add_info = centroids_normalized[..., num_features_x:]

        # Denormalize using inverse transform
        tokenized = preprocess_tensor(
            centroids_x,
            centroids_add_info,
            inverse=True,
            scale_factors_x=self.scale_factors_x.to(x.device),
            scale_factors_add_info=self.scale_factors_add_info.to(add_info.device),
        )

        # Split and return x and add_info parts separately
        tokenized_x = tokenized[:, :, :num_features_x]
        tokenized_add_info = tokenized[:, :, num_features_x:]
        return tokenized_x, tokenized_add_info

    def reconstruct(self, labels: torch.Tensor, num_features_x: int):
        """Reconstruct the data from the cluster labels.

        Parameters
        -----------
        labels: torch.Tensor
            Cluster labels of shape (batch_size, num_points).
        num_features_x: int
            Number of features in the x part (to split the centroids).

        Returns
        -------
        reconstructed_x: torch.Tensor
            Reconstructed x data of shape (batch_size, num_points, num_features_x).
        reconstructed_add_info: torch.Tensor
            Reconstructed add_info data of shape (batch_size, num_points, num_features_add_info).
        """
        batch_size, num_points = labels.shape

        # Get centroids for the labels (in normalized space)
        centroids_normalized = self.kmeans.centroids.to(labels.device)[labels]
        centroids_normalized = centroids_normalized.reshape(batch_size, num_points, -1)

        # Split centroids back into x and add_info parts
        centroids_x = centroids_normalized[..., :num_features_x]
        centroids_add_info = centroids_normalized[..., num_features_x:]

        # Denormalize using inverse transform
        reconstructed = preprocess_tensor(
            centroids_x,
            centroids_add_info,
            inverse=True,
            scale_factors_x=self.scale_factors_x.to(labels.device),
            scale_factors_add_info=self.scale_factors_add_info.to(labels.device),
        )

        # Split and return x and add_info parts separately
        reconstructed_x = reconstructed[..., :num_features_x]
        reconstructed_add_info = reconstructed[..., num_features_x:]
        return reconstructed_x, reconstructed_add_info
