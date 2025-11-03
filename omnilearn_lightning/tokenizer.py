import torch
from fast_pytorch_kmeans import KMeans

from .array_utils import preprocess_tensor


class KMeansTokenizer:
    def __init__(
        self,
        n_clusters: int,
        scale_factors_x: torch.Tensor,
        scale_factors_add_info: torch.Tensor | None,
        max_iter: int = 300,
        **kwargs,
    ):
        """KMeans-based tokenizer for particle features.

        Parameters
        ----------
        n_clusters: int
            Number of clusters (codebook size).
        scale_factors_x: torch.Tensor
            Scale factors for x features.
        scale_factors_add_info: torch.Tensor or None
            Scale factors for add_info features. If None, add_info is treated as absent
            throughout and only x features are used.
        max_iter: int, optional
            Maximum number of iterations for KMeans. Default is 100.
        kwargs:
            Additional keyword arguments for the KMeans model.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.kmeans_kwargs = kwargs
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            verbose=10,
            **self.kmeans_kwargs,
        )
        self.scale_factors_x = scale_factors_x
        self.scale_factors_add_info = scale_factors_add_info

    def _use_add_info(self, add_info):
        """Helper to determine if add_info features should be used."""
        return self.scale_factors_add_info is not None and add_info is not None

    def _get_scale_factors_add_info(self, device):
        if self.scale_factors_add_info is None:
            return None
        return self.scale_factors_add_info.to(device)

    def fit(
        self,
        x: torch.Tensor,
        add_info: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        fit_with_torchpq: bool = False,
    ):
        """Fit the KMeans model to the data.

        Parameters
        ----------
        x: torch.Tensor
            Input x data of shape (batch_size, num_points, num_features_x).
        add_info: torch.Tensor, optional
            Optional add_info data of shape (batch_size, num_points, num_features_add_info).
            If None (or if scale_factors_add_info is None), only x is used.
        mask: torch.Tensor, optional
            Boolean mask of shape (batch_size, num_points) indicating which points to
            include in the fitting process. If provided, only the points where mask is
            True will be used for fitting. Default is None.
        fit_with_torchpq: bool, optional
            Whether to use torchpq for KMeans fitting. Default is False.
        """
        # Normalize and concatenate x and add_info (if used)
        use_add_info = self._use_add_info(add_info)
        combined = preprocess_tensor(
            x,
            add_info if use_add_info else None,
            inverse=False,
            scale_factors_x=self.scale_factors_x.to(x.device),
            scale_factors_add_info=self._get_scale_factors_add_info(
                add_info.device if use_add_info else x.device
            ),
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

        if fit_with_torchpq:
            # Use torchpq for KMeans fitting for better performance
            import torchpq

            # raise error if cuda is not available
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "fit_with_torchpq=True requires CUDA. Please enable CUDA or "
                    "set fit_with_torchpq=False."
                )
            # ensure combined is on cuda
            combined = combined.to("cuda")

            _tmp_kmeans = torchpq.clustering.KMeans(
                n_clusters=self.n_clusters,
                distance="euclidean",
                max_iter=self.max_iter,
                verbose=10,
            )

            _tmp_kmeans.fit(combined.transpose(0, 1).contiguous())
            self.kmeans.centroids = _tmp_kmeans.centroids.transpose(0, 1).contiguous()
        else:
            self.kmeans.fit(combined)

    def predict(self, x: torch.Tensor, add_info: torch.Tensor | None = None):
        """Predict the closest cluster each sample belongs to.

        Parameters
        -----------
        x: torch.Tensor
            Input x data of shape (batch_size, num_points, num_features_x).
        add_info: torch.Tensor, optional
            Optional add_info data of shape (batch_size, num_points, num_features_add_info).
            If None (or if scale_factors_add_info is None), only x is used.

        Returns
        -------
        labels: torch.Tensor
            Index of the cluster each sample belongs to, of shape (batch_size, num_points).
        """
        # Normalize and concatenate x and add_info (if used)
        use_add_info = self._use_add_info(add_info)
        combined = preprocess_tensor(
            x,
            add_info if use_add_info else None,
            inverse=False,
            scale_factors_x=self.scale_factors_x.to(x.device),
            scale_factors_add_info=self._get_scale_factors_add_info(
                add_info.device if use_add_info else x.device
            ),
        )

        # move centroids to the same device as x
        self.kmeans.centroids = self.kmeans.centroids.to(x.device)
        batch_size, num_points, num_features = combined.shape
        combined_reshaped = combined.reshape(-1, num_features)
        labels = self.kmeans.predict(combined_reshaped)
        labels = labels.reshape(batch_size, num_points)
        return labels

    def transform(self, x: torch.Tensor, add_info: torch.Tensor | None = None):
        """Transform the data to the nearest cluster centroids.

        Parameters
        -----------
        x: torch.Tensor
            Input x data of shape (batch_size, num_points, num_features_x).
        add_info: torch.Tensor, optional
            Optional add_info data of shape (batch_size, num_points, num_features_add_info).
            If None (or if scale_factors_add_info is None), only x is used.

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
        centroids_normalized = self.kmeans.centroids.to(x.device)[labels.reshape(-1)]
        centroids_normalized = centroids_normalized.reshape(batch_size, num_points, -1)

        # Split centroids back into x and add_info parts
        num_features_x = x.shape[-1]
        use_add_info = self._use_add_info(add_info)
        centroids_x = centroids_normalized[..., :num_features_x]
        centroids_add_info = (
            centroids_normalized[..., num_features_x:] if use_add_info else None
        )

        # Denormalize using inverse transform
        tokenized = preprocess_tensor(
            centroids_x,
            centroids_add_info,
            inverse=True,
            scale_factors_x=self.scale_factors_x.to(x.device),
            scale_factors_add_info=self._get_scale_factors_add_info(
                add_info.device if use_add_info else x.device
            ),
        )

        # Split and return x and add_info parts separately
        tokenized_x = tokenized[:, :, :num_features_x]
        tokenized_add_info = tokenized[:, :, num_features_x:] if use_add_info else None
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
        use_add_info = self.scale_factors_add_info is not None
        centroids_x = centroids_normalized[..., :num_features_x]
        centroids_add_info = (
            centroids_normalized[..., num_features_x:] if use_add_info else None
        )

        # Denormalize using inverse transform
        reconstructed = preprocess_tensor(
            centroids_x,
            centroids_add_info,
            inverse=True,
            scale_factors_x=self.scale_factors_x.to(labels.device),
            scale_factors_add_info=self._get_scale_factors_add_info(labels.device),
        )

        # Split and return x and add_info parts separately
        reconstructed_x = reconstructed[..., :num_features_x]
        reconstructed_add_info = (
            reconstructed[..., num_features_x:] if use_add_info else None
        )
        return reconstructed_x, reconstructed_add_info
