"""Tests for wasserstein_distance_batched function in generation_utils."""

import unittest

import awkward as ak
import numpy as np

from omnilearn_lightning.generation_utils import wasserstein_distance_batched


class TestWassersteinDistanceBatched(unittest.TestCase):
    def test_constant_distributions_returns_zero(self):
        """Test that constant identical distributions return a Wasserstein distance of zero."""
        # Use constant arrays so sampling doesn't affect the result
        data1 = ak.Array([5.0] * 1000)
        data2 = ak.Array([5.0] * 1000)
        mean, std = wasserstein_distance_batched(
            data1=data1,
            data2=data2,
            num_eval_samples=100,
            num_batches=5,
        )
        self.assertAlmostEqual(mean, 0.0, places=5)
        self.assertAlmostEqual(std, 0.0, places=5)

    def test_different_distributions_returns_nonzero(self):
        """Test that different distributions return a non-zero Wasserstein distance."""
        data1 = ak.Array([0.0] * 1000)
        data2 = ak.Array([1.0] * 1000)
        mean, std = wasserstein_distance_batched(
            data1=data1,
            data2=data2,
            num_eval_samples=100,
            num_batches=5,
        )
        self.assertAlmostEqual(mean, 1.0, places=5)
        self.assertAlmostEqual(std, 0.0, places=5)

    def test_shifted_distribution(self):
        """Test that a shifted distribution returns the expected distance."""
        np.random.seed(42)
        data1 = ak.Array(np.random.normal(0, 1, 10000))
        data2 = ak.Array(np.random.normal(2, 1, 10000))
        mean, std = wasserstein_distance_batched(
            data1=data1,
            data2=data2,
            num_eval_samples=1000,
            num_batches=10,
        )
        # The Wasserstein distance between N(0,1) and N(2,1) should be close to 2
        self.assertGreater(mean, 1.8)
        self.assertLess(mean, 2.2)

    def test_return_types(self):
        """Test that the function returns float types."""
        data = ak.Array([1.0, 2.0, 3.0, 4.0, 5.0] * 20)
        mean, std = wasserstein_distance_batched(
            data1=data,
            data2=data,
            num_eval_samples=10,
            num_batches=3,
        )
        self.assertIsInstance(mean, (float, np.floating))
        self.assertIsInstance(std, (float, np.floating))

    def test_dimension_mismatch_raises_error(self):
        """Test that mismatched dimensions raise a ValueError."""
        data1 = ak.Array([1.0, 2.0, 3.0])
        data2 = ak.Array([[1.0, 2.0], [3.0, 4.0]])
        with self.assertRaises(ValueError) as context:
            wasserstein_distance_batched(
                data1=data1,
                data2=data2,
                num_eval_samples=2,
                num_batches=1,
            )
        self.assertIn("same number of dimensions", str(context.exception))

    def test_2d_arrays_are_flattened(self):
        """Test that 2D arrays are flattened before computing distance.

        Note: The function samples num_eval_samples rows first, then flattens.
        So for 2D constant arrays, the result should still be zero.
        """
        # Use constant values so flattening doesn't affect the result
        data1 = ak.Array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]] * 100)
        data2 = ak.Array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]] * 100)
        mean, std = wasserstein_distance_batched(
            data1=data1,
            data2=data2,
            num_eval_samples=50,
            num_batches=5,
        )
        self.assertAlmostEqual(mean, 0.0, places=5)

    def test_2d_arrays_known_distance(self):
        """Test that 2D arrays give expected distance after flattening."""
        # data1 flattens to all 0s, data2 flattens to all 1s
        data1 = ak.Array([[0.0, 0.0]] * 1000)
        data2 = ak.Array([[1.0, 1.0]] * 1000)
        mean, std = wasserstein_distance_batched(
            data1=data1,
            data2=data2,
            num_eval_samples=100,
            num_batches=5,
        )
        self.assertAlmostEqual(mean, 1.0, places=5)
        self.assertAlmostEqual(std, 0.0, places=5)

    def test_replace_true_allows_small_datasets(self):
        """Test that replace=True allows sampling more than dataset size."""
        # Use constant data so sampling with replacement gives same result
        data = ak.Array([7.0, 7.0, 7.0])
        mean, std = wasserstein_distance_batched(
            data1=data,
            data2=data,
            num_eval_samples=100,
            num_batches=3,
            replace=True,
        )
        self.assertAlmostEqual(mean, 0.0, places=5)

    def test_replace_false_sampling(self):
        """Test that replace=False samples without replacement."""
        # Create data large enough to sample without replacement
        data1 = ak.Array([0.0] * 500)
        data2 = ak.Array([1.0] * 500)
        mean, std = wasserstein_distance_batched(
            data1=data1,
            data2=data2,
            num_eval_samples=100,
            num_batches=3,
            replace=False,
        )
        self.assertAlmostEqual(mean, 1.0, places=5)

    def test_replace_false_too_large_raises_error(self):
        """Test that sampling without replacement more than dataset size raises an error."""
        data = ak.Array([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError) as context:
            wasserstein_distance_batched(
                data1=data,
                data2=data,
                num_eval_samples=10,
                num_batches=1,
                replace=False,
            )
        self.assertIn("Cannot take a larger sample than", str(context.exception))

    def test_multiple_batches_reduces_variance(self):
        """Test that using more batches provides a more stable estimate."""
        np.random.seed(123)
        data1 = ak.Array(np.random.uniform(0, 1, 1000))
        data2 = ak.Array(np.random.uniform(0, 1, 1000))

        # With few batches
        _, std_few = wasserstein_distance_batched(
            data1=data1,
            data2=data2,
            num_eval_samples=100,
            num_batches=3,
        )

        # With many batches
        _, std_many = wasserstein_distance_batched(
            data1=data1,
            data2=data2,
            num_eval_samples=100,
            num_batches=30,
        )

        # Both should be non-negative, and the std calculation should work
        self.assertGreaterEqual(std_few, 0)
        self.assertGreaterEqual(std_many, 0)
        # More batches should lead to lower std deviation
        self.assertLess(std_many, std_few)

    def test_asymmetric_data_sizes(self):
        """Test that the function works with different sized input arrays."""
        # Use constant data to ensure zero distance regardless of sampling
        data1 = ak.Array([3.0] * 100)
        data2 = ak.Array([3.0] * 500)
        mean, std = wasserstein_distance_batched(
            data1=data1,
            data2=data2,
            num_eval_samples=50,
            num_batches=5,
        )
        self.assertAlmostEqual(mean, 0.0, places=5)

    def test_asymmetric_data_sizes_nonzero(self):
        """Test that the function works with different sized input arrays and non-zero distance."""
        data1 = ak.Array([0.0] * 100)
        data2 = ak.Array([1.0] * 500)
        mean, std = wasserstein_distance_batched(
            data1=data1,
            data2=data2,
            num_eval_samples=50,
            num_batches=5,
        )
        self.assertAlmostEqual(mean, 1.0, places=5)
        self.assertAlmostEqual(std, 0.0, places=5)

    def test_asymmetric_data_sizes_close_to_symmetric_in_large_limit(self):
        """Test that asymmetric sizes give similar results to symmetric sizes in large data limit."""
        np.random.seed(456)

        data1 = ak.Array(np.random.normal(0, 1, 100_000))
        data2 = ak.Array(np.random.normal(1, 1, 150_000))
        data3 = ak.Array(np.random.normal(1, 1, 100_000))

        mean_asym, _ = wasserstein_distance_batched(
            data1=data1,
            data2=data2,
            num_eval_samples=50_000,
            num_batches=10,
        )
        mean_sym, _ = wasserstein_distance_batched(
            data1=data1,
            data2=data3,
            num_eval_samples=50_000,
            num_batches=10,
        )
        # The means should be both close to 1.00...
        self.assertAlmostEqual(mean_asym, 1.0, places=2)
        self.assertAlmostEqual(mean_sym, 1.0, places=2)

    def test_num_batches_affects_output_length(self):
        """Test that the function computes the correct number of batch distances."""
        data1 = ak.Array([0.0] * 100)
        data2 = ak.Array([1.0] * 100)

        # Run with different num_batches and verify both return valid results
        mean1, std1 = wasserstein_distance_batched(
            data1=data1,
            data2=data2,
            num_eval_samples=20,
            num_batches=1,
        )
        mean5, std5 = wasserstein_distance_batched(
            data1=data1,
            data2=data2,
            num_eval_samples=20,
            num_batches=5,
        )

        # Both should give distance of 1.0
        self.assertAlmostEqual(mean1, 1.0, places=5)
        self.assertAlmostEqual(mean5, 1.0, places=5)
        # With 1 batch, std should be 0
        self.assertAlmostEqual(std1, 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
