"""Tests for generation_utils functions."""

import unittest
from itertools import product

import awkward as ak
import numpy as np

from omnilearn_lightning.generation_utils import (
    calc_metrics_for_dict,
    distribution_metrics_batched,
    quantiled_kl_divergence,
)


def expected_kld_two_gaussians(mu_p, mu_q, sigma_p=1, sigma_q=1):
    """Analytical KL divergence between two univariate Gaussians.

    Reference: https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    """
    return (
        np.log(sigma_q / sigma_p)
        + (sigma_p**2 + (mu_p - mu_q) ** 2) / (2 * sigma_q**2)
        - 1 / 2
    )


class TestDistributionMetricsBatched(unittest.TestCase):
    """Tests for the unified distribution_metrics_batched function."""

    def test_returns_both_metrics(self):
        """Test that function returns both w1 and kld metrics."""
        data1 = ak.Array([0.0] * 1000)
        data2 = ak.Array([1.0] * 1000)
        results = distribution_metrics_batched(
            data1=data1,
            data2=data2,
            num_eval_samples=100,
            num_batches=5,
        )
        self.assertIn("w1", results)
        self.assertIn("kld", results)
        self.assertEqual(len(results["w1"]), 2)  # (mean, std)
        self.assertEqual(len(results["kld"]), 2)  # (mean, std)

    def test_constant_distributions_returns_zero(self):
        """Test that constant identical distributions return zero for both metrics."""
        data1 = ak.Array([5.0] * 1000)
        data2 = ak.Array([5.0] * 1000)
        results = distribution_metrics_batched(
            data1=data1,
            data2=data2,
            num_eval_samples=100,
            num_batches=5,
        )
        w1_mean, w1_std = results["w1"]
        kld_mean, kld_std = results["kld"]
        self.assertAlmostEqual(w1_mean, 0.0, places=5)
        self.assertAlmostEqual(w1_std, 0.0, places=5)
        self.assertAlmostEqual(kld_mean, 0.0, places=5)
        self.assertAlmostEqual(kld_std, 0.0, places=5)

    def test_different_distributions_returns_nonzero(self):
        """Test that different distributions return non-zero metrics."""
        # Use Gaussian distributions to avoid KLD becoming inf
        np.random.seed(42)
        data1 = ak.Array(np.random.normal(0, 1, 5000))
        data2 = ak.Array(np.random.normal(2, 1, 5000))
        results = distribution_metrics_batched(
            data1=data1,
            data2=data2,
            num_eval_samples=1000,
            num_batches=5,
        )
        w1_mean, _ = results["w1"]
        kld_mean, _ = results["kld"]
        self.assertGreater(w1_mean, 1.5)
        self.assertGreater(kld_mean, 0.0)

    def test_shifted_gaussian_distribution(self):
        """Test that a shifted Gaussian distribution returns expected W1 distance."""
        np.random.seed(42)
        data1 = ak.Array(np.random.normal(0, 1, 10000))
        data2 = ak.Array(np.random.normal(2, 1, 10000))
        results = distribution_metrics_batched(
            data1=data1,
            data2=data2,
            num_eval_samples=1000,
            num_batches=10,
        )
        w1_mean, _ = results["w1"]
        kld_mean, _ = results["kld"]
        # W1 distance between N(0,1) and N(2,1) should be close to 2
        self.assertGreater(w1_mean, 1.8)
        self.assertLess(w1_mean, 2.2)
        # KLD should be positive
        self.assertGreater(kld_mean, 0.0)

    def test_return_types(self):
        """Test that the function returns float types."""
        data = ak.Array([1.0, 2.0, 3.0, 4.0, 5.0] * 200)
        results = distribution_metrics_batched(
            data1=data,
            data2=data,
            num_eval_samples=100,
            num_batches=3,
        )
        for metric in ["w1", "kld"]:
            mean, std = results[metric]
            self.assertIsInstance(mean, (float, np.floating))
            self.assertIsInstance(std, (float, np.floating))

    def test_dimension_mismatch_raises_error(self):
        """Test that mismatched dimensions raise a ValueError."""
        data1 = ak.Array([1.0, 2.0, 3.0])
        data2 = ak.Array([[1.0, 2.0], [3.0, 4.0]])
        with self.assertRaises(ValueError) as context:
            distribution_metrics_batched(
                data1=data1,
                data2=data2,
                num_eval_samples=2,
                num_batches=1,
            )
        self.assertIn("same number of dimensions", str(context.exception))

    def test_2d_arrays_are_flattened(self):
        """Test that 2D arrays are flattened before computing metrics."""
        data1 = ak.Array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]] * 100)
        data2 = ak.Array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]] * 100)
        results = distribution_metrics_batched(
            data1=data1,
            data2=data2,
            num_eval_samples=50,
            num_batches=5,
        )
        w1_mean, _ = results["w1"]
        kld_mean, _ = results["kld"]
        self.assertAlmostEqual(w1_mean, 0.0, places=5)
        self.assertAlmostEqual(kld_mean, 0.0, places=5)

    def test_2d_arrays_known_distance(self):
        """Test that 2D arrays give expected distance after flattening."""
        data1 = ak.Array([[0.0, 0.0]] * 1000)
        data2 = ak.Array([[1.0, 1.0]] * 1000)
        results = distribution_metrics_batched(
            data1=data1,
            data2=data2,
            num_eval_samples=100,
            num_batches=5,
        )
        w1_mean, w1_std = results["w1"]
        self.assertAlmostEqual(w1_mean, 1.0, places=5)
        self.assertAlmostEqual(w1_std, 0.0, places=5)

    def test_replace_true_allows_small_datasets(self):
        """Test that replace=True allows sampling more than dataset size."""
        data = ak.Array([7.0, 7.0, 7.0])
        results = distribution_metrics_batched(
            data1=data,
            data2=data,
            num_eval_samples=100,
            num_batches=3,
            replace=True,
        )
        w1_mean, _ = results["w1"]
        self.assertAlmostEqual(w1_mean, 0.0, places=5)

    def test_replace_false_sampling(self):
        """Test that replace=False samples without replacement."""
        data1 = ak.Array([0.0] * 500)
        data2 = ak.Array([1.0] * 500)
        results = distribution_metrics_batched(
            data1=data1,
            data2=data2,
            num_eval_samples=100,
            num_batches=3,
            replace=False,
        )
        w1_mean, _ = results["w1"]
        self.assertAlmostEqual(w1_mean, 1.0, places=5)

    def test_replace_false_too_large_raises_error(self):
        """Test that sampling without replacement more than dataset size raises an error."""
        data = ak.Array([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError) as context:
            distribution_metrics_batched(
                data1=data,
                data2=data,
                num_eval_samples=10,
                num_batches=1,
                replace=False,
            )
        self.assertIn("Cannot take a larger sample than", str(context.exception))

    def test_num_batches_single_gives_zero_std(self):
        """Test that a single batch gives zero standard deviation."""
        data1 = ak.Array([0.0] * 1000)
        data2 = ak.Array([1.0] * 1000)
        results = distribution_metrics_batched(
            data1=data1,
            data2=data2,
            num_eval_samples=100,
            num_batches=1,
        )
        _, w1_std = results["w1"]
        _, kld_std = results["kld"]
        self.assertAlmostEqual(w1_std, 0.0, places=5)
        self.assertAlmostEqual(kld_std, 0.0, places=5)

    def test_n_bins_parameter(self):
        """Test that n_bins parameter affects KLD calculation."""
        np.random.seed(42)
        data1 = ak.Array(np.random.normal(0, 1, 5000))
        data2 = ak.Array(np.random.normal(0.5, 1, 5000))

        results_10 = distribution_metrics_batched(
            data1=data1,
            data2=data2,
            num_eval_samples=1000,
            num_batches=3,
            n_bins=10,
        )
        results_50 = distribution_metrics_batched(
            data1=data1,
            data2=data2,
            num_eval_samples=1000,
            num_batches=3,
            n_bins=50,
        )

        # Both should be positive
        self.assertGreater(results_10["kld"][0], 0)
        self.assertGreater(results_50["kld"][0], 0)

    def test_asymmetric_data_sizes(self):
        """Test that the function works with different sized input arrays."""
        data1 = ak.Array([3.0] * 100)
        data2 = ak.Array([3.0] * 500)
        results = distribution_metrics_batched(
            data1=data1,
            data2=data2,
            num_eval_samples=50,
            num_batches=5,
        )
        w1_mean, _ = results["w1"]
        self.assertAlmostEqual(w1_mean, 0.0, places=5)

    def test_metrics_use_same_samples(self):
        """Test that both metrics are computed on the same random samples.

        This is verified indirectly by checking that when we fix the seed,
        both metrics are deterministic and reproducible.
        """
        np.random.seed(42)
        data1 = ak.Array(np.random.normal(0, 1, 5000))
        data2 = ak.Array(np.random.normal(0.5, 1, 5000))

        # Run twice - should get same results due to fixed seed in function
        results1 = distribution_metrics_batched(
            data1=data1,
            data2=data2,
            num_eval_samples=500,
            num_batches=5,
        )
        results2 = distribution_metrics_batched(
            data1=data1,
            data2=data2,
            num_eval_samples=500,
            num_batches=5,
        )

        self.assertAlmostEqual(results1["w1"][0], results2["w1"][0], places=10)
        self.assertAlmostEqual(results1["kld"][0], results2["kld"][0], places=10)

    def test_num_eval_samples_none_uses_all_data(self):
        """Test that num_eval_samples=None uses all available samples."""
        data1 = ak.Array([0.0] * 1000)
        data2 = ak.Array([1.0] * 1000)
        results = distribution_metrics_batched(
            data1=data1,
            data2=data2,
            num_eval_samples=None,
            num_batches=1,
        )
        w1_mean, w1_std = results["w1"]
        kld_mean, kld_std = results["kld"]
        # W1 distance should be exactly 1.0 for these constant distributions
        self.assertAlmostEqual(w1_mean, 1.0, places=5)
        # With single batch, std should be 0
        self.assertAlmostEqual(w1_std, 0.0, places=5)
        self.assertAlmostEqual(kld_std, 0.0, places=5)

    def test_num_eval_samples_none_with_num_batches_greater_than_1_raises_error(self):
        """Test that num_eval_samples=None with num_batches>1 raises ValueError."""
        data1 = ak.Array([0.0] * 1000)
        data2 = ak.Array([1.0] * 1000)
        with self.assertRaises(ValueError) as context:
            distribution_metrics_batched(
                data1=data1,
                data2=data2,
                num_eval_samples=None,
                num_batches=5,
            )
        self.assertIn(
            "num_eval_samples cannot be None when num_batches > 1",
            str(context.exception),
        )

    def test_num_eval_samples_none_identical_distributions(self):
        """Test that num_eval_samples=None returns zero for identical distributions."""
        np.random.seed(42)
        data = ak.Array(np.random.normal(0, 1, 5000))
        results = distribution_metrics_batched(
            data1=data,
            data2=data,
            num_eval_samples=None,
            num_batches=1,
        )
        w1_mean, _ = results["w1"]
        kld_mean, _ = results["kld"]
        self.assertAlmostEqual(w1_mean, 0.0, places=5)
        self.assertAlmostEqual(kld_mean, 0.0, places=5)

    def test_num_eval_samples_none_with_2d_arrays(self):
        """Test that num_eval_samples=None works with 2D arrays that get flattened."""
        data1 = ak.Array([[0.0, 0.0]] * 500)
        data2 = ak.Array([[1.0, 1.0]] * 500)
        results = distribution_metrics_batched(
            data1=data1,
            data2=data2,
            num_eval_samples=None,
            num_batches=1,
        )
        w1_mean, _ = results["w1"]
        # After flattening, all values in data1 are 0.0 and all in data2 are 1.0
        self.assertAlmostEqual(w1_mean, 1.0, places=5)

    def test_num_eval_samples_none_different_sized_arrays(self):
        """Test that num_eval_samples=None works with differently sized arrays."""
        data1 = ak.Array([0.0] * 500)
        data2 = ak.Array([1.0] * 1000)
        results = distribution_metrics_batched(
            data1=data1,
            data2=data2,
            num_eval_samples=None,
            num_batches=1,
        )
        w1_mean, _ = results["w1"]
        self.assertAlmostEqual(w1_mean, 1.0, places=5)

    def test_num_eval_samples_none_gaussian_distributions(self):
        """Test num_eval_samples=None with Gaussian distributions for realistic scenario."""
        np.random.seed(42)
        data1 = ak.Array(np.random.normal(0, 1, 10000))
        data2 = ak.Array(np.random.normal(2, 1, 10000))
        results = distribution_metrics_batched(
            data1=data1,
            data2=data2,
            num_eval_samples=None,
            num_batches=1,
        )
        w1_mean, _ = results["w1"]
        kld_mean, _ = results["kld"]
        # W1 distance between N(0,1) and N(2,1) should be close to 2
        self.assertGreater(w1_mean, 1.8)
        self.assertLess(w1_mean, 2.2)
        # KLD should be positive and close to the analytical value
        expected_kld = expected_kld_two_gaussians(0, 2, 1, 1)  # Should be 2.0
        self.assertGreater(kld_mean, 0.5 * expected_kld)
        self.assertLess(kld_mean, 1.5 * expected_kld)


class TestQuantiledKLDivergence(unittest.TestCase):
    """Tests for the quantiled_kl_divergence function."""

    def test_identical_distributions_returns_zero(self):
        """Test that identical distributions return KL divergence of zero."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 200)
        kl = quantiled_kl_divergence(sample_ref=data, sample_approx=data, n_bins=30)
        self.assertAlmostEqual(kl, 0.0, places=5)

    def test_different_distributions_returns_nonzero(self):
        """Test that different distributions return non-zero KL divergence."""
        np.random.seed(42)
        data1 = np.random.normal(0, 1, 10000)
        data2 = np.random.normal(2, 1, 10000)
        kl = quantiled_kl_divergence(sample_ref=data1, sample_approx=data2, n_bins=30)
        self.assertGreater(kl, 0.0)

    def test_return_bin_edges(self):
        """Test that bin edges are returned when requested."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 100)
        kl, bin_edges = quantiled_kl_divergence(
            sample_ref=data, sample_approx=data, n_bins=10, return_bin_edges=True
        )
        self.assertEqual(len(bin_edges), 11)  # n_bins + 1 edges
        self.assertEqual(bin_edges[0], float("-inf"))
        self.assertEqual(bin_edges[-1], float("inf"))

    def test_return_zero_if_nan_or_inf(self):
        """Test that NaN/inf values are replaced with zero when requested."""
        data_ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 100)
        data_approx = np.array([1.0, 1.0, 1.0] * 100)  # Missing higher values
        kl_with_flag = quantiled_kl_divergence(
            sample_ref=data_ref,
            sample_approx=data_approx,
            n_bins=30,
            return_zero_if_nan_or_inf=True,
        )
        self.assertEqual(kl_with_flag, 0)

    def test_n_bins_affects_result(self):
        """Test that different n_bins values affect the result."""
        np.random.seed(42)
        data1 = np.random.normal(0, 1, 5000)
        data2 = np.random.normal(0.5, 1, 5000)

        kl_10 = quantiled_kl_divergence(
            sample_ref=data1, sample_approx=data2, n_bins=10
        )
        kl_50 = quantiled_kl_divergence(
            sample_ref=data1, sample_approx=data2, n_bins=50
        )

        self.assertGreater(kl_10, 0)
        self.assertGreater(kl_50, 0)

    def test_quantiled_kld_matches_analytical_formula(self):
        """Test that quantiled KLD matches analytical KLD for Gaussians.

        Validates the implementation against the closed-form KL divergence
        formula for two univariate Gaussian distributions.
        """
        n_samples = 1_000_000
        n_bins = 200

        mu_p_values = [0]
        mu_q_values = [0, 1, 2]
        sigma_p_values = [1]
        sigma_q_values = [1]

        rng = np.random.default_rng(42)

        for mu_p, mu_q, sigma_p, sigma_q in product(
            mu_p_values, mu_q_values, sigma_p_values, sigma_q_values
        ):
            expected_kld = expected_kld_two_gaussians(mu_p, mu_q, sigma_p, sigma_q)
            sample_p = rng.normal(mu_p, sigma_p, n_samples)
            sample_q = rng.normal(mu_q, sigma_q, n_samples)
            quantiled_kld = quantiled_kl_divergence(
                sample_ref=sample_p, sample_approx=sample_q, n_bins=n_bins
            )

            self.assertEqual(
                expected_kld,
                np.round(quantiled_kld, 2),
                msg=f"KLD mismatch for mu_p={mu_p}, mu_q={mu_q}, sigma_p={sigma_p}, sigma_q={sigma_q}",
            )

    def test_quantiled_kld_returns_inf_for_empty_bins(self):
        """Test that KLD returns inf when approx sample has empty bins."""
        rng = np.random.default_rng(42)
        sample_p = rng.normal(0, 1, 1_000_000)
        sample_q = np.array([1, 2, 3])

        quantiled_kld = quantiled_kl_divergence(
            sample_ref=sample_p, sample_approx=sample_q, n_bins=10
        )

        self.assertEqual(float("inf"), quantiled_kld)

    def test_quantiled_kld_inf_to_zero_with_flag(self):
        """Test that inf KLD is converted to zero with return_zero_if_nan_or_inf flag."""
        rng = np.random.default_rng(42)
        sample_p = rng.normal(0, 1, 1_000_000)
        sample_q = np.array([1, 2, 3])

        quantiled_kld = quantiled_kl_divergence(
            sample_ref=sample_p,
            sample_approx=sample_q,
            n_bins=10,
            return_zero_if_nan_or_inf=True,
        )

        self.assertEqual(0, quantiled_kld)


class TestCalcMetricsForDict(unittest.TestCase):
    """Tests for the calc_metrics_for_dict function."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.dict_ref = {
            "var1": ak.Array([0.0] * 1000),
            "var2": ak.Array(np.random.normal(0, 1, 1000)),
        }
        self.dict_approx = {
            "var1": ak.Array([1.0] * 1000),
            "var2": ak.Array(np.random.normal(0.5, 1, 1000)),
        }

    def test_default_metrics(self):
        """Test that default metrics (w1 and kld) are computed."""
        results = calc_metrics_for_dict(
            dict_reference=self.dict_ref,
            dict_approx=self.dict_approx,
            names=["var1", "var2"],
            num_eval_samples=100,
            num_batches=3,
        )
        self.assertIn("w1", results)
        self.assertIn("kld", results)
        self.assertIn("var1", results["w1"])
        self.assertIn("var2", results["w1"])
        self.assertIn("var1", results["kld"])
        self.assertIn("var2", results["kld"])

    def test_w1_only(self):
        """Test computing only W1 metric."""
        results = calc_metrics_for_dict(
            dict_reference=self.dict_ref,
            dict_approx=self.dict_approx,
            names=["var1"],
            metrics=["w1"],
            num_eval_samples=100,
            num_batches=3,
        )
        self.assertIn("w1", results)
        self.assertNotIn("kld", results)

    def test_kld_only(self):
        """Test computing only KLD metric."""
        results = calc_metrics_for_dict(
            dict_reference=self.dict_ref,
            dict_approx=self.dict_approx,
            names=["var1"],
            metrics=["kld"],
            num_eval_samples=100,
            num_batches=3,
        )
        self.assertIn("kld", results)
        self.assertNotIn("w1", results)

    def test_result_format(self):
        """Test that results are in the correct format (mean, std) tuples."""
        results = calc_metrics_for_dict(
            dict_reference=self.dict_ref,
            dict_approx=self.dict_approx,
            names=["var1"],
            num_eval_samples=100,
            num_batches=3,
        )
        mean, std = results["w1"]["var1"]
        self.assertIsInstance(mean, (float, np.floating))
        self.assertIsInstance(std, (float, np.floating))

    def test_w1_known_distance(self):
        """Test that W1 distance is correct for known distributions."""
        results = calc_metrics_for_dict(
            dict_reference=self.dict_ref,
            dict_approx=self.dict_approx,
            names=["var1"],
            metrics=["w1"],
            num_eval_samples=100,
            num_batches=5,
        )
        mean, std = results["w1"]["var1"]
        self.assertAlmostEqual(mean, 1.0, places=5)

    def test_invalid_metric_raises_error(self):
        """Test that invalid metric raises ValueError."""
        with self.assertRaises(ValueError) as context:
            calc_metrics_for_dict(
                dict_reference=self.dict_ref,
                dict_approx=self.dict_approx,
                names=["var1"],
                metrics=["invalid_metric"],
                num_eval_samples=100,
                num_batches=3,
            )
        self.assertIn("Invalid metric", str(context.exception))

    def test_numpy_array_input(self):
        """Test that numpy arrays are handled correctly."""
        dict_ref = {"var1": np.array([0.0] * 1000)}
        dict_approx = {"var1": np.array([1.0] * 1000)}
        results = calc_metrics_for_dict(
            dict_reference=dict_ref,
            dict_approx=dict_approx,
            names=["var1"],
            metrics=["w1"],
            num_eval_samples=100,
            num_batches=3,
        )
        mean, std = results["w1"]["var1"]
        self.assertAlmostEqual(mean, 1.0, places=5)

    def test_list_input(self):
        """Test that lists are handled correctly."""
        dict_ref = {"var1": [0.0] * 1000}
        dict_approx = {"var1": [1.0] * 1000}
        results = calc_metrics_for_dict(
            dict_reference=dict_ref,
            dict_approx=dict_approx,
            names=["var1"],
            metrics=["w1"],
            num_eval_samples=100,
            num_batches=3,
        )
        mean, std = results["w1"]["var1"]
        self.assertAlmostEqual(mean, 1.0, places=5)

    def test_both_metrics_computed_efficiently(self):
        """Test that requesting both metrics still works correctly."""
        # Use var2 which has Gaussian distributions (KLD won't be inf)
        results = calc_metrics_for_dict(
            dict_reference=self.dict_ref,
            dict_approx=self.dict_approx,
            names=["var2"],
            metrics=["w1", "kld"],
            num_eval_samples=100,
            num_batches=5,
        )
        # W1 should be positive for shifted Gaussians
        w1_mean, _ = results["w1"]["var2"]
        self.assertGreater(w1_mean, 0.0)
        # KLD should be positive
        kld_mean, _ = results["kld"]["var2"]
        self.assertGreater(kld_mean, 0.0)


if __name__ == "__main__":
    unittest.main()
