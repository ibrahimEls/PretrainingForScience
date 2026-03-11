import awkward as ak
import numpy as np
import scipy
import vector
from jetnet.evaluation import w1efp, w1m, w1p
from scipy.stats import wasserstein_distance

from omnilearn_lightning.array_utils import p4s_to_jetnet_eval_np_stack

vector.register_awkward()


def get_jetnet_default_metrics(
    p4s_ref: ak.Array,
    p4s_gen: ak.Array,
    num_eval_samples: int,
    num_batches: int,
    use_mask: bool = True,
):
    """Calculate default JetNet evaluation metrics for ref and generated jets.

    Expects p4s_ref and p4s_gen to be awkward arrays containing the 4-momenta of
    the jets, with fields corresponding to pt, eta, phi, and mass.

    For the metrics calculation, the pT of the particles is divided by the jet pT
    and the masses of the particles are set to zero, following the standard
    JetNet evaluation procedure.

    Parameters
    ----------
    p4s_ref : ak.Array
        Reference jets (e.g., real jets) as an awkward array of 4-momenta.
    p4s_gen : ak.Array
        Generated jets (e.g., from a model) as an awkward array of 4-momenta.
    num_eval_samples : int
        Number of jets per batch in the batched evaluation.
    num_batches : int
        Number of batches for evaluation.
    use_mask : bool
        Whether to use a mask for selecting specific jets. Default is True.

    """
    # Convert to numpy arrays for metric calculations
    p4s_ref_np, mask_ref_np = p4s_to_jetnet_eval_np_stack(
        p4s_ref, divide_pt_by_sum_pt=True
    )
    p4s_gen_np, mask_gen_np = p4s_to_jetnet_eval_np_stack(
        p4s_gen, divide_pt_by_sum_pt=True
    )
    mask_ref_np = ak.values_astype(mask_ref_np, int).to_numpy()
    mask_gen_np = ak.values_astype(mask_gen_np, int).to_numpy()

    # w1p output is of shape [(num_features,), (num_features,)]
    # with the first element being the mean and the second element being the std of
    # the W1 distance across the features.
    eval_kwargs = dict(num_eval_samples=num_eval_samples, num_batches=num_batches)
    w1p_ref_ref = w1p(
        jets1=p4s_ref_np,
        jets2=p4s_ref_np,
        mask1=mask_ref_np if use_mask else None,
        mask2=mask_ref_np if use_mask else None,
        **eval_kwargs,
    )
    w1p_ref_gen = w1p(
        jets1=p4s_ref_np,
        jets2=p4s_gen_np,
        mask1=mask_ref_np if use_mask else None,
        mask2=mask_gen_np if use_mask else None,
        **eval_kwargs,
    )

    # only take the first 3 features (pt, eta, phi) as the mass is not used in our
    # evaluation and would result misleadingly small values (due to perfect
    # agreement in the mass dimension from setting all particle masses to zero)
    w1p_mean_ref_ref = np.mean(w1p_ref_ref[0][:3])
    w1p_mean_ref_gen = np.mean(w1p_ref_gen[0][:3])
    w1p_std_ref_ref = np.mean(w1p_ref_ref[1][:3])
    w1p_std_ref_gen = np.mean(w1p_ref_gen[1][:3])

    # w1m returns tuple of (mean, std) for the jet mass
    w1m_ref_ref = w1m(p4s_ref_np, p4s_ref_np, **eval_kwargs)
    w1m_ref_gen = w1m(p4s_ref_np, p4s_gen_np, **eval_kwargs)

    # w1efp returns [(num_efps,), (num_efps,)] with the mean and std of the W1
    # distance across the EFPs
    w1efp_ref_ref = w1efp(p4s_ref_np, p4s_ref_np, **eval_kwargs)
    w1efp_ref_gen = w1efp(p4s_ref_np, p4s_gen_np, **eval_kwargs)

    return {
        "w1p_baseline": (
            w1p_mean_ref_ref,
            w1p_std_ref_ref,
        ),
        "w1p": (w1p_mean_ref_gen, w1p_std_ref_gen),
        "w1m_baseline": w1m_ref_ref,
        "w1m": w1m_ref_gen,
        "w1efp_baseline": (
            np.mean(w1efp_ref_ref[0]),
            np.mean(w1efp_ref_ref[1]),
        ),
        "w1efp": (
            np.mean(w1efp_ref_gen[0]),
            np.mean(w1efp_ref_gen[1]),
        ),
    }


def distribution_metrics_batched(
    data1: ak.Array,
    data2: ak.Array,
    num_eval_samples: int,
    num_batches: int,
    n_bins: int = 30,
    replace: bool = True,
    return_zero_if_nan_or_inf: bool = False,
):
    """Calculate W1 distance and KL divergence between two datasets with batched sampling.

    Computes both metrics using the same random samples for consistency and efficiency.

    Args:
        data1 (ak.Array): Reference data (e.g., real jets).
        data2 (ak.Array): Approximation data (e.g., generated jets).
        num_eval_samples (int): Number of samples to use for each metric calculation.
            If set to `None`, uses all available samples (which will not allow
            for mean/std estimation).
        num_batches (int): Number of batches for computing mean and std.
        n_bins (int): Number of bins for the quantiled KL divergence. Default is 30.
        replace (bool): Whether to sample with replacement. Default is True.
        return_zero_if_nan_or_inf (bool): If True, return 0 for NaN or inf KL values.

    Returns:
        dict: Dictionary with keys "w1" and "kld", each containing a tuple (mean, std).
    """
    w1_values = []
    kl_values = []
    rng = np.random.default_rng(seed=42)

    if data1.ndim != data2.ndim:
        raise ValueError(
            "data1 and data2 must have the same number of dimensions, "
            f"got {data1.ndim} and {data2.ndim}"
        )

    if num_eval_samples is None and num_batches > 1:
        raise ValueError(
            "num_eval_samples cannot be None when num_batches > 1, as this would "
            "prevent mean/std estimation."
        )

    for _ in range(num_batches):
        if num_eval_samples is None:
            rand_sample1 = data1
            rand_sample2 = data2
        else:
            rand1 = rng.choice(len(data1), size=num_eval_samples, replace=replace)
            rand2 = rng.choice(len(data2), size=num_eval_samples, replace=replace)
            rand_sample1 = data1[rand1]
            rand_sample2 = data2[rand2]

        # flatten in case the data is not 1D
        if rand_sample1.ndim > 1:
            rand_sample1 = ak.flatten(rand_sample1)
            rand_sample2 = ak.flatten(rand_sample2)

        # Convert to numpy for metric calculations
        sample1_np = np.array(rand_sample1)
        sample2_np = np.array(rand_sample2)

        # Compute W1 distance
        w1_values.append(wasserstein_distance(sample1_np, sample2_np))

        # Compute KL divergence
        kl = quantiled_kl_divergence(
            sample_ref=sample1_np,
            sample_approx=sample2_np,
            n_bins=n_bins,
            return_zero_if_nan_or_inf=return_zero_if_nan_or_inf,
        )
        kl_values.append(kl)

    return {
        "w1": (np.mean(w1_values), np.std(w1_values)),
        "kld": (np.mean(kl_values), np.std(kl_values)),
    }


def quantiled_kl_divergence(
    sample_ref: np.ndarray,
    sample_approx: np.ndarray,
    n_bins: int = 30,
    return_bin_edges=False,
    return_zero_if_nan_or_inf=False,
):
    """Calculate the KL divergence using quantiles on sample_ref to define the bounds.

    Parameters
    ----------
    sample_ref : np.ndarray
        The first sample to compare (this is the reference, so in the context of
        jet generation, those are the real jets).
    sample_approx : np.ndarray
        The second sample to compare (this is the model/approximation, so in the
        context of jet generation, those are the generated jets).
    n_bins : int
        The number of bins to use for the histogram. Those bins are defined by
        equiprobably quantiles of sample_ref.
    return_bin_edges : bool, optional
        If True, return the bins used to calculate the KL divergence.
    return_zero_if_nan_or_inf : bool, optional
        If True, return 0 if the KL divergence is NaN or inf. Default is False.
    """
    bin_edges = np.quantile(sample_ref, np.linspace(0.0, 1.0, n_bins + 1))
    bin_edges[0] = float("-inf")
    bin_edges[-1] = float("inf")
    pk = np.histogram(sample_ref, bin_edges)[0] / len(sample_ref)
    qk = np.histogram(sample_approx, bin_edges)[0] / len(sample_approx)
    kl = scipy.stats.entropy(pk, qk)
    # check if kl is nan or inf, if it is, return 0
    if return_zero_if_nan_or_inf and (np.isnan(kl) or np.isinf(kl)):
        kl = 0
    if return_bin_edges:
        return kl, bin_edges
    return kl


def calc_quantiled_kl_divergence_for_dict(
    dict_reference: dict,
    dict_approx: dict,
    names: list,
    n_bins: int = 30,
    return_zero_if_nan_or_inf=False,
):
    """Calculate the quantiled KL divergence for two dictionaries of samples.

    Parameters
    ----------
    dict_reference : dict
        The first dictionary of samples.
    dict_approx : dict
        The second dictionary of samples.
    names : list
        The names of the samples to compare. All names must be included in both dicts.
    return_zero_if_nan_or_inf : bool, optional
        If True, return 0 if the KL divergence is NaN or inf. Default is False
    """
    # loop over the names and calculate the quantiled kld for each name
    klds = {}
    for name in names:
        klds[name] = quantiled_kl_divergence(
            sample_ref=np.array(dict_reference[name]),
            sample_approx=np.array(dict_approx[name]),
            n_bins=n_bins,
            return_zero_if_nan_or_inf=return_zero_if_nan_or_inf,
        )
    return klds


def calc_metrics_for_dict(
    dict_reference: dict,
    dict_approx: dict,
    names: list,
    metrics: list[str] = None,
    num_eval_samples: int = 10000,
    num_batches: int = 10,
    n_bins: int = 30,
    replace: bool = True,
    return_zero_if_nan_or_inf: bool = False,
):
    """Calculate distribution metrics (W1 and/or KLD) for two dictionaries of samples.

    This function computes batched metrics with mean and standard deviation estimates
    for comparing distributions across multiple named variables.

    Parameters
    ----------
    dict_reference : dict
        Dictionary of reference samples (e.g., real data).
    dict_approx : dict
        Dictionary of approximation samples (e.g., generated data).
    names : list
        Names of the variables to compare. All names must exist in both dicts.
    metrics : list[str], optional
        List of metrics to compute. Options are "w1" (Wasserstein-1 distance) and
        "kld" (KL divergence). Default is ["w1", "kld"].
    num_eval_samples : int, optional
        Number of samples per batch for metric calculation. Default is 10000.
    num_batches : int, optional
        Number of batches for computing mean and std. Default is 10.
    n_bins : int, optional
        Number of bins for KL divergence calculation. Default is 30.
    replace : bool, optional
        Whether to sample with replacement. Default is True.
    return_zero_if_nan_or_inf : bool, optional
        If True, return 0 for NaN or inf KL values. Default is False.

    Returns
    -------
    dict
        Dictionary with structure {metric: {name: (mean, std), ...}, ...}
        where metric is "w1" or "kld" and name is the variable name.
    """
    if metrics is None:
        metrics = ["w1", "kld"]

    valid_metrics = {"w1", "kld"}
    for m in metrics:
        if m not in valid_metrics:
            raise ValueError(f"Invalid metric '{m}'. Must be one of {valid_metrics}.")

    # Initialize results structure
    results = {metric: {} for metric in metrics}

    for name in names:
        data1 = dict_reference[name]
        data2 = dict_approx[name]

        # Convert to ak.Array if not already
        if not isinstance(data1, ak.Array):
            data1 = ak.Array(data1)
        if not isinstance(data2, ak.Array):
            data2 = ak.Array(data2)

        # Compute both metrics in one pass
        batch_results = distribution_metrics_batched(
            data1=data1,
            data2=data2,
            num_eval_samples=num_eval_samples,
            num_batches=num_batches,
            n_bins=n_bins,
            replace=replace,
            return_zero_if_nan_or_inf=return_zero_if_nan_or_inf,
        )

        # Store only the requested metrics
        for metric in metrics:
            results[metric][name] = batch_results[metric]

    return results
