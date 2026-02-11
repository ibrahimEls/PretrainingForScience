from pathlib import Path

import awkward as ak
import fastjet
import numpy as np
import omnilearned
import scipy
import torch
import vector
from scipy.stats import wasserstein_distance
from tqdm import tqdm

from omnilearn_lightning.array_utils import np_to_ak, p4s_from_ptetaphimass

vector.register_awkward()

# Jet type definitions
jet_types_dict_jetclass = {
    0: {"label": "QCD", "tex_label": "$q/g$", "color": "C0"},
    1: {"label": "Hbb", "tex_label": "$H\\rightarrow b\\bar{b}$", "color": "C1"},
    2: {"label": "Hcc", "tex_label": "$H\\rightarrow c\\bar{c}$", "color": "C2"},
    3: {"label": "Hgg", "tex_label": "$H\\rightarrow gg$", "color": "C3"},
    4: {"label": "H4q", "tex_label": "$H\\rightarrow 4q$", "color": "C4"},
    5: {"label": "Hlnuqq", "tex_label": "$H\\rightarrow \\ell\\nu qq'$", "color": "C5"},
    6: {"label": "Zqq", "tex_label": "$Z\\rightarrow q\\bar{q}$", "color": "C6"},
    7: {"label": "Wqq", "tex_label": "$W\\rightarrow qq'$", "color": "C7"},
    8: {"label": "Tbqq", "tex_label": "$t\\rightarrow bqq'$", "color": "C8"},
    9: {"label": "Tbl", "tex_label": "$t\\rightarrow b\\ell\\nu$", "color": "C9"},
}

jet_types_dict_jetnet = {
    0: {"label": "gluon", "tex_label": "$g$", "color": "C0"},
    1: {"label": "light_quark", "tex_label": "$q$", "color": "C1"},
    2: {"label": "top", "tex_label": "$t$", "color": "C2"},
    3: {"label": "W", "tex_label": "$W$", "color": "C3"},
    4: {"label": "Z", "tex_label": "$Z$", "color": "C4"},
}


def get_numerical_label_from_name(jet_type_name: str, dataset_type: str) -> int:
    if dataset_type == "jetclass":
        jet_types_dict = jet_types_dict_jetclass
    elif "jetnet" in dataset_type:
        jet_types_dict = jet_types_dict_jetnet
    else:
        raise ValueError(
            f"Invalid dataset_type '{dataset_type}'. Must be 'jetclass' or 'jetnet'."
        )
    for label, info in jet_types_dict.items():
        if info["label"] == jet_type_name:
            return label
    raise ValueError(
        f"Jet type name '{jet_type_name}' not found in dataset '{dataset_type}'."
    )


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


def calc_deltaR(particles, jet):
    jet = ak.unflatten(ak.flatten(jet), counts=1)
    return particles.deltaR(jet)


class JetSubstructure:
    """Class to calculate and store the jet substructure variables.

    Definitions as in slide 7 here:
    https://indico.cern.ch/event/760557/contributions/3262382/attachments/1796645/2929179/lltalk.pdf
    """

    def __init__(
        self,
        particles,
        R=0.8,
        beta=1.0,
        use_wta_pt_scheme=False,
    ):
        """Run the jet clustering and calculate the substructure variables. The clustering is
        performed with the kt algorithm and the WTA pt scheme.

        Parameters
        ----------
        particles : awkward array
            The particles that are clustered into jets. Have to be vector Momentum4D objects
        R : float, optional
            The jet radius, by default 0.8
        beta : float, optional
            The beta parameter for N-subjettiness, by default 1.0
        use_wta_pt_scheme : bool, optional
            Whether to use the WTA pt scheme for the clustering, by default False
        """

        print(f"Calculating substructure for {len(particles)} jets")
        mask_too_few_particles = ak.num(particles) < 3
        n_jets_with_nparticles_too_small = ak.sum(mask_too_few_particles)
        if n_jets_with_nparticles_too_small > 0:
            print(
                f"There are {n_jets_with_nparticles_too_small} jets with less than 3 particles."
            )
            raise ValueError("Jets with too few particles are not allowed.")

        if ak.max(particles.pt) > 1e9:
            print(
                "The pt of (one or more) particles is larger than 1e9. We will clip the pt to 1e9. "
            )
            # particles = ak_select_and_preprocess(
            #     particles,
            #     pp_dict={
            #         "pt": {"clip_min_input_space": 0, "clip_max_input_space": 1e9},
            #         "eta": None,
            #         "phi": None,
            #         "mass": None,
            #     },
            # )
            particles = p4s_from_ptetaphimass(
                particles,
                field_name_pt="pt",
                field_name_eta="eta",
                field_name_phi="phi",
                field_name_mass="mass",
            )

        self.R = R
        self.beta = beta
        self.particles = particles
        self.particles_sum = ak.sum(particles, axis=1)
        self.jet_mass = self.particles_sum.mass
        self.jet_pt = self.particles_sum.pt
        self.jet_eta = self.particles_sum.eta
        self.jet_phi = self.particles_sum.phi
        self.jet_n_constituents = ak.num(particles)

        if use_wta_pt_scheme:
            jetdef = fastjet.JetDefinition(
                fastjet.kt_algorithm, self.R, fastjet.WTA_pt_scheme
            )
        else:
            jetdef = fastjet.JetDefinition(fastjet.kt_algorithm, self.R)
        print("Clustering jets with fastjet")
        print(f"Jet definition: {jetdef}")
        self.cluster = fastjet.ClusterSequence(particles, jetdef)
        self.inclusive_jets = self.cluster.inclusive_jets()
        self.exclusive_jets_1 = self.cluster.exclusive_jets(n_jets=1)
        self.exclusive_jets_2 = self.cluster.exclusive_jets(n_jets=2)
        self.exclusive_jets_3 = self.cluster.exclusive_jets(n_jets=3)

        print("Calculating N-subjettiness")
        self._calc_d0()
        self._calc_tau1()
        self._calc_tau2()
        self._calc_tau3()
        self.tau21 = self.tau2 / self.tau1
        self.tau32 = self.tau3 / self.tau2
        print("Calculating D2")
        # D2 as defined in https://arxiv.org/pdf/1409.6298.pdf
        self.d2 = self.cluster.exclusive_jets_energy_correlator(njets=1, func="d2")

    def _calc_d0(self):
        """Calculate the d0 values."""
        self.d0 = ak.sum(self.particles.pt * self.R**self.beta, axis=1)

    def _calc_tau1(self):
        """Calculate the tau1 values."""
        self.delta_r_1i = calc_deltaR(self.particles, self.exclusive_jets_1[:, :1])
        self.pt_i = self.particles.pt
        # calculate the tau1 values
        self.tau1 = ak.sum(self.pt_i * self.delta_r_1i**self.beta, axis=1) / self.d0

    def _calc_tau2(self):
        """Calculate the tau2 values."""
        delta_r_1i = calc_deltaR(self.particles, self.exclusive_jets_2[:, :1])
        delta_r_2i = calc_deltaR(self.particles, self.exclusive_jets_2[:, 1:2])
        self.pt_i = self.particles.pt
        # add new axis to make it broadcastable
        min_delta_r = ak.min(
            ak.concatenate(
                [
                    delta_r_1i[..., np.newaxis] ** self.beta,
                    delta_r_2i[..., np.newaxis] ** self.beta,
                ],
                axis=-1,
            ),
            axis=-1,
        )
        self.tau2 = ak.sum(self.pt_i * min_delta_r, axis=1) / self.d0

    def _calc_tau3(self):
        """Calculate the tau3 values."""
        delta_r_1i = calc_deltaR(self.particles, self.exclusive_jets_3[:, :1])
        delta_r_2i = calc_deltaR(self.particles, self.exclusive_jets_3[:, 1:2])
        delta_r_3i = calc_deltaR(self.particles, self.exclusive_jets_3[:, 2:3])
        self.pt_i = self.particles.pt
        min_delta_r = ak.min(
            ak.concatenate(
                [
                    delta_r_1i[..., np.newaxis] ** self.beta,
                    delta_r_2i[..., np.newaxis] ** self.beta,
                    delta_r_3i[..., np.newaxis] ** self.beta,
                ],
                axis=-1,
            ),
            axis=-1,
        )
        self.tau3 = ak.sum(self.pt_i * min_delta_r, axis=1) / self.d0

    def get_substructure_as_ak_array(self):
        """Return the substructure variables as a dictionary."""
        return ak.Array(
            {
                "tau1": self.tau1,
                "tau2": self.tau2,
                "tau3": self.tau3,
                "tau21": self.tau21,
                "tau32": self.tau32,
                "d2": self.d2,
                "jet_mass": self.jet_mass,
                "jet_pt": self.jet_pt,
                "jet_eta": self.jet_eta,
                "jet_phi": self.jet_phi,
                "jet_n_constituents": self.jet_n_constituents,
            }
        )


def get_batch_and_generate(
    batch,
    lightning_model=None,
    feature_names_x=None,
    verbose=True,
    n_sampling_steps=128,
    set_pid_to_zeros=False,
    set_add_info_to_zeros=False,
):
    X, y = (
        batch["X"].float().to(lightning_model.device),
        batch["y"].to(lightning_model.device),
    )
    model_kwargs = {
        k: (batch[k].to(lightning_model.device) if batch[k] is not None else None)
        for k in ("cond", "pid", "add_info")
        if (k in batch)
    }
    if verbose:
        print(f"X.shape: {X.shape}")
        print(f"y.shape: {y.shape}")
        for k, v in model_kwargs.items():
            if v is not None:
                print(f"{k}.shape: {v.shape}")
    mask = X[:, :, 0] != 0.0

    # if "cond" is not in batch, then create the multiplicity entry
    if model_kwargs.get("cond", None) is None:
        multiplicity = torch.sum(mask, dim=1).float() / 100.0
        model_kwargs["cond"] = multiplicity.unsqueeze(-1)

    # set pid and add_info to zeros
    if "pid" in model_kwargs and set_pid_to_zeros:
        # print(f"Setting pid to zeros with shape: {model_kwargs['pid'].shape}")
        model_kwargs["pid"] = torch.zeros_like(model_kwargs["pid"])
    if "add_info" in model_kwargs and set_add_info_to_zeros:
        # print(f"Setting add_info to zeros with shape: {model_kwargs['add_info'].shape}")
        model_kwargs["add_info"] = torch.zeros_like(model_kwargs["add_info"])

    with torch.no_grad():
        gen_output = omnilearned.diffusion.generate(
            lightning_model.model,
            y=y,
            shape=X.shape,
            nsteps=n_sampling_steps,
            **model_kwargs,
        )

    x_ak_original = np_to_ak(
        X.cpu().numpy(), names=feature_names_x.keys(), mask=mask.cpu().numpy()
    )
    x_ak_generated = np_to_ak(
        gen_output.cpu().numpy(), names=feature_names_x.keys(), mask=mask.cpu().numpy()
    )

    return x_ak_original, x_ak_generated


def generate_and_postprocess(
    n_batches,
    dataloader,
    **gen_kwargs,
):
    print(f"Generating {n_batches} batches...")
    n_batches_generated = 0
    original = []
    generated = []
    y_values = []

    for batch_i in tqdm(dataloader, total=n_batches):
        if n_batches_generated >= n_batches:
            break
        n_batches_generated += 1
        output = get_batch_and_generate(
            batch=batch_i,
            **gen_kwargs,
        )
        original.append(output[0])
        generated.append(output[1])
        y_values.append(batch_i["y"].cpu().numpy())

    # concatenate the lists into a single ak.Array per n_steps
    # n_steps=0 corresponds to the original data
    x_ak_dict = {
        "original": None,
        "generated": None,
    }
    x_ak_dict["original"] = ak.concatenate(original, axis=0)
    x_ak_dict["generated"] = ak.concatenate(generated, axis=0)

    # convert to p4s, based on pt, eta, phi, mass=0
    p4_conversion_kwargs = dict(
        field_name_eta="eta",
        field_name_phi="phi",
        field_name_pt="log_pt",
        pt_is_log=True,
    )
    p4s_dict = {
        "original": p4s_from_ptetaphimass(
            x_ak_dict["original"], **p4_conversion_kwargs
        ),
        "generated": p4s_from_ptetaphimass(
            x_ak_dict["generated"], **p4_conversion_kwargs
        ),
    }
    # calculate sum of constituent p4s = jet p4s
    p4s_sum_dict = {
        "original": ak.sum(p4s_dict["original"], axis=1),
        "generated": ak.sum(p4s_dict["generated"], axis=1),
    }
    # calculate jet substructure (subjettiness ratios etc)
    substructure_calculator_original = JetSubstructure(p4s_dict["original"])
    substructure_calculator_generated = JetSubstructure(p4s_dict["generated"])

    substructure_dict = {
        "original": (substructure_calculator_original.get_substructure_as_ak_array()),
        "generated": (substructure_calculator_generated.get_substructure_as_ak_array()),
    }

    return ak.Array(
        {
            "x_ak": x_ak_dict,
            "p4s": p4s_dict,
            "p4s_sum": p4s_sum_dict,
            "substructure": substructure_dict,
            "y": ak.concatenate(y_values, axis=0),
        }
    )


def save_gen_output(gen_output: ak.Array, output_path: str | Path) -> None:
    """Save generated jet output to parquet file.

    Parameters
    ----------
    gen_output : ak.Array
        The output from generate_and_postprocess containing x_ak, substructure, y, etc.
    output_path : str | Path
        Path to save the parquet file (should end with .parquet)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ak.to_parquet(gen_output, output_path)
    print(f"Saved generated jets to {output_path}")


def load_gen_output(output_path: str | Path) -> ak.Array:
    """Load generated jet output from parquet file.

    Parameters
    ----------
    output_path : str | Path
        Path to the parquet file

    Returns
    -------
    ak.Array
        The loaded generation output
    """
    output_path = Path(output_path)
    gen_output = ak.from_parquet(output_path)
    print(f"Loaded generated jets from {output_path}")
    return gen_output


def generate_or_load(
    output_path: str | Path,
    lightning_model,
    gen_kwargs: dict,
    set_pid_to_zeros: bool = False,
    set_add_info_to_zeros: bool = False,
    force_regenerate: bool = False,
) -> ak.Array:
    """Generate jets or load from cache if already exists.

    Parameters
    ----------
    output_path : str | Path
        Path to save/load the parquet file
    lightning_model : PETLightning
        The model to use for generation (only used if file doesn't exist)
    gen_kwargs : dict
        Keyword arguments for generate_and_postprocess
    set_pid_to_zeros : bool
        Whether to zero out PID features
    set_add_info_to_zeros : bool
        Whether to zero out additional info features
    force_regenerate : bool
        If True, regenerate even if file exists

    Returns
    -------
    ak.Array
        The generation output (either loaded or freshly generated)
    """
    output_path = Path(output_path)

    if output_path.exists() and not force_regenerate:
        return load_gen_output(output_path)

    print(f"Generating jets (will save to {output_path})...")
    gen_output = generate_and_postprocess(
        lightning_model=lightning_model,
        **gen_kwargs,
        set_pid_to_zeros=set_pid_to_zeros,
        set_add_info_to_zeros=set_add_info_to_zeros,
    )
    save_gen_output(gen_output, output_path)
    return gen_output
