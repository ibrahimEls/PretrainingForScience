from pathlib import Path

import awkward as ak
import fastjet
import numpy as np
import omnilearned
import torch
import vector
from scipy.stats import wasserstein_distance
from tqdm import tqdm

from omnilearn_lightning.array_utils import np_to_ak, p4s_from_ptetaphimass

vector.register_awkward()


def wasserstein_distance_batched(
    data1: ak.Array,
    data2: ak.Array,
    num_eval_samples: int,
    num_batches: int,
    replace: bool = True,
):
    """Calculate the Wasserstein distance between two datasets multiple times and return mean and
    std.

    Args:
        data1 (ak.Array): Data1 usually the real data, can be num_batches times smaller than data2
        data2 (ak.Array): Data2 usually the generated data, can be num_batches times larger than data1
        num_eval_samples (int): Number of samples to use for each Wasserstein distance calculation
        num_batches (int): Number of batches to split the data into
        replace (bool): Whether to sample with replacement. Default is True.

    Returns:
        float: Mean Wasserstein distance of all batches
        float: Standard deviation of the Wasserstein distances of all batches
    """
    w1 = []
    rng = np.random.default_rng(seed=42)
    if data1.ndim != data2.ndim:
        raise ValueError(
            f"data1 and data2 must have the same number of dimensions, got {data1.ndim} and {data2.ndim}"
        )
    for _ in range(num_batches):
        rand1 = rng.choice(len(data1), size=num_eval_samples, replace=replace)
        rand2 = rng.choice(len(data2), size=num_eval_samples, replace=replace)
        rand_sample1 = data1[rand1]
        rand_sample2 = data2[rand2]
        # flatten in case the data is not 1D
        if rand_sample1.ndim > 1:
            rand_sample1 = ak.flatten(rand_sample1)
            rand_sample2 = ak.flatten(rand_sample2)

        w1.append(wasserstein_distance(rand_sample1, rand_sample2))

    return np.mean(w1), np.std(w1)


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
