import awkward as ak
import fastjet
import numpy as np
import omnilearned
import torch
import vector
from tqdm import tqdm

from omnilearn_lightning.array_utils import np_to_ak, p4s_from_ptetaphimass

vector.register_awkward()


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
        }
    )
