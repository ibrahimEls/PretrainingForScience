import awkward as ak
import numpy as np
import torch
import vector

vector.register_awkward()


def preprocess_tensor(
    x: torch.Tensor,
    add_info: torch.Tensor | None = None,
    inverse: bool = False,
    scale_factors_x: torch.Tensor | None = None,
    scale_factors_add_info: torch.Tensor | None = None,
):
    """Preprocess tensors by scaling/unscaling and concatenating features.

    Parameters
    ----------
    x : torch.Tensor
        Tensor of shape (batch_size, num_points, num_features_x).
    add_info : torch.Tensor or None
        Optional tensor of shape (batch_size, num_points, num_features_add_info).
        If ``None`` or if ``scale_factors_add_info`` is ``None``, only ``x`` is processed.
    inverse : bool, optional
        If True, multiply by scale factors (inverse transform). If False, divide by scale factors
        (forward normalization). Default is False.
    scale_factors_x : torch.Tensor, optional
        Scale factors for ``x`` with shape (num_features_x,). Required.
    scale_factors_add_info : torch.Tensor, optional
        Scale factors for ``add_info`` with shape (num_features_add_info,). If ``None``,
        ``add_info`` is ignored and only ``x`` is processed.

    Returns
    -------
    torch.Tensor
        If both ``x`` and ``add_info`` are processed, returns a combined tensor of shape
        (batch_size, num_points, num_features_x + num_features_add_info). Otherwise returns
        only the processed ``x`` tensor with shape (batch_size, num_points, num_features_x).

    Raises
    ------
    ValueError
        If required scale factors are missing or shapes are incompatible.
    """

    # Validate and process x
    if scale_factors_x is None:
        raise ValueError("scale_factors_x must be provided")
    if x.dim() != 3:
        raise ValueError(
            "x must be a 3D tensor of shape (batch_size, num_points, num_features)"
        )
    if scale_factors_x.shape[-1] != x.shape[-1]:
        raise ValueError(
            f"scale_factors_x last dimension must match x features: {scale_factors_x.shape[-1]} != {x.shape[-1]}"
        )
    sfx = torch.as_tensor(scale_factors_x, device=x.device, dtype=x.dtype)
    x_proc = x * sfx if inverse else x / sfx

    # If add_info or its scale factors are missing, return only processed x
    if add_info is None or scale_factors_add_info is None:
        return x_proc

    # Otherwise, validate and process add_info and concatenate
    if add_info.dim() != 3:
        raise ValueError(
            "add_info must be a 3D tensor of shape (batch_size, num_points, num_features)"
        )
    if scale_factors_add_info.shape[-1] != add_info.shape[-1]:
        raise ValueError(
            "scale_factors_add_info last dimension must match add_info features: "
            f"{scale_factors_add_info.shape[-1]} != {add_info.shape[-1]}"
        )
    sfa = torch.as_tensor(
        scale_factors_add_info, device=add_info.device, dtype=add_info.dtype
    )
    add_proc = add_info * sfa if inverse else add_info / sfa

    return torch.cat([x_proc, add_proc], dim=-1)


def p4s_from_ptetaphimass(
    ak_arr,
    field_name_pt="part_pt",
    field_name_eta="part_etarel",
    field_name_phi="part_phirel",
    field_name_mass="part_mass",
    pt_is_log=False,
):
    """Create a Momentum4D array from pt, eta, phi, mass fields in an awkward array.

    Parameters
    ----------
    ak_arr : ak.Array
        Array with fields part_pt, part_etarel, part_phirel, part_mass.
    field_name_pt : str, optional
        Name of the field containing the transverse momentum, by default "part_pt".
    field_name_eta : str, optional
        Name of the field containing the pseudorapidity, by default "part_etarel".
    field_name_phi : str, optional
        Name of the field containing the azimuthal angle, by default "part_phirel".
    field_name_mass : str, optional
        Name of the field containing the mass, by default
    pt_is_log : bool, optional
        Whether the pt field is in log scale, by default False. If True, the pt
        values are exponentiated before creating the Momentum4D array.
    """
    return ak.zip(
        {
            "pt": ak_arr[field_name_pt]
            if not pt_is_log
            else np.exp(ak_arr[field_name_pt]),
            "eta": ak_arr[field_name_eta],
            "phi": ak_arr[field_name_phi],
            "mass": ak_arr[field_name_mass]
            if field_name_mass in ak_arr.fields
            else ak.zeros_like(ak_arr[field_name_pt]),
        },
        with_name="Momentum4D",
    )


def ak_pad(x: ak.Array, maxlen: int, axis: int = 1, fill_value=0, return_mask=False):
    """Function to pad an awkward array to a specified length. The array is padded along the
    specified axis.

    Parameters
    ----------
    x : awkward array
        Array to pad.
    maxlen : int
        Length to pad to.
    axis : int, optional
        Axis along which to pad. Default is 1.
    fill_value : float or int, optional
        Value to use for padding. Default is 0.
    return_mask : bool, optional
        If True, also return a mask array indicating which values are padded.
        Default is False.
        If the input array has fields, the mask is created from the first field.

    Returns
    -------
    awkward array
        Padded array.
    mask : awkward array
        Mask array indicating which values are padded. Only returned if return_mask is True.
    """
    padded_x = ak.fill_none(ak.pad_none(x, maxlen, axis=axis, clip=True), fill_value)
    if return_mask:
        if len(x.fields) >= 1:
            mask = ak.ones_like(x[x.fields[0]], dtype="bool")
        else:
            mask = ak.ones_like(x, dtype="bool")
        mask = ak.fill_none(ak.pad_none(mask, maxlen, axis=axis, clip=True), False)
        return padded_x, mask
    return padded_x


def combine_ak_arrays(*arrays):
    """Function to combine multiple awkward arrays. The arrays should have different fields.

    Parameters
    ----------
    *arrays : ak.Array
        Input arrays to combine.

    Returns
    -------
    ak.Array
        Combined array.
    """
    combined_fields = {}
    for arr in arrays:
        if arr is None:
            continue
        if set(combined_fields.keys()) & set(arr.fields):
            dict_with_fields = {f"arr{i}": arr.fields for i, arr in enumerate(arrays)}
            raise ValueError(
                "You are trying to merge multiple ak.Arrays but they have common field names. "
                f"The common field names are: {set(combined_fields.keys()) & set(arr.fields)} "
                f"The individual field names are: {dict_with_fields}"
            )
        combined_fields.update({field: arr[field] for field in arr.fields})
    return ak.Array(combined_fields)


def np_to_ak(x: np.ndarray, names: list, mask: np.ndarray = None, dtype="float32"):
    """Function to convert a numpy array and its mask to an awkward array. The features
    corresponding to the names are assumed to correspond to the last axis of the array.

    Parameters
    ----------
    x : np.ndarray
        Array to convert.
    names : list
        List of field names (corresponding to the features in x along the last dimension).
    mask : np.ndarray, optional
        Mask array. Default is None. If x is an array of shape (n, m, k), the mask should
        be of shape (n, m).
    dtype : str, optional
        Data type to convert the values to. Default is "float32".
    """

    if mask is None:
        mask = np.ones_like(x[..., 0], dtype="bool")

    return ak.Array(
        {
            name: ak.values_astype(
                ak.drop_none(ak.mask(ak.Array(x[..., i]), mask != 0)),
                dtype,
            )
            for i, name in enumerate(names)
        }
    )


def ak_to_np_stack(
    ak_array: ak.Array, names: list = None, axis: int = -1, use_attrs: bool = False
):
    """Function to convert an awkward array to a numpy array by stacking the values of the
    specified fields. This is much faster than ak.to_numpy(ak_array) for large arrays.

    Parameters
    ----------
    ak_array : awkward array
        Array to convert.
    names : list, optional
        List of field names to convert. Default is None.
    axis : int, optional
        Axis along which to stack the values. Default is -1.
    use_attrs : bool, optional
        Whether to use the attributes of the awkward array. Default is False. If True, the
        function will access the fields using getattr(ak_array, name) instead of
        ak_array[name]. This useful for e.g. Momentum4D arrays.
    """
    if names is None:
        raise ValueError("names must be specified")
    return ak.to_numpy(
        np.stack(
            [
                ak.to_numpy(
                    ak.values_astype(
                        getattr(ak_array, name) if use_attrs else ak_array[name],
                        "float32",
                    )
                )
                for name in names
            ],
            axis=axis,
        )
    )


def ak_smear(arr, sigma=0, seed=42):
    """Helper function to smear an array of values by a given sigma.

    Parameters
    ----------
    arr : awkward array
        The array to smear
    sigma : float, optional
        The sigma of the smearing, by default 0 (i.e. no smearing)
    seed : int, optional
        Seed for the random number generator, by default 42
    """
    # Convert it to a 1D numpy array and perform smearing
    numpy_arr = ak.to_numpy(arr.layout.content)

    if sigma != 0:
        rng = np.random.default_rng(seed)
        numpy_arr = rng.normal(numpy_arr, sigma)

    # Convert it back to awkward form
    return ak.Array(
        ak.contents.ListOffsetArray(arr.layout.offsets, ak.Array(numpy_arr).layout)
    )


def ak_clip(arr, clip_min=None, clip_max=None):
    """Helper function to clip the values of an array.

    Parameters
    ----------
    arr : awkward array
        The array to clip
    clip_min : float, optional
        Minimum value to clip to, by default None
    clip_max : float, optional
        Maximum value to clip to, by default None
    """
    ndim = arr.ndim
    # Convert it to a 1D numpy array and perform clipping
    if ndim > 1:
        numpy_arr = ak.to_numpy(arr.layout.content)
    else:
        numpy_arr = ak.to_numpy(arr)

    if clip_min is not None:
        numpy_arr = np.clip(numpy_arr, clip_min, None)

    if clip_max is not None:
        numpy_arr = np.clip(numpy_arr, None, clip_max)

    if ndim > 1:
        # Convert it back to awkward form
        return ak.Array(
            ak.contents.ListOffsetArray(arr.layout.offsets, ak.Array(numpy_arr).layout)
        )
    return numpy_arr


def ak_subtract(arr1, arr2):
    """Helper function to subtract two awkward arrays with names fields, i.e. arr1 - arr2

    Parameters
    ----------
    arr1 : ak.Array
        First array
    arr2 : ak.Array
        Second array

    Returns
    -------
    ak.Array
        Array with the fields of arr1 - arr2
    """

    if arr1.fields != arr2.fields:
        raise ValueError(
            "The two arrays do not have the same fields. Array 1 has fields: "
            f"{arr1.fields}, while array 2 has fields: {arr2.fields}"
        )

    if len(arr1) != len(arr2):
        raise ValueError(
            "The two arrays do not have the same length. Array 1 has length: "
            f"{len(arr1)}, while array 2 has length: {len(arr2)}"
        )

    if len(arr1.fields) == 0 or len(arr2.fields) == 0:
        raise ValueError("One or both arrays have no fields.")

    return ak.Array(
        {name: getattr(arr1, name) - getattr(arr2, name) for name in arr1.fields}
    )


def ak_mean(arr, axis=None):
    """Helper function to calculate the mean of an awkward array with field names along a certain
    axis.

    Parameters
    ----------
    arr : ak.Array
        Array to calculate the mean of
    axis : int, optional
        Axis along which to calculate the mean, by default None, which
        calculates the mean over all dimensions.

    Returns
    -------
    dict
        Dictionary with the mean of each field in the array. If the mean is still
        an awkward array, the values of the dict will be awkward arrays as well.
    """

    if not isinstance(arr, ak.Array):
        raise TypeError("Input arr must be an awkward array.")

    if axis is not None:
        if not isinstance(axis, int):
            raise TypeError("Input axis must be an integer.")
        # elif axis < 0 or axis >= arr.ndim:
        #     raise ValueError("Input axis is out of range.")

    return {name: ak.mean(getattr(arr, name), axis=axis) for name in arr.fields}


def ak_abs(arr):
    """Helper function to calculate the absolute value of an awkward array with field names.

    Parameters
    ----------
    arr : ak.Array
        Array to calculate the absolute value of

    Returns
    -------
    ak.Array
        Array with the absolute values of each field
    """

    if not isinstance(arr, ak.Array):
        raise TypeError("Input arr must be an awkward array.")
    if len(arr.fields) == 0:
        raise ValueError("Input arr has no fields.")

    return ak.Array({name: np.abs(getattr(arr, name)) for name in arr.fields})


def set_fraction_ones_to_zeros(mask: torch.Tensor, fraction: torch.Tensor):
    """Set a fraction of the ones in a mask to zero. A new mask is returned while
    the input is kept unchanged.

    Parameters
    ----------
    mask : torch.Tensor
        Mask tensor.
    fraction : float
        Fraction of ones to set to zero

    Returns
    -------
    torch.Tensor
        Modified mask tensor
    """
    mask = mask.clone()
    ones_indices = (mask == 1).nonzero(as_tuple=True)
    num_ones = len(ones_indices[0])
    num_to_zero = int(num_ones * fraction)
    indices_to_zero = torch.randperm(num_ones)[:num_to_zero]
    mask[ones_indices[0][indices_to_zero], ones_indices[1][indices_to_zero]] = 0
    return mask


def replace_masked_positions(
    x: torch.Tensor,
    mask_is_valid: torch.Tensor,
    mask_is_valid_corrupted: torch.Tensor,
    mask_is_valid_but_masked: torch.Tensor,
    vectors_to_insert: torch.Tensor,
    pos_encoding_type: str,
    pos_encoding_feature: torch.Tensor = None,
):
    supported_pos_enc_types = [
        None,
        "sort_descending_all",
        "sort_descending_in_masked_subset",
    ]
    if pos_encoding_type not in supported_pos_enc_types:
        raise ValueError(
            "Positional encoding type not supported. Supported types are: "
            f"{supported_pos_enc_types}, but got {pos_encoding_type}"
        )

    # check that the feature dimension (last) matches
    if x.shape[-1] != vectors_to_insert.shape[-1]:
        raise ValueError(
            "Feature dimension (last) of x and vectors_to_insert must match. "
            f"Got {x.shape[-1]} and {vectors_to_insert.shape[-1]}"
        )
    # check that masks have correct shape (should have the same len in dim 0 and 1)
    if x.shape[0] != mask_is_valid.shape[0] or x.shape[1] != mask_is_valid.shape[1]:
        raise ValueError(
            f"x and mask_is_valid have incompatible shape:{x.shape} and {mask_is_valid.shape}"
        )

    if pos_encoding_type is not None and pos_encoding_feature is None:
        raise ValueError(
            "Positional encoding feature must be provided if pos_encoding_type is not None."
        )

    if pos_encoding_feature is not None:
        if mask_is_valid.shape != pos_encoding_feature.shape:
            raise ValueError(
                "Shape of mask_is_valid and pos_encoding_feature must match. "
                f"Got {mask_is_valid.shape} and {pos_encoding_feature.shape}"
            )

    # check that there are enough vectors to insert
    max_n_points = mask_is_valid.shape[1]
    n_vectors_to_insert = vectors_to_insert.shape[0]
    if n_vectors_to_insert < max_n_points:
        raise ValueError(
            "Not enough vectors to insert. "
            f"Got {n_vectors_to_insert}, but need at least {max_n_points}."
        )

    indices = torch.nonzero(mask_is_valid_but_masked, as_tuple=True)

    if pos_encoding_type is None:
        x[indices] = vectors_to_insert[indices[1]]
    elif pos_encoding_type in [
        "sort_descending_all",
        "sort_descending_in_masked_subset",
    ]:
        # ensure the sort feature is available and clone it
        pos_encoding_feature = pos_encoding_feature.clone()
        # set sorting feature of invalid particles to -inf to ensure they are at
        # the end after sorting (and thus don't affect sorting)
        pos_encoding_feature[mask_is_valid == 0] = float("-inf")

        if pos_encoding_type == "sort_descending_in_masked_subset":
            # set sorting feature of valid but unmasked particles to -inf to ensure they are at
            # the end after sorting (and thus don't affect sorting)
            pos_encoding_feature[mask_is_valid_corrupted == 1] = float("-inf")

        sort_index = torch.argsort(pos_encoding_feature, dim=1, descending=True)
        rank_index = torch.argsort(sort_index, dim=1)

        # replace masked entries in x with vectors_to_insert
        x[indices] = vectors_to_insert[rank_index[indices]]
    else:
        raise ValueError("This should never happen.")

    return None


def p4s_to_jetnet_eval_np_stack(p4s_ak, maxlen=100, divide_pt_by_sum_pt=True):
    """
    Helper function to convert an awkward array of Momentum4D objects to a numpy
    array with the format expected by the JetNet evaluation code. The output array
    has shape (n_jets, maxlen, 4) and the features are ordered as (eta, phi, pt,
    mass).

    Parameters
    ----------
    p4s_ak : ak.Array
        Awkward array of Momentum4D objects with fields eta, phi, pt, mass
    maxlen : int, optional
        Maximum number of particles per jet. If a jet has fewer particles, it will be padded with zeros. If a jet has more particles, it will be truncated. Default is 100.
    divide_pt_by_sum_pt : bool, optional
        Whether to divide the pt of each particle by the sum of pt of all particles in the jet. Default is True.
    """
    p4s_ak_sum = ak.sum(p4s_ak, axis=1)
    return ak_to_np_stack(
        ak_pad(
            ak.Array(
                {
                    "eta": p4s_ak.eta,
                    "phi": p4s_ak.phi,
                    "pt": p4s_ak.pt / p4s_ak_sum.pt
                    if divide_pt_by_sum_pt
                    else p4s_ak.pt,
                    "mass": ak.zeros_like(p4s_ak.pt),
                }
            ),
            maxlen=maxlen,
        ),
        names=["eta", "phi", "pt", "mass"],
    )
