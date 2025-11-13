"""Tests for functions in gabbro/utils."""

import unittest

import awkward as ak
import numpy as np
import torch

from omnilearn_lightning.array_utils import (
    ak_abs,
    ak_clip,
    ak_mean,
    ak_pad,
    ak_smear,
    ak_subtract,
    ak_to_np_stack,
    combine_ak_arrays,
    np_to_ak,
    preprocess_tensor,
    replace_masked_positions,
    set_fraction_ones_to_zeros,
)


class TestCombineAkArrays(unittest.TestCase):
    def test_combine_ak_arrays(self):
        """Test the function combine_ak_arrays()."""
        ak_arr1 = ak.Array(
            {
                "part_pt": [[1, 2, 3], [4, 5]],
                "part_eta": [[0, 0, 1], [0, 1]],
            }
        )
        ak_arr2 = ak.Array(
            {
                "part_phi": [[6, 7, 8], [9, 10]],
            }
        )

        combined = combine_ak_arrays(ak_arr1, ak_arr2)
        self.assertEqual(combined.part_pt.tolist(), [[1, 2, 3], [4, 5]])
        self.assertEqual(combined.part_eta.tolist(), [[0, 0, 1], [0, 1]])
        self.assertEqual(combined.part_phi.tolist(), [[6, 7, 8], [9, 10]])

    def test_None_is_skipped(self):
        """Test that None is skipped."""
        ak_arr1 = ak.Array(
            {
                "part_pt": [[1, 2, 3], [4, 5]],
                "part_eta": [[0, 0, 1], [0, 1]],
            }
        )
        ak_arr2 = ak.Array(
            {
                "part_phi": [[6, 7, 8], [9, 10]],
            }
        )

        combined = combine_ak_arrays(ak_arr1, None, ak_arr2)
        self.assertEqual(combined.part_pt.tolist(), [[1, 2, 3], [4, 5]])
        self.assertEqual(combined.part_eta.tolist(), [[0, 0, 1], [0, 1]])
        self.assertEqual(combined.part_phi.tolist(), [[6, 7, 8], [9, 10]])

    def test_combine_ak_arrays_three_arrays(self):
        ak_arr1 = ak.Array(
            {"part_pt": [[1, 2, 3], [4, 5]], "part_eta": [[0, 0, 1], [0, 1]]}
        )
        ak_arr2 = ak.Array({"part_phi": [[6, 7, 8], [9, 10]]})
        ak_arr3 = ak.Array({"part_E": [[11, 12, 13], [14, 15]]})

        combined = combine_ak_arrays(ak_arr1, ak_arr2, ak_arr3)
        self.assertEqual(combined.part_pt.tolist(), [[1, 2, 3], [4, 5]])
        self.assertEqual(combined.part_eta.tolist(), [[0, 0, 1], [0, 1]])
        self.assertEqual(combined.part_phi.tolist(), [[6, 7, 8], [9, 10]])
        self.assertEqual(combined.part_E.tolist(), [[11, 12, 13], [14, 15]])

    def test_same_field_raises_error(self):
        """Check that an error is raised if the same field is present in both arrays."""
        ak_arr1 = ak.Array(
            {
                "part_pt": [[1, 2, 3], [4, 5]],
                "part_eta": [[0, 0, 1], [0, 1]],
            }
        )
        ak_arr2 = ak.Array(
            {
                "part_pt": [[6, 7, 8], [9, 10]],
            }
        )

        with self.assertRaises(ValueError):
            combine_ak_arrays(ak_arr1, ak_arr2)


class TestPreprocessTensor(unittest.TestCase):
    def setUp(self):
        # Create simple inputs: x has 3 features, add_info has 2 features
        self.x = torch.tensor(
            [
                [
                    [1.0, 2.0, 4.0],
                    [10.0, 20.0, 40.0],
                ]
            ],
            dtype=torch.float32,
        )
        self.add_info = torch.tensor(
            [
                [
                    [5.0, 15.0],
                    [50.0, 150.0],
                ]
            ],
            dtype=torch.float32,
        )
        self.sfx = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float32)
        self.sfa = torch.tensor([10.0, 30.0], dtype=torch.float32)

    def test_forward_concat_scaled(self):
        out = preprocess_tensor(
            x=self.x,
            add_info=self.add_info,
            inverse=False,
            scale_factors_x=self.sfx,
            scale_factors_add_info=self.sfa,
        )
        expected = torch.tensor(
            [
                [
                    [1.0 / 1.0, 2.0 / 2.0, 4.0 / 4.0, 5.0 / 10.0, 15.0 / 30.0],
                    [10.0 / 1.0, 20.0 / 2.0, 40.0 / 4.0, 50.0 / 10.0, 150.0 / 30.0],
                ]
            ]
        )
        self.assertTrue(torch.allclose(out, expected))

    def test_inverse_concat_unscaled(self):
        out = preprocess_tensor(
            x=self.x,
            add_info=self.add_info,
            inverse=True,
            scale_factors_x=self.sfx,
            scale_factors_add_info=self.sfa,
        )
        expected = torch.cat([self.x * self.sfx, self.add_info * self.sfa], dim=-1)
        self.assertTrue(torch.allclose(out, expected))

    def test_raises_if_scale_factors_missing(self):
        # Missing x scale factors should raise
        with self.assertRaises(ValueError):
            preprocess_tensor(
                x=self.x,
                add_info=self.add_info,
                inverse=False,
                scale_factors_x=None,
                scale_factors_add_info=self.sfa,
            )
        # Missing add_info scale factors should be treated as "ignore add_info"
        out = preprocess_tensor(
            x=self.x,
            add_info=self.add_info,
            inverse=False,
            scale_factors_x=self.sfx,
            scale_factors_add_info=None,
        )
        self.assertTrue(torch.allclose(out, self.x / self.sfx))

    def test_raises_if_wrong_feature_sizes(self):
        # Wrong size for x scale factors
        with self.assertRaises(ValueError):
            preprocess_tensor(
                x=self.x,
                add_info=self.add_info,
                inverse=False,
                scale_factors_x=torch.tensor([1.0, 2.0], dtype=torch.float32),
                scale_factors_add_info=self.sfa,
            )
        # Wrong size for add_info scale factors
        with self.assertRaises(ValueError):
            preprocess_tensor(
                x=self.x,
                add_info=self.add_info,
                inverse=False,
                scale_factors_x=self.sfx,
                scale_factors_add_info=torch.tensor([10.0], dtype=torch.float32),
            )

    def test_raises_if_wrong_dims(self):
        # Provide 2D x
        with self.assertRaises(ValueError):
            preprocess_tensor(
                x=self.x.squeeze(0),
                add_info=self.add_info,
                inverse=False,
                scale_factors_x=self.sfx,
                scale_factors_add_info=self.sfa,
            )
        # Provide 2D add_info
        with self.assertRaises(ValueError):
            preprocess_tensor(
                x=self.x,
                add_info=self.add_info.squeeze(0),
                inverse=False,
                scale_factors_x=self.sfx,
                scale_factors_add_info=self.sfa,
            )


class TestNpToAk(unittest.TestCase):
    def setUp(self):
        # np array of shape (2, 3, 2) (2 jets, 3 constituents, 2 features)
        self.np_array = np.array(
            [
                [[1, 2], [3, 3], [0, 0]],
                # also want to mask the "4", but checking if this is corrected by the function
                [[2, 2], [4, 0], [0, 0]],
            ]
        )
        self.np_mask = np.array(
            [
                [True, True, False],
                [True, False, False],
            ]
        )
        self.names = ["pt", "eta"]
        self.ak_arrary_expected = ak.Array(
            {
                "pt": [[1, 3], [2]],
                "eta": [[2, 3], [2]],
            }
        )
        self.ak_arrary_expected_without_mask = ak.Array(
            {
                "pt": [[1, 3, 0], [2, 4, 0]],
                "eta": [[2, 3, 0], [2, 0, 0]],
            }
        )

    def test_np_to_ak_with_mask(self):
        result = np_to_ak(self.np_array, mask=self.np_mask, names=self.names)
        for i, name in enumerate(self.names):
            self.assertTrue(ak.all(result[name] == self.ak_arrary_expected[name]))

    def test_np_to_ak_without_mask(self):
        result = np_to_ak(self.np_array, names=self.names)
        for i, name in enumerate(self.names):
            self.assertTrue(
                ak.all(result[name] == self.ak_arrary_expected_without_mask[name])
            )


class TestAkToNpStack(unittest.TestCase):
    def setUp(self):
        self.ak_array = ak.Array(
            {
                "pt": [[1, 2, 3], [2, 4]],
                "eta": [[0, 0, 0], [2, 2]],
                "phi": [[0, 0, 0], [3, 3]],
                "E": [[1, 1, 1], [4, 4]],
            }
        )
        # use as the arget array a version where the pt and eta are swapped
        # --> this check both that the order of the stacked fields is correct
        #     and that not all features have to be selected
        self.np_array_padded_len5_eta_pt = np.array(
            [
                [[0, 1], [0, 2], [0, 3], [0, 0], [0, 0]],
                [[2, 2], [2, 4], [0, 0], [0, 0], [0, 0]],
            ]
        )

    def test_ak_to_np_stack(self):
        input_data = ak_pad(self.ak_array, maxlen=5, axis=1)
        result = ak_to_np_stack(input_data, axis=2, names=["eta", "pt"])

        try:
            self.assertTrue(np.array_equal(result, self.np_array_padded_len5_eta_pt))
        except AssertionError:
            print("Arrays are not equal:")
            print("Expected:", self.np_array_padded_len5_eta_pt)
            print("Actual:", result)
            raise AssertionError


class TestAkSmearAndClip(unittest.TestCase):
    def setUp(self):
        self.input_array = ak.Array(
            {
                "pt": [[2, 1], [2]],
            }
        )

    def test_smear(self):
        """Test that the function smears the input array."""
        result = ak_smear(self.input_array["pt"], sigma=0.05, seed=101)
        expected_result = [
            [1.9604923750018493, 0.8982687259084063],
            [2.030165087346238],
        ]
        self.assertEqual(result.tolist(), expected_result)

    def test_clipmin(self):
        """Test that the function clips the input array to min value."""
        result = ak_clip(self.input_array["pt"], clip_min=1.5)
        expected_result = [
            [2, 1.5],
            [2],
        ]
        self.assertEqual(result.tolist(), expected_result)

    def test_clipmax(self):
        """Test that the function clips the input array to max value."""
        result = ak_clip(self.input_array["pt"], clip_max=1.5)
        expected_result = [
            [1.5, 1],
            [1.5],
        ]
        self.assertEqual(result.tolist(), expected_result)

    def test_clipminmax(self):
        """Test that the function clips the input array to min and max value."""
        result = ak_clip(self.input_array["pt"], clip_min=1.5, clip_max=1.8)
        expected_result = [
            [1.8, 1.5],
            [1.8],
        ]
        self.assertEqual(result.tolist(), expected_result)

    def test_clipminmax_non_nested(self):
        """Test for correct clipping of non-nested array."""
        input_array = ak.Array([2, 1, 3, 4])
        result = ak_clip(input_array, clip_min=2.5, clip_max=3.5)
        expected_result = [2.5, 2.5, 3, 3.5]
        self.assertEqual(result.tolist(), expected_result)

    def test_smear_and_clip(self):
        """Test that the function smears and clips the input array."""
        result = ak_clip(
            ak_smear(
                self.input_array["pt"],
                sigma=0.05,
                seed=101,
            ),
            clip_min=0.9,
            clip_max=2.01,
        )
        expected_result = [
            [1.9604923750018493, 0.9],
            [2.01],
        ]
        self.assertEqual(result.tolist(), expected_result)


class TestAkSubtract(unittest.TestCase):
    def test_ak_subtract(self):
        """Test the function ak_subtract() with valid inputs."""
        arr1 = ak.Array(
            {
                "pt": [[2, 1, 0], [1]],
                "eta": [[2, 0, 0], [1]],
            }
        )
        arr2 = ak.Array(
            {
                "pt": [[2, 2, 0], [1]],
                "eta": [[2, 0, 0], [1]],
            }
        )

        expected_diff = ak.Array(
            {
                "pt": [[0, -1, 0], [0]],
                "eta": [[0, 0, 0], [0]],
            }
        )

        diff = ak_subtract(arr1, arr2)

        for field in arr1.fields:
            self.assertEqual(diff[field].tolist(), expected_diff[field].tolist())

    def test_raises_error_if_different_length(self):
        """Test that an error is raised if the arrays have different lengths."""
        arr1 = ak.Array(
            {
                "pt": [[2, 1, 0], [1]],
                "eta": [[2, 0, 0], [1]],
            }
        )
        arr2 = ak.Array(
            {
                "pt": [[2, 2, 0], [1], [3]],
                "eta": [[2, 0, 0], [1], [3]],
            }
        )

        with self.assertRaises(ValueError):
            ak_subtract(arr1, arr2)

    def test_raises_error_if_different_fields(self):
        """Test that an error is raised if the arrays have different fields."""
        arr1 = ak.Array(
            {
                "pt": [[2, 1, 0], [1]],
                "eta": [[2, 0, 0], [1]],
            }
        )
        arr2 = ak.Array(
            {
                "pt": [[2, 2, 0], [1]],
                "phi": [[2, 0, 0], [1]],
            }
        )

        with self.assertRaises(ValueError):
            ak_subtract(arr1, arr2)

    def test_raises_error_if_has_no_fields(self):
        """Test that an error is raised if the arrays have no fields."""
        arr1 = ak.Array([1, 2, 3])
        arr2 = ak.Array([4, 5, 6])

        with self.assertRaises(ValueError):
            ak_subtract(arr1, arr2)


class TestAkMean(unittest.TestCase):
    def test_overall_case(self):
        """Test the function ak_mean() with a valid input."""
        arr = ak.Array(
            {
                "pt": [[2, 1, 0], [1]],
                "eta": [[2, 0, 0], [1]],
            }
        )

        expected_mean = {
            "pt": 1.0,
            "eta": 0.75,
        }

        mean = ak_mean(arr)

        for field in arr.fields:
            self.assertEqual(mean[field], expected_mean[field])

    def test_with_axis_specification(self):
        """Test the function ak_mean() with a valid input and axis specification."""
        arr = ak.Array(
            {
                "pt": [[2, 1, 0], [1]],
                "eta": [[2, 0, 0], [1]],
            }
        )

        expected_mean = {
            "pt": ak.Array([1.0, 1.0]),
            "eta": ak.Array([2 / 3, 1.0]),
        }

        mean = ak_mean(arr, axis=1)

        for field in arr.fields:
            self.assertEqual(mean[field].tolist(), expected_mean[field].tolist())

    def test_raises_error_wrong_input_type(self):
        """Test that an error is raised if the input is not an ak array."""
        with self.assertRaises(TypeError):
            ak_mean([1, 2, 3])

    def test_raises_error_axis_non_int(self):
        """Test that an error is raised if the axis is not an integer."""
        arr = ak.Array(
            {
                "pt": [[2, 1, 0], [1]],
                "eta": [[2, 0, 0], [1]],
            }
        )

        with self.assertRaises(TypeError):
            ak_mean(arr, axis="1")


class TestAkAbs(unittest.TestCase):
    def test_correct_usage_nested(self):
        """Test the function ak_abs() with a valid input (but nested, such that the returned dict
        contains lists/ak.Arrays as values)."""
        arr = ak.Array(
            {
                "pt": [[-2, 1, 0], [1]],
                "eta": [[-2, 0, 0], [1]],
            }
        )

        expected_abs = ak.Array(
            {
                "pt": [[2, 1, 0], [1]],
                "eta": [[2, 0, 0], [1]],
            }
        )

        abs_arr = ak_abs(arr)

        for field in arr.fields:
            self.assertEqual(abs_arr[field].tolist(), expected_abs[field].tolist())

    def test_correct_usage_flat(self):
        """Test the function ak_abs() with a valid input."""
        arr = ak.Array(
            {
                "pt": [1, -2, 3, -4],
                "eta": [0, 0, 1, 1],
            }
        )
        expected_abs = ak.Array(
            {
                "pt": [1, 2, 3, 4],
                "eta": [0, 0, 1, 1],
            }
        )

        abs_arr = ak_abs(arr)

        self.assertEqual(abs_arr.tolist(), expected_abs.tolist())

    def test_raises_error_if_no_fields(self):
        """Test that an error is raised if the array has no fields."""
        arr = ak.Array([1, 2, 3])

        with self.assertRaises(ValueError):
            ak_abs(arr)

    def test_raises_error_if_wrong_input_type(self):
        """Test that an error is raised if the input is not an ak array."""
        with self.assertRaises(TypeError):
            ak_abs([1, 2, 3])


class TestMasking(unittest.TestCase):
    def test_no_changes_if_fraction_zero(self):
        """Test that the mask is not changed if the fraction is zero."""
        mask = torch.tensor(
            [
                [0, 0, 0, 1, 0],
                [0, 1, 0, 0, 1],
            ]
        )
        mask_initial = mask.clone()
        mask = set_fraction_ones_to_zeros(mask, 0)
        assert torch.equal(mask, mask_initial), (
            "Mask was modified, but should not have been."
        )

    def test_no_changes_if_all_entries_zero(self):
        """Test that the mask is not changed if all entries are zero."""
        mask = torch.zeros(2, 5)
        mask_initial = mask.clone()
        mask = set_fraction_ones_to_zeros(mask, 0.5)
        assert torch.equal(mask, mask_initial), (
            "Mask was modified, but should not have been cause it was all zeros."
        )

    def test_correct_masking(self):
        """Test one seeded example where the mask is modified as expected."""
        mask = torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            ]
        )

        # these are currently 15 ones in total (of 20 entries)
        torch.manual_seed(0)
        mask_expected = torch.tensor(
            [
                [0, 1, 0, 0, 0, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            ]
        )
        mask = set_fraction_ones_to_zeros(mask, 0.5)
        assert torch.equal(mask, mask_expected), "Mask was not modified as expected."

    def test_fraction_correct(self):
        """Test that the fraction of ones that survive is correct."""
        n_0, n_1 = 1000, 10
        mask = torch.ones(n_0, n_1)
        fraction = 0.3
        mask = set_fraction_ones_to_zeros(mask, fraction)
        n_ones_survived = torch.sum(mask)

        # the fraction is exact if the number of ones is a multiple of the fraction

        assert n_ones_survived == n_0 * n_1 * (1 - fraction), (
            f"Expected {n_0 * n_1 * (1 - fraction)} ones to survive, but got {n_ones_survived}."
        )

    def test_that_input_remains_unchanged(self):
        mask = torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            ]
        )
        mask_initial = mask.clone()
        _ = set_fraction_ones_to_zeros(mask, 0.5)
        assert torch.equal(mask, mask_initial), (
            "Input mask was modified, but should not have been."
        )


# Tests for the replace_masked_positions function, which replaces masked positions in a tensor
# with provided vectors and optionally sorts them according to positional encoding.
class TestReplaceMaskedPositions(unittest.TestCase):
    def setUp(self):
        # Example input tensor with shape (1, 8, 2)
        # Each row represents a particle with two features
        self.x = torch.tensor(
            [
                [
                    [7.5, 2.0],  # <- #2
                    [3.0, 4.0],  # <- #5
                    [5.0, 6.0],  # <- #4
                    [7.0, 8.0],  # <- #3
                    [9.0, 1.0],  # <- #1
                    [1.0, 1.0],  # <- #6
                    [8.0, 0.0],  # <- # should be ignored
                    [8.0, 7.0],  # <- # should be ignored
                ],
                [
                    [1.2, 1.1],  # <- #6
                    [6.5, 8.0],  # <- #3
                    [4.4, 2.3],  # <- #4
                    [7.0, 8.0],  # <- #2
                    [9.0, 1.0],  # <- #1
                    [1.0, 1.0],  # <- #7
                    [2.0, 1.0],  # <- #5
                    [9.9, 1.0],  # <- # should be ignored
                ],
            ]
        )
        # Mask indicating valid particles (1 = valid, 0 = invalid)
        self.mask_is_valid = torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 0],
            ],
            dtype=torch.int,
        )
        self.mask_is_valid_corrupted = torch.tensor(
            [
                [0, 1, 0, 1, 1, 0, 0, 0],
                [0, 1, 0, 0, 1, 0, 0, 0],
            ],
            dtype=torch.int,
        )
        # Mask for valid but masked particles (1 = valid and masked, 0 = otherwise)
        self.mask_is_valid_but_masked = self.mask_is_valid * (
            1 - self.mask_is_valid_corrupted
        )
        # Vectors to insert in place of masked positions
        self.vectors_to_insert = torch.tensor(
            [
                [10, 10],
                [20, 20],
                [30, 30],
                [40, 40],
                [50, 50],
                [60, 60],
                [70, 70],
                [80, 80],
            ],
            dtype=torch.float,
        )

    def test_error_invalid_mask_shape(self):
        """Test error is raised if x and mask shapes do not match."""
        # Use self.x but change mask shape
        mask_is_valid = torch.ones(1, self.x.shape[1], dtype=torch.int)
        with self.assertRaises(ValueError):
            replace_masked_positions(
                x=self.x,
                mask_is_valid=mask_is_valid,
                mask_is_valid_corrupted=mask_is_valid,
                mask_is_valid_but_masked=mask_is_valid,
                vectors_to_insert=self.vectors_to_insert,
                pos_encoding_type=None,
            )

    def test_error_invalid_vectors_to_insert_shape(self):
        """Test error is raised if vectors_to_insert shape does not match x shape."""
        vectors_to_insert = torch.randn(3, self.x.shape[2])  # Should be (8, 2)
        with self.assertRaises(ValueError):
            replace_masked_positions(
                x=self.x,
                mask_is_valid=self.mask_is_valid,
                mask_is_valid_corrupted=self.mask_is_valid_corrupted,
                mask_is_valid_but_masked=self.mask_is_valid_but_masked,
                vectors_to_insert=vectors_to_insert,
                pos_encoding_type=None,
            )

    def test_error_invalid_pos_encoding_type(self):
        """Test error is raised if pos_encoding_type is not recognized."""
        with self.assertRaises(ValueError):
            replace_masked_positions(
                x=self.x,
                mask_is_valid=self.mask_is_valid,
                mask_is_valid_corrupted=self.mask_is_valid_corrupted,
                mask_is_valid_but_masked=self.mask_is_valid_but_masked,
                vectors_to_insert=self.vectors_to_insert,
                pos_encoding_type="invalid_type",
            )

    def test_error_missing_pos_encoding_feature(self):
        """Test error is raised if pos_encoding_type requires pos_encoding_feature but it is None."""
        for pos_encoding_type in [
            "sort_descending_all",
            "sort_descending_in_masked_subset",
        ]:
            with self.assertRaises(ValueError):
                replace_masked_positions(
                    x=self.x,
                    mask_is_valid=self.mask_is_valid,
                    mask_is_valid_corrupted=self.mask_is_valid_corrupted,
                    mask_is_valid_but_masked=self.mask_is_valid_but_masked,
                    vectors_to_insert=self.vectors_to_insert,
                    pos_encoding_type=pos_encoding_type,
                    pos_encoding_feature=None,
                )

    def test_replace_masked_positions_without_pos_encoding(self):
        # Test replacing masked positions without any positional encoding or sorting
        x = self.x.clone()
        replace_masked_positions(
            x=x,
            mask_is_valid=self.mask_is_valid,
            mask_is_valid_corrupted=self.mask_is_valid_corrupted,
            mask_is_valid_but_masked=self.mask_is_valid_but_masked,
            vectors_to_insert=self.vectors_to_insert,
            pos_encoding_type=None,
            pos_encoding_feature=None,
        )
        # Expected output: masked positions replaced by corresponding vectors
        x_expected = torch.tensor(
            [
                [
                    [10, 10],  # <- #2 (was masked)
                    [3.0, 4.0],  # <- #5
                    [30, 30],  # <- #4 (was masked)
                    [7.0, 8.0],  # <- #3
                    [9.0, 1.0],  # <- #1
                    [60, 60],  # <- #6 (was masked)
                    [8.0, 0.0],  # <- # should be ignored
                    [8.0, 7.0],  # <- # should be ignored
                ],
                [
                    [10, 10],  # <- #6 (was masked)
                    [6.5, 8.0],  # <- #3
                    [30, 30],  # <- #4 (was masked)
                    [40, 40],  # <- #2 (was masked)
                    [9.0, 1.0],  # <- #1
                    [60, 60],  # <- #7 (was masked)
                    [70, 70],  # <- #5 (was masked)
                    [9.9, 1.0],  # <- # should be ignored
                ],
            ]
        )
        self.assertTrue(torch.equal(x, x_expected))

    def test_replace_masked_positions_with_full_sorting(self):
        # Test replacing masked positions with sorting applied to all valid positions
        x = self.x.clone()
        replace_masked_positions(
            x=x,
            mask_is_valid=self.mask_is_valid,
            mask_is_valid_corrupted=self.mask_is_valid_corrupted,
            mask_is_valid_but_masked=self.mask_is_valid_but_masked,
            vectors_to_insert=self.vectors_to_insert,
            pos_encoding_type="sort_descending_all",
            pos_encoding_feature=x[..., 0],
        )
        x_expected = torch.tensor(
            [
                [
                    [20, 20],  # <- #2 (was masked)
                    [3.0, 4.0],  # <- #5
                    [40, 40],  # <- #4 (was masked)
                    [7.0, 8.0],  # <- #3
                    [9.0, 1.0],  # <- #1
                    [60, 60],  # <- #6 (was masked)
                    [8.0, 0.0],  # <- # should be ignored
                    [8.0, 7.0],  # <- # should be ignored
                ],
                [
                    [60, 60],  # <- #6 (was masked)
                    [6.5, 8.0],  # <- #3
                    [40, 40],  # <- #4 (was masked)
                    [20, 20],  # <- #2 (was masked)
                    [9.0, 1.0],  # <- #1
                    [70, 70],  # <- #7 (was masked)
                    [50, 50],  # <- #5 (was masked)
                    [9.9, 1.0],  # <- # should be ignored
                ],
            ]
        )
        self.assertTrue(torch.equal(x, x_expected))

    def test_replace_masked_positions_with_sorting_masked_subset(self):
        # Test replacing masked positions with sorting only within the masked subset
        x = self.x.clone()
        replace_masked_positions(
            x=x,
            mask_is_valid=self.mask_is_valid,
            mask_is_valid_corrupted=self.mask_is_valid_corrupted,
            mask_is_valid_but_masked=self.mask_is_valid_but_masked,
            vectors_to_insert=self.vectors_to_insert,
            pos_encoding_type="sort_descending_in_masked_subset",
            pos_encoding_feature=x[..., 0],
        )
        x_expected = torch.tensor(
            [
                [
                    [10, 10],  # <- #2 (was masked) #1 in masked subset
                    [3.0, 4.0],  # <- #5
                    [20, 20],  # <- #4 (was masked) #2 in masked subset
                    [7.0, 8.0],  # <- #3
                    [9.0, 1.0],  # <- #1
                    [30, 30],  # <- #6 (was masked) #3 in masked subset
                    [8.0, 0.0],  # <- # should be ignored
                    [8.0, 7.0],  # <- # should be ignored
                ],
                [
                    [40, 40],  # <- #6 (was masked)
                    [6.5, 8.0],  # <- #3
                    [20, 20],  # <- #4 (was masked)
                    [10, 10],  # <- #2 (was masked)
                    [9.0, 1.0],  # <- #1
                    [50, 50],  # <- #7 (was masked)
                    [30, 30],  # <- #5 (was masked)
                    [9.9, 1.0],  # <- # should be ignored
                ],
            ]
        )
        self.assertTrue(torch.equal(x, x_expected))

    def test_sorting_none_equals_sort_descending_all_if_already_sorted(self):
        """If the positional feature is already sorted descending in the input, then
        using no positional encoding and using sort_descending_all should produce
        the same output after replacement.
        """
        # Small example: one batch, 5 positions, 2 features
        x = torch.tensor(
            [
                [
                    [9.0, 0.0],  # valid, largest
                    [7.0, 0.0],  # valid
                    [5.0, 0.0],  # valid
                    [3.0, 0.0],  # valid, smallest among valids
                    [0.0, 0.0],  # invalid (ignored)
                ]
            ]
        )

        # valid mask: first 4 positions valid
        mask_is_valid = torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.int)
        # corrupted mask: mark second position as corrupted so that
        # mask_is_valid_but_masked selects positions 0,2,3 to be replaced
        mask_is_valid_corrupted = torch.tensor([[0, 1, 0, 0, 0]], dtype=torch.int)
        mask_is_valid_but_masked = mask_is_valid * (1 - mask_is_valid_corrupted)

        vectors_to_insert = torch.tensor(
            [[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0], [50.0, 50.0]],
            dtype=torch.float,
        )

        # Replace without positional encoding
        x_none = x.clone()
        replace_masked_positions(
            x=x_none,
            mask_is_valid=mask_is_valid,
            mask_is_valid_corrupted=mask_is_valid_corrupted,
            mask_is_valid_but_masked=mask_is_valid_but_masked,
            vectors_to_insert=vectors_to_insert,
            pos_encoding_type=None,
        )

        # Replace with full sorting by the first feature (descending). The feature
        # is already in descending order among valid positions, so result should
        # be identical.
        x_sorted = x.clone()
        replace_masked_positions(
            x=x_sorted,
            mask_is_valid=mask_is_valid,
            mask_is_valid_corrupted=mask_is_valid_corrupted,
            mask_is_valid_but_masked=mask_is_valid_but_masked,
            vectors_to_insert=vectors_to_insert,
            pos_encoding_type="sort_descending_all",
            pos_encoding_feature=x_sorted[..., 0],
        )

        # The two outputs should be equal
        self.assertTrue(torch.equal(x_none, x_sorted))
