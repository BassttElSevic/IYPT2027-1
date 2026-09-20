import math
import unittest
from types import SimpleNamespace

import cupy as cp

from single_hole_PSF import SimulationConfig as PinholePSFSimulationConfig
from single_hole_retinal_image import (
    RetinaSimulationConfig,
    build_optotype,
    build_optotype_coordinates,
    convolve_optotype,
    interpolate_threshold_stroke_arcmin,
    optotype_contrast_ratio,
    rectangle_coverage,
    resample_psf_to_optotype_grid,
)


def make_test_config(
    grid_size: int = 256,
    pixels_per_arcmin: float = 16.0,
) -> RetinaSimulationConfig:
    return RetinaSimulationConfig(
        optotype_grid_size=grid_size,
        optotype_pixels_per_arcmin=pixels_per_arcmin,
    )


class OptotypeGeometryTest(unittest.TestCase):
    def test_rectangle_coverage_is_one_inside_and_zero_outside(self) -> None:
        extent_arcmin = 5.0
        pixel_arcmin = 0.25
        axis = cp.arange(-40, 41, dtype=cp.float32) * pixel_arcmin
        x_arcmin, y_arcmin = cp.meshgrid(axis, axis, indexing="xy")

        coverage = rectangle_coverage(
            x_arcmin,
            y_arcmin,
            -extent_arcmin,
            extent_arcmin,
            -extent_arcmin,
            extent_arcmin,
            pixel_arcmin,
        )
        centre = coverage.shape[0] // 2

        self.assertAlmostEqual(float(coverage[centre, centre]), 1.0, places=6)
        self.assertAlmostEqual(
            float(coverage[centre + 4, centre + 4]),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(coverage[centre, centre + 38]),
            0.0,
            places=6,
        )

    def test_e_optotype_has_white_strokes_and_black_gaps(self) -> None:
        config = make_test_config()
        x_arcmin, y_arcmin, _ = build_optotype_coordinates(config)
        stroke_arcmin = 2.0
        optotype = build_optotype(
            x_arcmin,
            y_arcmin,
            stroke_arcmin,
            config,
        )
        centre = config.optotype_grid_size // 2
        blank_x = int(round(0.5 * stroke_arcmin / config.target_angular_pixel_arcmin))
        blank_y = int(round(stroke_arcmin / config.target_angular_pixel_arcmin))

        self.assertAlmostEqual(
            float(optotype[centre, centre]),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(optotype[centre + blank_y, centre + blank_x]),
            0.0,
            places=6,
        )
        self.assertAlmostEqual(float(optotype.max()), 1.0, places=6)
        self.assertAlmostEqual(float(optotype.min()), 0.0, places=6)

    def test_clear_e_optotype_has_unit_contrast(self) -> None:
        config = make_test_config()
        x_arcmin, y_arcmin, _ = build_optotype_coordinates(config)
        stroke_arcmin = 2.0
        optotype = build_optotype(
            x_arcmin,
            y_arcmin,
            stroke_arcmin,
            config,
        )
        contrast = optotype_contrast_ratio(
            x_arcmin,
            y_arcmin,
            optotype,
            optotype,
            stroke_arcmin,
            config,
        )
        self.assertAlmostEqual(contrast, 1.0, places=6)


class ConvolutionTest(unittest.TestCase):
    def test_non_centred_impulse_keeps_relative_kernel_offset(self) -> None:
        config = make_test_config(grid_size=64)
        impulse = cp.zeros(
            (config.optotype_grid_size, config.optotype_grid_size),
            dtype=cp.float32,
        )
        impulse[40, 20] = 1.0

        axis_y = cp.arange(config.optotype_grid_size, dtype=cp.float32) - 20
        axis_x = cp.arange(config.optotype_grid_size, dtype=cp.float32) - 30
        kernel_y, kernel_x = cp.meshgrid(axis_y, axis_x, indexing="ij")
        sigma_pixels = 1.3
        psf = cp.exp(
            -(kernel_y * kernel_y + kernel_x * kernel_x)
            / (2.0 * sigma_pixels * sigma_pixels)
        )
        psf /= psf.sum(dtype=cp.float64)

        blurred = convolve_optotype(impulse, psf, config)
        peak = cp.unravel_index(cp.argmax(blurred), blurred.shape)
        self.assertEqual(tuple(int(value) for value in peak), (28, 18))

    def test_psf_resampling_keeps_the_centre(self) -> None:
        native_grid_size = 256
        config = RetinaSimulationConfig(
            psf_config=PinholePSFSimulationConfig(
                grid_size=native_grid_size,
            ),
            optotype_grid_size=512,
            optotype_pixels_per_arcmin=16.0,
        )
        centre = native_grid_size // 2
        axis = cp.arange(native_grid_size, dtype=cp.float32) - centre
        x_arcmin, y_arcmin = cp.meshgrid(axis, axis, indexing="xy")
        sigma_pixels = 2.0
        psf = cp.exp(
            -(x_arcmin * x_arcmin + y_arcmin * y_arcmin)
            / (2.0 * sigma_pixels * sigma_pixels)
        )
        psf /= psf.sum(dtype=cp.float64)
        sampling = SimpleNamespace(angular_pixel_arcmin=0.05)

        resampled = resample_psf_to_optotype_grid(psf, sampling, config)
        peak = cp.unravel_index(cp.argmax(resampled), resampled.shape)
        expected_centre = config.optotype_grid_size // 2
        self.assertEqual(
            tuple(int(value) for value in peak),
            (expected_centre, expected_centre),
        )

    def test_centred_impulse_remains_centred(self) -> None:
        config = make_test_config(grid_size=64)
        centre = config.optotype_grid_size // 2
        impulse = cp.zeros(
            (config.optotype_grid_size, config.optotype_grid_size),
            dtype=cp.float32,
        )
        impulse[centre, centre] = 1.0

        axis = cp.arange(config.optotype_grid_size, dtype=cp.float32) - centre
        x_arcmin, y_arcmin = cp.meshgrid(axis, axis, indexing="xy")
        sigma_pixels = 1.5
        psf = cp.exp(
            -(x_arcmin * x_arcmin + y_arcmin * y_arcmin)
            / (2.0 * sigma_pixels * sigma_pixels)
        )
        psf /= psf.sum(dtype=cp.float64)

        blurred = convolve_optotype(impulse, psf, config)
        peak = cp.unravel_index(cp.argmax(blurred), blurred.shape)
        self.assertEqual(tuple(int(value) for value in peak), (centre, centre))
        self.assertTrue(
            math.isclose(
                float(blurred.sum(dtype=cp.float64)),
                1.0,
                rel_tol=1.0e-5,
                abs_tol=1.0e-5,
            )
        )

    def test_gaussian_convolution_preserves_total_intensity(self) -> None:
        config = make_test_config(grid_size=256)
        x_arcmin, y_arcmin, _ = build_optotype_coordinates(config)
        optotype = build_optotype(x_arcmin, y_arcmin, 2.0, config)

        centre = config.optotype_grid_size // 2
        axis = cp.arange(config.optotype_grid_size, dtype=cp.float32) - centre
        kernel_x, kernel_y = cp.meshgrid(axis, axis, indexing="xy")
        sigma_pixels = 1.2
        psf = cp.exp(
            -(kernel_x * kernel_x + kernel_y * kernel_y)
            / (2.0 * sigma_pixels * sigma_pixels)
        )
        psf /= psf.sum(dtype=cp.float64)

        blurred = convolve_optotype(optotype, psf, config)
        expected = float(optotype.sum(dtype=cp.float64))
        actual = float(blurred.sum(dtype=cp.float64))
        self.assertTrue(
            math.isclose(actual, expected, rel_tol=1.0e-4, abs_tol=1.0e-4)
        )

        image_axis = cp.arange(config.optotype_grid_size, dtype=cp.float64)
        optotype_sum = optotype.sum(dtype=cp.float64)
        image_sum = blurred.sum(dtype=cp.float64)
        object_centre_x = float(
            (optotype.sum(axis=0, dtype=cp.float64) * image_axis).sum()
            / optotype_sum
        )
        object_centre_y = float(
            (optotype.sum(axis=1, dtype=cp.float64) * image_axis).sum()
            / optotype_sum
        )
        blurred_centre_x = float(
            (blurred.sum(axis=0, dtype=cp.float64) * image_axis).sum()
            / image_sum
        )
        blurred_centre_y = float(
            (blurred.sum(axis=1, dtype=cp.float64) * image_axis).sum()
            / image_sum
        )
        self.assertLess(abs(blurred_centre_x - object_centre_x), 0.1)
        self.assertLess(abs(blurred_centre_y - object_centre_y), 0.1)


class ThresholdAndConfigurationTest(unittest.TestCase):
    def test_threshold_interpolation_reports_censoring(self) -> None:
        lower_value, lower_state = interpolate_threshold_stroke_arcmin(
            (0.5, 1.0),
            (0.3, 0.1),
            0.2,
        )
        upper_value, upper_state = interpolate_threshold_stroke_arcmin(
            (0.5, 1.0),
            (0.0, 0.1),
            0.2,
        )
        resolved_value, resolved_state = interpolate_threshold_stroke_arcmin(
            (0.5, 1.0),
            (0.0, 0.3),
            0.2,
        )

        self.assertEqual(lower_value, 0.5)
        self.assertEqual(lower_state, "lower_censored")
        self.assertTrue(math.isnan(upper_value))
        self.assertEqual(upper_state, "upper_censored")
        self.assertEqual(resolved_state, "resolved")
        self.assertGreater(resolved_value, 0.5)
        self.assertLess(resolved_value, 1.0)

    def test_invalid_config_branches_are_rejected(self) -> None:
        invalid_overrides = (
            {"optotype_kind": "LandoltC"},
            {"convolution_mode": "same"},
            {"simulation_directory_name": "../outside"},
            {"gpu_device_id": 1},
            {"primary_wavelength_nm": 600.0},
            {"representative_myopia_d": 0.75},
        )
        for overrides in invalid_overrides:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    RetinaSimulationConfig(**overrides)


if __name__ == "__main__":
    unittest.main()
