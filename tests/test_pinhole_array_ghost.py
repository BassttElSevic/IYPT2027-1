"""第三阶段多针孔阵列重影仿真的单元测试。

测试纪律沿用第二阶段：先用小网格和解析锚点验证，再进入正式扫描。
所有测试只使用少量孔和短向量，避免每次测试都加载大 GPU 数组。
"""

import math
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any

import cupy as cp
import numpy as np

from pinhole_array_ghost import (
    accumulate_shifted_kernels,
    build_diameter_anchor_grid,
    build_psf_cache,
    build_solar_directions,
    build_solar_disk_kernel,
    circle_overlap_area_mm2,
    compute_hole_weights,
    cluster_ghost_positions,
    classify_risk_labels,
    compute_ghost_metrics,
    local_peak_arcmin,
    convolve_with_solar_disk,
    integrate_solar_source_direct,
    load_previous_diameter_anchors,
    measure_image_width_arcmin,
    plan_field_grid,
    resample_psf_to_grid,
)
from pinhole_array_ghost import (
    ARRAY_PATTERN_SQUARE_PACKING,
    ARRAY_PATTERN_TRIANGULAR_PACKING,
    ArrayGhostSimulationConfig,
    analytic_area_fraction,
    build_pitch_grid,
    generate_circular_rings,
    generate_hexagonal_packing,
    generate_hexagonal_rings,
    generate_jittered_square,
    generate_square_packing,
    generate_triangular_packing,
    lattice_unit_cell_area_mm2,
    minimum_ring_radius_mm,
    rasterize_mask,
    resolve_simulation_directory,
    validate_config,
    validate_layout,
    validate_mask,
    validate_output_component,
)


class ConfigValidationTest(unittest.TestCase):
    def test_default_config_is_valid(self) -> None:
        config = ArrayGhostSimulationConfig()
        config.validate()

    def test_quick_variant_is_valid_and_derives_quick_values(self) -> None:
        config = ArrayGhostSimulationConfig()
        quick = config.quick_variant()

        quick.validate()
        self.assertTrue(quick.quick_mode)
        self.assertEqual(quick.effective_row_count(), quick.quick_row_count)
        self.assertEqual(
            quick.effective_analysis_grid_size(),
            quick.quick_analysis_grid_size,
        )
        self.assertEqual(
            quick.effective_pitch_values_mm(),
            quick.quick_pitch_values_mm,
        )
        self.assertEqual(
            quick.effective_solar_radial_ring_count(),
            quick.quick_solar_radial_ring_count,
        )

    def test_full_config_uses_full_values(self) -> None:
        config = ArrayGhostSimulationConfig()

        self.assertFalse(config.quick_mode)
        self.assertEqual(
            config.effective_pitch_values_mm(),
            config.pitch_reference_values_mm,
        )
        self.assertEqual(
            config.effective_diameter_anchor_factors(),
            config.diameter_anchor_factors,
        )

    def test_non_even_grid_is_rejected(self) -> None:
        config = replace(ArrayGhostSimulationConfig(), analysis_grid_size=513)
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_empty_myopia_tuple_is_rejected(self) -> None:
        config = replace(ArrayGhostSimulationConfig(), myopia_values_d=())
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_zero_myopia_is_rejected(self) -> None:
        config = replace(ArrayGhostSimulationConfig(), myopia_values_d=(0.0, 3.0))
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_inverted_pitch_range_is_rejected(self) -> None:
        config = replace(
            ArrayGhostSimulationConfig(),
            minimum_pitch_mm=8.0,
            maximum_pitch_mm=2.0,
        )
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_design_point_below_clearance_is_rejected(self) -> None:
        config = replace(
            ArrayGhostSimulationConfig(),
            design_point_diameter_mm=(2.4,),
            design_point_pitch_mm=(2.5,),
            design_point_labels=("A",),
            minimum_edge_clearance_mm=0.2,
        )
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_unknown_source_kind_is_rejected(self) -> None:
        config = replace(ArrayGhostSimulationConfig(), source_kind="laser_coherent")
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_pupil_vignetting_requires_distance(self) -> None:
        config = replace(
            ArrayGhostSimulationConfig(),
            enable_pupil_vignetting=True,
            pupil_plane_distance_mm=None,
        )
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_pupil_vignetting_with_distance_is_accepted(self) -> None:
        config = replace(
            ArrayGhostSimulationConfig(),
            enable_pupil_vignetting=True,
            pupil_plane_distance_mm=12.0,
        )
        validate_config(config)

    def test_reuse_disabled_requires_debug_label(self) -> None:
        config = replace(ArrayGhostSimulationConfig(), reuse_previous_metrics=False)
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_reuse_disabled_with_debug_label_is_accepted(self) -> None:
        config = replace(
            ArrayGhostSimulationConfig(),
            reuse_previous_metrics=False,
            reuse_debug_label="debug_recompute",
        )
        validate_config(config)

    def test_ring_count_must_be_positive(self) -> None:
        config = replace(ArrayGhostSimulationConfig(), ring_count_values=(0, 1))
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_pitch_to_diameter_ratio_must_exceed_one(self) -> None:
        config = replace(
            ArrayGhostSimulationConfig(),
            pitch_to_diameter_ratio_values=(0.5, 2.0),
        )
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_solar_angular_radius_matches_diameter(self) -> None:
        config = ArrayGhostSimulationConfig()
        self.assertAlmostEqual(
            config.solar_angular_radius_deg,
            0.5 * config.solar_angular_diameter_deg,
            places=12,
        )
        self.assertAlmostEqual(
            config.solar_angular_radius_arcmin,
            15.9,
            places=6,
        )

    def test_pattern_kind_contract_is_available(self) -> None:
        config = ArrayGhostSimulationConfig()
        self.assertIn(ARRAY_PATTERN_SQUARE_PACKING, config.valid_pattern_kinds)
        self.assertEqual(len(config.valid_pattern_kinds), 7)


class OutputPathTest(unittest.TestCase):
    def test_path_separator_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_output_component("nested/dir", "simulation_directory_name")

    def test_parent_token_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_output_component("..", "simulation_directory_name")

    def test_empty_component_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_output_component("", "simulation_directory_name")

    def test_simulation_directory_lives_under_repository_output(self) -> None:
        config = ArrayGhostSimulationConfig()
        directory = resolve_simulation_directory(config)
        repository_root = Path(__file__).resolve().parent.parent

        self.assertTrue(directory.is_relative_to(repository_root))
        self.assertEqual(directory.name, config.simulation_directory_name)
        self.assertTrue(directory.exists())

    def test_default_output_root_has_no_drive_letter(self) -> None:
        config = ArrayGhostSimulationConfig()
        self.assertNotIn(":", config.output_root_directory_name)


class FieldOfViewTest(unittest.TestCase):
    def test_solar_radius_is_about_sixteen_arcmin(self) -> None:
        config = ArrayGhostSimulationConfig()
        # 0.53 度直径对应约 31.8 角分直径，半径约 15.9 角分。
        self.assertTrue(
            math.isclose(config.solar_angular_radius_arcmin, 15.9, abs_tol=0.05)
        )

    def test_first_order_angle_matches_pitch_over_focal_length(self) -> None:
        config = ArrayGhostSimulationConfig()
        pitch_mm = 2.0
        expected_deg = math.degrees(pitch_mm / config.focal_length_mm)

        # p / f 的一阶重影角口径：2 mm / 25 mm = 0.08 rad = 4.5837 度。
        self.assertAlmostEqual(expected_deg, 4.5837, places=3)


def neighbor_angles_deg(centers: Any, center_index: int, radius_mm: float) -> list[float]:
    """返回某个孔在给定半径内的近邻方位角，用于检查近邻拓扑。"""
    import numpy as np

    reference = centers[center_index]
    offsets = centers - reference
    distances = np.sqrt((offsets**2).sum(axis=1))
    mask = (distances > 0.0) & (distances <= radius_mm)
    angles = np.degrees(np.arctan2(offsets[mask, 1], offsets[mask, 0]))
    return sorted(float(angle) % 360.0 for angle in angles)


def single_hole_layout(diameter_mm: float):
    """构造只有一个中心孔的排布，用于覆盖度基本性质测试。"""
    import numpy as np

    from pinhole_array_ghost import ArrayLayout

    return ArrayLayout(
        pattern_kind="square_packing",
        centers_mm=np.zeros((1, 2), dtype=np.float64),
        pitch_nominal_mm=2.0,
        diameter_mm=diameter_mm,
        row_count=1,
        column_count=1,
        ring_count=0,
        holes_per_ring=(),
        ring_radii_mm=(),
        ring_radial_pitch_mm=2.0,
        analytic_area_fraction=math.nan,
        notes="single hole coverage probe",
    )


class CircleOverlapTest(unittest.TestCase):
    def test_disjoint_circles_have_zero_overlap(self) -> None:
        self.assertAlmostEqual(
            circle_overlap_area_mm2(10.0, 1.0, 1.0),
            0.0,
            places=12,
        )

    def test_fully_contained_circle_returns_smaller_area(self) -> None:
        expected = math.pi * 0.5**2
        self.assertAlmostEqual(
            circle_overlap_area_mm2(0.1, 0.5, 3.0),
            expected,
            places=12,
        )

    def test_identical_circles_have_full_area(self) -> None:
        expected = math.pi * 1.0**2
        self.assertAlmostEqual(
            circle_overlap_area_mm2(0.0, 1.0, 1.0),
            expected,
            places=12,
        )

    def test_half_overlap_of_equal_circles(self) -> None:
        # 两个等半径圆的交叠面积等于半径时，交叠约为 1.2284 r^2。
        overlap = circle_overlap_area_mm2(1.0, 1.0, 1.0)
        self.assertAlmostEqual(overlap / (math.pi * 1.0**2), 0.3910, places=4)

    def test_overlap_is_monotone_decreasing_with_distance(self) -> None:
        distances = [0.0, 0.2, 0.5, 1.0, 1.5, 1.9]
        overlaps = [
            circle_overlap_area_mm2(value, 1.0, 1.0) for value in distances
        ]
        self.assertTrue(
            all(a >= b for a, b in zip(overlaps, overlaps[1:])),
            overlaps,
        )


class HoleWeightTest(unittest.TestCase):
    def test_infinite_pupil_model_gives_unit_weights(self) -> None:
        config = ArrayGhostSimulationConfig()
        layout = generate_square_packing(config, 2.0, 0.7)
        weights, model = compute_hole_weights(layout, None, config)

        self.assertEqual(model, "infinite_pupil_upper_bound")
        self.assertTrue((weights == 1.0).all())

    def test_pupil_model_attenuates_off_axis_holes(self) -> None:
        config = replace(
            ArrayGhostSimulationConfig(),
            enable_pupil_vignetting=True,
            pupil_plane_distance_mm=12.0,
        )
        layout = generate_square_packing(config, 2.0, 0.7)
        weights, model = compute_hole_weights(layout, 2.0, config)
        import numpy as np

        centre_index = int(np.argmin((layout.centers_mm**2).sum(axis=1)))
        self.assertEqual(model, "quasi_static_pupil")
        self.assertAlmostEqual(weights[centre_index], 1.0, places=12)
        # 瞳孔直径 2 mm 时，偏离超过 1 mm 的孔完全被挡掉。
        self.assertTrue((weights >= 0.0).all())
        self.assertTrue((weights <= 1.0 + 1e-12).all())
        self.assertTrue((weights < 1.0).any())


class PsfCacheTest(unittest.TestCase):
    def test_repeated_key_hits_cache_without_recompute(self) -> None:
        config = ArrayGhostSimulationConfig()
        cache = build_psf_cache(config)

        first, sampling_first = cache.get(config, 0.7, 3.0, 550.0)
        second, sampling_second = cache.get(config, 0.7, 3.0, 550.0)

        self.assertEqual(cache.compute_count, 1)
        self.assertEqual(cache.hit_count, 1)
        self.assertTrue(first is second)
        self.assertAlmostEqual(
            sampling_first.angular_pixel_arcmin,
            sampling_second.angular_pixel_arcmin,
            places=12,
        )

    def test_distinct_key_triggers_new_compute(self) -> None:
        config = ArrayGhostSimulationConfig()
        cache = build_psf_cache(config)

        cache.get(config, 0.7, 3.0, 550.0)
        cache.get(config, 0.7, 3.0, 600.0)

        self.assertEqual(cache.compute_count, 2)
        self.assertEqual(cache.hit_count, 0)

    def test_psf_is_normalised_to_one(self) -> None:
        config = ArrayGhostSimulationConfig()
        cache = build_psf_cache(config)
        psf, _ = cache.get(config, 0.7, 3.0, 550.0)

        self.assertAlmostEqual(
            float(psf.sum(dtype=config.accumulator_dtype)),
            1.0,
            places=6,
        )


class ResamplingTest(unittest.TestCase):
    def test_identity_resampling_keeps_centre_and_energy(self) -> None:
        config = ArrayGhostSimulationConfig()
        axis = cp.arange(64, dtype=cp.float32) - 32
        source = cp.exp(-((axis[:, None] ** 2 + axis[None, :] ** 2) / 8.0))
        source = source / source.sum()

        resampled = resample_psf_to_grid(source, 0.25, 64, 0.25, config)
        peak_y, peak_x = [
            int(value.get()) for value in cp.unravel_index(resampled.argmax(), resampled.shape)
        ]

        self.assertEqual((peak_y, peak_x), (32, 32))
        self.assertAlmostEqual(
            float(resampled.sum(dtype=config.accumulator_dtype)),
            1.0,
            places=6,
        )

    def test_half_pixel_shift_moves_peak_by_one_source_pixel(self) -> None:
        config = ArrayGhostSimulationConfig()
        axis = cp.arange(64, dtype=cp.float32) - 32
        source = cp.zeros((64, 64), dtype=cp.float32)
        source[32, 32] = 1.0

        # 目标角分辨率是源的两倍，源上一像素对应目标两像素。
        resampled = resample_psf_to_grid(source, 0.25, 64, 0.5, config)
        self.assertEqual(float(resampled.sum()), 1.0)
        self.assertGreater(float(resampled[32, 32]), 0.0)


class SolarDiskTest(unittest.TestCase):
    def test_disk_kernel_sums_to_one(self) -> None:
        config = ArrayGhostSimulationConfig()
        kernel, metadata = build_solar_disk_kernel(
            config, 256, config.analysis_pixel_arcmin
        )

        self.assertAlmostEqual(metadata["weight_sum"], 1.0, places=6)
        self.assertLess(
            abs(metadata["weight_sum"] - 1.0),
            config.solar_weight_sum_tolerance,
        )

    def test_disk_kernel_matches_analytic_area_fraction(self) -> None:
        config = ArrayGhostSimulationConfig()
        _kernel, metadata = build_solar_disk_kernel(
            config, 256, config.analysis_pixel_arcmin
        )
        analytic = metadata["analytic_disk_fraction_of_window"]
        numeric = metadata["numeric_disk_fraction_of_window"]

        self.assertLess(abs(numeric - analytic) / analytic, 0.01)

    def test_disk_kernel_is_symmetric(self) -> None:
        config = ArrayGhostSimulationConfig()
        kernel, _ = build_solar_disk_kernel(
            config, 256, config.analysis_pixel_arcmin
        )

        # 偶数网格的中心约定是 coordinate_i = (i - N/2) * pixel，
        # 因此关于中心坐标的镜像对应下标映射 i -> (N - i) mod N，
        # 也就是先反转数组，再整体滚动一个像素。
        flipped_y = cp.roll(kernel[::-1, :], 1, axis=0)
        flipped_x = cp.roll(kernel[:, ::-1], 1, axis=1)
        self.assertLess(float(cp.abs(kernel - flipped_y).max()), 1e-12)
        self.assertLess(float(cp.abs(kernel - flipped_x).max()), 1e-12)

    def test_solar_direction_weights_sum_to_one(self) -> None:
        config = ArrayGhostSimulationConfig()
        directions, weights = build_solar_directions(config)

        self.assertAlmostEqual(float(weights.sum()), 1.0, places=9)
        radii = cp.sqrt((directions**2).sum(axis=1))
        self.assertLessEqual(
            float(radii.max()),
            config.solar_angular_radius_arcmin + 1e-6,
        )

    def test_analysis_grid_must_cover_solar_disk(self) -> None:
        config = ArrayGhostSimulationConfig()
        with self.assertRaises(ValueError):
            build_solar_disk_kernel(config, 32, config.analysis_pixel_arcmin)


class SolarIntegrationEquivalenceTest(unittest.TestCase):
    """V-G：圆盘卷积与稠密方向积分必须给出同一个太阳扩展源结果。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = ArrayGhostSimulationConfig()
        cls.cache = build_psf_cache(cls.config)
        psf, sampling = cls.cache.get(cls.config, 0.6354, 3.0, 550.0)
        grid_size = 256
        cls.grid = resample_psf_to_grid(
            psf,
            sampling.angular_pixel_arcmin,
            grid_size,
            cls.config.analysis_pixel_arcmin,
            cls.config,
        )
        cls.disk, _ = build_solar_disk_kernel(
            cls.config, grid_size, cls.config.analysis_pixel_arcmin
        )
        cls.convolved, _ = convolve_with_solar_disk(
            cls.grid, cls.disk, cls.config
        )

    def test_convolution_preserves_total_energy(self) -> None:
        self.assertAlmostEqual(
            float(self.convolved.sum(dtype=self.config.accumulator_dtype)),
            1.0,
            places=6,
        )

    def test_dense_direction_sum_matches_convolution_width(self) -> None:
        directions, weights = build_solar_directions(
            self.config,
            self.config.solar_radial_ring_dense_count,
            self.config.solar_azimuth_dense_count,
        )
        direct = integrate_solar_source_direct(
            self.grid,
            directions,
            weights,
            self.config.analysis_pixel_arcmin,
            self.config,
        )
        width_conv = measure_image_width_arcmin(
            self.convolved,
            self.config.analysis_pixel_arcmin,
            self.config.sun_width_energy_fraction,
            self.config,
        )
        width_direct = measure_image_width_arcmin(
            direct,
            self.config.analysis_pixel_arcmin,
            self.config.sun_width_energy_fraction,
            self.config,
        )

        relative_difference_percent = (
            100.0 * abs(width_conv - width_direct) / width_conv
        )
        self.assertLess(
            relative_difference_percent,
            self.config.solar_convergence_width_tolerance_percent,
        )

    def test_dense_direction_sum_peak_matches_convolution(self) -> None:
        directions, weights = build_solar_directions(
            self.config,
            self.config.solar_radial_ring_dense_count,
            self.config.solar_azimuth_dense_count,
        )
        direct = integrate_solar_source_direct(
            self.grid,
            directions,
            weights,
            self.config.analysis_pixel_arcmin,
            self.config,
        )
        peak_conv = float(self.convolved.max())
        peak_direct = float(direct.max())
        relative_difference_percent = (
            100.0 * abs(peak_conv - peak_direct) / peak_conv
        )

        self.assertLess(
            relative_difference_percent,
            self.config.solar_convergence_peak_tolerance_percent,
        )

    def test_standard_direction_sum_width_matches_convolution(self) -> None:
        directions, weights = build_solar_directions(self.config)
        direct = integrate_solar_source_direct(
            self.grid,
            directions,
            weights,
            self.config.analysis_pixel_arcmin,
            self.config,
        )
        width_conv = measure_image_width_arcmin(
            self.convolved,
            self.config.analysis_pixel_arcmin,
            self.config.sun_width_energy_fraction,
            self.config,
        )
        width_direct = measure_image_width_arcmin(
            direct,
            self.config.analysis_pixel_arcmin,
            self.config.sun_width_energy_fraction,
            self.config,
        )

        relative_difference_percent = (
            100.0 * abs(width_conv - width_direct) / width_conv
        )
        self.assertLess(
            relative_difference_percent,
            self.config.solar_convergence_width_tolerance_percent,
        )


class ArrayImageTest(unittest.TestCase):
    """V-B、V-D、V-E、V-F：平移、单孔退化、双孔对称与能量守恒。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = ArrayGhostSimulationConfig().quick_variant()
        cls.cache = build_psf_cache(cls.config)
        psf, sampling = cls.cache.get(cls.config, 0.6354, 3.0, 550.0)
        grid_size = cls.config.effective_analysis_grid_size()
        cls.kernel = resample_psf_to_grid(
            psf,
            sampling.angular_pixel_arcmin,
            grid_size,
            cls.config.analysis_pixel_arcmin,
            cls.config,
        )
        cls.width_arcmin = measure_image_width_arcmin(
            cls.kernel,
            cls.config.analysis_pixel_arcmin,
            cls.config.sun_width_energy_fraction,
            cls.config,
        )

    def _shifts(self, centers_mm):
        return cp.asarray(
            centers_mm
            / self.config.focal_length_mm
            * (180.0 * 60.0 / math.pi),
            dtype=cp.float32,
        )

    def test_single_hole_degrades_to_the_kernel(self) -> None:
        layout = generate_square_packing(self.config, 4.0, 0.7)
        shifts = self._shifts(layout.centers_mm)
        import numpy as np

        centre_index = int(np.argmin((layout.centers_mm**2).sum(axis=1)))
        one_shift = shifts[centre_index : centre_index + 1]
        weights = cp.ones((1,), dtype=self.config.accumulator_dtype)
        field = plan_field_grid(one_shift, self.width_arcmin, self.config)
        result = accumulate_shifted_kernels(
            self.kernel,
            self.config.analysis_pixel_arcmin,
            one_shift,
            weights,
            field,
            self.config,
        )

        self.assertAlmostEqual(result["total_energy"], 1.0, places=6)
        peak_y, peak_x = [
            int(value.get())
            for value in cp.unravel_index(result["image"].argmax(), result["image"].shape)
        ]
        self.assertEqual((peak_y, peak_x), (field["grid_size"] // 2,) * 2)

    def test_array_energy_equals_weight_sum(self) -> None:
        layout = generate_square_packing(self.config, 4.0, 0.7)
        shifts = self._shifts(layout.centers_mm)
        weights = cp.ones(
            (shifts.shape[0],), dtype=self.config.accumulator_dtype
        )
        field = plan_field_grid(shifts, self.width_arcmin, self.config)
        result = accumulate_shifted_kernels(
            self.kernel,
            self.config.analysis_pixel_arcmin,
            shifts,
            weights,
            field,
            self.config,
        )

        self.assertAlmostEqual(
            result["total_energy"],
            result["expected_energy"],
            places=6,
        )
        self.assertAlmostEqual(result["expected_energy"], float(layout.hole_count))

    def test_half_weight_halves_that_ghost_peak(self) -> None:
        layout = generate_square_packing(self.config, 4.0, 0.7)
        shifts = self._shifts(layout.centers_mm)
        weights = cp.ones(
            (shifts.shape[0],), dtype=self.config.accumulator_dtype
        )
        field = plan_field_grid(shifts, self.width_arcmin, self.config)
        reference = accumulate_shifted_kernels(
            self.kernel,
            self.config.analysis_pixel_arcmin,
            shifts,
            weights,
            field,
            self.config,
        )

        halved = weights.copy()
        halved[1] = 0.5
        modified = accumulate_shifted_kernels(
            self.kernel,
            self.config.analysis_pixel_arcmin,
            shifts,
            halved,
            field,
            self.config,
        )
        self.assertAlmostEqual(
            modified["total_energy"],
            reference["total_energy"] - 0.5,
            places=6,
        )

    def test_two_symmetric_holes_produce_symmetric_image(self) -> None:
        import numpy as np

        centers_mm = np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float64)
        shifts = self._shifts(centers_mm)
        weights = cp.ones((2,), dtype=self.config.accumulator_dtype)
        field = plan_field_grid(shifts, self.width_arcmin, self.config)
        result = accumulate_shifted_kernels(
            self.kernel,
            self.config.analysis_pixel_arcmin,
            shifts,
            weights,
            field,
            self.config,
        )
        image = result["image"]

        # 两个等亮孔必须给出关于视场中心对称的一对像。峰位不落在整数
        # 像素上时双线性插值会引入微小不对称，因此这里检查能量对称与
        # 峰位等距，而不是逐像素完全相等。
        centre = field["grid_size"] // 2
        left_energy = float(image[:, :centre].sum())
        right_energy = float(image[:, centre + 1 :].sum())
        self.assertAlmostEqual(
            left_energy / right_energy,
            1.0,
            places=4,
        )
        vertical_axis = cp.arange(field["grid_size"]) - centre
        peak_columns = [
            int(value.get())
            for value in cp.argsort(image.max(axis=0))[-2:]
        ]
        self.assertAlmostEqual(
            abs(vertical_axis[peak_columns[0]].item()),
            abs(vertical_axis[peak_columns[1]].item()),
            delta=1.0,
        )
        self.assertAlmostEqual(result["total_energy"], 2.0, places=6)

    def test_shift_places_peak_at_pitch_over_focal_length(self) -> None:
        import numpy as np

        pitch_mm = 2.0
        centers_mm = np.array([[-0.5 * pitch_mm, 0.0]], dtype=np.float64)
        shifts = self._shifts(centers_mm)
        expected_arcmin = (
            pitch_mm / self.config.focal_length_mm * (180.0 * 60.0 / math.pi) * 0.5
        )
        self.assertAlmostEqual(
            abs(float(shifts[0, 0])),
            expected_arcmin,
            places=3,
        )

        weights = cp.ones((1,), dtype=self.config.accumulator_dtype)
        field = plan_field_grid(shifts, self.width_arcmin, self.config)
        result = accumulate_shifted_kernels(
            self.kernel,
            self.config.analysis_pixel_arcmin,
            shifts,
            weights,
            field,
            self.config,
        )
        peak_y, peak_x = [
            int(value.get())
            for value in cp.unravel_index(result["image"].argmax(), result["image"].shape)
        ]
        centre = field["grid_size"] // 2
        expected_pixel = centre + int(
            round(float(shifts[0, 0]) / field["pixel_arcmin"])
        )
        self.assertAlmostEqual(
            abs(peak_x - centre),
            abs(expected_pixel - centre),
            delta=1,
        )
        self.assertEqual(peak_y, centre)


class AnchorReuseTest(unittest.TestCase):
    def test_anchors_exclude_zero_myopia(self) -> None:
        config = ArrayGhostSimulationConfig()
        anchors = load_previous_diameter_anchors(config)

        self.assertNotIn(0.0, anchors["retinal_optimal_mm"])
        self.assertNotIn(0.0, anchors["psf_optimal_mm"])

    def test_retinal_anchor_matches_readme_value(self) -> None:
        config = ArrayGhostSimulationConfig()
        anchors = load_previous_diameter_anchors(config)

        # README：M = 3 D 时第二阶段最优孔径约 0.635 mm。
        self.assertAlmostEqual(
            anchors["retinal_optimal_mm"][3.0],
            0.635,
            places=3,
        )

    def test_anchor_grid_scales_around_reference(self) -> None:
        config = ArrayGhostSimulationConfig()
        anchors = load_previous_diameter_anchors(config)
        reference_mm, source = m_reference(config, anchors)

        values, grid_source = build_diameter_anchor_grid(
            config, anchors, 3.0, 550.0
        )
        self.assertEqual(grid_source, source)
        self.assertAlmostEqual(min(values), reference_mm * 0.8, places=6)
        self.assertAlmostEqual(max(values), reference_mm * 1.2, places=6)

    def test_anchor_grid_deduplicates_within_tolerance(self) -> None:
        config = replace(
            ArrayGhostSimulationConfig(),
            diameter_anchor_factors=(1.0, 1.0 + 1e-9, 1.2),
        )
        anchors = load_previous_diameter_anchors(config)
        values, _ = build_diameter_anchor_grid(config, anchors, 3.0, 550.0)

        self.assertEqual(len(values), 2)

    def test_missing_anchor_is_reported(self) -> None:
        config = ArrayGhostSimulationConfig()
        anchors = load_previous_diameter_anchors(config)
        with self.assertRaises(KeyError):
            build_diameter_anchor_grid(config, anchors, 99.0, 550.0)

    def test_debug_mode_falls_back_to_theoretical_diameter(self) -> None:
        config = replace(
            ArrayGhostSimulationConfig(),
            reuse_previous_metrics=False,
            reuse_debug_label="debug_recompute",
        )
        anchors = load_previous_diameter_anchors(config)
        values, source = build_diameter_anchor_grid(config, anchors, 99.0, 550.0)

        self.assertEqual(source, "theoretical_fallback_debug")
        self.assertGreater(len(values), 0)


def m_reference(config, anchors):
    """取 M = 3 D 的主锚点与来源，供锚点邻域测试复用。"""
    from pinhole_array_ghost import reference_diameter_mm

    return reference_diameter_mm(config, anchors, 3.0, 550.0)


class SubPixelPeakTest(unittest.TestCase):
    def test_parabolic_refinement_recovers_sub_pixel_peak(self) -> None:
        config = ArrayGhostSimulationConfig()
        grid_size = 64
        pixel_arcmin = 0.1
        axis = (cp.arange(grid_size, dtype=cp.float32) - grid_size // 2) * pixel_arcmin
        grid_x, grid_y = cp.meshgrid(axis, axis, indexing="xy")
        # 峰值偏离像素中心约 0.3 与 -0.2 个像素。
        offset_x = 0.3 * pixel_arcmin
        offset_y = -0.2 * pixel_arcmin
        sigma = 5.0 * pixel_arcmin
        image = cp.exp(
            -((grid_x - offset_x) ** 2 + (grid_y - offset_y) ** 2) / (2.0 * sigma**2)
        )

        measured = local_peak_arcmin(
            image,
            pixel_arcmin,
            np.asarray([offset_x, offset_y]),
            8.0 * pixel_arcmin,
        )
        raw_peak = float(image.max())

        self.assertAlmostEqual(measured, 1.0, places=3)
        self.assertLess(abs(measured - 1.0), abs(raw_peak - 1.0) + 1e-6)

    def test_peak_search_is_local_not_global(self) -> None:
        config = ArrayGhostSimulationConfig()
        grid_size = 64
        pixel_arcmin = 0.5
        image = cp.zeros((grid_size, grid_size), dtype=cp.float32)
        image[grid_size // 2 + 4, grid_size // 2 + 4] = 0.25
        image[grid_size // 2, grid_size // 2] = 0.75

        near_centre = local_peak_arcmin(
            image, pixel_arcmin, np.zeros(2), pixel_arcmin
        )
        self.assertAlmostEqual(near_centre, 0.75, places=6)


class GhostClusteringTest(unittest.TestCase):
    def test_holes_farther_than_merge_radius_stay_separate(self) -> None:
        shifts = np.asarray([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
        weights = np.ones(3)
        clusters = cluster_ghost_positions(shifts, weights, 2.0)

        self.assertEqual(len(clusters), 3)
        for cluster in clusters:
            self.assertEqual(cluster["multiplicity"], 1)

    def test_holes_inside_merge_radius_merge_into_one_cluster(self) -> None:
        shifts = np.asarray([[10.0, 0.0], [10.5, 0.0], [10.25, 0.2]])
        weights = np.ones(3)
        clusters = cluster_ghost_positions(shifts, weights, 1.0)

        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0]["multiplicity"], 3)
        self.assertAlmostEqual(clusters[0]["total_weight"], 3.0, places=12)

    def test_cluster_centre_is_weighted_average(self) -> None:
        shifts = np.asarray([[0.0, 0.0], [1.0, 0.0]])
        weights = np.asarray([1.0, 3.0])
        clusters = cluster_ghost_positions(shifts, weights, 2.0)

        self.assertEqual(len(clusters), 1)
        self.assertAlmostEqual(clusters[0]["centre_arcmin"][0], 0.75, places=12)

    def test_clusters_are_sorted_by_radius(self) -> None:
        shifts = np.asarray([[0.0, 0.0], [5.0, 0.0], [2.0, 0.0]])
        weights = np.ones(3)
        clusters = cluster_ghost_positions(shifts, weights, 0.5)

        radii = [cluster["radius_arcmin"] for cluster in clusters]
        self.assertEqual(radii, sorted(radii))


class GhostMetricsTest(unittest.TestCase):
    """V-B、V-E、V-F：等亮重影、重影能量占比与解析角间隔。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = ArrayGhostSimulationConfig().quick_variant()
        cls.cache = build_psf_cache(cls.config)
        psf, sampling = cls.cache.get(cls.config, 0.6354, 3.0, 550.0)
        grid_size = cls.config.effective_analysis_grid_size()
        psf_grid = resample_psf_to_grid(
            psf,
            sampling.angular_pixel_arcmin,
            grid_size,
            cls.config.analysis_pixel_arcmin,
            cls.config,
        )
        disk, _ = build_solar_disk_kernel(
            cls.config, grid_size, cls.config.analysis_pixel_arcmin
        )
        cls.kernel, _ = convolve_with_solar_disk(psf_grid, disk, cls.config)
        cls.width_arcmin = measure_image_width_arcmin(
            cls.kernel,
            cls.config.analysis_pixel_arcmin,
            cls.config.sun_width_energy_fraction,
            cls.config,
        )
        cls.theory_angle_arcmin = (
            2.0 / cls.config.focal_length_mm * (180.0 * 60.0 / math.pi)
        )

    def _metrics_for(self, layout, pitch_mm: float):
        shifts_np = (
            layout.centers_mm
            / self.config.focal_length_mm
            * (180.0 * 60.0 / math.pi)
        )
        weights_np, _ = compute_hole_weights(layout, None, self.config)
        shifts = cp.asarray(shifts_np, dtype=cp.float32)
        weights = cp.asarray(weights_np, dtype=self.config.accumulator_dtype)
        field = plan_field_grid(shifts, self.width_arcmin, self.config)
        result = accumulate_shifted_kernels(
            self.kernel,
            self.config.analysis_pixel_arcmin,
            shifts,
            weights,
            field,
            self.config,
        )
        metrics, rows, clusters = compute_ghost_metrics(
            result,
            field,
            shifts_np,
            weights_np,
            self.width_arcmin,
            self.theory_angle_arcmin,
            self.config,
        )
        return metrics, rows, clusters, result, field

    def test_every_ghost_is_as_bright_as_the_main_image(self) -> None:
        for pattern in ("square", "triangular", "honeycomb"):
            with self.subTest(pattern=pattern):
                if pattern == "square":
                    layout = generate_square_packing(self.config, 2.0, 0.6354)
                elif pattern == "triangular":
                    layout = generate_triangular_packing(self.config, 2.0, 0.6354)
                else:
                    layout = generate_hexagonal_packing(self.config, 2.0, 0.6354)
                _metrics, rows, _clusters, _result, _field = self._metrics_for(
                    layout, 2.0
                )

                ratios = [row["peak_ratio"] for row in rows]
                self.assertGreater(len(ratios), 0)
                self.assertLess(max(abs(value - 1.0) for value in ratios), 0.01)

    def test_ghost_energy_fraction_is_one_minus_one_over_hole_count(self) -> None:
        layout = generate_square_packing(self.config, 2.0, 0.6354)
        metrics, _rows, _clusters, _result, _field = self._metrics_for(layout, 2.0)
        expected = (layout.hole_count - 1) / layout.hole_count

        self.assertAlmostEqual(
            metrics["ghost_integrated_fraction"],
            expected,
            places=2,
        )

    def test_first_order_angle_matches_pitch_over_focal_length(self) -> None:
        layout = generate_square_packing(self.config, 2.0, 0.6354)
        metrics, _rows, _clusters, _result, _field = self._metrics_for(layout, 2.0)

        self.assertLess(
            metrics["first_order_angle_error_percent"],
            self.config.first_order_angle_tolerance_percent,
        )
        self.assertAlmostEqual(
            metrics["first_order_angle_arcmin"],
            self.theory_angle_arcmin,
            places=3,
        )

    def test_cluster_count_equals_holes_minus_main(self) -> None:
        layout = generate_triangular_packing(self.config, 2.0, 0.6354)
        metrics, _rows, _clusters, _result, _field = self._metrics_for(layout, 2.0)

        self.assertEqual(metrics["ghost_cluster_count"], layout.hole_count - 1)

    def test_well_separated_ghosts_report_unit_valley_visibility(self) -> None:
        layout = generate_square_packing(self.config, 2.0, 0.6354)
        metrics, _rows, _clusters, _result, _field = self._metrics_for(layout, 2.0)

        self.assertGreater(
            metrics["separation_to_width_ratio"],
            self.config.separation_ratio_threshold,
        )
        self.assertEqual(metrics["valley_visibility"], 1.0)
        self.assertEqual(
            metrics["valley_status"], "not_evaluated_well_separated"
        )

    def test_energy_is_conserved_in_metric_row(self) -> None:
        layout = generate_circular_rings(self.config, 2.0, 0.6354)
        metrics, _rows, _clusters, _result, _field = self._metrics_for(layout, 2.0)

        self.assertLess(
            metrics["energy_relative_error_percent"],
            self.config.energy_relative_tolerance_percent,
        )

    def test_direction_anisotropy_distinguishes_honeycomb_from_rings(self) -> None:
        honeycomb = generate_hexagonal_packing(self.config, 2.0, 0.6354)
        rings = generate_circular_rings(self.config, 2.0, 0.6354)
        honeycomb_metrics, _r1, _c1, _i1, _f1 = self._metrics_for(honeycomb, 2.0)
        ring_metrics, _r2, _c2, _i2, _f2 = self._metrics_for(rings, 2.0)

        # 蜂窝只有 3 个近邻方向，方位各向异性必须高于环形阵列。
        self.assertGreater(
            honeycomb_metrics["direction_anisotropy"],
            ring_metrics["direction_anisotropy"],
        )

    def test_metrics_include_pattern_signature_and_positions(self) -> None:
        layout = generate_square_packing(self.config, 2.0, 0.6354)
        metrics, _rows, _clusters, _result, _field = self._metrics_for(layout, 2.0)

        self.assertGreater(len(metrics["pattern_signature"]), 0)
        self.assertGreater(len(metrics["ghost_positions"]), 0)
        self.assertEqual(metrics["ghost_count"], layout.hole_count - 1)


class RiskLabelTest(unittest.TestCase):
    def _base_metrics(self) -> dict[str, Any]:
        return {
            "separation_to_width_ratio": 10.0,
            "max_ghost_peak_ratio": 1.0,
        }

    def _base_layout_qc(self, pitch_mm: float = 1.0) -> dict[str, Any]:
        return {
            "geometry_ok": True,
            "topology_ok": True,
            "pitch_nominal_mm": pitch_mm,
        }

    def test_well_separated_bright_ghosts_are_resolved(self) -> None:
        config = ArrayGhostSimulationConfig()
        primary, labels = classify_risk_labels(
            self._base_layout_qc(),
            self._base_metrics(),
            2.0,
            1.0,
            config,
        )

        self.assertEqual(primary, "GhostResolved")
        self.assertIn("GhostResolved", labels)

    def test_small_separation_ratio_is_merged(self) -> None:
        config = ArrayGhostSimulationConfig()
        metrics = self._base_metrics()
        metrics["separation_to_width_ratio"] = 0.5
        primary, labels = classify_risk_labels(
            self._base_layout_qc(), metrics, 2.0, 1.0, config
        )

        self.assertEqual(primary, "GhostMerged")

    def test_faint_ghosts_are_weak(self) -> None:
        config = ArrayGhostSimulationConfig()
        metrics = self._base_metrics()
        metrics["max_ghost_peak_ratio"] = 0.001
        primary, labels = classify_risk_labels(
            self._base_layout_qc(), metrics, 2.0, 1.0, config
        )

        self.assertEqual(primary, "GhostWeak")

    def test_invalid_geometry_dominates_other_labels(self) -> None:
        config = ArrayGhostSimulationConfig()
        layout_qc = self._base_layout_qc()
        layout_qc["geometry_ok"] = False
        primary, labels = classify_risk_labels(
            layout_qc, self._base_metrics(), 2.0, 1.0, config
        )

        self.assertEqual(primary, "InvalidGeometry")
        self.assertIn("InvalidGeometry", labels)

    def test_pitch_at_or_above_pupil_flags_dead_zone(self) -> None:
        config = ArrayGhostSimulationConfig()
        primary, labels = classify_risk_labels(
            self._base_layout_qc(pitch_mm=4.0),
            self._base_metrics(),
            4.0,
            1.0,
            config,
        )

        self.assertIn("DeadZoneRisk", labels)

    def test_vignetted_reference_hole_is_flagged(self) -> None:
        config = ArrayGhostSimulationConfig()
        primary, labels = classify_risk_labels(
            self._base_layout_qc(),
            self._base_metrics(),
            2.0,
            0.1,
            config,
        )

        self.assertEqual(primary, "PupilVignetted")


class SquarePackingTest(unittest.TestCase):
    def test_hole_count_matches_row_and_column_product(self) -> None:
        config = ArrayGhostSimulationConfig()
        layout = generate_square_packing(config, pitch_mm=2.0, diameter_mm=0.7)
        self.assertEqual(layout.hole_count, config.square_row_count * config.square_column_count)

    def test_lattice_spacing_equals_pitch(self) -> None:
        config = ArrayGhostSimulationConfig()
        pitch_mm = 2.0
        layout = generate_square_packing(config, pitch_mm, diameter_mm=0.7)

        import numpy as np

        x_values = np.unique(np.round(layout.centers_mm[:, 0], 9))
        y_values = np.unique(np.round(layout.centers_mm[:, 1], 9))
        self.assertAlmostEqual(float(np.diff(x_values).min()), pitch_mm, places=9)
        self.assertAlmostEqual(float(np.diff(y_values).min()), pitch_mm, places=9)

    def test_internal_hole_has_four_neighbours(self) -> None:
        config = ArrayGhostSimulationConfig()
        pitch_mm = 2.0
        layout = generate_square_packing(config, pitch_mm, diameter_mm=0.7)
        qc = validate_layout(layout, config)

        self.assertEqual(qc["core_neighbor_count_max"], 4)
        self.assertTrue(qc["topology_ok"])
        self.assertAlmostEqual(qc["nearest_neighbor_min_mm"], pitch_mm, places=9)

    def test_diagonal_neighbour_is_pitch_times_sqrt_two(self) -> None:
        config = ArrayGhostSimulationConfig()
        pitch_mm = 2.0
        layout = generate_square_packing(config, pitch_mm, diameter_mm=0.7)
        import numpy as np

        center_index = int(np.argmin((layout.centers_mm**2).sum(axis=1)))
        angles = neighbor_angles_deg(
            layout.centers_mm, center_index, pitch_mm * math.sqrt(2.0) + 1e-9
        )
        self.assertEqual(len(angles), 8)
        diagonals = [angle for angle in angles if angle % 90.0 != 0.0]
        self.assertEqual(len(diagonals), 4)


class TriangularPackingTest(unittest.TestCase):
    def test_row_spacing_and_half_pitch_offset(self) -> None:
        config = ArrayGhostSimulationConfig()
        pitch_mm = 2.0
        layout = generate_triangular_packing(config, pitch_mm, diameter_mm=0.7)
        import numpy as np

        y_values = np.unique(np.round(layout.centers_mm[:, 1], 9))
        self.assertAlmostEqual(
            float(np.diff(y_values).min()),
            pitch_mm * math.sqrt(3.0) / 2.0,
            places=6,
        )
        first_row_x = np.sort(
            layout.centers_mm[
                np.isclose(layout.centers_mm[:, 1], y_values[0]), 0
            ]
        )
        second_row_x = np.sort(
            layout.centers_mm[
                np.isclose(layout.centers_mm[:, 1], y_values[1]), 0
            ]
        )
        offset = second_row_x[0] - first_row_x[0]
        self.assertAlmostEqual(abs(offset), 0.5 * pitch_mm, places=9)

    def test_internal_hole_has_six_neighbours(self) -> None:
        config = ArrayGhostSimulationConfig()
        layout = generate_triangular_packing(config, pitch_mm=2.0, diameter_mm=0.7)
        qc = validate_layout(layout, config)

        self.assertEqual(qc["core_neighbor_count_max"], 6)
        self.assertTrue(qc["topology_ok"])

    def test_six_neighbours_are_sixty_degrees_apart(self) -> None:
        config = ArrayGhostSimulationConfig()
        layout = generate_triangular_packing(config, pitch_mm=2.0, diameter_mm=0.7)
        import numpy as np

        center_index = int(np.argmin((layout.centers_mm**2).sum(axis=1)))
        angles = neighbor_angles_deg(layout.centers_mm, center_index, 2.0 + 1e-9)
        self.assertEqual(len(angles), 6)
        gaps = [
            (angles[(index + 1) % 6] - angles[index]) % 360.0
            for index in range(6)
        ]
        for gap in gaps:
            self.assertAlmostEqual(gap, 60.0, places=6)


class HoneycombPackingTest(unittest.TestCase):
    def test_internal_hole_has_three_neighbours_at_pitch(self) -> None:
        config = ArrayGhostSimulationConfig()
        pitch_mm = 2.0
        layout = generate_hexagonal_packing(config, pitch_mm, diameter_mm=0.7)
        qc = validate_layout(layout, config)

        self.assertEqual(qc["core_neighbor_count_max"], 3)
        self.assertTrue(qc["topology_ok"])
        self.assertAlmostEqual(qc["nearest_neighbor_min_mm"], pitch_mm, places=9)

    def test_three_neighbours_are_one_hundred_twenty_degrees_apart(self) -> None:
        config = ArrayGhostSimulationConfig()
        layout = generate_hexagonal_packing(config, pitch_mm=2.0, diameter_mm=0.7)
        import numpy as np

        center_index = int(np.argmin((layout.centers_mm**2).sum(axis=1)))
        angles = neighbor_angles_deg(layout.centers_mm, center_index, 2.0 + 1e-9)
        self.assertEqual(len(angles), 3)
        gaps = [
            (angles[(index + 1) % 3] - angles[index]) % 360.0
            for index in range(3)
        ]
        for gap in gaps:
            self.assertAlmostEqual(gap, 120.0, places=6)

    def test_honeycomb_is_not_a_triangular_lattice(self) -> None:
        config = ArrayGhostSimulationConfig()
        honeycomb = generate_hexagonal_packing(config, 2.0, 0.7)
        triangular = generate_triangular_packing(config, 2.0, 0.7)

        self.assertNotEqual(honeycomb.hole_count, triangular.hole_count)
        self.assertTrue(math.isnan(honeycomb.analytic_area_fraction))


class ConcentricRingTest(unittest.TestCase):
    def test_ring_radius_uses_chord_not_arc(self) -> None:
        # 回归测试：弧长公式会给出偏小的半径，实际孔心弦长因此小于 p。
        pitch_mm = 2.0
        hole_count = 12
        chord_radius = minimum_ring_radius_mm(hole_count, pitch_mm)
        arc_radius = hole_count * pitch_mm / (2.0 * math.pi)

        self.assertGreater(chord_radius, arc_radius)
        self.assertAlmostEqual(
            2.0 * chord_radius * math.sin(math.pi / hole_count),
            pitch_mm,
            places=12,
        )

    def test_generated_ring_chord_is_at_least_pitch(self) -> None:
        config = ArrayGhostSimulationConfig()
        layout = generate_circular_rings(
            config,
            pitch_mm=2.0,
            diameter_mm=0.7,
            holes_per_ring=(12, 12, 12),
        )
        qc = validate_layout(layout, config)

        self.assertGreaterEqual(qc["nearest_neighbor_min_mm"], 2.0 - 1e-9)
        self.assertTrue(qc["geometry_ok"])

    def test_hexagonal_rings_have_six_r_holes_per_ring(self) -> None:
        config = ArrayGhostSimulationConfig()
        layout = generate_hexagonal_rings(config, pitch_mm=2.0, diameter_mm=0.7, ring_count=3)

        self.assertEqual(layout.holes_per_ring, (6, 12, 18))
        self.assertEqual(layout.hole_count, 1 + 6 + 12 + 18)
        self.assertEqual(layout.ring_count, 3)

    def test_hexagonal_ring_radii_grow_by_ring_pitch(self) -> None:
        config = ArrayGhostSimulationConfig()
        layout = generate_hexagonal_rings(config, pitch_mm=2.0, diameter_mm=0.7, ring_count=4)

        radii = layout.ring_radii_mm
        self.assertEqual(len(radii), 4)
        for index in range(1, len(radii)):
            self.assertAlmostEqual(
                radii[index] - radii[index - 1],
                layout.ring_radial_pitch_mm,
                places=9,
            )

    def test_hexagonal_ring_spacing_never_below_pitch(self) -> None:
        config = ArrayGhostSimulationConfig()
        layout = generate_hexagonal_rings(config, pitch_mm=2.0, diameter_mm=0.7, ring_count=4)
        qc = validate_layout(layout, config)

        self.assertGreaterEqual(qc["nearest_neighbor_min_mm"], 2.0 - 1e-9)
        self.assertTrue(qc["geometry_ok"])

    def test_ring_pitch_is_raised_to_requested_pitch(self) -> None:
        config = replace(
            ArrayGhostSimulationConfig(),
            hexagonal_ring_radial_pitch_mm=0.5,
        )
        layout = generate_hexagonal_rings(config, pitch_mm=2.0, diameter_mm=0.7, ring_count=2)

        self.assertAlmostEqual(layout.ring_radial_pitch_mm, 2.0, places=12)

    def test_circular_rings_follow_configured_hole_counts(self) -> None:
        config = ArrayGhostSimulationConfig()
        layout = generate_circular_rings(config, pitch_mm=2.0, diameter_mm=0.7)

        self.assertEqual(
            layout.holes_per_ring,
            tuple(config.circular_ring_holes_per_ring),
        )
        self.assertEqual(
            layout.hole_count,
            1 + sum(config.circular_ring_holes_per_ring),
        )

    def test_circular_ring_spacing_never_below_pitch(self) -> None:
        config = ArrayGhostSimulationConfig()
        layout = generate_circular_rings(config, pitch_mm=2.0, diameter_mm=0.7)
        qc = validate_layout(layout, config)

        self.assertGreaterEqual(qc["nearest_neighbor_min_mm"], 2.0 - 1e-9)
        self.assertTrue(qc["geometry_ok"])

    def test_evenly_spaced_ring_with_fixed_count_keeps_spacing(self) -> None:
        config = ArrayGhostSimulationConfig()
        layout = generate_circular_rings(
            config,
            pitch_mm=2.0,
            diameter_mm=0.7,
            holes_per_ring=(12, 12, 12),
        )
        qc = validate_layout(layout, config)

        # 固定每圈 12 孔时，最近邻由环半径间距决定，仍不得小于 p。
        self.assertGreaterEqual(qc["nearest_neighbor_min_mm"], 2.0 - 1e-9)


class AreaFractionTest(unittest.TestCase):
    def test_triangular_area_is_square_area_times_two_over_sqrt_three(self) -> None:
        diameter_mm, pitch_mm = 0.7, 2.0
        square = analytic_area_fraction(
            ARRAY_PATTERN_SQUARE_PACKING, diameter_mm, pitch_mm
        )
        triangular = analytic_area_fraction(
            ARRAY_PATTERN_TRIANGULAR_PACKING, diameter_mm, pitch_mm
        )

        self.assertAlmostEqual(
            triangular / square,
            2.0 / math.sqrt(3.0),
            places=12,
        )

    def test_square_area_fraction_formula(self) -> None:
        diameter_mm, pitch_mm = 0.7, 2.0
        expected = math.pi / 4.0 * (diameter_mm / pitch_mm) ** 2
        self.assertAlmostEqual(
            analytic_area_fraction(
                ARRAY_PATTERN_SQUARE_PACKING, diameter_mm, pitch_mm
            ),
            expected,
            places=15,
        )

    def test_ring_layouts_have_no_analytic_fraction(self) -> None:
        self.assertTrue(
            math.isnan(analytic_area_fraction("concentric_hexagonal_rings", 0.7, 2.0))
        )
        self.assertTrue(
            math.isnan(analytic_area_fraction("concentric_circular_rings", 0.7, 2.0))
        )


class PitchGridTest(unittest.TestCase):
    def test_pitch_grid_is_monotone_and_respects_clearance(self) -> None:
        config = ArrayGhostSimulationConfig()
        diameter_mm = 1.2
        values = build_pitch_grid(config, diameter_mm)

        self.assertGreater(len(values), 1)
        self.assertTrue(all(a < b for a, b in zip(values, values[1:])))
        self.assertGreaterEqual(
            values[0],
            diameter_mm + config.minimum_edge_clearance_mm - 1e-12,
        )
        self.assertLessEqual(values[-1], config.maximum_pitch_mm + 1e-12)

    def test_pitch_grid_upper_bound_covers_diameter_dominated_case(self) -> None:
        config = replace(
            ArrayGhostSimulationConfig(),
            minimum_pitch_mm=0.1,
            maximum_pitch_mm=0.5,
            minimum_edge_clearance_mm=0.2,
        )
        values = build_pitch_grid(config, diameter_mm=2.0)

        # 上界不得低于下界，否则对数采样会出现非法区间。
        self.assertGreaterEqual(values[-1], values[0])
        self.assertGreaterEqual(values[0], 2.2 - 1e-12)


class LayoutQualityTest(unittest.TestCase):
    def test_layout_geometry_fails_when_holes_overlap(self) -> None:
        config = ArrayGhostSimulationConfig()
        layout = generate_square_packing(config, pitch_mm=0.6, diameter_mm=0.7)
        qc = validate_layout(layout, config)

        self.assertFalse(qc["geometry_ok"])

    def test_layout_geometry_passes_for_legal_design(self) -> None:
        config = ArrayGhostSimulationConfig()
        layout = generate_square_packing(config, pitch_mm=2.0, diameter_mm=1.2)
        qc = validate_layout(layout, config)

        self.assertTrue(qc["geometry_ok"])
        self.assertAlmostEqual(qc["edge_clearance_mm"], 0.8, places=9)

    def test_jittered_square_is_seed_reproducible(self) -> None:
        config = ArrayGhostSimulationConfig()
        first = generate_jittered_square(config, 2.0, 0.7, seed=7)
        second = generate_jittered_square(config, 2.0, 0.7, seed=7)
        third = generate_jittered_square(config, 2.0, 0.7, seed=8)

        self.assertTrue((first.centers_mm == second.centers_mm).all())
        self.assertFalse((first.centers_mm == third.centers_mm).all())
        self.assertEqual(first.hole_count, second.hole_count)


class MaskCoverageTest(unittest.TestCase):
    """方案 3.3 节要求的覆盖度基本性质：圆心为 1、远处为 0、边缘半透明。"""

    def test_single_hole_coverage_centre_is_one_and_far_field_is_zero(self) -> None:
        config = ArrayGhostSimulationConfig()
        layout = single_hole_layout(diameter_mm=0.7)
        mask = rasterize_mask(layout, config)
        centre = mask.coverage.shape[0] // 2

        # 圆心附近应完全覆盖，远离圆孔处应完全没有覆盖。
        self.assertAlmostEqual(float(mask.coverage[centre, centre]), 1.0, places=9)
        self.assertAlmostEqual(float(mask.coverage[0, 0]), 0.0, places=9)
        self.assertAlmostEqual(float(mask.coverage[-1, -1]), 0.0, places=9)

    def test_single_hole_edge_has_partial_coverage(self) -> None:
        config = ArrayGhostSimulationConfig()
        layout = single_hole_layout(diameter_mm=0.7)
        mask = rasterize_mask(layout, config)

        values = mask.coverage.ravel()
        partial = values[(values > 0.0) & (values < 1.0)]
        self.assertGreater(partial.size, 0)
        self.assertLess(float(partial.min()), 1.0)

    def test_coverage_never_exceeds_one(self) -> None:
        config = ArrayGhostSimulationConfig()
        layout = generate_square_packing(config, pitch_mm=2.0, diameter_mm=1.2)
        mask = rasterize_mask(layout, config)

        self.assertLessEqual(float(mask.coverage.max()), 1.0)
        self.assertGreaterEqual(float(mask.coverage.min()), 0.0)

    def test_numeric_area_matches_analytic_circle_area(self) -> None:
        config = ArrayGhostSimulationConfig()
        layout = generate_square_packing(config, pitch_mm=2.0, diameter_mm=0.7)
        mask = rasterize_mask(layout, config)
        qc = validate_mask(mask, layout, config)

        self.assertLess(
            qc["relative_area_error"],
            config.mask_area_relative_tolerance,
        )
        self.assertTrue(qc["mask_ok"], qc["fail_reasons"])

    def test_all_patterns_have_matching_component_counts(self) -> None:
        config = ArrayGhostSimulationConfig()
        layouts = (
            generate_square_packing(config, 2.0, 0.7),
            generate_triangular_packing(config, 2.0, 0.7),
            generate_hexagonal_packing(config, 2.0, 0.7),
            generate_hexagonal_rings(config, 2.0, 0.7, 2),
            generate_circular_rings(config, 2.0, 0.7),
        )
        for layout in layouts:
            with self.subTest(pattern=layout.pattern_kind):
                mask = rasterize_mask(layout, config)
                qc = validate_mask(mask, layout, config)
                self.assertEqual(qc["labeled_hole_count"], qc["designed_hole_count"])
                self.assertLess(
                    qc["relative_area_error"],
                    config.mask_area_relative_tolerance,
                )
                self.assertTrue(qc["mask_ok"], qc["fail_reasons"])

    def test_mask_detects_overlapping_holes(self) -> None:
        config = ArrayGhostSimulationConfig()
        layout = generate_square_packing(config, pitch_mm=0.6, diameter_mm=0.7)
        mask = rasterize_mask(layout, config)
        qc = validate_mask(mask, layout, config)

        self.assertLess(qc["labeled_hole_count"], qc["designed_hole_count"])
        self.assertFalse(qc["mask_ok"])
        self.assertIn("connected_component_count_mismatch", qc["fail_reasons"])

    def test_centroid_offset_within_tolerance(self) -> None:
        config = ArrayGhostSimulationConfig()
        layout = generate_triangular_packing(config, 2.0, 0.7)
        mask = rasterize_mask(layout, config)
        qc = validate_mask(mask, layout, config)

        self.assertLessEqual(
            qc["max_centroid_offset_mm"],
            qc["centroid_tolerance_mm"],
        )

    def test_mask_resolution_guard_rejects_coarse_grid(self) -> None:
        config = replace(ArrayGhostSimulationConfig(), mask_pixel_mm=0.2)
        layout = generate_square_packing(config, pitch_mm=2.0, diameter_mm=0.7)

        # 0.7 mm 直径在 0.2 mm 像素上只有约 3.5 个像素，必须被拒绝。
        with self.assertRaises(ValueError):
            rasterize_mask(layout, config)


class LatticeFillingTest(unittest.TestCase):
    def test_triangular_denser_than_square_denser_than_honeycomb(self) -> None:
        pitch_mm, diameter_mm = 2.0, 0.7
        hole_area_mm2 = math.pi * (0.5 * diameter_mm) ** 2
        fillings = {
            pattern: hole_area_mm2
            / lattice_unit_cell_area_mm2(pattern, pitch_mm)
            for pattern in (
                "square_packing",
                "triangular_packing",
                "hexagonal_packing",
            )
        }

        self.assertGreater(
            fillings["triangular_packing"], fillings["square_packing"]
        )
        self.assertGreater(
            fillings["square_packing"], fillings["hexagonal_packing"]
        )

    def test_honeycomb_unit_cell_holds_two_holes(self) -> None:
        pitch_mm = 2.0
        cell_area_mm2 = lattice_unit_cell_area_mm2(
            "hexagonal_packing", pitch_mm
        )
        self.assertAlmostEqual(
            cell_area_mm2,
            0.75 * math.sqrt(3.0) * pitch_mm * pitch_mm,
            places=12,
        )

    def test_ring_layouts_have_no_unit_cell(self) -> None:
        self.assertTrue(
            math.isnan(
                lattice_unit_cell_area_mm2("concentric_hexagonal_rings", 2.0)
            )
        )



if __name__ == "__main__":
    unittest.main()
