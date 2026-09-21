"""第三阶段多针孔阵列重影仿真的单元测试。

测试纪律沿用第二阶段：先用小网格和解析锚点验证，再进入正式扫描。
所有测试只使用少量孔和短向量，避免每次测试都加载大 GPU 数组。
"""

import math
import unittest
from dataclasses import replace
from pathlib import Path

import cupy as cp

from pinhole_array_ghost import (
    ARRAY_PATTERN_SQUARE_PACKING,
    ArrayGhostSimulationConfig,
    resolve_simulation_directory,
    validate_config,
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


if __name__ == "__main__":
    unittest.main()
