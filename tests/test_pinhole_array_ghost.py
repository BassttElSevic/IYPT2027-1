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
    minimum_ring_radius_mm,
    resolve_simulation_directory,
    validate_config,
    validate_layout,
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


if __name__ == "__main__":
    unittest.main()
