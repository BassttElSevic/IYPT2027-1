# 任务②：离焦视网膜像与视标卷积
# ============================================================
# 本文件完成第二个仿真任务：把任务①得到的单孔 PSF 作为模糊核，
# 对 E 字或 Landolt C 视标做卷积，得到近视者通过针孔看到的模糊视网膜像。
#
# 数据流：
#   视标数组（角坐标，非相干强度）
#     ↓
#   任务① PSF（同一 d、M、λ）
#     ↓ 重采样到统一角分辨率
#   二维卷积
#     ↓
#   模糊视网膜像
#     ↓ 提取平均对比度、模型阈值、MAR、logMAR
#
# 本文件只写任务②的输入、输出、采样、扫描、指标和绘图。
# 任务①的 PSF 物理模型直接从 single_hole_PSF.py 导入，
# 避免重复实现圆孔、离焦波前和 FFT 传播。
#
# 数值纪律：
#   1. 所有参数都放在 RetinaSimulationConfig 中。
#   2. 光学计算内部使用 SI 单位，绘图和报告使用角分、mm、nm。
#   3. 卷积与视标重采样使用 CuPy，在 GPU 上完成。
#   4. 生成文件统一放在 output/single_hole_retinal_image/。


import csv
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cupy as cp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
from cupyx.scipy.ndimage import map_coordinates
from cupyx.scipy.signal import fftconvolve

from single_hole_PSF import (
    SimulationConfig as PinholePSFSimulationConfig,
    build_diameter_grid,
    build_gpu_grids,
    build_myopia_values,
    build_sampling,
    build_soft_aperture,
    compute_psf,
    configure_bilingual_plot_font,
    enclosing_diameter_arcmin,
    nearest_diameter,
    theoretical_optimal_diameter_mm,
)


@dataclass(frozen=True)
class RetinaSimulationConfig:
    """任务②的全部输入、输出、采样与绘图参数。"""

    gpu_device_id: int = 0
    psf_config: PinholePSFSimulationConfig = field(
        default_factory=PinholePSFSimulationConfig
    )

    # 输出目录与文件。
    output_root_directory_name: str = "output"
    simulation_directory_name: str = "single_hole_retinal_image"
    metrics_filename: str = "optotype_metrics.csv"
    stroke_metrics_filename: str = "optotype_stroke_metrics.csv"
    config_filename: str = "retinal_image_config.json"
    primary_figure_filename: str = "primary_summary.png"
    wavelength_figure_filename: str = "wavelength_comparison.png"
    contrast_figure_filename: str = "contrast_curves.png"
    numerical_table_filename: str = "numerical_table.png"

    # 视标采样。像素数越大，FFT 卷积的内存和计算量越大。
    optotype_kind: str = "E"
    optotype_grid_size: int = 2048
    optotype_pixels_per_arcmin: float = 16.0
    optotype_anti_alias_fraction: float = 0.75
    psf_resampling_order: int = 1
    psf_resampling_margin_arcmin: float = 1.0
    convolution_mode: str = "full"
    floating_comparison_tolerance: float = 1.0e-9

    # E 字或 Landolt C 的笔画与空白宽度扫描点。
    # 在标准视标中，笔画与空白宽度等于最小分辨角 MAR。
    optotype_stroke_values_arcmin: tuple[float, ...] = (
        0.50,
        0.80,
        1.20,
        1.80,
        2.50,
        3.50,
        5.00,
        7.00,
        10.00,
        14.00,
        20.00,
    )

    # 模型对比度阈值。该值用于定义模型内部的比较判据，
    # 不代表视觉心理物理实验得到的真实可辨认阈值。
    model_contrast_threshold: float = 0.20
    optotype_ink_threshold: float = 0.50
    optotype_extent_strokes: float = 2.50

    # 重点波长与代表近视度。
    primary_wavelength_nm: float = 550.0
    representative_myopia_d: float = 3.0
    representative_diameters_mm: tuple[float, ...] = (
        0.30,
        0.60,
        1.20,
        2.00,
        3.00,
    )
    summary_myopia_values: tuple[float, ...] = (
        0.50,
        1.00,
        2.00,
        3.00,
        4.00,
        6.00,
    )

    # 绘图参数。
    figure_dpi: int = 300
    primary_figure_size_inches: tuple[float, float] = (30.0, 23.0)
    wavelength_figure_size_inches: tuple[float, float] = (30.0, 20.0)
    contrast_figure_size_inches: tuple[float, float] = (24.0, 15.0)
    validation_figure_size_inches: tuple[float, float] = (22.0, 12.0)
    primary_grid_rows: int = 4
    primary_grid_columns: int = 2
    primary_grid_height_ratios: tuple[float, ...] = (1.20, 0.90, 0.72, 0.58)
    primary_grid_hspace: float = 0.42
    primary_grid_wspace: float = 0.22
    primary_mosaic_wspace: float = 0.18
    wavelength_grid_rows: int = 3
    wavelength_grid_columns: int = 2
    wavelength_grid_height_ratios: tuple[float, ...] = (1.00, 0.80, 0.52)
    wavelength_grid_hspace: float = 0.40
    wavelength_grid_wspace: float = 0.22
    wavelength_mosaic_wspace: float = 0.20
    figure_suptitle_font_size: float = 16.0
    axis_title_font_size: float = 12.0
    axis_label_font_size: float = 11.0
    legend_font_size: float = 8.0
    primary_curve_legend_columns: int = 2
    panel_title_font_size: float = 8.0
    conclusion_font_size: float = 10.5
    conclusion_linespacing: float = 1.5
    conclusion_text_position: tuple[float, float] = (0.01, 0.98)
    figure_caption_position: tuple[float, float] = (0.01, 0.01)
    heatmap_note_position: tuple[float, float] = (0.98, 0.02)
    table_font_size: float = 9.0
    table_title_pad: float = 16.0
    table_scale_x: float = 1.0
    table_scale_y: float = 1.45
    table_column_widths: tuple[float, ...] = (
        0.22,
        0.12,
        0.10,
        0.16,
        0.16,
        0.12,
        0.12,
    )

    image_dynamic_range: float = 4.0
    image_display_half_width_strokes: float = 3.25
    image_display_minimum_half_width_arcmin: float = 2.0
    image_colormap_name: str = "magma"
    image_interpolation: str = "nearest"
    image_axis_tick_count: int = 4
    image_axis_label_font_size: float = 8.0
    image_tick_font_size: float = 7.0
    heatmap_colormap_name: str = "viridis"
    primary_line_color: str = "#111111"
    reference_line_color: str = "#d62728"
    line_marker: str = "o"
    primary_line_width: float = 1.3
    reference_line_width: float = 1.8
    reference_line_style: str = "--"
    grid_alpha: float = 0.25
    colorbar_pad: float = 0.02

    @property
    def target_angular_pixel_arcmin(self) -> float:
        return 1.0 / self.optotype_pixels_per_arcmin

    def __post_init__(self) -> None:
        if self.optotype_grid_size <= 0:
            raise ValueError("optotype_grid_size must be positive")
        if self.optotype_grid_size % 2 != 0:
            raise ValueError("optotype_grid_size must be even")
        if self.optotype_pixels_per_arcmin < 6.0:
            raise ValueError("optotype_pixels_per_arcmin must be at least 6")
        if self.optotype_kind != "E":
            raise ValueError("this simulation currently supports optotype_kind='E'")
        if not 0.0 < self.optotype_anti_alias_fraction:
            raise ValueError("optotype_anti_alias_fraction must be positive")
        if not self.optotype_stroke_values_arcmin:
            raise ValueError("optotype_stroke_values_arcmin cannot be empty")
        if any(value <= 0.0 for value in self.optotype_stroke_values_arcmin):
            raise ValueError("optotype stroke values must be positive")
        if any(
            right <= left
            for left, right in zip(
                self.optotype_stroke_values_arcmin,
                self.optotype_stroke_values_arcmin[1:],
            )
        ):
            raise ValueError("optotype stroke values must be strictly increasing")
        if not 0.0 < self.model_contrast_threshold < 1.0:
            raise ValueError("model_contrast_threshold must be in (0, 1)")
        if not 0.0 < self.optotype_ink_threshold < 1.0:
            raise ValueError("optotype_ink_threshold must be in (0, 1)")
        if self.optotype_extent_strokes <= 0.0:
            raise ValueError("optotype_extent_strokes must be positive")
        if self.convolution_mode != "full":
            raise ValueError("convolution_mode must be 'full'")
        if self.image_display_half_width_strokes <= 0.0:
            raise ValueError("image_display_half_width_strokes must be positive")
        if self.image_display_minimum_half_width_arcmin <= 0.0:
            raise ValueError(
                "image_display_minimum_half_width_arcmin must be positive"
            )
        if self.image_axis_tick_count < 2:
            raise ValueError("image_axis_tick_count must be at least two")
        if self.primary_curve_legend_columns < 1:
            raise ValueError("primary_curve_legend_columns must be positive")
        if self.primary_grid_rows < 1 or self.primary_grid_columns < 1:
            raise ValueError("primary grid dimensions must be positive")
        if len(self.primary_grid_height_ratios) != self.primary_grid_rows:
            raise ValueError(
                "primary_grid_height_ratios must match primary_grid_rows"
            )
        if self.wavelength_grid_rows < 1 or self.wavelength_grid_columns < 1:
            raise ValueError("wavelength grid dimensions must be positive")
        if len(self.wavelength_grid_height_ratios) != self.wavelength_grid_rows:
            raise ValueError(
                "wavelength_grid_height_ratios must match wavelength_grid_rows"
            )
        if self.gpu_device_id != self.psf_config.gpu_device_id:
            raise ValueError(
                "gpu_device_id must match psf_config.gpu_device_id"
            )
        if not math.isclose(
            self.primary_wavelength_nm,
            self.psf_config.primary_wavelength_nm,
            rel_tol=0.0,
            abs_tol=self.floating_comparison_tolerance,
        ):
            raise ValueError(
                "primary_wavelength_nm must match psf_config.primary_wavelength_nm"
            )
        if not any(
            math.isclose(
                self.primary_wavelength_nm,
                wavelength_nm,
                rel_tol=0.0,
                abs_tol=self.floating_comparison_tolerance,
            )
            for wavelength_nm in self.psf_config.wavelengths_nm
        ):
            raise ValueError(
                "primary_wavelength_nm must belong to psf_config.wavelengths_nm"
            )
        if not any(
            math.isclose(
                self.representative_myopia_d,
                myopia_d,
                rel_tol=0.0,
                abs_tol=self.floating_comparison_tolerance,
            )
            and myopia_d > 0.0
            for myopia_d in build_myopia_values(self.psf_config)
        ):
            raise ValueError(
                "representative_myopia_d must belong to the positive scan grid"
            )
        if self.output_root_directory_name != "output":
            raise ValueError("output_root_directory_name must be 'output'")
        if (
            Path(self.simulation_directory_name).name
            != self.simulation_directory_name
            or self.simulation_directory_name in {"", ".", ".."}
        ):
            raise ValueError(
                "simulation_directory_name must be one safe path component"
            )
        output_filenames = (
            self.metrics_filename,
            self.stroke_metrics_filename,
            self.config_filename,
            self.primary_figure_filename,
            self.wavelength_figure_filename,
            self.contrast_figure_filename,
            self.numerical_table_filename,
        )
        for filename in output_filenames:
            if Path(filename).name != filename or filename in {"", ".", ".."}:
                raise ValueError("output filenames must be safe path components")


def build_optotype_coordinates(
    config: RetinaSimulationConfig,
) -> tuple[cp.ndarray, cp.ndarray, float]:
    """返回以角分为单位的坐标网格和单个像素角尺寸。"""
    pixel_arcmin = config.target_angular_pixel_arcmin
    axis = (
        cp.arange(config.optotype_grid_size, dtype=cp.float32)
        - config.optotype_grid_size // 2
    ) * pixel_arcmin
    x_arcmin, y_arcmin = cp.meshgrid(axis, axis, indexing="xy")
    return x_arcmin, y_arcmin, pixel_arcmin


def rectangle_coverage(
    x_arcmin: cp.ndarray,
    y_arcmin: cp.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    soft_width_arcmin: float,
) -> cp.ndarray:
    """生成带抗锯齿边缘的矩形覆盖度，范围 0 到 1。"""
    signed_inside_distance = cp.minimum(
        x_arcmin - x_min,
        cp.minimum(
            x_max - x_arcmin,
            cp.minimum(y_arcmin - y_min, y_max - y_arcmin),
        ),
    )
    return cp.clip(
        0.5 + signed_inside_distance / soft_width_arcmin,
        0.0,
        1.0,
    )


def build_e_optotype(
    x_arcmin: cp.ndarray,
    y_arcmin: cp.ndarray,
    stroke_arcmin: float,
    config: RetinaSimulationConfig,
) -> cp.ndarray:
    """生成标准 5 x 5 网格 E 字视标。"""
    soft_width = config.target_angular_pixel_arcmin * config.optotype_anti_alias_fraction
    stroke = stroke_arcmin
    vertical = rectangle_coverage(
        x_arcmin,
        y_arcmin,
        -2.5 * stroke,
        -1.5 * stroke,
        -2.5 * stroke,
        2.5 * stroke,
        soft_width,
    )
    top_bar = rectangle_coverage(
        x_arcmin,
        y_arcmin,
        -2.5 * stroke,
        2.5 * stroke,
        1.5 * stroke,
        2.5 * stroke,
        soft_width,
    )
    middle_bar = rectangle_coverage(
        x_arcmin,
        y_arcmin,
        -2.5 * stroke,
        2.5 * stroke,
        -0.5 * stroke,
        0.5 * stroke,
        soft_width,
    )
    bottom_bar = rectangle_coverage(
        x_arcmin,
        y_arcmin,
        -2.5 * stroke,
        2.5 * stroke,
        -2.5 * stroke,
        -1.5 * stroke,
        soft_width,
    )
    return cp.maximum(cp.maximum(vertical, top_bar), cp.maximum(middle_bar, bottom_bar))


def circle_coverage(
    x_arcmin: cp.ndarray,
    y_arcmin: cp.ndarray,
    radius_arcmin: float,
    soft_width_arcmin: float,
) -> cp.ndarray:
    """生成圆形覆盖度，用于 Landolt C 的内外圆。"""
    distance = cp.sqrt(x_arcmin * x_arcmin + y_arcmin * y_arcmin)
    return cp.clip((radius_arcmin - distance + 0.5 * soft_width_arcmin) / soft_width_arcmin, 0.0, 1.0)


def build_landolt_c_optotype(
    x_arcmin: cp.ndarray,
    y_arcmin: cp.ndarray,
    stroke_arcmin: float,
    config: RetinaSimulationConfig,
) -> cp.ndarray:
    """生成标准 Landolt C 视标，开口位于右侧。"""
    soft_width = config.target_angular_pixel_arcmin * config.optotype_anti_alias_fraction
    stroke = stroke_arcmin
    outer = circle_coverage(x_arcmin, y_arcmin, 2.5 * stroke, soft_width)
    inner = circle_coverage(x_arcmin, y_arcmin, 1.5 * stroke, soft_width)
    ring = cp.clip(outer - inner, 0.0, 1.0)
    gap = rectangle_coverage(
        x_arcmin,
        y_arcmin,
        1.5 * stroke,
        2.5 * stroke,
        -0.5 * stroke,
        0.5 * stroke,
        soft_width,
    )
    return cp.clip(ring - gap, 0.0, 1.0)


def build_optotype(
    x_arcmin: cp.ndarray,
    y_arcmin: cp.ndarray,
    stroke_arcmin: float,
    config: RetinaSimulationConfig,
) -> cp.ndarray:
    if config.optotype_kind == "LandoltC":
        return build_landolt_c_optotype(x_arcmin, y_arcmin, stroke_arcmin, config)
    return build_e_optotype(x_arcmin, y_arcmin, stroke_arcmin, config)


def resample_psf_to_optotype_grid(
    psf: cp.ndarray,
    sampling: Any,
    config: RetinaSimulationConfig,
) -> cp.ndarray:
    """把任务①的 PSF 插值到统一角分辨率网格。"""
    target_half_arcmin = (
        config.optotype_grid_size / 2.0
        + config.psf_resampling_margin_arcmin
    ) * config.target_angular_pixel_arcmin
    native_half_px = int(
        math.ceil(target_half_arcmin / sampling.angular_pixel_arcmin)
    )
    native_centre = config.psf_config.grid_size // 2
    native_half_px = min(native_half_px, native_centre)
    start = native_centre - native_half_px
    stop = native_centre + native_half_px
    cropped = psf[start:stop, start:stop]

    source_centre_y = native_centre - start
    source_centre_x = native_centre - start
    target_axis = (
        cp.arange(config.optotype_grid_size, dtype=cp.float32)
        - config.optotype_grid_size // 2
    )
    source_step = (
        config.target_angular_pixel_arcmin
        / sampling.angular_pixel_arcmin
    )
    source_y = target_axis * source_step + source_centre_y
    source_x = target_axis * source_step + source_centre_x
    source_y_grid, source_x_grid = cp.meshgrid(
        source_y,
        source_x,
        indexing="ij",
    )
    resampled = map_coordinates(
        cropped,
        cp.stack((source_y_grid, source_x_grid), axis=0),
        order=config.psf_resampling_order,
        mode="constant",
        cval=0.0,
    )
    energy = resampled.sum(dtype=cp.float64)
    if energy > 0.0:
        resampled = resampled / energy
    return resampled.astype(cp.float32)


def optotype_contrast_ratio(
    x_arcmin: cp.ndarray,
    y_arcmin: cp.ndarray,
    reference_optotype: cp.ndarray,
    blurred_optotype: cp.ndarray,
    stroke_arcmin: float,
    config: RetinaSimulationConfig,
) -> float:
    """计算全部 E 字笔画与背景区域的平均对比度。"""
    ink_mask = reference_optotype >= config.optotype_ink_threshold
    half_extent = config.optotype_extent_strokes * stroke_arcmin
    optotype_region = (cp.abs(x_arcmin) <= half_extent) & (
        cp.abs(y_arcmin) <= half_extent
    )
    gap_mask = optotype_region & ~ink_mask
    if not bool(ink_mask.any()) or not bool(gap_mask.any()):
        return 0.0
    bar_mean = float(blurred_optotype[ink_mask].mean().item())
    gap_mean = float(blurred_optotype[gap_mask].mean().item())
    denominator = bar_mean + gap_mean
    if denominator == 0.0:
        return 0.0
    return (bar_mean - gap_mean) / denominator


def interpolate_threshold_stroke_arcmin(
    stroke_values: tuple[float, ...],
    contrast_values: tuple[float, ...],
    threshold: float,
) -> tuple[float, str]:
    """在对比度曲线上插值找到达到阈值的视标大小。"""
    if contrast_values[0] >= threshold:
        return stroke_values[0], "lower_censored"
    if contrast_values[-1] <= threshold:
        return math.nan, "upper_censored"
    log_strokes = [math.log(value) for value in stroke_values]
    for index in range(1, len(contrast_values)):
        left = contrast_values[index - 1]
        right = contrast_values[index]
        if left <= threshold <= right or right <= threshold <= left:
            denominator = right - left
            if denominator == 0.0:
                return stroke_values[index - 1], "resolved"
            fraction = (threshold - left) / denominator
            log_value = log_strokes[index - 1] + fraction * (
                log_strokes[index] - log_strokes[index - 1]
            )
            return math.exp(log_value), "resolved"
    return math.nan, "upper_censored"


def convolve_optotype(
    optotype: cp.ndarray,
    resampled_psf: cp.ndarray,
    config: RetinaSimulationConfig,
) -> cp.ndarray:
    """在 GPU 上做二维 FFT 卷积。"""
    result = fftconvolve(
        optotype,
        resampled_psf,
        mode=config.convolution_mode,
    )

    # fftconvolve treats the kernel origin as resampled_psf.shape // 2.
    # Cropping from that index keeps the object centre at its original index.
    crop_start_y = resampled_psf.shape[0] // 2
    crop_start_x = resampled_psf.shape[1] // 2
    crop_end_y = crop_start_y + optotype.shape[0]
    crop_end_x = crop_start_x + optotype.shape[1]
    return result[crop_start_y:crop_end_y, crop_start_x:crop_end_x]


def blurred_optotype_display_half_width_arcmin(
    stroke_arcmin: float,
    config: RetinaSimulationConfig,
) -> float:
    return max(
        config.image_display_half_width_strokes * stroke_arcmin,
        config.image_display_minimum_half_width_arcmin,
    )


def crop_blurred_optotype_for_display(
    blurred_optotype: cp.ndarray,
    stroke_arcmin: float,
    config: RetinaSimulationConfig,
) -> tuple[cp.ndarray, tuple[float, float, float, float]]:
    """裁出中心视窗，并返回以角分为单位的显示范围。"""
    pixel_arcmin = config.target_angular_pixel_arcmin
    half_width_arcmin = blurred_optotype_display_half_width_arcmin(
        stroke_arcmin,
        config,
    )
    half_width_pixels = max(
        1,
        int(math.ceil(half_width_arcmin / pixel_arcmin)),
    )
    centre_y = blurred_optotype.shape[0] // 2
    centre_x = blurred_optotype.shape[1] // 2
    start_y = max(0, centre_y - half_width_pixels)
    end_y = min(blurred_optotype.shape[0], centre_y + half_width_pixels)
    start_x = max(0, centre_x - half_width_pixels)
    end_x = min(blurred_optotype.shape[1], centre_x + half_width_pixels)
    cropped = blurred_optotype[start_y:end_y, start_x:end_x]

    x_min = (start_x - centre_x) * pixel_arcmin
    x_max = (end_x - centre_x) * pixel_arcmin
    y_min = (start_y - centre_y) * pixel_arcmin
    y_max = (end_y - centre_y) * pixel_arcmin
    return cropped, (x_min, x_max, y_min, y_max)


def draw_blurred_optotype_panel(
    axis: plt.Axes,
    item: dict[str, Any],
    config: RetinaSimulationConfig,
) -> None:
    data, extent = crop_blurred_optotype_for_display(
        item["blurred_image"],
        item["display_stroke_arcmin"],
        config,
    )
    peak = float(data.max())
    floor = peak * (10.0 ** (-config.image_dynamic_range))
    axis.imshow(
        data,
        origin="lower",
        extent=extent,
        cmap=config.image_colormap_name,
        norm=LogNorm(vmin=floor, vmax=peak),
        interpolation=config.image_interpolation,
    )
    axis.set_xlim(extent[0], extent[1])
    axis.set_ylim(extent[2], extent[3])
    axis.set_aspect("equal", adjustable="box")
    axis.xaxis.set_major_locator(
        MaxNLocator(nbins=config.image_axis_tick_count)
    )
    axis.yaxis.set_major_locator(
        MaxNLocator(nbins=config.image_axis_tick_count)
    )
    axis.tick_params(
        axis="both",
        labelsize=config.image_tick_font_size,
    )
    axis.set_xlabel(
        "角坐标 / Angular x (arcmin)",
        fontsize=config.image_axis_label_font_size,
    )
    axis.set_ylabel(
        "角坐标 / Angular y (arcmin)",
        fontsize=config.image_axis_label_font_size,
    )


def save_primary_summary_figure(
    output_path: Path,
    config: RetinaSimulationConfig,
    primary_rows: list[dict[str, Any]],
    representative_images: list[dict[str, Any]],
) -> None:
    if not primary_rows:
        raise RuntimeError("primary summary requires at least one metric row")
    if not representative_images:
        raise RuntimeError("primary summary requires representative images")
    figure = plt.figure(figsize=config.primary_figure_size_inches, dpi=config.figure_dpi)
    grid = figure.add_gridspec(
        config.primary_grid_rows,
        config.primary_grid_columns,
        height_ratios=config.primary_grid_height_ratios,
        hspace=config.primary_grid_hspace,
        wspace=config.primary_grid_wspace,
    )

    heatmap_axis = figure.add_subplot(grid[0, 0])
    curve_axis = figure.add_subplot(grid[0, 1])
    mosaic_spec = grid[1, :].subgridspec(
        1,
        len(representative_images),
        wspace=config.primary_mosaic_wspace,
    )
    table_axis = figure.add_subplot(grid[2, :])
    conclusion_axis = figure.add_subplot(grid[3, :])
    conclusion_axis.axis("off")

    plotted_myopia = sorted(
        {row["myopia_d"] for row in primary_rows if row["myopia_d"] > 0.0}
    )
    if not plotted_myopia:
        raise RuntimeError("primary summary requires at least one myopia value")
    diameter_grid = sorted(
        {
            row["diameter_mm"]
            for row in primary_rows
            if abs(row["myopia_d"] - plotted_myopia[0])
            < config.floating_comparison_tolerance
        }
    )
    finite_matrix_values = [
        row["threshold_log_mar"]
        for row in primary_rows
        if math.isfinite(row["threshold_log_mar"])
    ]
    if not finite_matrix_values:
        raise RuntimeError("primary summary has no finite convolution logMAR")
    colour_min = min(finite_matrix_values)
    colour_max = max(finite_matrix_values)
    matrix = [
        [
            next(
                row["threshold_log_mar"]
                for row in primary_rows
                if abs(row["myopia_d"] - myopia_d)
                < config.floating_comparison_tolerance
                and abs(row["diameter_mm"] - diameter_mm)
                < config.floating_comparison_tolerance
            )
            for diameter_mm in diameter_grid
        ]
        for myopia_d in plotted_myopia
    ]
    image = heatmap_axis.pcolormesh(
        diameter_grid,
        plotted_myopia,
        matrix,
        shading="auto",
        cmap=config.heatmap_colormap_name,
        vmin=colour_min,
        vmax=colour_max,
    )
    heatmap_axis.set_xscale("log")
    heatmap_axis.set_xlabel("针孔直径 / Pinhole diameter d (mm)")
    heatmap_axis.set_ylabel("近视度数 / Myopia M (D)")
    heatmap_axis.set_title(
        "模型对比度阈值 logMAR / Model contrast-threshold logMAR"
    )
    heatmap_axis.text(
        config.heatmap_note_position[0],
        config.heatmap_note_position[1],
        f"白色格：{max(config.optotype_stroke_values_arcmin):g} 角分内未达阈值\n"
        "White: unresolved within scan range",
        transform=heatmap_axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=config.image_tick_font_size,
    )
    figure.colorbar(image, ax=heatmap_axis, pad=config.colorbar_pad).set_label(
        "logMAR（卷积阈值 / convolution threshold）",
        fontsize=config.axis_label_font_size,
    )

    for myopia_d in plotted_myopia:
        myopia_rows = [
            row
            for row in primary_rows
            if abs(row["myopia_d"] - myopia_d)
            < config.floating_comparison_tolerance
        ]
        myopia_rows = sorted(myopia_rows, key=lambda row: row["diameter_mm"])
        simulated = [row["threshold_stroke_arcmin"] for row in myopia_rows]
        curve_axis.plot(
            diameter_grid,
            simulated,
            marker=config.line_marker,
            color=config.primary_line_color,
            linewidth=config.primary_line_width,
            label=f"M={myopia_d:g} D",
        )
    curve_axis.set_xscale("log")
    curve_axis.set_xlabel("针孔直径 / Pinhole diameter d (mm)")
    curve_axis.set_ylabel(
        "模型阈值 MAR / Model-threshold MAR (arcmin)"
    )
    curve_axis.set_title(
        "模型阈值随孔径变化 / Model-threshold MAR versus diameter"
    )
    curve_axis.grid(alpha=config.grid_alpha)
    curve_axis.legend(
        fontsize=config.legend_font_size,
        ncol=config.primary_curve_legend_columns,
    )

    for index, item in enumerate(representative_images):
        axis = figure.add_subplot(mosaic_spec[0, index])
        draw_blurred_optotype_panel(axis, item, config)
        if item["threshold_status"] == "resolved":
            mar_label = f"MAR={item['threshold_stroke_arcmin']:.2f}'"
        elif item["threshold_status"] == "lower_censored":
            mar_label = f"MAR≤{item['threshold_stroke_arcmin']:.2f}'"
        else:
            mar_label = (
                f"MAR>{item['display_stroke_arcmin']:.2f}'"
                "（未分辨 / not resolved）"
            )
        axis.set_title(
            f"d={item['diameter_mm']:.3f} mm\n{mar_label}",
            fontsize=config.panel_title_font_size,
        )

    table_axis.axis("off")
    headers = [
        "近视 M",
        "仿真孔径",
        "理论孔径",
        "模型阈值 MAR",
        "模型阈值 logMAR",
        "PSF D50",
        "PSF logMAR",
    ]
    table_rows = []
    for myopia_d in config.summary_myopia_values:
        candidates = [
            row
            for row in primary_rows
            if abs(row["myopia_d"] - myopia_d)
            < config.floating_comparison_tolerance
        ]
        finite_candidates = [
            row for row in candidates if math.isfinite(row["threshold_log_mar"])
        ]
        if not finite_candidates:
            continue
        best = min(finite_candidates, key=lambda row: row["threshold_log_mar"])
        theory = theoretical_optimal_diameter_mm(
            config.psf_config,
            config.primary_wavelength_nm,
            myopia_d,
        )
        table_rows.append(
            [
                f"{myopia_d:g}",
                f"{best['diameter_mm']:.3f}",
                f"{theory:.3f}",
                f"{best['threshold_stroke_arcmin']:.3f}",
                f"{best['threshold_log_mar']:.3f}",
                f"{best['optical_d50_arcmin']:.3f}",
                f"{best['optical_log_mar']:.3f}",
            ]
        )
    table = table_axis.table(
        cellText=table_rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(config.table_font_size)
    table.scale(config.table_scale_x, config.table_scale_y)
    for (_, column_index), cell in table.get_celld().items():
        if column_index < len(config.table_column_widths):
            cell.set_width(config.table_column_widths[column_index])
    table_axis.set_title(
        f"{config.primary_wavelength_nm:g} nm 数值锚点 / Numerical anchors",
        pad=config.table_title_pad,
        fontsize=config.axis_title_font_size,
    )

    finite_summary_rows = [
        row
        for row in primary_rows
        if any(
            abs(row["myopia_d"] - myopia_d) < config.floating_comparison_tolerance
            for myopia_d in config.summary_myopia_values
        )
        and math.isfinite(row["threshold_log_mar"])
    ]
    if not finite_summary_rows:
        raise RuntimeError("primary summary has no finite numerical anchor")
    best_overall = min(finite_summary_rows, key=lambda row: row["threshold_log_mar"])
    conclusion_text = (
        "结论 / Conclusions：\n"
        "1. 对同一近视度数，针孔过小由衍射限制，针孔过大由离焦限制，"
        "模型阈值 MAR 因此存在单峰最小值。\n"
        "2. 近视越深，最优针孔整体减小，模型阈值整体升高；"
        "离散孔径网格会带来局部非单调。\n"
        f"3. {config.primary_wavelength_nm:g} nm 锚点中，最低卷积 logMAR "
        f"出现在 M={best_overall['myopia_d']:g} D、"
        f"d={best_overall['diameter_mm']:.3f} mm，"
        f"MAR={best_overall['threshold_stroke_arcmin']:.3f} 角分。\n"
        "4. 当前为单孔、单色光学代理，不含明视觉权重、眼内色差、视网膜采样和多孔重叠。"
    )
    conclusion_axis.text(
        config.conclusion_text_position[0],
        config.conclusion_text_position[1],
        conclusion_text,
        transform=conclusion_axis.transAxes,
        ha="left",
        va="top",
        fontsize=config.conclusion_font_size,
        linespacing=config.conclusion_linespacing,
    )
    figure.suptitle(
        "单孔 E 字视网膜像仿真 / Single-hole optotype retinal imaging",
        fontsize=config.figure_suptitle_font_size,
    )
    figure.savefig(output_path, dpi=config.figure_dpi, bbox_inches="tight")
    plt.close(figure)


def save_wavelength_comparison_figure(
    output_path: Path,
    config: RetinaSimulationConfig,
    wavelength_rows: list[dict[str, Any]],
    wavelength_images: list[dict[str, Any]],
) -> None:
    if not wavelength_rows:
        raise RuntimeError("wavelength comparison requires metric rows")
    if not wavelength_images:
        raise RuntimeError("wavelength comparison requires image panels")
    figure = plt.figure(figsize=config.wavelength_figure_size_inches, dpi=config.figure_dpi)
    grid = figure.add_gridspec(
        config.wavelength_grid_rows,
        config.wavelength_grid_columns,
        height_ratios=config.wavelength_grid_height_ratios,
        hspace=config.wavelength_grid_hspace,
        wspace=config.wavelength_grid_wspace,
    )
    threshold_axis = figure.add_subplot(grid[0, 0])
    optical_axis = figure.add_subplot(grid[0, 1])
    mosaic_spec = grid[1, :].subgridspec(
        1,
        len(wavelength_images),
        wspace=config.wavelength_mosaic_wspace,
    )
    conclusion_axis = figure.add_subplot(grid[2, :])
    conclusion_axis.axis("off")

    wavelengths = sorted({row["wavelength_nm"] for row in wavelength_rows})
    best_rows_by_wavelength = []
    for wavelength in wavelengths:
        finite_rows = [
            row
            for row in wavelength_rows
            if abs(row["wavelength_nm"] - wavelength)
            < config.floating_comparison_tolerance
            and math.isfinite(row["threshold_log_mar"])
        ]
        if not finite_rows:
            continue
        best_rows_by_wavelength.append(
            min(finite_rows, key=lambda row: row["threshold_log_mar"])
        )
    if not best_rows_by_wavelength:
        raise RuntimeError("wavelength comparison has no finite threshold")
    best_wavelengths = [
        row["wavelength_nm"] for row in best_rows_by_wavelength
    ]
    threshold_mar = [
        row["threshold_stroke_arcmin"] for row in best_rows_by_wavelength
    ]
    optical_mar = [
        row["optical_mar_arcmin"] for row in best_rows_by_wavelength
    ]
    threshold_axis.plot(
        best_wavelengths,
        threshold_mar,
        marker=config.line_marker,
        color=config.primary_line_color,
        linewidth=config.primary_line_width,
        label="卷积阈值 / Convolution",
    )
    threshold_axis.plot(
        best_wavelengths,
        optical_mar,
        linestyle=config.reference_line_style,
        color=config.reference_line_color,
        linewidth=config.reference_line_width,
        label=f"PSF D50/{config.psf_config.mar_to_d50_divisor:g}",
    )
    threshold_axis.set_xlabel("波长 / Wavelength (nm)")
    threshold_axis.set_ylabel("MAR / Minimum angle of resolution (arcmin)")
    threshold_axis.set_title(
        f"波长对模型阈值的影响 / Model threshold at "
        f"M={config.representative_myopia_d:g} D"
    )
    threshold_axis.grid(alpha=config.grid_alpha)
    threshold_axis.legend(fontsize=config.legend_font_size)

    theory_ratio = [
        math.sqrt(wavelength / config.primary_wavelength_nm)
        for wavelength in best_wavelengths
    ]
    reference_nm = config.primary_wavelength_nm
    reference_candidates = [
        row["diameter_mm"]
        for row in best_rows_by_wavelength
        if abs(row["wavelength_nm"] - reference_nm)
        < config.floating_comparison_tolerance
    ]
    if not reference_candidates:
        raise RuntimeError("primary wavelength has no finite comparison row")
    reference_diameter = reference_candidates[0]
    simulated_ratio = [
        row["diameter_mm"] / reference_diameter for row in best_rows_by_wavelength
    ]
    optical_axis.plot(
        best_wavelengths,
        simulated_ratio,
        marker=config.line_marker,
        color=config.primary_line_color,
        linewidth=config.primary_line_width,
        label="卷积最优 / Convolution",
    )
    optical_axis.plot(
        best_wavelengths,
        theory_ratio,
        linestyle=config.reference_line_style,
        color=config.reference_line_color,
        linewidth=config.reference_line_width,
        label="λ^0.5 理论 / Theory",
    )
    optical_axis.set_xlabel("波长 / Wavelength (nm)")
    optical_axis.set_ylabel("相对最优孔径 / Relative optimal diameter")
    optical_axis.set_title("最优孔径随波长的变化 / Optimum-diameter wavelength shift")
    optical_axis.grid(alpha=config.grid_alpha)
    optical_axis.legend(fontsize=config.legend_font_size)

    for index, item in enumerate(wavelength_images):
        axis = figure.add_subplot(mosaic_spec[0, index])
        draw_blurred_optotype_panel(axis, item, config)
        axis.set_title(
            f"{item['wavelength_nm']:g} nm\n"
            f"d={item['diameter_mm']:.3f} mm, MAR={item['threshold_stroke_arcmin']:.2f}'",
            fontsize=config.panel_title_font_size,
        )

    conclusion_text = (
        "结论 / Conclusions：\n"
        f"在 M={config.representative_myopia_d:g} D 下，最优针孔和模型阈值"
        "整体随波长增加；离散孔径网格会造成局部起伏。\n"
        "卷积阈值来自模糊 E 字的笔画-空白对比度；PSF 的 D50/4 仅作为趋势参照。\n"
        "当前比较使用等强度单色 PSF，不含明视觉权重、眼内色差和视网膜采样。"
    )
    conclusion_axis.text(
        config.conclusion_text_position[0],
        config.conclusion_text_position[1],
        conclusion_text,
        transform=conclusion_axis.transAxes,
        ha="left",
        va="top",
        fontsize=config.conclusion_font_size,
        linespacing=config.conclusion_linespacing,
    )
    figure.suptitle(
        "多波长单孔 E 字仿真 / Wavelength-dependent optotype simulation",
        fontsize=config.figure_suptitle_font_size,
    )
    figure.savefig(output_path, dpi=config.figure_dpi, bbox_inches="tight")
    plt.close(figure)


def save_contrast_curve_figure(
    output_path: Path,
    config: RetinaSimulationConfig,
    contrast_rows: list[dict[str, Any]],
    representative_diameters_mm: tuple[float, ...],
) -> None:
    figure = plt.figure(figsize=config.contrast_figure_size_inches, dpi=config.figure_dpi)
    axis = figure.add_subplot(111)
    for diameter_mm in representative_diameters_mm:
        rows = [
            row
            for row in contrast_rows
            if abs(row["wavelength_nm"] - config.primary_wavelength_nm)
            < config.floating_comparison_tolerance
            and abs(row["myopia_d"] - config.representative_myopia_d)
            < config.floating_comparison_tolerance
            and abs(row["diameter_mm"] - diameter_mm)
            < config.floating_comparison_tolerance
        ]
        if not rows:
            continue
        rows = sorted(rows, key=lambda row: row["stroke_arcmin"])
        axis.plot(
            [row["stroke_arcmin"] for row in rows],
            [row["contrast_ratio"] for row in rows],
            marker=config.line_marker,
            linewidth=config.primary_line_width,
            label=f"d={diameter_mm:.3f} mm",
        )
    axis.axhline(
        config.model_contrast_threshold,
        color="black",
        linestyle="--",
        linewidth=config.reference_line_width,
        label="模型对比度阈值 / Model contrast threshold",
    )
    axis.set_xscale("log")
    axis.set_xlabel("E 字笔画与空白宽度 / Stroke and gap width (arcmin)")
    axis.set_ylabel("笔画与空白对比度 / Stroke-to-gap contrast")
    axis.set_title(
        f"视标对比度扫描 / Contrast scan at M={config.representative_myopia_d:g} D"
    )
    axis.grid(alpha=config.grid_alpha)
    axis.legend(fontsize=config.legend_font_size)
    figure.text(
        config.figure_caption_position[0],
        config.figure_caption_position[1],
        "曲线与模型阈值线的交点给出用于比较孔径的模型阈值 MAR。",
        ha="left",
        va="bottom",
        fontsize=config.conclusion_font_size,
    )
    figure.savefig(output_path, dpi=config.figure_dpi, bbox_inches="tight")
    plt.close(figure)


def save_numerical_table_figure(
    output_path: Path,
    config: RetinaSimulationConfig,
    rows: list[dict[str, Any]],
) -> None:
    figure = plt.figure(figsize=config.validation_figure_size_inches, dpi=config.figure_dpi)
    axis = figure.add_subplot(111)
    axis.axis("off")
    headers = [
        "波长 (nm)",
        "近视 (D)",
        "针孔 (mm)",
        "模型阈值 MAR (arcmin)",
        "模型阈值 logMAR",
        "PSF MAR (arcmin)",
        "PSF logMAR",
    ]
    selected_rows = []
    for wavelength_nm in config.psf_config.wavelengths_nm:
        candidates = [
            row
            for row in rows
            if abs(row["wavelength_nm"] - wavelength_nm)
            < config.floating_comparison_tolerance
            and abs(row["myopia_d"] - config.representative_myopia_d)
            < config.floating_comparison_tolerance
        ]
        finite_candidates = [
            row for row in candidates if math.isfinite(row["threshold_log_mar"])
        ]
        if not finite_candidates:
            continue
        best = min(finite_candidates, key=lambda row: row["threshold_log_mar"])
        selected_rows.append(
            [
                f"{wavelength_nm:g}",
                f"{config.representative_myopia_d:g}",
                f"{best['diameter_mm']:.3f}",
                f"{best['threshold_stroke_arcmin']:.3f}",
                f"{best['threshold_log_mar']:.3f}",
                f"{best['optical_mar_arcmin']:.3f}",
                f"{best['optical_log_mar']:.3f}",
            ]
        )
    table = axis.table(
        cellText=selected_rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(config.table_font_size)
    table.scale(config.table_scale_x, config.table_scale_y)
    for (_, column_index), cell in table.get_celld().items():
        if column_index < len(config.table_column_widths):
            cell.set_width(config.table_column_widths[column_index])
    axis.set_title(
        f"M={config.representative_myopia_d:g} D 多波长数值表 / "
        "Wavelength numerical table",
        pad=config.table_title_pad,
    )
    figure.savefig(output_path, dpi=config.figure_dpi, bbox_inches="tight")
    plt.close(figure)


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def print_progress(
    completed: int,
    total: int,
    started_at: float,
) -> None:
    elapsed = time.perf_counter() - started_at
    rate = completed / elapsed if elapsed > 0.0 else 0.0
    print(
        f"[{completed:4d}/{total:4d}] rate={rate:.1f} parameter scans/s",
        flush=True,
    )


def main() -> None:
    config = RetinaSimulationConfig()
    configure_bilingual_plot_font()
    cp.cuda.Device(config.gpu_device_id).use()
    gpu_properties = cp.cuda.runtime.getDeviceProperties(config.gpu_device_id)
    gpu_name = gpu_properties["name"]
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode()

    script_directory = Path(__file__).resolve().parent
    output_directory = (
        script_directory
        / config.output_root_directory_name
        / config.simulation_directory_name
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    x_arcmin, y_arcmin, _ = build_optotype_coordinates(config)
    normalized_radius, normalized_radius_squared, radial_bin_index, radial_pixel_count = (
        build_gpu_grids(config.psf_config)
    )
    aperture = build_soft_aperture(config.psf_config, normalized_radius)
    optotype_templates = {
        stroke_arcmin: build_optotype(
            x_arcmin,
            y_arcmin,
            stroke_arcmin,
            config,
        )
        for stroke_arcmin in config.optotype_stroke_values_arcmin
    }

    wavelengths = config.psf_config.wavelengths_nm
    all_myopia_values = build_myopia_values(config.psf_config)
    primary_myopia_values = tuple(
        value for value in all_myopia_values if value > 0.0
    )
    green_diameter_grid = build_diameter_grid(
        config.psf_config,
        config.primary_wavelength_nm,
    )

    scan_tasks: list[tuple[float, float, float]] = []
    for myopia_d in primary_myopia_values:
        for diameter_mm in green_diameter_grid:
            scan_tasks.append((config.primary_wavelength_nm, myopia_d, diameter_mm))
    for wavelength_nm in wavelengths:
        if abs(wavelength_nm - config.primary_wavelength_nm) < config.floating_comparison_tolerance:
            continue
        wavelength_grid = build_diameter_grid(config.psf_config, wavelength_nm)
        for diameter_mm in wavelength_grid:
            scan_tasks.append(
                (wavelength_nm, config.representative_myopia_d, diameter_mm)
            )

    all_rows: list[dict[str, Any]] = []
    green_rows: list[dict[str, Any]] = []
    stroke_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    representative_images: list[dict[str, Any]] = []
    wavelength_images: list[dict[str, Any]] = []
    representative_diameter_targets = {
        diameter: nearest_diameter(green_diameter_grid, diameter)
        for diameter in config.representative_diameters_mm
    }

    started_at = time.perf_counter()
    total_scans = len(scan_tasks)
    print(
        f"GPU: {gpu_name}; optotype_grid={config.optotype_grid_size}; "
        f"pixels_per_arcmin={config.optotype_pixels_per_arcmin:g}; "
        f"scans={total_scans}; output={output_directory}",
        flush=True,
    )

    for completed, (wavelength_nm, myopia_d, diameter_mm) in enumerate(
        scan_tasks,
        start=1,
    ):
        sampling = build_sampling(config.psf_config, wavelength_nm, diameter_mm)
        psf = compute_psf(
            config.psf_config,
            aperture,
            normalized_radius_squared,
            wavelength_nm,
            diameter_mm,
            myopia_d,
        )
        radial_bin_count = int(radial_pixel_count.size)
        optical_d50_arcmin = enclosing_diameter_arcmin(
            psf,
            radial_bin_index,
            radial_bin_count,
            sampling.angular_pixel_arcmin,
            config.psf_config,
        )
        optical_mar_arcmin = (
            optical_d50_arcmin / config.psf_config.mar_to_d50_divisor
        )
        optical_log_mar = math.log10(
            max(optical_mar_arcmin, config.psf_config.mar_floor_arcmin)
        )
        resampled_psf = resample_psf_to_optotype_grid(psf, sampling, config)

        scores: list[float] = []
        for stroke_arcmin in config.optotype_stroke_values_arcmin:
            reference = optotype_templates[stroke_arcmin]
            blurred = convolve_optotype(reference, resampled_psf, config)
            contrast = optotype_contrast_ratio(
                x_arcmin,
                y_arcmin,
                reference,
                blurred,
                stroke_arcmin,
                config,
            )
            scores.append(contrast)
            stroke_rows.append(
                {
                    "wavelength_nm": wavelength_nm,
                    "diameter_mm": diameter_mm,
                    "myopia_d": myopia_d,
                    "stroke_arcmin": stroke_arcmin,
                    "contrast_ratio": contrast,
                    "optical_d50_arcmin": optical_d50_arcmin,
                    "optical_log_mar": optical_log_mar,
                }
            )

        threshold_stroke, threshold_status = interpolate_threshold_stroke_arcmin(
            config.optotype_stroke_values_arcmin,
            tuple(scores),
            config.model_contrast_threshold,
        )
        if threshold_status == "resolved":
            threshold_log_mar = math.log10(
                max(threshold_stroke, config.psf_config.mar_floor_arcmin)
            )
        else:
            threshold_log_mar = math.nan
        threshold_detected = threshold_status == "resolved"

        is_green = (
            abs(wavelength_nm - config.primary_wavelength_nm)
            < config.floating_comparison_tolerance
        )
        is_representative_myopia = (
            abs(myopia_d - config.representative_myopia_d)
            < config.floating_comparison_tolerance
        )
        if (
            is_green
            and is_representative_myopia
            and diameter_mm in representative_diameter_targets.values()
        ):
            display_stroke = (
                threshold_stroke
                if math.isfinite(threshold_stroke)
                else max(config.optotype_stroke_values_arcmin)
            )
            threshold_reference = build_optotype(
                x_arcmin,
                y_arcmin,
                display_stroke,
                config,
            )
            threshold_blurred = convolve_optotype(
                threshold_reference,
                resampled_psf,
                config,
            )
            representative_images.append(
                {
                    "wavelength_nm": wavelength_nm,
                    "myopia_d": myopia_d,
                    "diameter_mm": diameter_mm,
                    "stroke_arcmin": threshold_stroke,
                    "threshold_stroke_arcmin": threshold_stroke,
                    "threshold_log_mar": threshold_log_mar,
                    "display_stroke_arcmin": display_stroke,
                    "threshold_detected": threshold_detected,
                    "threshold_status": threshold_status,
                    "blurred_image": cp.asnumpy(threshold_blurred),
                }
            )

        aggregate_row = {
            "wavelength_nm": wavelength_nm,
            "diameter_mm": diameter_mm,
            "myopia_d": myopia_d,
            "threshold_stroke_arcmin": threshold_stroke,
            "threshold_log_mar": threshold_log_mar,
            "threshold_detected": threshold_detected,
            "threshold_status": threshold_status,
            "optical_d50_arcmin": optical_d50_arcmin,
            "optical_mar_arcmin": optical_mar_arcmin,
            "optical_log_mar": optical_log_mar,
        }
        all_rows.append(aggregate_row)
        if is_green:
            green_rows.append(aggregate_row)
        if is_green and is_representative_myopia:
            contrast_rows.extend(
                {
                    "wavelength_nm": wavelength_nm,
                    "diameter_mm": diameter_mm,
                    "myopia_d": myopia_d,
                    "stroke_arcmin": stroke_arcmin,
                    "contrast_ratio": contrast,
                }
                for stroke_arcmin, contrast in zip(
                    config.optotype_stroke_values_arcmin,
                    scores,
                )
            )
        if completed % 25 == 0 or completed == total_scans:
            print_progress(completed, total_scans, started_at)

    comparison_rows = [
        row
        for row in all_rows
        if abs(row["myopia_d"] - config.representative_myopia_d) < config.floating_comparison_tolerance
    ]
    for wavelength_nm in wavelengths:
        finite_candidates = [
            row
            for row in comparison_rows
            if abs(row["wavelength_nm"] - wavelength_nm) < config.floating_comparison_tolerance
            and math.isfinite(row["threshold_log_mar"])
        ]
        if not finite_candidates:
            continue
        best_row = min(finite_candidates, key=lambda row: row["threshold_log_mar"])
        diameter_mm = best_row["diameter_mm"]
        sampling = build_sampling(config.psf_config, wavelength_nm, diameter_mm)
        psf = compute_psf(
            config.psf_config,
            aperture,
            normalized_radius_squared,
            wavelength_nm,
            diameter_mm,
            config.representative_myopia_d,
        )
        resampled_psf = resample_psf_to_optotype_grid(psf, sampling, config)
        reference = build_optotype(
            x_arcmin,
            y_arcmin,
            best_row["threshold_stroke_arcmin"],
            config,
        )
        blurred = convolve_optotype(reference, resampled_psf, config)
        wavelength_images.append(
            {
                "wavelength_nm": wavelength_nm,
                "diameter_mm": diameter_mm,
                "threshold_stroke_arcmin": best_row["threshold_stroke_arcmin"],
                "display_stroke_arcmin": best_row["threshold_stroke_arcmin"],
                "threshold_detected": True,
                "threshold_status": best_row["threshold_status"],
                "blurred_image": cp.asnumpy(blurred),
            }
        )

    cp.cuda.Stream.null.synchronize()

    primary_fields = list(all_rows[0].keys())
    stroke_fields = list(stroke_rows[0].keys())
    write_csv(
        output_directory / config.metrics_filename,
        all_rows,
        primary_fields,
    )
    write_csv(
        output_directory / config.stroke_metrics_filename,
        stroke_rows,
        stroke_fields,
    )

    config_payload = asdict(config)
    config_payload["psf_config"] = asdict(config.psf_config)
    config_payload["psf_config"]["real_dtype"] = config.psf_config.real_dtype.__name__
    config_payload["psf_config"]["complex_dtype"] = config.psf_config.complex_dtype.__name__
    config_payload["psf_config"]["accumulator_dtype"] = config.psf_config.accumulator_dtype.__name__
    config_payload["psf_config"]["wavelengths_nm"] = list(config.psf_config.wavelengths_nm)
    config_payload["gpu_name"] = gpu_name
    config_payload["script_sha256"] = file_sha256(Path(__file__).resolve())
    test_file = script_directory / "tests" / "test_single_hole_retinal_image.py"
    if test_file.exists():
        config_payload["test_sha256"] = file_sha256(test_file)
    with (output_directory / config.config_filename).open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(config_payload, file, ensure_ascii=False, indent=2)

    save_primary_summary_figure(
        output_directory / config.primary_figure_filename,
        config,
        green_rows,
        representative_images,
    )
    save_wavelength_comparison_figure(
        output_directory / config.wavelength_figure_filename,
        config,
        comparison_rows,
        wavelength_images,
    )
    save_contrast_curve_figure(
        output_directory / config.contrast_figure_filename,
        config,
        contrast_rows,
        tuple(sorted(set(representative_diameter_targets.values()))),
    )
    save_numerical_table_figure(
        output_directory / config.numerical_table_filename,
        config,
        all_rows,
    )

    finite_rows = [
        row for row in all_rows if math.isfinite(row["threshold_log_mar"])
    ]
    if not finite_rows:
        raise RuntimeError("no finite optotype threshold was found")
    best_row = min(finite_rows, key=lambda row: row["threshold_log_mar"])
    print(
        "Best optotype point: "
        f"lambda={best_row['wavelength_nm']:g} nm, "
        f"d={best_row['diameter_mm']:.4f} mm, "
        f"M={best_row['myopia_d']:g} D, "
        f"MAR={best_row['threshold_stroke_arcmin']:.3f} arcmin, "
        f"logMAR={best_row['threshold_log_mar']:.3f}",
        flush=True,
    )
    print(f"Saved outputs to: {output_directory}", flush=True)


if __name__ == "__main__":
    main()
