# 任务①（单孔 PSF 扫描）找出"对每种近视度数，最清晰的针孔直径是多大"，并用仿真定量验证"针孔不是越小越好"这条结论
# d是针孔致敬，M是近视度数，λ是波长，f是针孔到视网膜的等效距离

# 我们仿真的第一个任务，单孔PSF的仿真

# 什么是PSF？点扩散函数（Point Spread Function，PSF）是光学系统对一个点光源的响应。它描述了光学系统如何将一个理想的点光源成像到图像平面上。PSF的形状和大小直接影响成像质量。
# 一个理想的点光源（比如远处的星星），经过一个光学系统（比如眼睛、相机镜头）后，在像面上不会还是一个点，而是会扩散成一个有一定大小的光斑。这个光斑的强度分布，就叫 点扩散函数 PSF。

# 在波动光学模拟里面，光是一个复数数组
# 数组的每个元素，是这个位置上的复振幅：振幅 + 相位
# 另外，本仿真只关注横截面，所以只有两个元素(x, y)，没有纵向(z)的变化
# 由于这里的任务一里面，我们只是先研究系统本身，而不是人眼，我们先把光定义成一个无穷远的点源发出的单色平面波

# 原本的是

"""针孔面场 P
  ↓ ASM 传播到角膜
角膜前场
  ↓ 加角膜/晶状体透镜相位
晶状体后场
  ↓ ASM 传播到视网膜
视网膜场 U
  ↓ |U|²
PSF

# 平行光照射一个孔径，经过一个理想透镜后，在透镜的焦平面上，场分布正比于孔径函数的傅里叶变换

针孔面场 P（针孔=有效入瞳）
  ↓ 等效透镜：直接做 FFT
视网膜场 U
  ↓ |U|²
PSF
"""

# 什么是瞳孔函数
# 瞳孔函数（pupil function）是波动光学里描述"光从孔径到像点之间经历了什么相位修饰"的函数：

# 整个流程

r"""

## 第 1 步：定参数

**物理量**：扫描变量 d（0.3–3mm，对数等距取 24 点，在临界孔径 d* 附近加密）和 M（0.5–3D）；先让 λ=550nm（人眼最敏感的绿光，单色先把趋势跑通）、f≈25mm（针孔到视网膜的等效距离，眼球的光学长度）、λf 的乘积决定"空间频率 ↔ 视网膜位置"的换算比例。

要写两个for循环，一个是d的，另一个是λ的，λ的只要绿光和红光和蓝光的

**数值量**：采样网格。瞳孔函数数组里，圆孔直径要占 64–128 像素——太少，圆边缘锯齿化，艾里环结构画不出来；太多，后面 FFT 的数组太大浪费。还要零填充 4–8 倍，也就是把数组撑大到实际孔径的好几倍再补零。FFT 的输出采样间距是 λf/(N·Δx)，数组越大，视网膜面上的像素越细，才能覆盖最大弥散圆（M=3D、d=3mm 时约 0.15mm）并分辨艾里环。

## 第 2 步：造瞳孔函数 P

**振幅部分 A(ρ)**：在 N×N 网格上算每个像素到中心的距离，距离 ≤ d/2 的写 1，否则写 0。这个二值圆盘就是"针孔"。注意它在整个光链里的地位：针孔比瞳孔小得多，所以它取代解剖瞳孔成为系统的有效入瞳——角膜晶状体不再单独建模，合并成后面那个理想透镜（FFT 本身）。

**相位部分**：先算 W₂₀ = Md²/8。这个公式的来历是：近视 M 屈光度 ⇔ 远物点像面在视网膜前方 Δz = Mf² 处 ⇔ 折算到孔径面上是一块二次相位板，峰谷波前误差 W₂₀（SI 单位，米）。M=1D、d=2mm 时 W₂₀≈0.91λ，相位已经绕了近一圈；M=3D、d=3mm 时约 6.1λ，相位疯狂缠绕——这正是光斑糊开的根源。然后对圆盘内每个像素乘 e^(i·k·W₂₀·ρ²)，k=2π/λ，ρ 是到圆心的归一化距离（圆边处 ρ=1）。

两部分相乘得到复数数组 P。**这一步完全没做传播**，它只是"把针孔和近视这两个物理事实编码成一个可以送进衍射积分的输入"。

## 第 3 步：FFT——这一步就是传播

这里用CuPy做，不然太慢了

## 第 4 步：模平方，得 PSF

视网膜感光细胞、以及任何探测器，响应的都是强度不是振幅。所以 `PSF = |U|²`，再除以总和归一化（ΣPSF=1）。

## 第 5 步：从光斑里读指标

- **D50 法**：以光斑中心为圆心画圈，圈内能量达到总能量 50% 的那个圆的角直径，就是弥散角 β。比"半高全宽"稳健，对离焦斑这种非高斯形状更合理；
- **MTF 路线**：PSF 自相关 → 光学传递函数 OTF → 取模得 MTF → 找 MTF 降到 50% 的空间频率 MTF50。这是系统响应函数的语言，后续和"6px/角分"的视标卷积（任务②）直接衔接。

## 第 6 步：换算成视力语言

β 是物理量，临床用 MAR（最小分辨角，角分）。两者关系有现成公式：MAR ≈ β/4 ≈ 0.86·M·d（d 以 mm 计），logMAR = log₁₀(MAR）。检验锚点：M=3D、d=1.2mm → MAR≈3.1′ → logMAR≈0.49。这把"仿真光斑"翻译成"视力表上几行"

## 第 7 步：扫描

(d, M) 网格约 507 个点。每个点：**重新算 W₂₀ → 重新乘相位（圆盘不用重画）→ FFT → 模平方 → D50**。注意扫描时圆盘数组只需生成一次，变的只是相位因子，这是这个流程最快的实现方式。每个点算完只存一行指标 (d, M, β, logMAR)，PSF 数组立即释放——507 张 2048² 图全留着就是几个 GB 的浪费。

因为是5070Ti，所以用CuPy 

## 第 8 步：拼热图，回答物理问题

按 (d, M) 摆成网格画热图。会看到：

- **清晰区**：logMAR 最小的那块，对应最优针孔 d_opt(M)。预期 d_opt 落在 1–1.5mm 附近——和临床"1.2mm 针孔中和约 3D"互证；
- **左下角（小孔区）**：d < d* 的区域，衍射主导，孔越小 logMAR 越差——证明"针孔不是越小越好"；
- **右上角（大孔+高度数）**：离焦主导，β → Md 的几何极限——波动模型在这里应该退化成几何圆盘，可以拿几何公式对照（V8 锚点）。

最终交付：一张 logMAR 热图 + 一条 d_opt(M) 曲线 + d*(M) 验证表。这就是"单孔 PSF 扫描"这个任务的全部产出。

## 贯穿全程的验证锚点


- 第 3 步后：M=0 的暗环位置 vs 理论 1.22λ/d，偏差 <2%（验 FFT）；
- 第 2 步后：±M 的光斑直径必须相同，不对称就是离焦符号写反了；



## 一副针孔眼镜里有两股"糊"的力

光穿过针孔成像，模糊来自两个独立的物理机制，而且它们对着干：

**① 衍射模糊**——孔越小越糊。圆孔的衍射极限是艾里斑，角直径：

$$\theta_{\mathrm{diff}} = \frac{2.44\,\lambda}{d}$$

λ 是波长，d 是孔径。孔砍到一半，衍射斑大一倍。这是波动光学给的硬下限，神仙也没法绕过。

**② 离焦模糊**——孔越大越糊。近视 M 屈光度的人，远物点在视网膜上糊成一个几何弥散圆，角直径：

$$\theta_{\mathrm{def}} = M\,d$$

孔越大，收集光线的范围越宽，同一束光在视网膜上扫开的范围越宽。M 越大（近视越深），同样孔径下糊得越开。

## 关键一步：让两股力打平

衍射项 ∝ 1/d，随孔径**下降**；离焦项 ∝ d，随孔径**上升**。画出来是两条方向相反的曲线，必然相交一次。交点处总模糊最小——这就是"最优针孔"的理论位置。

设它们相等，解出临界孔径：

$$Md = \frac{2.44\lambda}{d} \;\;\Longrightarrow\;\; d^2 = \frac{2.44\lambda}{M} \;\;\Longrightarrow\;\; \boxed{d^* = \sqrt{\dfrac{2.44\lambda}{M}}}$$

系数 2.44 是圆孔衍射特有的（它来自艾里斑第一零点半径 1.22λ/d，乘 2 得直径）。方孔的话这个系数就不同。

## 代入数字，五个值就出来了

取 λ = 550 nm（人眼最敏感的绿光，也是仿真的基准波长），注意统一到米再开方：

$$2.44\lambda = 2.44 \times 550\,\mathrm{nm} = 1.342\,\mu\mathrm{m} = 1.342\times10^{-6}\,\mathrm{m^2}\cdot\mathrm{D}$$

（除以 M，单位正好是 m²，开方得 m。）

| M | 计算 | d* |
|---|---|---|
| 1 D | √(1.342×10⁻⁶/1) = √(1.342×10⁻⁶) | **1.16 mm** |
| 2 D | √(1.342×10⁻⁶/2) = √(0.671×10⁻⁶) | **0.82 mm** |
| 3 D | √(1.342×10⁻⁶/3) = √(0.447×10⁻⁶) | **0.67 mm** |
| 4 D | √(1.342×10⁻⁶/4) | **0.58 mm** |
| 6 D | √(1.342×10⁻⁶/6) | **0.47 mm** |

开平方让衰减变慢，所以近视从 1D 加到 6D（6 倍），最优孔径只从 1.16 缩到 0.47（约 2.5 倍）。这就是表格里那一列数的全部来历。

## 这公式还说了三件事

**它自带波动/几何分界线。** d < d* 的区域衍射占优，几何光学会把 PSF 算成亚角分小点、严重夸大针孔收益，必须用波动（FFT）算；d > d* 且 M 大时几何圆盘就足够准。文档里"全程用波动光学，几何只作对照"的建议，判据就是这条线。

**它解释了临床经验。** 文献说"1.2mm 针孔约中和 3D近视"——代入 M=3D 得 d*=0.67mm，量纲一致、数值同量级；更精确的临床最优区间 0.94–1.75mm 则对应 1D 上下（d*=1.16mm）附近。一个两行推导就和两百年临床经验对上了，这正是它被选为锚点的原因。

**它告诉你 λ 的角色。** d* ∝ √λ，所以红光和蓝光算出来的最优孔径只差百分之几——这就是为什么任务①用单色 550nm 就够，色散留到任务②、③再加多波长。

**五值表 = 令"衍射角 2.44λ/d"等于"离焦角 Md"解出的 d*，代入 λ=550nm 和 M=1,2,3,4,6D，开平方得到的五个理论最优孔径。**

- 第 8 步后：d*(M) 五值表 {1.16, 0.82, 0.67, 0.58, 0.47}mm @ {1,2,3,4,6}D，仿真 vs 理论 <5%。

重点做550nm的，红光和蓝光的，就没必要对这个五值的

最后生成两个图，重点研究550nm的，红光和蓝光放一张图

"""

# 要求

# 所有的物理量，哪怕是网格大小，都不准写成硬编码！必须要规范！这里所有的量，都要写成变量的形式！任何量都不准硬编码
# 生成的图放在output这份文件夹里，图的命名规则是：PSF_550nm.png、PSF_450nm.png、PSF_650nm.png
# 更新：可见光波长数量由单一配置项控制；每个波长独立出图和独立保存五值验证；另增加跨波长对比图。

import os
import warnings
warnings.filterwarnings("ignore", message="CUDA path could not be detected")
# 强制指定 CuPy 的缓存和工作目录，绕开系统 TEMP 的干扰！
os.environ["CUPY_CACHE_DIR"] = r"C:\Temp\cupy_cache"
os.environ["TEMP"] = r"C:\Temp"
os.environ["TMP"] = r"C:\Temp"

# 确保目录存在
os.makedirs(r"C:\Temp\cupy_cache", exist_ok=True)

# 用matplotlib画图

import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cupy as cp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LogNorm


# 单位换算常数：公式内部统一使用 SI 单位，只在输入、输出和绘图边界换算。
METRES_PER_MILLIMETRE = 1.0e-3
METRES_PER_NANOMETRE = 1.0e-9
ARCMINUTES_PER_RADIAN = 180.0 * 60.0 / math.pi
ARCMINUTES_PER_DEGREE = 60.0


def configure_bilingual_plot_font() -> None:
    preferred_fonts = ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC")
    installed_fonts = {font.name for font in font_manager.fontManager.ttflist}
    selected_font = next(
        (font for font in preferred_fonts if font in installed_fonts),
        "DejaVu Sans",
    )
    plt.rcParams["font.sans-serif"] = [selected_font, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


@dataclass(frozen=True)
class SimulationConfig:
    """单孔 PSF 扫描的全部物理量、数值量和输出参数。"""

    # GPU 与数值精度
    gpu_device_id: int = 0
    real_dtype: Any = cp.float32
    complex_dtype: Any = cp.complex64
    accumulator_dtype: Any = cp.float64

    # 波长与眼模型：可见光范围与采样数量均由配置控制。
    visible_min_nm: float = 400.0
    visible_max_nm: float = 700.0
    # 这是波长数量的唯一入口，循环、目录和验证表都不写固定数量。
    wavelength_sample_count: int = 7
    wavelength_rounding_decimals: int = 6
    primary_wavelength_nm: float = 550.0
    focal_length_mm: float = 25.0
    airy_first_zero_radius_coefficient: float = 1.22
    defocus_wavefront_denominator: float = 8.0

    # 孔径扫描：24 个对数点，并在每个理论 d* 附近加密 3 个点。
    diameter_min_mm: float = 0.3
    diameter_max_mm: float = 3.0
    diameter_log_count: int = 24
    diameter_refinement_factors: tuple[float, ...] = (0.95, 1.0, 1.05)
    diameter_refinement_myopia_d: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 6.0)
    diameter_deduplication_tolerance_mm: float = 1.0e-6

    # 近视扫描：13 个点，与 39 个孔径点组成设计中的 507 点网格。
    myopia_min_d: float = 0.0
    myopia_max_d: float = 6.0
    myopia_step_d: float = 0.5

    # 瞳孔采样：圆孔直径占 256 px，整个计算窗口为零填充后的 2048 x 2048。
    grid_size: int = 2048
    aperture_diameter_pixels: float = 256.0
    aperture_edge_softness_pixels: float = 1.0

    # 能量、MTF 与验算判据
    d50_energy_fraction: float = 0.50
    mar_to_d50_divisor: float = 4.0
    mar_floor_arcmin: float = 1.0e-3
    mtf50_threshold: float = 0.50
    first_minimum_profile_fraction: float = 0.20
    theory_acceptance_percent: float = 5.0
    airy_acceptance_percent: float = 2.0
    energy_acceptance_percent: float = 0.1
    symmetry_acceptance_percent: float = 0.1
    validation_diameter_mm: float = 1.0
    symmetry_myopia_d: float = 2.0

    # 出图用的代表点与坐标范围
    psf_snapshot_myopia_d: float = 3.0
    psf_snapshot_diameters_mm: tuple[float, ...] = (0.3, 0.6, 1.2, 2.0, 3.0)
    psf_snapshot_half_width_arcmin: float = 32.0
    psf_log_dynamic_range: float = 3.0
    heatmap_logmar_min: float = -0.5
    heatmap_logmar_max: float = 1.5
    main_figure_size_inches: tuple[float, float] = (24.0, 17.0)
    comparison_figure_size_inches: tuple[float, float] = (26.0, 18.0)
    conclusion_font_size: float = 11.0
    validation_table_font_size: float = 9.0
    comparison_profile_half_width_arcmin: float = 16.0
    comparison_profile_dynamic_range: float = 4.0
    wavelength_colormap_name: str = "turbo"
    figure_dpi: int = 300

    # 输出命名与目录
    output_root_directory_name: str = "output"
    simulation_directory_name: str = "single_hole_PSF"
    wavelength_directory_name_template: str = "PSF_{wavelength_nm:g}nm"
    comparison_directory_name: str = "wavelength_comparison"
    psf_figure_filename_template: str = "PSF_{wavelength_nm:g}nm.png"
    comparison_figure_filename: str = "wavelength_comparison.png"
    metrics_filename: str = "single_hole_metrics.csv"
    validation_filename: str = "single_hole_validation.csv"
    aggregate_metrics_filename: str = "all_wavelength_metrics.csv"
    aggregate_validation_filename: str = "all_wavelength_validation.csv"
    config_filename: str = "single_hole_config.json"

    @property
    def wavelengths_nm(self) -> tuple[float, ...]:
        if self.wavelength_sample_count < 1:
            raise ValueError("wavelength_sample_count must be positive")
        sampled = cp.linspace(
            self.visible_min_nm,
            self.visible_max_nm,
            self.wavelength_sample_count,
            dtype=cp.float64,
        )
        return tuple(
            round(float(value), self.wavelength_rounding_decimals)
            for value in sampled.get()
        )

    def __post_init__(self) -> None:
        if self.visible_min_nm <= 0.0:
            raise ValueError("visible_min_nm must be positive")
        if self.visible_max_nm <= self.visible_min_nm:
            raise ValueError("visible_max_nm must exceed visible_min_nm")
        if self.wavelength_sample_count < 1:
            raise ValueError("wavelength_sample_count must be positive")
        if self.wavelength_rounding_decimals < 0:
            raise ValueError("wavelength_rounding_decimals cannot be negative")
        if self.diameter_min_mm <= 0.0:
            raise ValueError("diameter_min_mm must be positive")
        if self.diameter_max_mm <= self.diameter_min_mm:
            raise ValueError("diameter_max_mm must exceed diameter_min_mm")
        if self.diameter_log_count < 1:
            raise ValueError("diameter_log_count must be positive")
        if self.grid_size < int(math.ceil(self.aperture_diameter_pixels)):
            raise ValueError("grid_size must contain the sampled aperture")
        if self.grid_size % 2 != 0:
            raise ValueError("grid_size must be even for a centred FFT")
        if not 0.0 < self.d50_energy_fraction < 1.0:
            raise ValueError("d50_energy_fraction must lie between zero and one")
        if not 0.0 < self.mtf50_threshold < 1.0:
            raise ValueError("mtf50_threshold must lie between zero and one")
        if not self.visible_min_nm <= self.primary_wavelength_nm <= self.visible_max_nm:
            raise ValueError("primary_wavelength_nm must lie in the visible range")


@dataclass(frozen=True)
class OpticalSampling:
    """给定孔径和波长时的瞳孔面、视网膜面采样关系。"""

    wavelength_m: float
    pupil_pixel_m: float
    retinal_pixel_m: float
    angular_pixel_rad: float
    angular_pixel_arcmin: float
    retinal_window_m: float


def build_myopia_values(config: SimulationConfig) -> tuple[float, ...]:
    point_count = int(
        round((config.myopia_max_d - config.myopia_min_d) / config.myopia_step_d)
    ) + 1
    return tuple(
        config.myopia_min_d + index * config.myopia_step_d
        for index in range(point_count)
    )


def theoretical_optimal_diameter_mm(
    config: SimulationConfig, wavelength_nm: float, myopia_d: float
) -> float:
    if myopia_d <= 0.0:
        raise ValueError("theoretical_optimal_diameter_mm requires positive myopia")
    wavelength_m = wavelength_nm * METRES_PER_NANOMETRE
    diffraction_diameter_coefficient = (
        2.0 * config.airy_first_zero_radius_coefficient
    )
    diameter_m = math.sqrt(
        diffraction_diameter_coefficient * wavelength_m / myopia_d
    )
    return diameter_m / METRES_PER_MILLIMETRE


def build_diameter_grid(
    config: SimulationConfig, wavelength_nm: float
) -> tuple[float, ...]:
    if config.diameter_log_count == 1:
        base_values = [config.diameter_min_mm]
    else:
        log_min = math.log(config.diameter_min_mm)
        log_max = math.log(config.diameter_max_mm)
        log_step = (log_max - log_min) / (config.diameter_log_count - 1)
        base_values = [
            math.exp(log_min + index * log_step)
            for index in range(config.diameter_log_count)
        ]

    refined_values: list[float] = []
    for myopia_d in config.diameter_refinement_myopia_d:
        optimum_mm = theoretical_optimal_diameter_mm(
            config, wavelength_nm, myopia_d
        )
        for factor in config.diameter_refinement_factors:
            candidate_mm = optimum_mm * factor
            if config.diameter_min_mm <= candidate_mm <= config.diameter_max_mm:
                refined_values.append(candidate_mm)

    ordered_values = sorted(base_values + refined_values)
    deduplicated: list[float] = []
    for value in ordered_values:
        if (
            not deduplicated
            or abs(value - deduplicated[-1])
            > config.diameter_deduplication_tolerance_mm
        ):
            deduplicated.append(value)
    return tuple(deduplicated)


def build_sampling(
    config: SimulationConfig, wavelength_nm: float, diameter_mm: float
) -> OpticalSampling:
    wavelength_m = wavelength_nm * METRES_PER_NANOMETRE
    focal_length_m = config.focal_length_mm * METRES_PER_MILLIMETRE
    diameter_m = diameter_mm * METRES_PER_MILLIMETRE
    pupil_pixel_m = diameter_m / config.aperture_diameter_pixels
    retinal_pixel_m = (
        wavelength_m * focal_length_m / (config.grid_size * pupil_pixel_m)
    )
    angular_pixel_rad = retinal_pixel_m / focal_length_m
    angular_pixel_arcmin = angular_pixel_rad * ARCMINUTES_PER_RADIAN
    retinal_window_m = config.grid_size * retinal_pixel_m
    return OpticalSampling(
        wavelength_m=wavelength_m,
        pupil_pixel_m=pupil_pixel_m,
        retinal_pixel_m=retinal_pixel_m,
        angular_pixel_rad=angular_pixel_rad,
        angular_pixel_arcmin=angular_pixel_arcmin,
        retinal_window_m=retinal_window_m,
    )


def build_gpu_grids(
    config: SimulationConfig,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    coordinate_px = (
        cp.arange(config.grid_size, dtype=config.real_dtype)
        - config.grid_size // 2
    )
    grid_x, grid_y = cp.meshgrid(coordinate_px, coordinate_px, indexing="xy")
    radius_px = cp.sqrt(grid_x * grid_x + grid_y * grid_y)
    normalized_radius = radius_px / (config.aperture_diameter_pixels / 2.0)
    normalized_radius_squared = normalized_radius * normalized_radius

    radial_bin_index = cp.floor(radius_px).astype(cp.int32)
    max_radial_index = int(cp.max(radial_bin_index).item())
    radial_bin_count = max_radial_index + 1
    radial_pixel_count = cp.bincount(
        radial_bin_index.ravel(), minlength=radial_bin_count
    ).astype(config.accumulator_dtype)

    return (
        normalized_radius,
        normalized_radius_squared,
        radial_bin_index,
        radial_pixel_count,
    )


def build_soft_aperture(
    config: SimulationConfig, normalized_radius: cp.ndarray
) -> cp.ndarray:
    edge_half_width = (
        config.aperture_edge_softness_pixels / config.aperture_diameter_pixels
    )
    aperture = cp.clip(
        (1.0 - normalized_radius) / (2.0 * edge_half_width) + 0.5,
        0.0,
        1.0,
    )
    return aperture.astype(config.real_dtype)


def compute_psf(
    config: SimulationConfig,
    aperture: cp.ndarray,
    normalized_radius_squared: cp.ndarray,
    wavelength_nm: float,
    diameter_mm: float,
    myopia_d: float,
) -> cp.ndarray:
    diameter_m = diameter_mm * METRES_PER_MILLIMETRE
    wavelength_m = wavelength_nm * METRES_PER_NANOMETRE
    defocus_wavefront_m = (
        myopia_d
        * diameter_m
        * diameter_m
        / config.defocus_wavefront_denominator
    )
    wave_number = 2.0 * math.pi / wavelength_m
    phase = cp.exp(
        1j
        * wave_number
        * defocus_wavefront_m
        * normalized_radius_squared
    )
    pupil_function = aperture * phase
    field = cp.fft.fftshift(cp.fft.fft2(cp.fft.ifftshift(pupil_function)))
    intensity = cp.abs(field) ** 2
    intensity_sum = intensity.sum(dtype=config.accumulator_dtype)
    return (intensity / intensity_sum).astype(config.real_dtype)


def radial_bin_sum(
    values: cp.ndarray,
    radial_bin_index: cp.ndarray,
    radial_bin_count: int,
    accumulator_dtype: Any,
) -> cp.ndarray:
    return cp.bincount(
        radial_bin_index.ravel(),
        weights=values.ravel(),
        minlength=radial_bin_count,
    ).astype(accumulator_dtype)


def enclosing_diameter_arcmin(
    psf: cp.ndarray,
    radial_bin_index: cp.ndarray,
    radial_bin_count: int,
    angular_pixel_arcmin: float,
    config: SimulationConfig,
) -> float:
    radial_energy = radial_bin_sum(
        psf,
        radial_bin_index,
        radial_bin_count,
        config.accumulator_dtype,
    )
    cumulative_energy = cp.cumsum(radial_energy)
    total_energy = cumulative_energy[-1]
    target_energy = total_energy * config.d50_energy_fraction
    upper_index = int(cp.searchsorted(cumulative_energy, target_energy).item())

    if upper_index == 0:
        radius_px = 0.0
    else:
        lower_energy = cumulative_energy[upper_index - 1]
        upper_energy = cumulative_energy[upper_index]
        interpolation = float(
            ((target_energy - lower_energy) / (upper_energy - lower_energy)).item()
        )
        radius_px = upper_index - 1.0 + interpolation

    diameter_arcmin = 2.0 * radius_px * angular_pixel_arcmin
    return diameter_arcmin


def first_dark_ring_arcmin(
    psf: cp.ndarray,
    angular_pixel_arcmin: float,
    config: SimulationConfig,
) -> float:
    centre = config.grid_size // 2
    profile = psf[centre, centre:].get().tolist()
    if len(profile) < 3:
        return math.nan

    peak_index = max(range(len(profile)), key=lambda index: profile[index])
    threshold = profile[peak_index] * config.first_minimum_profile_fraction
    for index in range(max(peak_index + 1, 1), len(profile) - 1):
        is_local_minimum = (
            profile[index] <= profile[index - 1]
            and profile[index] <= profile[index + 1]
        )
        if is_local_minimum and profile[index] <= threshold:
            left = profile[index - 1]
            middle = profile[index]
            right = profile[index + 1]
            denominator = left - 2.0 * middle + right
            radius_px = float(index)
            if denominator != 0.0:
                interpolation = 0.5 * (left - right) / denominator
                if -1.0 <= interpolation <= 1.0:
                    radius_px += interpolation
            return radius_px * angular_pixel_arcmin
    return math.nan


def interpolated_minimum_diameter(
    diameter_values: list[float], metric_values: list[float]
) -> float:
    if len(diameter_values) != len(metric_values):
        raise ValueError("diameter_values and metric_values must have equal length")
    if len(diameter_values) < 3:
        return diameter_values[
            min(range(len(metric_values)), key=metric_values.__getitem__)
        ]

    minimum_index = min(
        range(len(metric_values)), key=metric_values.__getitem__
    )
    if minimum_index == 0 or minimum_index == len(metric_values) - 1:
        return diameter_values[minimum_index]

    log_diameters = [math.log(value) for value in diameter_values]
    x0 = log_diameters[minimum_index - 1] - log_diameters[minimum_index]
    x2 = log_diameters[minimum_index + 1] - log_diameters[minimum_index]
    y0 = metric_values[minimum_index - 1]
    y1 = metric_values[minimum_index]
    y2 = metric_values[minimum_index + 1]

    slope_left = (y0 - y1) / x0
    slope_right = (y2 - y1) / x2
    quadratic = (slope_right - slope_left) / (x2 - x0)
    if quadratic <= 0.0:
        return diameter_values[minimum_index]
    linear = slope_right - quadratic * x2
    vertex = -linear / (2.0 * quadratic)
    if not x0 <= vertex <= x2:
        return diameter_values[minimum_index]
    return math.exp(log_diameters[minimum_index] + vertex)


def mtf50_cycles_per_degree(
    psf: cp.ndarray,
    radial_bin_index: cp.ndarray,
    radial_pixel_count: cp.ndarray,
    radial_bin_count: int,
    sampling: OpticalSampling,
    config: SimulationConfig,
) -> float:
    centred_psf = cp.fft.ifftshift(psf)
    spectrum = cp.fft.fft2(centred_psf)
    otf = cp.fft.fftshift(cp.fft.ifft2(cp.abs(spectrum) ** 2).real)
    centre = config.grid_size // 2
    centre_value = otf[centre, centre]
    if float(cp.abs(centre_value).item()) == 0.0:
        return math.nan
    otf = otf / centre_value

    ring_sum = radial_bin_sum(
        otf,
        radial_bin_index,
        radial_bin_count,
        config.accumulator_dtype,
    )
    valid_bins = radial_pixel_count > 0.0
    mtf_profile = cp.zeros_like(ring_sum)
    mtf_profile[valid_bins] = ring_sum[valid_bins] / radial_pixel_count[valid_bins]
    mtf_profile = cp.clip(cp.abs(mtf_profile), 0.0, 1.0)

    profile = mtf_profile.get().tolist()
    for index in range(1, len(profile)):
        if profile[index] <= config.mtf50_threshold:
            previous = profile[index - 1]
            current = profile[index]
            if previous == current:
                radius_px = float(index)
            else:
                fraction = (
                    (previous - config.mtf50_threshold) / (previous - current)
                )
                radius_px = (index - 1) + fraction
            frequency_per_m = radius_px / (
                config.grid_size * sampling.retinal_pixel_m
            )
            focal_length_m = config.focal_length_mm * METRES_PER_MILLIMETRE
            cycles_per_radian = frequency_per_m * focal_length_m
            return cycles_per_radian / ARCMINUTES_PER_DEGREE
    return math.nan


def snapshot_from_psf(
    psf: cp.ndarray,
    sampling: OpticalSampling,
    config: SimulationConfig,
) -> cp.ndarray:
    half_width_px = int(
        math.ceil(
            config.psf_snapshot_half_width_arcmin
            / sampling.angular_pixel_arcmin
        )
    )
    maximum_half_width_px = config.grid_size // 2 - 1
    half_width_px = min(half_width_px, maximum_half_width_px)
    centre = config.grid_size // 2
    crop_start = centre - half_width_px
    crop_stop = centre + half_width_px
    return psf[crop_start:crop_stop, crop_start:crop_stop]


def nearest_diameter(
    diameter_grid: tuple[float, ...], target_diameter_mm: float
) -> float:
    return min(diameter_grid, key=lambda value: abs(value - target_diameter_mm))


def nearest_wavelength(
    config: SimulationConfig, target_wavelength_nm: float
) -> float:
    return min(
        config.wavelengths_nm,
        key=lambda value: abs(value - target_wavelength_nm),
    )


def wavelength_colour(
    config: SimulationConfig, wavelength_nm: float
) -> tuple[float, float, float, float]:
    visible_span = config.visible_max_nm - config.visible_min_nm
    normalized = (wavelength_nm - config.visible_min_nm) / visible_span
    normalized = min(max(normalized, 0.0), 1.0)
    return plt.get_cmap(config.wavelength_colormap_name)(normalized)


def simulated_optimum_diameter_mm(
    rows: list[dict[str, Any]],
    wavelength_nm: float,
    myopia_d: float,
) -> float:
    candidates = sorted(
        (
            row
            for row in rows
            if row["wavelength_nm"] == wavelength_nm
            and row["myopia_d"] == myopia_d
        ),
        key=lambda row: row["diameter_mm"],
    )
    return interpolated_minimum_diameter(
        [row["diameter_mm"] for row in candidates],
        [row["log_mar"] for row in candidates],
    )


def minimum_logmar(
    rows: list[dict[str, Any]], wavelength_nm: float, myopia_d: float
) -> float:
    return min(
        row["log_mar"]
        for row in rows
        if row["wavelength_nm"] == wavelength_nm
        and row["myopia_d"] == myopia_d
    )


def wavelength_conclusion_lines(
    rows: list[dict[str, Any]],
    wavelength_nm: float,
    myopia_values: tuple[float, ...],
    config: SimulationConfig,
) -> list[str]:
    conclusion_lines: list[str] = []
    for myopia_d in myopia_values:
        optimum_mm = simulated_optimum_diameter_mm(
            rows, wavelength_nm, myopia_d
        )
        best_log_mar = minimum_logmar(rows, wavelength_nm, myopia_d)
        conclusion_lines.append(
            f"在 {wavelength_nm:g} nm、近视 {myopia_d:g} D"
            f"（等效针孔-视网膜距离 {config.focal_length_mm:g} mm）时，"
            f"最合理孔径 d_opt = {optimum_mm:.4f} mm，"
            f"对应 logMAR = {best_log_mar:.3f}。"
        )
        conclusion_lines.append(
            f"At {wavelength_nm:g} nm and {myopia_d:g} D myopia "
            f"(equivalent pinhole-retina distance "
            f"{config.focal_length_mm:g} mm), the optimal pinhole "
            f"diameter is {optimum_mm:.4f} mm, giving "
            f"logMAR = {best_log_mar:.3f}."
        )
    return conclusion_lines


def rows_for_wavelength(
    rows: list[dict[str, Any]], wavelength_nm: float
) -> list[dict[str, Any]]:
    return [row for row in rows if row["wavelength_nm"] == wavelength_nm]


def build_validation_rows(
    config: SimulationConfig,
    rows: list[dict[str, Any]],
    myopia_values: tuple[float, ...],
) -> list[dict[str, Any]]:
    validation_rows: list[dict[str, Any]] = []
    for wavelength_nm in config.wavelengths_nm:
        wavelength_rows = rows_for_wavelength(rows, wavelength_nm)
        wavelength_rows_at_zero = [
            row for row in wavelength_rows if row["myopia_d"] == 0.0
        ]
        airy_row = min(
            wavelength_rows_at_zero,
            key=lambda row: abs(
                row["diameter_mm"] - config.validation_diameter_mm
            ),
        )
        airy_theory = (
            config.airy_first_zero_radius_coefficient
            * wavelength_nm
            * METRES_PER_NANOMETRE
            / (airy_row["diameter_mm"] * METRES_PER_MILLIMETRE)
            * ARCMINUTES_PER_RADIAN
        )
        airy_error = (
            (airy_row["first_dark_ring_arcmin"] - airy_theory)
            / airy_theory
            * 100.0
        )
        validation_rows.append(
            {
                "validation": "Airy first dark ring",
                "wavelength_nm": wavelength_nm,
                "myopia_d": 0.0,
                "diameter_mm": airy_row["diameter_mm"],
                "simulation_value": airy_row["first_dark_ring_arcmin"],
                "theory_value": airy_theory,
                "error_percent": airy_error,
                "acceptance_percent": config.airy_acceptance_percent,
                "passed": abs(airy_error) <= config.airy_acceptance_percent,
            }
        )

        for myopia_d in config.diameter_refinement_myopia_d:
            if myopia_d not in myopia_values:
                continue
            candidates = sorted(
                (
                    row
                    for row in wavelength_rows
                    if row["myopia_d"] == myopia_d
                ),
                key=lambda row: row["diameter_mm"],
            )
            simulated_mm = interpolated_minimum_diameter(
                [row["diameter_mm"] for row in candidates],
                [row["log_mar"] for row in candidates],
            )
            theory_mm = theoretical_optimal_diameter_mm(
                config, wavelength_nm, myopia_d
            )
            error = (
                (simulated_mm - theory_mm)
                / theory_mm
                * 100.0
            )
            validation_rows.append(
                {
                    "validation": "optimal diameter",
                    "wavelength_nm": wavelength_nm,
                    "myopia_d": myopia_d,
                    "diameter_mm": simulated_mm,
                    "simulation_value": simulated_mm,
                    "theory_value": theory_mm,
                    "error_percent": error,
                    "acceptance_percent": config.theory_acceptance_percent,
                    "passed": abs(error) <= config.theory_acceptance_percent,
                }
            )
    return validation_rows


def build_integrity_validation_rows(
    config: SimulationConfig,
    rows: list[dict[str, Any]],
    aperture: cp.ndarray,
    normalized_radius_squared: cp.ndarray,
    radial_bin_index: cp.ndarray,
    radial_bin_count: int,
) -> list[dict[str, Any]]:
    reference_wavelength_nm = nearest_wavelength(
        config, config.primary_wavelength_nm
    )
    maximum_energy_error = max(
        abs(row["psf_total"] - 1.0) * 100.0 for row in rows
    )
    energy_row = {
        "validation": "PSF energy normalization",
        "wavelength_nm": reference_wavelength_nm,
        "myopia_d": config.symmetry_myopia_d,
        "diameter_mm": nearest_diameter(
            build_diameter_grid(config, reference_wavelength_nm),
            config.validation_diameter_mm,
        ),
        "simulation_value": maximum_energy_error,
        "theory_value": 0.0,
        "error_percent": maximum_energy_error,
        "acceptance_percent": config.energy_acceptance_percent,
        "passed": maximum_energy_error <= config.energy_acceptance_percent,
    }

    symmetry_diameter_mm = nearest_diameter(
        build_diameter_grid(config, reference_wavelength_nm),
        config.validation_diameter_mm,
    )
    sampling = build_sampling(
        config, reference_wavelength_nm, symmetry_diameter_mm
    )
    positive_psf = compute_psf(
        config,
        aperture,
        normalized_radius_squared,
        reference_wavelength_nm,
        symmetry_diameter_mm,
        config.symmetry_myopia_d,
    )
    negative_psf = compute_psf(
        config,
        aperture,
        normalized_radius_squared,
        reference_wavelength_nm,
        symmetry_diameter_mm,
        -config.symmetry_myopia_d,
    )
    positive_d50 = enclosing_diameter_arcmin(
        positive_psf,
        radial_bin_index,
        radial_bin_count,
        sampling.angular_pixel_arcmin,
        config,
    )
    negative_d50 = enclosing_diameter_arcmin(
        negative_psf,
        radial_bin_index,
        radial_bin_count,
        sampling.angular_pixel_arcmin,
        config,
    )
    symmetry_error = (negative_d50 - positive_d50) / positive_d50 * 100.0
    symmetry_row = {
        "validation": "defocus sign symmetry",
        "wavelength_nm": reference_wavelength_nm,
        "myopia_d": config.symmetry_myopia_d,
        "diameter_mm": symmetry_diameter_mm,
        "simulation_value": positive_d50,
        "theory_value": negative_d50,
        "error_percent": symmetry_error,
        "acceptance_percent": config.symmetry_acceptance_percent,
        "passed": abs(symmetry_error) <= config.symmetry_acceptance_percent,
    }
    return [energy_row, symmetry_row]


def write_csv(
    path: Path, rows: list[dict[str, Any]], fieldnames: list[str]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_logmar_matrix(
    rows: list[dict[str, Any]],
    wavelength_nm: float,
    diameter_grid: tuple[float, ...],
    myopia_values: tuple[float, ...],
) -> list[list[float]]:
    row_index = {
        (row["wavelength_nm"], row["diameter_mm"], row["myopia_d"]): row["log_mar"]
        for row in rows
    }
    plotted_myopia = [value for value in myopia_values if value > 0.0]
    return [
        [
            row_index[(wavelength_nm, diameter_mm, myopia_d)]
            for diameter_mm in diameter_grid
        ]
        for myopia_d in plotted_myopia
    ]


def plot_logmar_heatmap(
    axis: plt.Axes,
    rows: list[dict[str, Any]],
    wavelength_nm: float,
    diameter_grid: tuple[float, ...],
    myopia_values: tuple[float, ...],
    config: SimulationConfig,
) -> None:
    plotted_myopia = [value for value in myopia_values if value > 0.0]
    matrix = build_logmar_matrix(
        rows, wavelength_nm, diameter_grid, myopia_values
    )
    image = axis.pcolormesh(
        diameter_grid,
        plotted_myopia,
        matrix,
        shading="auto",
        cmap="viridis",
        vmin=config.heatmap_logmar_min,
        vmax=config.heatmap_logmar_max,
    )
    theory_curve = [
        theoretical_optimal_diameter_mm(config, wavelength_nm, myopia_d)
        for myopia_d in plotted_myopia
    ]
    axis.plot(
        theory_curve,
        plotted_myopia,
        color="white",
        linestyle="--",
        linewidth=1.8,
        label="理论 d* / Theory d*",
    )
    axis.set_xlabel("针孔直径 / Pinhole diameter d (mm)")
    axis.set_ylabel("近视度数 / Myopia M (D)")
    axis.set_title(f"logMAR 热图 / logMAR map ({wavelength_nm:g} nm)")
    axis.set_xscale("log")
    axis.legend(loc="lower right", fontsize=9, framealpha=0.92)
    colorbar = axis.figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("logMAR（D50 路径 / D50 route）")


def plot_optimal_diameter_curve(
    axis: plt.Axes,
    rows: list[dict[str, Any]],
    wavelengths_nm: tuple[float, ...],
    myopia_values: tuple[float, ...],
    config: SimulationConfig,
    show_theory_legend: bool = True,
    legend_columns: int = 2,
) -> None:
    positive_myopia = [value for value in myopia_values if value > 0.0]
    multiple_wavelengths = len(wavelengths_nm) > 1
    for wavelength_nm in wavelengths_nm:
        wavelength_rows = rows_for_wavelength(rows, wavelength_nm)
        simulated = []
        theory = []
        for myopia_d in positive_myopia:
            simulated.append(
                simulated_optimum_diameter_mm(
                    wavelength_rows, wavelength_nm, myopia_d
                )
            )
            theory.append(
                theoretical_optimal_diameter_mm(
                    config, wavelength_nm, myopia_d
                )
            )
        colour = wavelength_colour(config, wavelength_nm)
        axis.plot(
            positive_myopia,
            simulated,
            marker="o",
            color=colour,
            linewidth=1.8,
            label=f"{wavelength_nm:g} nm 仿真 / Simulation",
        )
        axis.plot(
            positive_myopia,
            theory,
            color=colour,
            linestyle="--",
            linewidth=1.4,
            label=(
                f"{wavelength_nm:g} nm 理论 / Theory"
                if show_theory_legend
                else "_nolegend_"
            ),
        )
    axis.set_xlabel("近视度数 / Myopia M (D)")
    axis.set_ylabel("最优孔径 / Optimal diameter (mm)")
    axis.set_title("最优孔径 / Optimal diameter d_opt(M)")
    axis.set_yscale("log")
    axis.grid(alpha=0.25)
    axis.legend(
        fontsize=7 if multiple_wavelengths else 9,
        ncol=legend_columns,
        loc="best",
    )


def draw_psf_mosaic(
    figure: plt.Figure,
    subplot_spec: Any,
    snapshots: dict[tuple[float, float], cp.ndarray],
    wavelength_nm: float,
    representative_diameters_mm: tuple[float, ...],
    samplings: dict[tuple[float, float], OpticalSampling],
    config: SimulationConfig,
) -> None:
    mosaic = subplot_spec.subgridspec(
        1, len(representative_diameters_mm), wspace=0.05
    )
    for index, diameter_mm in enumerate(representative_diameters_mm):
        axis = figure.add_subplot(mosaic[0, index])
        key = (wavelength_nm, diameter_mm)
        snapshot = snapshots[key]
        sampling = samplings[key]
        host_snapshot = snapshot.get()
        peak = float(host_snapshot.max())
        dynamic_floor = peak * (10.0 ** (-config.psf_log_dynamic_range))
        half_width_arcmin = (
            snapshot.shape[0] / 2.0 * sampling.angular_pixel_arcmin
        )
        image = axis.imshow(
            host_snapshot,
            origin="lower",
            extent=(
                -half_width_arcmin,
                half_width_arcmin,
                -half_width_arcmin,
                half_width_arcmin,
            ),
            cmap="magma",
            norm=LogNorm(vmin=dynamic_floor, vmax=peak),
        )
        axis.set_title(f"d = {diameter_mm:.3f} mm", fontsize=9)
        axis.set_xlabel("角分 / arcmin")
        if index == 0:
            axis.set_ylabel("角分 / arcmin")
        else:
            axis.set_yticklabels([])
        axis.set_aspect("equal")
        colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
        colorbar.ax.tick_params(labelsize=7)


def plot_validation_table(
    axis: plt.Axes,
    validation_rows: list[dict[str, Any]],
    config: SimulationConfig,
) -> None:
    axis.axis("off")
    headers = [
        "检查 / Check",
        "波长 / λ",
        "近视 / M",
        "仿真 / Simulation",
        "理论 / Theory",
        "误差 / Error",
    ]
    table_rows = []
    for row in validation_rows:
        validation_label = {
            "Airy first dark ring": "Airy 暗环 / Airy ring",
            "optimal diameter": "最优孔径 / Optimal diameter",
            "PSF energy normalization": "能量归一化 / Energy",
            "defocus sign symmetry": "离焦对称 / Defocus symmetry",
        }.get(row["validation"], row["validation"])
        table_rows.append(
            [
                validation_label,
                f"{row['wavelength_nm']:g} nm",
                f"{row['myopia_d']:g} D",
                f"{row['simulation_value']:.4g}",
                f"{row['theory_value']:.4g}",
                f"{row['error_percent']:+.2f}%",
            ]
        )
    table = axis.table(
        cellText=table_rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(config.validation_table_font_size)
    table.scale(1.0, 1.35)
    axis.set_title("数值锚点 / Numerical anchors", pad=14)


def save_wavelength_figure(
    output_path: Path,
    rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    wavelength_nm: float,
    diameter_grid: tuple[float, ...],
    myopia_values: tuple[float, ...],
    snapshots: dict[tuple[float, float], cp.ndarray],
    representative_diameters_mm: tuple[float, ...],
    samplings: dict[tuple[float, float], OpticalSampling],
    config: SimulationConfig,
) -> None:
    figure = plt.figure(figsize=config.main_figure_size_inches, dpi=config.figure_dpi)
    grid = figure.add_gridspec(
        3,
        2,
        height_ratios=(1.10, 0.92, 0.70),
        hspace=0.36,
        wspace=0.18,
    )
    heatmap_axis = figure.add_subplot(grid[0, 0])
    right_column = grid[0, 1].subgridspec(
        2, 1, height_ratios=(1.0, 0.82), hspace=0.22
    )
    curve_axis = figure.add_subplot(right_column[0, 0])
    validation_axis = figure.add_subplot(right_column[1, 0])
    mosaic_spec = grid[1, :]
    conclusion_axis = figure.add_subplot(grid[2, :])
    conclusion_axis.axis("off")

    plot_logmar_heatmap(
        heatmap_axis,
        rows,
        wavelength_nm,
        diameter_grid,
        myopia_values,
        config,
    )
    plot_optimal_diameter_curve(
        curve_axis,
        rows,
        (wavelength_nm,),
        myopia_values,
        config,
    )
    draw_psf_mosaic(
        figure,
        mosaic_spec,
        snapshots,
        wavelength_nm,
        representative_diameters_mm,
        samplings,
        config,
    )
    wavelength_validation = [
        row
        for row in validation_rows
        if row["wavelength_nm"] == wavelength_nm
    ]
    plot_validation_table(
        validation_axis,
        wavelength_validation,
        config,
    )
    conclusion_lines = wavelength_conclusion_lines(
        rows,
        wavelength_nm,
        config.diameter_refinement_myopia_d,
        config,
    )
    conclusion_axis.set_title(
        "结论 / Conclusions", fontsize=14, pad=12
    )
    conclusion_axis.text(
        0.01,
        0.98,
        "\n".join(conclusion_lines),
        transform=conclusion_axis.transAxes,
        ha="left",
        va="top",
        fontsize=config.conclusion_font_size,
        linespacing=1.45,
    )
    figure.suptitle(
        f"单孔 PSF 扫描 / Single-hole PSF scan: {wavelength_nm:g} nm",
        fontsize=17,
    )
    figure.savefig(output_path, dpi=config.figure_dpi, bbox_inches="tight")
    plt.close(figure)


def plot_optimum_shift_vs_wavelength(
    axis: plt.Axes,
    rows: list[dict[str, Any]],
    wavelengths_nm: tuple[float, ...],
    myopia_d: float,
    config: SimulationConfig,
) -> None:
    reference_nm = nearest_wavelength(config, config.primary_wavelength_nm)
    reference_diameter_mm = simulated_optimum_diameter_mm(
        rows, reference_nm, myopia_d
    )
    simulated_ratio = []
    theory_ratio = []
    for wavelength_nm in wavelengths_nm:
        optimum_mm = simulated_optimum_diameter_mm(
            rows, wavelength_nm, myopia_d
        )
        simulated_ratio.append(optimum_mm / reference_diameter_mm)
        theory_ratio.append(math.sqrt(wavelength_nm / reference_nm))

    axis.plot(
        wavelengths_nm,
        simulated_ratio,
        marker="o",
        color="#111111",
        linewidth=2.0,
        label="仿真 / Simulation",
    )
    axis.plot(
        wavelengths_nm,
        theory_ratio,
        linestyle="--",
        color="#d62728",
        linewidth=1.8,
        label="λ^0.5 理论 / λ^0.5 theory",
    )
    axis.set_xlabel("波长 / Wavelength (nm)")
    axis.set_ylabel(
        f"相对 d_opt / Relative d_opt\n（基准 / Reference: {reference_nm:g} nm）"
    )
    axis.set_title(
        f"波长对最优孔径的影响 / Wavelength shift at M = {myopia_d:g} D"
    )
    axis.grid(alpha=0.25)
    axis.legend()


def plot_best_logmar_vs_wavelength(
    axis: plt.Axes,
    rows: list[dict[str, Any]],
    wavelengths_nm: tuple[float, ...],
    myopia_d: float,
    config: SimulationConfig,
) -> None:
    values = [
        minimum_logmar(rows, wavelength_nm, myopia_d)
        for wavelength_nm in wavelengths_nm
    ]
    axis.plot(
        wavelengths_nm,
        values,
        marker="o",
        color="#174ea6",
        linewidth=2.0,
    )
    axis.set_xlabel("波长 / Wavelength (nm)")
    axis.set_ylabel("最佳 logMAR / Best logMAR")
    axis.set_title(
        f"各波长最佳视锐度 / Best acuity at M = {myopia_d:g} D"
    )
    axis.grid(alpha=0.25)


def build_wavelength_comparison_profiles(
    config: SimulationConfig,
    rows: list[dict[str, Any]],
    aperture: cp.ndarray,
    normalized_radius_squared: cp.ndarray,
    myopia_d: float,
) -> dict[float, tuple[list[float], list[float], float]]:
    profiles: dict[float, tuple[list[float], list[float], float]] = {}
    for wavelength_nm in config.wavelengths_nm:
        optimum_mm = simulated_optimum_diameter_mm(
            rows, wavelength_nm, myopia_d
        )
        sampling = build_sampling(config, wavelength_nm, optimum_mm)
        psf = compute_psf(
            config,
            aperture,
            normalized_radius_squared,
            wavelength_nm,
            optimum_mm,
            myopia_d,
        )
        centre = config.grid_size // 2
        host_profile = psf[centre, centre:].get().tolist()
        max_index = min(
            len(host_profile),
            int(
                math.ceil(
                    config.comparison_profile_half_width_arcmin
                    / sampling.angular_pixel_arcmin
                )
            )
            + 1,
        )
        angles = [
            index * sampling.angular_pixel_arcmin for index in range(max_index)
        ]
        floor = 10.0 ** (-config.comparison_profile_dynamic_range)
        intensities = [
            max(host_profile[index], floor) for index in range(max_index)
        ]
        profiles[wavelength_nm] = (angles, intensities, optimum_mm)
    cp.cuda.Stream.null.synchronize()
    return profiles


def save_wavelength_comparison_figure(
    output_path: Path,
    rows: list[dict[str, Any]],
    wavelengths_nm: tuple[float, ...],
    myopia_values: tuple[float, ...],
    profiles: dict[float, tuple[list[float], list[float], float]],
    config: SimulationConfig,
) -> None:
    figure = plt.figure(
        figsize=config.comparison_figure_size_inches, dpi=config.figure_dpi
    )
    grid = figure.add_gridspec(
        4,
        2,
        height_ratios=(1.05, 0.80, 1.05, 0.78),
        hspace=0.34,
        wspace=0.22,
    )
    optimum_axis = figure.add_subplot(grid[0, :])
    acuity_axis = figure.add_subplot(grid[1, 0])
    shift_axis = figure.add_subplot(grid[1, 1])
    profile_axis = figure.add_subplot(grid[2, :])
    conclusion_spec = grid[3, :].subgridspec(1, 2, wspace=0.08)
    conclusion_left_axis = figure.add_subplot(conclusion_spec[0, 0])
    conclusion_right_axis = figure.add_subplot(conclusion_spec[0, 1])

    plot_optimal_diameter_curve(
        optimum_axis,
        rows,
        wavelengths_nm,
        myopia_values,
        config,
        show_theory_legend=False,
        legend_columns=4,
    )
    optimum_axis.text(
        0.99,
        0.03,
        "同色虚线 / Dotted curves: theory",
        transform=optimum_axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
    )
    plot_best_logmar_vs_wavelength(
        acuity_axis,
        rows,
        wavelengths_nm,
        config.psf_snapshot_myopia_d,
        config,
    )
    plot_optimum_shift_vs_wavelength(
        shift_axis,
        rows,
        wavelengths_nm,
        config.psf_snapshot_myopia_d,
        config,
    )
    for wavelength_nm in wavelengths_nm:
        angles, intensities, optimum_mm = profiles[wavelength_nm]
        profile_axis.semilogy(
            angles,
            intensities,
            color=wavelength_colour(config, wavelength_nm),
            linewidth=1.6,
            label=f"{wavelength_nm:g} nm (d={optimum_mm:.3f} mm)",
        )
    profile_axis.set_xlabel("角半径 / Angular radius (arcmin)")
    profile_axis.set_ylabel("归一化强度 / Normalized intensity")
    profile_axis.set_title(
        f"各波长最优 PSF 剖面 / PSF profiles at M = {config.psf_snapshot_myopia_d:g} D"
    )
    profile_axis.grid(alpha=0.25)
    profile_axis.legend(fontsize=8, ncol=4)

    comparison_conclusions: list[str] = []
    for wavelength_nm in wavelengths_nm:
        comparison_conclusions.append(
            "\n".join(
                wavelength_conclusion_lines(
                    rows,
                    wavelength_nm,
                    (config.psf_snapshot_myopia_d,),
                    config,
                )
            )
        )
    split_index = math.ceil(len(comparison_conclusions) / 2)
    conclusion_columns = (
        comparison_conclusions[:split_index],
        comparison_conclusions[split_index:],
    )
    for axis, conclusion_column in zip(
        (conclusion_left_axis, conclusion_right_axis),
        conclusion_columns,
    ):
        axis.axis("off")
        axis.text(
            0.0,
            1.0,
            "\n\n".join(conclusion_column),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=config.conclusion_font_size,
            linespacing=1.35,
        )
    conclusion_left_axis.set_title(
        "代表近视度数下的结论 / Conclusions at representative myopia",
        fontsize=13,
        pad=10,
    )

    figure.suptitle(
        f"可见光 {len(wavelengths_nm)} 波长单孔 PSF 对比 / "
        "Single-hole PSF comparison across visible wavelengths",
        fontsize=17,
    )
    figure.savefig(output_path, dpi=config.figure_dpi, bbox_inches="tight")
    plt.close(figure)


def print_progress(
    wavelength_nm: float,
    diameter_mm: float,
    completed: int,
    total: int,
    started_at: float,
) -> None:
    elapsed_s = time.perf_counter() - started_at
    rate = completed / elapsed_s if elapsed_s > 0.0 else 0.0
    print(
        f"[{completed:4d}/{total:4d}] "
        f"lambda={wavelength_nm:g} nm d={diameter_mm:.4f} mm "
        f"rate={rate:.1f} points/s",
        flush=True,
    )


def main() -> None:
    config = SimulationConfig()
    configure_bilingual_plot_font()
    cp.cuda.Device(config.gpu_device_id).use()
    gpu_properties = cp.cuda.runtime.getDeviceProperties(config.gpu_device_id)
    gpu_name = gpu_properties["name"]
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode()

    script_directory = Path(__file__).resolve().parent
    output_root = script_directory / config.output_root_directory_name
    simulation_directory = output_root / config.simulation_directory_name
    comparison_directory = (
        simulation_directory / config.comparison_directory_name
    )
    simulation_directory.mkdir(parents=True, exist_ok=True)
    comparison_directory.mkdir(parents=True, exist_ok=True)

    wavelengths_nm = config.wavelengths_nm
    myopia_values = build_myopia_values(config)
    diameter_grids = {
        wavelength_nm: build_diameter_grid(config, wavelength_nm)
        for wavelength_nm in wavelengths_nm
    }
    total_points = sum(len(grid) for grid in diameter_grids.values()) * len(
        myopia_values
    )

    (
        normalized_radius,
        normalized_radius_squared,
        radial_bin_index,
        radial_pixel_count,
    ) = build_gpu_grids(config)
    radial_bin_count = int(radial_pixel_count.size)

    representative_diameters_by_wavelength = {
        wavelength_nm: tuple(
            sorted(
                {
                    nearest_diameter(grid, target)
                    for target in config.psf_snapshot_diameters_mm
                }
            )
        )
        for wavelength_nm, grid in diameter_grids.items()
    }
    rows: list[dict[str, Any]] = []
    snapshots: dict[tuple[float, float], cp.ndarray] = {}
    samplings: dict[tuple[float, float], OpticalSampling] = {}
    started_at = time.perf_counter()
    completed = 0

    print(
        f"GPU: {gpu_name}; grid={config.grid_size}; "
        f"wavelengths={len(wavelengths_nm)}; points={total_points}; "
        f"output={simulation_directory}",
        flush=True,
    )

    aperture = build_soft_aperture(config, normalized_radius)
    for diameter_mm in sorted(
        {diameter for grid in diameter_grids.values() for diameter in grid}
    ):
        for wavelength_nm in wavelengths_nm:
            diameter_grid = diameter_grids[wavelength_nm]
            if diameter_mm not in diameter_grid:
                continue
            sampling = build_sampling(config, wavelength_nm, diameter_mm)
            for myopia_d in myopia_values:
                psf = compute_psf(
                    config,
                    aperture,
                    normalized_radius_squared,
                    wavelength_nm,
                    diameter_mm,
                    myopia_d,
                )
                d50_arcmin = enclosing_diameter_arcmin(
                    psf,
                    radial_bin_index,
                    radial_bin_count,
                    sampling.angular_pixel_arcmin,
                    config,
                )
                log_mar = math.log10(
                    max(d50_arcmin / config.mar_to_d50_divisor, config.mar_floor_arcmin)
                )
                first_ring = first_dark_ring_arcmin(
                    psf,
                    sampling.angular_pixel_arcmin,
                    config,
                )
                mtf50 = mtf50_cycles_per_degree(
                    psf,
                    radial_bin_index,
                    radial_pixel_count,
                    radial_bin_count,
                    sampling,
                    config,
                )
                theory_airy_radius_arcmin = (
                    config.airy_first_zero_radius_coefficient
                    * sampling.wavelength_m
                    / (diameter_mm * METRES_PER_MILLIMETRE)
                    * ARCMINUTES_PER_RADIAN
                )
                theory_geometric_diameter_arcmin = (
                    myopia_d
                    * diameter_mm
                    * ARCMINUTES_PER_RADIAN
                    * METRES_PER_MILLIMETRE
                )
                rows.append(
                    {
                        "wavelength_nm": wavelength_nm,
                        "diameter_mm": diameter_mm,
                        "myopia_d": myopia_d,
                        "d50_arcmin": d50_arcmin,
                        "log_mar": log_mar,
                        "mtf50_cpd": mtf50,
                        "first_dark_ring_arcmin": first_ring,
                        "theory_airy_radius_arcmin": theory_airy_radius_arcmin,
                        "theory_geometric_diameter_arcmin": theory_geometric_diameter_arcmin,
                        "angular_pixel_arcmin": sampling.angular_pixel_arcmin,
                        "retinal_pixel_m": sampling.retinal_pixel_m,
                        "retinal_window_mm": sampling.retinal_window_m
                        / METRES_PER_MILLIMETRE,
                        "psf_total": float(
                            psf.sum(dtype=config.accumulator_dtype).item()
                        ),
                    }
                )

                should_save_snapshot = (
                    diameter_mm
                    in representative_diameters_by_wavelength[wavelength_nm]
                    and myopia_d == config.psf_snapshot_myopia_d
                )
                if should_save_snapshot:
                    snapshots[(wavelength_nm, diameter_mm)] = snapshot_from_psf(
                        psf, sampling, config
                    )
                    samplings[(wavelength_nm, diameter_mm)] = sampling

                completed += 1
                if completed % len(myopia_values) == 0:
                    print_progress(
                        wavelength_nm,
                        diameter_mm,
                        completed,
                        total_points,
                        started_at,
                    )

    cp.cuda.Stream.null.synchronize()

    validation_rows = build_validation_rows(config, rows, myopia_values)
    validation_rows.extend(
        build_integrity_validation_rows(
            config,
            rows,
            aperture,
            normalized_radius_squared,
            radial_bin_index,
            radial_bin_count,
        )
    )

    metrics_fields = list(rows[0].keys())
    validation_fields = list(validation_rows[0].keys())
    write_csv(
        simulation_directory / config.aggregate_metrics_filename,
        rows,
        metrics_fields,
    )
    write_csv(
        simulation_directory / config.aggregate_validation_filename,
        validation_rows,
        validation_fields,
    )

    config_payload = asdict(config)
    config_payload["real_dtype"] = config.real_dtype.__name__
    config_payload["complex_dtype"] = config.complex_dtype.__name__
    config_payload["accumulator_dtype"] = config.accumulator_dtype.__name__
    config_payload["gpu_name"] = gpu_name
    config_payload["wavelengths_nm"] = list(wavelengths_nm)
    with (simulation_directory / config.config_filename).open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(config_payload, file, ensure_ascii=False, indent=2)

    for wavelength_nm in wavelengths_nm:
        wavelength_directory = simulation_directory / (
            config.wavelength_directory_name_template.format(
                wavelength_nm=wavelength_nm
            )
        )
        wavelength_directory.mkdir(parents=True, exist_ok=True)
        wavelength_rows = rows_for_wavelength(rows, wavelength_nm)
        wavelength_validation = [
            row
            for row in validation_rows
            if row["wavelength_nm"] == wavelength_nm
        ]
        write_csv(
            wavelength_directory / config.metrics_filename,
            wavelength_rows,
            metrics_fields,
        )
        write_csv(
            wavelength_directory / config.validation_filename,
            wavelength_validation,
            validation_fields,
        )
        wavelength_output = (
            wavelength_directory
            / config.psf_figure_filename_template.format(
                wavelength_nm=wavelength_nm
            )
        )
        save_wavelength_figure(
            wavelength_output,
            rows,
            validation_rows,
            wavelength_nm,
            diameter_grids[wavelength_nm],
            myopia_values,
            snapshots,
            representative_diameters_by_wavelength[wavelength_nm],
            samplings,
            config,
        )

    comparison_profiles = build_wavelength_comparison_profiles(
        config,
        rows,
        aperture,
        normalized_radius_squared,
        config.psf_snapshot_myopia_d,
    )
    save_wavelength_comparison_figure(
        comparison_directory / config.comparison_figure_filename,
        rows,
        wavelengths_nm,
        myopia_values,
        comparison_profiles,
        config,
    )

    print("\nValidation anchors:", flush=True)
    for row in validation_rows:
        status = "PASS" if row["passed"] else "FAIL"
        print(
            f"  [{status}] {row['validation']} | "
            f"{row['wavelength_nm']:g} nm | M={row['myopia_d']:g} D | "
            f"error={row['error_percent']:+.2f}%",
            flush=True,
        )

    optical_rows = [row for row in rows if row["myopia_d"] > 0.0]
    best_row = min(optical_rows, key=lambda row: row["log_mar"])
    print(
        "Best scanned point: "
        f"lambda={best_row['wavelength_nm']:g} nm, "
        f"d={best_row['diameter_mm']:.4f} mm, "
        f"M={best_row['myopia_d']:g} D, "
        f"logMAR={best_row['log_mar']:.4f}",
        flush=True,
    )
    print(f"Saved outputs to: {simulation_directory}", flush=True)


if __name__ == "__main__":
    main()
