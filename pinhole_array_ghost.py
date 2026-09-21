# 第三阶段仿真：多针孔阵列重影与可分辨性
# ============================================================================
# 本文件分两部分：
#   (A) 上方是第三阶段的实现方案、公式口径、验证门槛和踩坑提示；
#   (B) 下方是实现部分，按方案第 14 节的函数顺序落地。
# 编码时不要跳过验证，也不要把第三步的物理模型和第一步、第二步的模型
# 重复实现。
# 编码过程中的版本控制行为必须遵守第 17 节的 Git 提交与推送规范：
# 写一点就 commit 并 push，提交信息必须多角度详细展开。
#
# 开始编码前必须阅读：
# 1. single_hole_PSF.py
#    - 已提供单孔瞳孔函数、离焦相位、FFT、PSF、采样和 D50 指标。
#    - 第三步必须复用这些函数，不能重新写一份“近似相同”的单孔传播器。
# 2. single_hole_retinal_image.py
#    - 已提供 PSF 重采样、全尺寸卷积、中心裁切和验证纪律。
#    - 第三步的平移叠加必须沿用其中心对齐和能量守恒逻辑。
# 3. tests/test_single_hole_retinal_image.py
#    - 已覆盖卷积核中心、非中心脉冲、质心、总能量和配置边界。
#    - 第三步先补同等级的小尺寸测试，再跑大图。
# 4. README.md 与 git log
#    - 第一步确认了圆孔、离焦、D50、最优孔径和波长趋势。
#    - 第二步修复了覆盖度、卷积中心、重采样中心和显示窗口四类问题。
#    - 第三步要吸收这些教训，不能重新引入同一类错误。
# 5. 两份 PDF
#    - 《仿真概念澄清_重影光源眼模型与库选型.pdf》第 1 节的非相干重影机制。
#    - 《针孔太阳镜_仿真技术栈与工作流程.pdf》第 1、3、4、5、6 节。
#
# 本阶段要回答的问题：
# - 多针孔阵列会在视网膜上产生几组重影，相邻重影的角间隔是多少？
# - 主像和重影谁亮，重影是否可与主像分离，太阳扩展源会不会把它们抹成一片？
# - 正方形密铺、三角形密铺、六边形密铺，哪一种更容易产生可识别栅格？
# - 中心孔加一环、两环、三环的同心环阵列，随着圈数和每圈孔数增加，
#   重影怎样从离散点变成环状或准连续背景？
# - 孔径 d、孔距 p 和相对关系 p/d 分别如何改变重影机制？
# - 在固定 p、固定 d、固定 d/p 比三种切法下，重影数量、位置、峰比、
#   分离角和融合程度如何变化？
# - 哪些参数组合应直接判定为不可接受，而不是靠“图看起来还行”判断？
#
# 范围锁定：
# - 对话新增的主要求是：孔心排布、孔径 d、孔距 p、p/d 相对关系和重影机制。
# - PDF 的任务③硬约束是：孔间非相干、重影角 p/f、太阳扩展源、
#   单孔 PSF 平移平铺、重影分离可量化、输出可验证。
# - PDF 的“六边形密排”在数学上是六近邻三角晶格；对话中要求的
#   “六边形图案”另行定义为蜂窝顶点型三近邻阵列，二者都保留，
#   但报告和配置中必须区分。
# - 随机阵列、绝对照度和镜片轮廓不属于本阶段主线，只作可选对照或后续阶段。
#
# 前两阶段结论的复用边界：
# - 第一阶段已回答 d 对单孔 PSF/D50 的影响；第二阶段已回答 d 对 E 字
#   视网膜像的影响。第三步不再从 0.3 到 3.0 mm 全范围重算孔径物理。
# - 第二步 README 的 550 nm 最优孔径作为主锚点：M = 0.5, 1, 1.5, 2,
#   2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6 D 分别约为 1.489, 1.101, 0.902,
#   0.778, 0.702, 0.635, 0.579, 0.550, 0.497, 0.495, 0.473, 0.449 mm。
# - 第一阶段 PSF/D50 的五个锚点用于交叉校验：M = 1, 2, 3, 4, 6 D
#   分别约为 1.1167, 0.7891, 0.6443, 0.5583, 0.4559 mm。
# - 第二步 M = 3 D 的波长最优孔径用于彩色敏感性：400 到 700 nm 的
#   7 个波长约为 0.542, 0.575, 0.605, 0.635, 0.664, 0.691, 0.717 mm。
# - 第三步的新计算重点是 p、阵列和重影；d 只在上述锚点及少量邻域点
#   上加密，不在本阶段重复证明“最优孔径随 M 变化”。
#
# 术语约定：
# - “孔的大小”和“孔径”在本阶段都指圆孔直径 d；
# - “孔距”指相邻孔心距离 p；
# - “二者的相对关系”主要记录为 p/d、d/p、p/f 和 d/f；
# - 如果有意研究椭圆孔、矩形孔或不同方向孔径，必须另设参数和新模型，
#   不能把它们悄悄混进圆孔直径 d。
#
# 本阶段的主线只有一个：重影机制。面积分数、透过率、孔边缘间隙和瞳孔
# 约束用于解释和筛除参数点，不用来取代重影位置、峰比和融合指标。
#
# 本阶段不重复承担的任务：
# - 不做整块 40 mm 镜片的相干全场角谱传播。太阳空间相干口径约 72 um，
#   远小于毫米级孔距，不同孔之间按强度叠加，不做 N 缝式复振幅干涉。
# - 不做完整 ISO 12312-1 绝对亮度判定，留到第四阶段透过率和照度标定。
# - 不做完整人眼高阶像差、色差和视网膜视锥采样；这些只能作为敏感性讨论。
# - 不把规则阵列的仿真写成激光相干演示；普通太阳照明与激光实验要明确分开。
# - 当前不把镜片轮廓作为设计变量。无论后续成品大致是圆形、方形还是其他
#   外形，本阶段只比较孔心排布；有限边界只用于把孔数截断到可计算规模。
#
# ============================================================================
# 0. 总体模型：先点源，再扩展源；先几何，再渐晕
# ============================================================================
#
# 0.1 点源阵列响应
# 设第 i 个孔心相对当前注视主孔的坐标为 r_i，针孔到视网膜等效距离为 f。
# 同一个远处点源经不同孔进入眼睛后，各像点的角位移满足：
#     Δθ_i = r_i / f
# 一阶相邻孔的间距大小为：
#     Δθ_neighbor = p / f
# 其中 p 是相邻孔距，f 使用第一步已有的 25 mm 等效距离口径。
#
# 点源下的阵列 PSF 写成：
#     PSF_array(θ) = Σ_i w_i PSF_single(θ - Δθ_i)
# 这里的加法必须是强度加法，不是复振幅加法。若写成复数相加，会错误地
# 产生孔间干涉条纹，并得到与太阳照明不符的结论。
#
# 必须先固定符号约定，再写任何平移代码：
# - 定义孔心 r_i 的图像位移为 +r_i / f，或统一取负号；
# - 在全流程、测试、图注和 CSV 中使用同一个约定；
# - 测试一个非中心单孔，断言峰值落在 p/f 对应的位置；
# - 不要一会儿平移 PSF、一会儿平移坐标轴，避免双重反向。
#
# 0.2 太阳扩展源
# 太阳角直径取 0.53°，对应约 31.8 角分。把太阳盘分解为角位置样本：
#     I_array(θ; λ) = Σ_s B_s PSF_array(θ - θ_s; λ)
# 角样本按太阳圆盘面积加权，不能只沿直径取点。
# 推荐分层圆环采样：
# - 快速档：直径方向 5 到 7 个径向尺度；
# - 标准档：直径方向 9 到 15 个径向尺度；
# - 每个圆环再取足够多的方位角，总计约 60 到 150 个方向；
# - 对所有二维方向做面积权重归一化。
#
# 若模型中暂不含随入射角变化的渐晕，则“太阳盘与点源阵列 PSF 卷积”和
# “逐太阳角样本相加”在数学上等价。此时可用一次卷积降低成本，但必须用
# 少量点逐项求和复算，证明两种实现一致。
#
# 若加入随视角变化的瞳孔投影、孔壁遮挡或孔有效面积，系统不再平移不变，
# 不能只做一次简单卷积。此时对每个太阳角样本计算孔权重，再做强度求和。
#
# 0.3 波动还是几何
# 本阶段不重新选择传播算法，直接继承第一步：
# - 单孔 PSF 始终由第一步的波动模型给出；
# - 几何弥散圆只用于极限检验和结果解释；
# - 当 d 较大且 M 较大时，可验证波动 PSF 的 D50 是否趋近几何直径；
# - 若为了速度使用几何核，必须显式标记“近似模式”，不能混入正式主结果。
#
# 0.4 复用前两阶段的孔径结论
# d 的扫描策略先读取前两个阶段的结果，而不是重新从全孔径网格证明：
# - 以第二步 E 字卷积最优孔径为主锚点，因为它更接近“看起来是否清晰”；
# - 以第一阶段 PSF/D50 最优孔径为交叉校验，检查纯光学判据和视标判据
#   是否给出相容的 d；
# - 仅在缺失的 (d, M, λ) 组合上调用第一步 PSF；
# - 对每个 M 只保留 d_opt 及其小邻域，例如乘以 0.8, 0.9, 1.0, 1.1, 1.2；
# - 保留一个低分辨率全范围对照组，只用于确认没有错过新最优区。
#
# 0.5 重影机制如何由 d、p 和 p/d 控制
# 一阶重影的绝对角间隔只由孔距和等效焦距决定：
#     Δθ_first = p / f
# 单孔像本身的角宽度 W 由第一步 PSF 和太阳盘共同决定：
#     W = W(d, M, λ, solar_diameter)
# 因此重影是否看起来分离，不是只看 p，也不是只看 d，而是看：
#     R = Δθ_first / W
# 其中：
# - d 主要控制单孔像的宽度。衍射区 W 随 d 增大而减小，
#   离焦区 W 随 d 增大而增大，不能只写成一个固定比例。
# - p 主要控制重影位置。p 越大，相邻重影角间隔 p/f 越大。
# - p/d 是重要的无量纲尺度，但不足以单独决定结果。即使 p/d 不变，
#   同时放大 d 和 p 也会改变绝对重影角、绝对 PSF 宽度、太阳盘融合程度
#   和衍射/离焦所处区域。
#
# 因此正式扫描必须同时保留独立的 d、独立的 p 和比值 p/d 三个数据列。
# 这里“完整”指 d 和 p 不被压成单一比值，各自作为独立维度保留；
# d 的取值仍按 0.4 节只取 d_opt 锚点及其邻域，不回到 0.3 到 3.0 mm
# 全范围重扫：
# 1. 固定 d，只扫 p：观察重影位置如何拉开；
# 2. 固定 p，只扫 d：观察每个重影像本身如何变宽或变窄；
# 3. 扫 p/d：观察几何相似缩放下的趋势；
# 4. 在 refined 网格内扫 d-p 联合平面，coarse 网格只作对照检查：
#    检查是否存在只靠 p/d 无法解释的交叉现象。
#
# 重影机制的最小指标集合：
# - ghost_count：在视场和亮度门槛内的可识别峰数量；
# - ghost_positions：每个峰相对主像的 (Δθ_x, Δθ_y)；
# - first_order_angle：最邻近一阶峰的 p/f 角度；
# - ghost_peak_ratio：每个重影峰相对主峰的强度；
# - ghost_integrated_fraction：扣除主像窗口后的总能量占比；
# - valley_visibility：相邻重影之间的谷值可见度；
# - merge_ratio：峰间距除以太阳盘卷积后的像宽；
# - pattern_signature：正方、三角、六边和环阵列各自的角向峰分布。
#
# 面积分数 T 和 edge_clearance 是约束与解释变量：
# - T 说明孔大一点或小一点的减光差异；
# - edge_clearance 说明制造上是否能做出该 d、p；
# - 它们不能替代重影机制指标作为主结论。
#
# 0.6 工具链与库选型：沿用“自写为主、库为对照”
# - 主计算复用第一、二阶段的 NumPy/SciPy 数据结构与 CuPy GPU 路径；
# - 目标硬件沿用第一阶段的 RTX 5070 Ti（16 GB 显存）：FFT、平移累加、
#   太阳盘积分和成批扫描等重计算必须走 CuPy GPU，慢速 CPU 路径只用于
#   掩膜 QC、小尺寸单元测试和无 GPU 环境下的调试档；无 GPU 不得出
#   正式结论图；
# - 单孔 PSF 不重写，直接从 single_hole_PSF.py 导入；
# - 卷积中心规则复用 single_hole_retinal_image.py；
# - scipy 用于掩膜质检、最近邻距离和必要的小型对照；
# - matplotlib 使用 Agg 输出，不引入交互窗口；
# - diffractsim 只可作为展示层可选工具；
# - diffractio 只用于少量代表点交叉验证，不能成为主流程依赖；
# - POPPY 和 LightPipes 只保留为文献级交叉验证备选，不进入本阶段主扫描；
# - BL-ASM 只在 coherent control 或对照实验中启用，不用于太阳照明的
#   多孔主结果。
#
# ============================================================================
# 1. 配置设计：先定义数据契约，再写算法
# ============================================================================
#
# 新建 ArrayGhostSimulationConfig，并让它嵌套复用 PinholePSFSimulationConfig。
# 沿用第一阶段规范的绝对口径：任何量都不准硬编码。物理量、扫描量、
# 排布量、光源量、网格大小、指标阈值、绘图尺寸、字号、颜色和路径都
# 必须集中在这个配置中，一律写成有名字的变量。禁止在函数体内写 550、
# 25、0.53、2048、(28, 20)、300、9 号字等散落字面量。第一阶段遗留的
# 个别函数内 fontsize 字面量不得在第三阶段复制，统一按第二阶段的分
# 角色字号配置字段执行。
#
# 1.1 物理输入
# - gpu_device_id、real_dtype、complex_dtype、accumulator_dtype；
# - focal_length_mm：沿用第一步，默认 25 mm；
# - primary_wavelength_nm：默认 550 nm；
# - myopia_values_d：本阶段至少覆盖 0.5、1、2、3、4、6 D；
# - diameter_reference_kind：retinal_optotype_primary 为主，psf_d50_crosscheck 为辅；
# - diameter_anchor_factors：围绕每个 M 的 d_opt 取少量倍率点；
# - coarse_diameter_check_values：只用于低分辨率全范围对照，数量要少；
# - pitch_reference_values_mm：建议覆盖 2.0、2.5、4.0、8.0 mm；
# - pitch_to_diameter_ratio_values：覆盖 p/d 的相对关系；
# - row_count、column_count：方格、三角和蜂窝周期阵列的有限截取规模；
# - ring_count：同心环阵列从中心孔向外包含的环数；
# - holes_per_ring：各圈孔数；既支持固定每圈孔数，也支持逐圈递增列表；
# - ring_radial_pitch_mm：相邻环半径差，通常等于或大于孔距 p；
# - array_extent_radius_mm：仅用于截断可计算孔数的数值边界，不代表镜片轮廓；
# - pupil_diameter_mm：至少设 2、4、8 mm 三档，检查 p 与瞳孔的相对关系；
# - pupil_plane_distance_mm：镜片到瞳孔的近似距离；没有这个量就不要假装
#   已经算出了真实的离轴渐晕；
# - film_thickness_mm：孔壁厚度相关的可选输入；只有一个厚度值时只能做
#   参数敏感性，不能宣称是实际成品结论。
#
# 1.2 排布参数
# - array_pattern_kind 只取以下主模式：
#   square_packing、triangular_packing、hexagonal_packing、
#   concentric_hexagonal_rings、concentric_circular_rings；
# - minimum_pitch_mm 与 maximum_pitch_mm；
# - diameter_to_pitch_ratio_values：d / p 的扫描点；
# - reuse_previous_metrics 必须为 true，缺失组合才允许重新计算；
# - minimum_edge_clearance_mm：孔边缘之间要求保留的制造余量；
# - target_open_area_fractions：希望保持相近透过率时的面积分数；
# - jittered_square 和 poisson_disk 只作为可选对照，不是本阶段主阵列；
# - 若启用随机对照，jitter_fraction 不能大到破坏最小孔距；
# - random_seed：随机对照必须固定种子；
# - 近邻统计的 bins、百分位数和最少样本数。
#
# 1.3 光源参数
# - source_kind：point 或 solar_disk；
# - solar_angular_diameter_deg：默认 0.53°；
# - radial_ring_count、azimuth_sample_count、source_weight_normalization；
# - spectral_sample_nm：正式彩色结果使用 400 到 700 nm 间隔约 50 nm 的 7 个点；
# - solar_spectrum_weight 与 photopic_weight 的来源必须在配置中说明；
# - 快速模式允许等权单色，但输出名、图注和 CSV 必须标明为单色代理。
#
# 1.4 数值与输出参数
# - overview_pixel_arcmin：宽视场总览网格；
# - local_pixel_arcmin：主像和每个重影附近的高分辨率局部网格；
# - local_half_width_arcmin：局部窗口必须至少覆盖单孔 PSF 的若干倍宽度；
# - field_margin_arcmin 或 field_margin_deg；
# - image_dynamic_range、colormaps、figure_dpi（沿用前两阶段的 300）；
# - 每类图独立的 figure_size_inches：沿用前两阶段“大图”惯例，主对比图
#   取 26 到 30 英寸宽、18 到 23 英寸高量级，验证表图不小于 22 x 12
#   英寸，保证多子图并排时单个子图和表格文字仍清晰可读；
# - 分角色字号字段：suptitle、axis_title、axis_label、legend、
#   panel_title、tick、conclusion、table，按第二阶段模式全部入配置；
# - output 根目录必须固定为仓库下的 output；
# - simulation_directory_name 必须是单个安全路径分量；
# - 所有输出文件名也必须通过路径安全检查。
#
# 1.5 配置校验
# - d > 0、p > d、数组网格为偶数；
# - 若定义 edge_clearance = p - d，则 edge_clearance 必须大于等于
#   minimum_edge_clearance_mm；报告时明确这是孔心距减孔径，不等同于
#   每个孔边界各自到另一孔边界的半间隙；
# - p >= pupil_diameter 是“避免经典 Scheiner 双像”的主约束；
# - p <= (pupil_diameter + d) / sqrt(2) 只作为文档给出的正方形覆盖参考；
#  三角形和六边形必须用各自的孔心覆盖条件，不能照套该式；
# - 若不满足上两条，不直接静默运行，应输出 fail 原因或放入单独的
#   counterexample 组，并明确标注这是有意展示的坏设计；
# - 太阳角直径、径向环数、方位采样数必须为正；
# - row_count、column_count、ring_count 和每圈孔数必须足够大，以体现排布规律；
# - 视场必须覆盖最大孔距造成的最大一阶重影，再加太阳半径和边距；
# - reuse_previous_metrics 必须为 true；若设为 false，必须给出明确的
#   debugging 标签，不能生成正式结论图；
# - 配置中不能调用 GPU 分配内存，也不要把 CuPy 数组作为配置字段。
#
# 重要教训：第二步用浮点数在扫描网格中做成员检查时依赖 tolerance。
# 第三步也不要使用裸 `in`、`==` 或字符串格式化后的数值做扫描键。
# 优先用整数索引或显式 isclose 建立稳定键，避免 1.0 与 1.0000000001
# 造成漏算、重复算和图中缺格。
#
# ============================================================================
# 2. 孔径排布生成：先输出孔心，再生成掩膜
# ============================================================================
#
# 2.1 统一数据契约
# 排布函数只返回：
# - centers_mm：形状为 (N, 2) 的孔心坐标；
# - layout_metadata：array_pattern_kind、row_count、column_count、ring_count、
#   holes_per_ring、p_nominal、p_min、p_max、diameter_mm、
#   edge_clearance_mm、N、area_fraction；
# - 不允许函数内部直接画图或写文件。
#
# 2.2 正方形密铺 square_packing
# - 用一个孔对应一个正方形单元；
# - x、y 方向相邻孔心距都为 p；
# - 生成坐标可写为 (i * p, j * p)，i、j 取整数；
# - 内部孔通常有 4 个最近邻，沿 x、y 的重影角间隔都是 p / f；
# - 面积分数解析值 T = (pi / 4) * (d / p)^2；
# - 重点检查对角线方向的第二个近邻，其距离为 p * sqrt(2)，
#   对应重影更强地落在斜方向上。
#
# 2.3 三角形密铺 triangular_packing
# - 孔心按等边三角晶格放置，研究对象为三角形顶点阵列；
# - 同一行孔心距为 p；相邻行距为 p * sqrt(3) / 2；
# - 奇数行相对偶数行平移 p / 2；
# - 内部孔通常有 6 个最近邻，近邻距离都为 p；
# - 六个一阶重影方向相隔 60°，不是正方形阵列的 90°；
# - 面积分数解析值 T = (pi / (2 * sqrt(3))) * (d / p)^2；
# - 同一 d / p 下，其解析面积分数约为正方形密铺的
#   2 / sqrt(3) = 1.1547 倍。
# - 该模式同时承担 PDF 中“六边形密排”的兼容性验证，因为 PDF 的
#   六边形密排就是把孔心放在六边形单元中心，其对偶关系等价于
#   这里的三角晶格。
#
# 2.4 六边形密铺 hexagonal_packing
# - 这里把六边形密铺定义为蜂窝孔心，而不是再次使用三角晶格的别名；
# - 一个蜂窝基元含两个孔心，相邻孔心距离为 p；
# - 内部孔通常有 3 个最近邻，三个一阶重影方向相隔约 120°；
# - 若研究目标其实是“孔心占满六边形单元中心”，那属于 triangular_packing，
#   必须在配置和报告中明确写成 triangular_packing，不能重复叫六边形；
# - 蜂窝面积分数不能用三角密铺的 pi / (2 * sqrt(3)) 公式；
# - 六边形密铺的孔密度和透过率必须从有限孔心与掩膜面积数值统计。
#
# 2.5 同心六边形环 concentric_hexagonal_rings
# - 始终保留中心主孔；
# - 第 r 圈半径优先取 r * ring_radial_pitch_mm，默认一圈一圈向外增加；
# - 第 r 圈放 6r 个孔，角度间隔为 360° / (6r)；
# - 圈间可以加入 30° / r 的角向旋转，使环与环不形成同一径向辐条；
# - 该模式重点回答“中心孔加一环、两环、三环时，重影如何增加”；
# - ring_count = 1、2、3 分别对应总共 1 + 6、1 + 6 + 12、
#   1 + 6 + 12 + 18 个孔；
# - 每一圈都记录 ring_index、ring_radius_mm、holes_in_ring、
#   angular_step_deg 和最近邻距离。
#
# 2.6 同心等孔数环 concentric_circular_rings
# - 始终保留中心主孔；
# - holes_per_ring 允许配置为固定值，例如每圈 8、12 或 16 个孔；
# - 也可配置为列表，例如 [6, 12, 18, 24]，用于逐圈加密测试；
# - 若 holes_per_ring 为固定值，需要增大半径才能保持最小孔距，
#   因此必须同时记录环周长、相邻孔弧距和真实孔心距；
# - 若每圈孔数随 r 线性增加，则与同心六边形环接近，但角度偏移不同；
# - 该模式用于区分“环数增加”和“每圈孔数增加”各自带来的重影变化。
#
# 2.7 有限截取和对照
# - 周期密铺用 row_count、column_count 截取有限孔心；
# - 环状排布用 ring_count 和 holes_per_ring 截取；
# - array_extent_radius_mm 只用于防止生成过多孔，不作为镜片轮廓变量；
# - 每个布局先做最小近邻距离和边缘间隙检查，再进入光学计算；
# - 随机抖动和泊松盘只作为可选对照，不参与正方形/三角形/六边形的主排名；
# - 任何有限截取造成的边界缺失都要记录，不能把 N 差异误写成排布差异。
#
# 2.8 推荐设计组
# 先固定 550 nm、M = 3 D，对以下文档设计做可比计算：
# - A：d = 1.2 mm，p = 2.0 mm；
# - B：d = 1.0 mm，p = 2.5 mm；
# - C：d = 0.7 mm，p = 2.0 mm。
# 再至少比较 square_packing、triangular_packing、hexagonal_packing、
# concentric_hexagonal_rings 和 concentric_circular_rings。
# 比较时保持参考主孔位置一致；孔数一致和透过率一致不能同时强求，
# 因此要分三组：
# - 第一组固定孔数，比较排布拓扑；
# - 第二组固定实际面积分数 T，比较真实减光条件下的重影。
# - 第三组固定 p 和 d，只增加 ring_count，观察重影从几个点扩展到背景的路径。
# 若把孔数和面积分数同时改变，就无法判断差异来自排布还是来自透过率。
#
# 2.9 孔距 p 与孔径 d 的二维关系：本阶段的核心敏感性实验
# 先把孔径维度分成三层，避免重复前两阶段工作：
# - 精确层：使用第二步 README 的 d_opt(M) 和 d_opt(λ, M=3D)，
#   不加新全扫描；
# - 邻域层：对每个 M 在 d_opt 附近取少量倍率点，验证重影结论对
#   孔径误差和打孔公差是否稳健；
# - 对照层：只在低分辨率全范围网格上做一次解析或粗 PSF 对照，
#   用来确认前两阶段没有漏掉值得复查的孔径区。
#
# 第三步的 d-p 平面必须按上述分层输出：
# - refined_diameter_pitch_grid：只在 d_opt 邻域内精细扫描；
# - coarse_diameter_pitch_grid：覆盖较宽 d，但只用少量解析指标；
# - 两个网格分别写 CSV，不能把粗对照图和精修结论混在同一张主图中。
#
# 后续描述中的“d-p 关系”默认指 refined grid；若使用 coarse grid，
# 文件名、图注和 README 必须明确写 coarse check。
#
# 不要把 p 和 d 当成两个孤立的单变量。真正需要扫描的是耦合关系，并同时
# 跟踪至少五个量：
# - 几何余量：edge_clearance = p - d；
# - 面积分数：T(d, p, array_pattern_kind)；
# - 重影间隔：Δθ ≈ p / f；
# - 单孔模糊宽度：W(d, M, λ)，从第一步 D50 或第二步阈值读取；
# - 分离充分度：R = Δθ / W，必要时再除以太阳角直径的影响。
#
# 固定一个变量时的预期趋势要在代码和图注中逐条验证：
# - 固定 d 增大 p：T 下降，孔数可能减少，重影间隔 p / f 增大；
#   若 p 超过视场覆盖约束，死区风险上升，因此不是越大越好。
# - 固定 p 增大 d：T 上升，几何余量下降；大 d 在近视下会增加单孔模糊，
#   使每个重影本身变大，分离充分度反而可能下降。
# - 固定 d / p 同时放大 d 和 p：解析面积分数通常不变，但重影角增大，
#   单孔模糊也随 d 变化；因此“固定填充率”并不等于“固定视觉结果”。
# - 固定 p / f：重影角不变，但 d 独立决定 PSF 宽度，必须单独报告。
#
# 至少输出三种切法的重影关系图，避免只画一个总热图：
# 1. d-p 平面：
#    x = d、y = p，色标分别为 R、first_order_angle、ghost_peak_ratio、
#    valley_visibility 和 GhostResolved 标签；
# 2. 固定 p 的 d 曲线：
#    观察 d 如何改变单孔模糊 W、重影峰宽、峰比和融合程度；
# 3. 固定 d 的 p 曲线：
#    观察 p 如何改变重影间隔、重影数量和视场覆盖。
#
# 推荐的 d-p 扫描范围：
# - refined d 从 d_opt(M) 的 0.8 到 1.2 倍加密，不重复全范围孔径研究；
# - coarse d 只取少量全范围检查点，并只计算解析几何或粗 PSF；
# - p 从 max(pupil_diameter, 1.5 * d) 到 8.0 mm，对数采样；
# - d / p 比至少在 0.15、0.20、0.25、0.30、0.40、0.50 附近取点；
# - 对每个合法点计算 R、T、重影峰比和风险标签；
# - 将不满足 p >= pupil_diameter、edge_clearance 或视场覆盖的点保留在
#   CSV 中并标记 fail，不要直接删掉。
#
# 条件筛选和 Pareto 前沿：
# - 硬约束：p > d + minimum_edge_clearance；
# - 重影约束：p / f >= separation_ratio_threshold * W；
# - 瞳孔约束：p >= pupil_diameter；
# - 覆盖约束：p <= geometric_dead_zone_limit，按具体晶格数值计算；
# - 光学约束：d 的 PSF 和第二步模型阈值放在可接受范围；
# - 主目标：最大化 R、最小化 ghost_peak_ratio、最大化 valley_visibility；
# - 次目标：在满足重影指标时不牺牲过多 T，并满足 edge_clearance；
# - T 不作为主目标与 R 平权，否则会把“减光更强”误写成“重影更小”；
# - 多目标没有唯一最优解，必须给 Pareto 前沿，并用实际佩戴条件选点。
#
# 解析极限只用于解释，不用于替代数值：
# - 几何离焦极限下 W ≈ M * d，因此 R ≈ p / (f * M * d)；
# - 当 M、f 固定时，R 主要受 p / d 比控制；
# - 衍射主导时 W 随 λ / d 增大，单独提高 p / d 比可能仍无法分离；
# - 太阳盘本身约 0.53°，因此即使 p / f 大于重影间隔，太阳像也可能融合；
# - 分离条件应使用太阳盘卷积后的谷值，而不是只看两个点峰位置。
#
# ============================================================================
# 3. 掩膜生成与 QC：第三步需要几何，不需要巨型相干传播
# ============================================================================
#
# 3.1 抗锯齿掩膜
# - 在目标像素的 4x4 子网格上画二值圆孔，再块平均回目标网格；
# - 得到 [0, 1] 灰度覆盖度，而不是只有 0 和 1 的锯齿圆；
# - 掩膜窗口要覆盖全部孔心并留边，不研究窗口外形对光学结果的贡献；
# - 以灰度的圆心向各方向检查覆盖度；
# - 这一步只为几何、透过率和图形质量服务，不作为整块镜片的波动传播输入。
#
# 3.2 必做 QC
# - 实际面积分数与解析填充率比较，要求相对误差 < 1%；
# - 连通域数量必须等于设计孔数，若减少说明孔重叠或掩膜边界截断；
# - regionprops 检查等效直径、Feret 直径和圆度；
# - cKDTree 最近邻距离的分布检查最小间距；
# - 统计孔数、p_min、p_max、T、array_extent 利用率；
# - QC fail 时不得继续出主结果图。
#
# 3.3 踩坑提示
# - 第二步曾出现矩形覆盖度内部不是 1、外部不是 0 的 bug。
#   第三步的圆孔覆盖度必须测试圆心为 1、远离圆心为 0、边缘在 0 到 1 间。
# - 不要把“几何掩膜采样尺度”和“单孔 PSF 的瞳孔采样尺度”混为一个网格。
#   两者服务不同对象，第二步已经明确这种双尺度做法并不矛盾。
# - 不要为了画出所有孔而把单孔 PSF 数组一起扩到超大尺寸。
#   掩膜用一套几何网格，PSF 用第一步的傅里叶网格，显示用第三套视图。
#
# ============================================================================
# 4. 复用第一、二步的光学接口
# ============================================================================
#
# 4.1 单孔 PSF
# - 从 single_hole_PSF.py 导入：
#   PinholePSFSimulationConfig、build_gpu_grids、build_soft_aperture、
#   compute_psf、build_sampling、enclosing_diameter_arcmin、
#   build_diameter_grid、build_myopia_values、configure_bilingual_plot_font。
# - 对每个 (d, M, λ) 只算一次单孔 PSF；
# - 同一 PSF 可被多个孔平移复用，不要每孔重算 FFT；
# - PSF 在平移前已经归一化到 ΣPSF = 1，平移不应改变峰值和总能量。
#
# 4.2 视网膜像与卷积
# - 从 single_hole_retinal_image.py 复用或抽取中心对齐规则；
# - 若继续使用 fftconvolve，必须沿用第二步的“不预先 ifftshift，
#   full 卷积后从 kernel.shape // 2 开始裁切”的中心约定；
# - 对中心脉冲、非中心脉冲、总能量和质心分别测试；
# - 不要直接使用 mode='same' 并假设 kernel 原点就在数组中心。
#
# 4.3 可能的导入副作用
# - single_hole_PSF.py 当前在导入时设置 CuPy 缓存和临时目录；
#   第三步导入它时会触发该行为。不要把这个副作用误认为第三步自己写的；
# - 若后续要重构缓存设置，先单独提交并回归前两步，不能在本阶段顺手大改；
# - 第三步不要重复设置另一套互相冲突的缓存目录。
#
# 4.4 浮点精度
# - 继续使用 float32 / complex64 做 GPU 主计算，float64 做能量累计；
# - 固定扫描键时保留 tolerance，避免浮点相等判断；
# - 累计多个孔的能量时不要反复用 float32 累加后再与 1 比较；
# - 所有 pass/fail 判定都写成“误差百分比 <= 容差”，不要写裸相等。
#
# ============================================================================
# 5. 平移叠加：宽视场总览与局部高分辨双网格
# ============================================================================
#
# 5.1 为什么不能只建一张超大图
# p = 8 mm、f = 25 mm 时，一阶重影约为 18.3°。若用 0.05 角分像素覆盖
# 正负 20°，单张二维图会超过 2 万像素见方，和第二步的 2048 网格不在同
# 一个量级。不要靠一张巨图解决所有问题。
#
# 5.2 推荐的三种 grid
# - geometry_grid：
#   宽视场、较低角分辨率，用来画所有孔的一阶/二阶重影位置和太阳盘重叠；
#   像素可选约 0.25 到 0.50 角分，但必须由配置给出并做收敛检查。
# - local_psf_grid：
#   在每个主像和一阶重影中心附近建立小窗口，角分辨率按单孔 PSF 采样；
#   用来测峰值、D50、重影间隔和分离度；
# - source_integration_grid：
#   只覆盖一个太阳盘叠加重影的局部区域，用足够细的太阳样本做积分验证。
#
# 5.3 平移放置规则
# - 平移量在连续角坐标中计算：shift = r_i / f；
# - 通过目标角坐标到源 PSF 像素坐标的显式映射采样；
# - 源中心必须严格映射到目标平移后的中心，不能在重采样中再产生半像素偏移；
# - 非整数像素位置要用线性或三次插值，并把插值阶数写入配置；
# - 平移后再次归一化前，先记录 sum 和 peak，判断是否丢失能量；
# - 若孔的有效权重 w_i 不同，先乘权重再累加；
# - 累加完成后只做一次总归一化，避免每孔各自归一化后错误放大边缘孔。
#
# 5.4 物理权重
# 至少区分以下权重来源：
# - 孔面积相同：w_i 相同；
# - 孔径变化设计：w_i 正比于开孔面积或实际掩膜积分；
# - 离轴瞳孔投影：按孔相对瞳孔中心的位置计算有效重叠面积；
# - 孔壁渐晕：在 film_thickness > 0 时才启用，并做厚度敏感性；
# - 镜片边界：被裁掉的孔不应进入重影求和。
#
# 如果当前版本还没有 pupil_plane_distance 和 thickness，就把 w_i = 1，
# 并在输出中明确写“无离轴渐晕的几何上限”。不要悄悄把缺失因素当作已建模。
#
# ============================================================================
# 6. 光源采样：点源负责几何，太阳盘负责真实融合
# ============================================================================
#
# 6.1 点源模式
# - 只用 θ = 0 的单点，用于验证 Δθ = r_i / f；
# - 输出每个孔对应的峰值位置，而不是只看总图；
# - 一阶相邻孔的重影角与 p / f 做解析对照；
# - 这个模式应得到清晰、可数、位置准确的离散峰。
#
# 6.2 太阳盘模式
# - 太阳不是点源。每个太阳角样本都要先形成一组阵列 PSF，再按亮度加权；
# - 权重来自圆盘面积，归一化后总和为 1；
# - 单色主实验用 550 nm；彩色实验再对 7 个波长按太阳光谱乘 V(λ) 加权；
# - 不同波长分别做平移和叠加，最后加的是强度，不是复振幅；
# - 波长循环完成后统一归一化，防止某些波长因采样点少而占额外权重。
#
# 6.3 收敛性
# - 对至少三个代表点做“标准档”和“加密档”比较；
# - 加密档的径向环数和方位数各约翻倍；
# - 太阳盘积分总能量、主峰高度和重影谷值的变化都应 < 2%；
# - 如果未通过，优先增加角采样，不要用图像模糊来掩盖欠采样。
#
# 6.4 不同入射角下的非相干性
# - 普通太阳照明下，孔与孔之间按强度叠加；
# - 同一太阳样本经过多个孔时仍按各孔独立成像处理；
# - 不要从某个孔和另一个孔之间建立复数互相关项；
# - 如果要做激光相干对照，只能另设 laser_coherent 模式，并使用完整
#   角谱传播和小孔阵列网格；其结果不得与太阳结果画在同一物理结论中。
#
# ============================================================================
# 7. 重影指标：把“能看到几个像”变成可复算数字
# ============================================================================
#
# 7.1 几何指标
# - hole_diameter_mm：圆孔直径 d；
# - hole_pitch_mm：相邻孔心距 p；
# - pitch_to_diameter_ratio = p / d；
# - diameter_to_pitch_ratio = d / p；
# - ghost_vector_arcmin：每个孔相对参考主孔的 (Δθ_x, Δθ_y)；
# - neighbor_ghost_angle_arcmin：相邻孔的最大或平均角间隔；
# - neighbor_ghost_angle_deg；
# - theory_ghost_angle = p / f，使用与仿真完全相同的 f；
# - ghost_angle_error_percent；
# - first_order_ghost_count 和 unique_overlap_count。
#
# 7.2 亮度与分离指标
# - main_peak_normalized：参考主像峰值归一为 1；
# - ghost_peak_normalized：各局部的最大重影峰值；
# - ghost_integrated_fraction：扣除主像窗口后，重影窗口内的能量占比；
# - sun_image_width_arcmin：单孔或参考主像经太阳盘卷积后的等效宽度；
# - separation_to_width_ratio = ghost_angle / sun_image_width；
# - valley_visibility = (I_low - I_valley) / (I_low + I_valley)，
#   其中 I_low 是邻近两个峰中的较低峰，I_valley 是连线中点；
# - 当 separation_to_width_ratio 小于配置阈值时，不报“重影可分辨”，
#   只报“融合成增亮区域”。阈值必须在报告中写清，不可事后调参。
#
# 7.3 规则性指标
# - 规则网格的阵列自相关应出现明显旁瓣栅格，随机排布应更平滑；
# - 用给定半径内的自相关峰旁瓣比、频率域最大峰与中位背景之比量化；
# - 增加“方向性”指标，检查正方形、三角晶格和蜂窝晶格的方向各向异性；
# - 检查三角晶格的 6 个最近邻与蜂窝晶格的 3 个最近邻是否被正确计数；
# - 所有指标都基于同一视场、同一归一化和同一孔数；
# - 不要只凭肉眼判断“随机看起来更自然”。
#
# 7.4 风险标签
# 每个参数点至少输出一个离散标签：
# - GhostResolved：重影分离超过阈值且相对亮度高于可见性门槛；
# - GhostMerged：重影位置可分但太阳盘卷积后谷值不足；
# - GhostWeak：重影存在但峰值低于模型可见性门槛；
# - DeadZoneRisk：存在视线落入孔间而无主孔的几何风险；
# - PupilVignetted：孔超出瞳孔有效投影或长宽比明显下降；
# - InvalidGeometry：孔重叠、越界或不满足最小间距。
#
# ============================================================================
# 8. 扫描策略：先小样本筛掉坏组合，再放大主实验
# ============================================================================
#
# 8.1 第一阶段：解析和脉冲测试
# - 用两个孔和一个非中心脉冲验证平移符号；
# - 用 2x2 与 3x3 方格验证一阶和二阶重影位置；
# - 用最小三角密铺和最小六边蜂窝验证最近邻方向数；
# - 用中心孔加一圈验证 6 个一阶重影的角向分布；
# - 用解析 p / f 对照全部孔心的重影位置；
# - 不运行 GPU 大扫描。
#
# 8.2 第二阶段：单色 550 nm、M = 3 D
# - 先跑设计 A、B、C；
# - 对每个设计跑 point source 和 solar disk 两种光源；
# - 对每个设计跑 2、4、8 mm 三档瞳孔直径；
# - 对每种排布跑 square_packing、triangular_packing、hexagonal_packing、
#   concentric_hexagonal_rings、concentric_circular_rings；
# - 环状模式至少跑 ring_count = 1、2、3、4；
# - 每圈孔数模式至少跑固定 6、12、18，以及随圈线性增加；
# - 把 pattern、ring_count 和 holes_per_ring 分列写入 CSV，
#   不能合并成一个含义模糊的 layout 字符串。
#
# 8.3 第三阶段：近视与孔径敏感性
# - M 取 0.5、1、2、3、4、6 D；
# - d 只取第二步 d_opt(M) 及少量邻域倍率点；
# - 在 M = 3 D 和 550 nm 下做 refined (d, p) 热图；
# - 热图用离散风险标签或连续重影指标，不要只画二维 PSF 缩略图。
#
# 8.3.1 d-p 关系扫描
# - 先对每种排布分别做 d-p 合法性矩阵；
# - 对每个合法点计算 edge_clearance、T、p / f、W 和 R；
# - 输出固定 p、固定 d、固定 d / p 三组切片；
# - 对每个 (layout, pupil_diameter, M) 生成一张 d-p 风险图；
# - 再选出 pareto_front_diameter_pitch.csv，而不是人工挑一组“最好”。
#
# 8.4 第四阶段：波长
# - 只在代表性 (d, p, M, layout) 上跑 7 个波长；
# - d 先取第二步 M = 3 D 的 7 个波长最优孔径，只在缺失时重算 PSF；
# - 检查红端较大 PSF 是否降低谷值、增加重影融合；
# - 彩色 RGB 只用于展示，定量结论仍以各波长 CSV 为准。
#
# 8.5 第五阶段：随机排布稳定性
# - 随机排布只是可选对照；若启用，jittered 和 Poisson 至少各跑 10 个 seed；
# - 报告指标的中位数、四分位区间和最坏 seed；
# - 若不同 seed 的结论方向改变，不能只挑选最好看的那张图；
# - 固定随机种子并写入配置哈希。
#
# ============================================================================
# 9. 输出目录与文件
# ============================================================================
#
# 所有输出放在：
#     output/pinhole_array_ghost/
#
# 推荐文件：
# - array_ghost_config.json：完整配置、依赖脚本哈希、测试文件哈希、GPU 名；
# - previous_diameter_anchors.csv：前两阶段 d_opt、来源、误差和复用状态；
# - refined_diameter_pitch_grid.csv：d_opt 邻域内的 d-p 精细网格；
# - coarse_diameter_pitch_grid.csv：低分辨率全范围对照网格；
# - array_geometry.csv：布局、孔数、p 统计、T_num、T_analytic、QC 通过状态；
# - ghost_metrics.csv：每个 (layout, d, p, pupil, M, wavelength, source) 一行；
# - ring_geometry.csv：ring_index、ring_radius、holes_in_ring、角步进和孔距；
# - ghost_peak_rows.csv：每个主要峰的位置、峰值、面积和相对亮度；
# - solar_convergence.csv：太阳采样收敛结果；
# - validation_anchors.csv：每个理论锚点的仿真值、理论值、误差和 pass；
# - array_patterns.png：正方形、三角形、六边形和同心环孔心图；
# - array_masks.png：不同主模式的抗锯齿掩膜；
# - point_source_array_psf.png：点源阵列 PSF，标出一阶/二阶重影；
# - solar_ghost_comparison.png：正方形、三角形、六边形和同心环的结果；
# - ring_count_comparison.png：中心孔加 1、2、3、4 圈的重影演化；
# - ghost_angle_heatmap.png：重影间距和分离比热图；
# - ghost_diameter_pitch_relation.png：d-p 重影指标、R 和风险标签；
# - pitch_diameter_slices.png：固定 p、固定 d、固定 d / p 的曲线；
# - pareto_front_diameter_pitch.csv：分离、峰比和融合指标的 Pareto 点；
# - design_comparison.png：A、B、C 三设计的风险指标对比；
# - validation_table.png：V5 及扩展锚点汇总。
#
# 文件命名和目录名也必须集中配置。不要在多个函数中重复写字符串；
# 输出目录创建前先做安全路径校验；所有图关闭 figure，避免长时间扫描
# 造成内存持续增长。
#
# ============================================================================
# 10. 验证锚点：先验证账，再验证物理，最后验证外观
# ============================================================================
#
# V-A 掩膜面积
# - T_num 与解析 T_analytic 相对误差 < 1%；
# - 抗锯齿后应优于硬阈值掩膜；
# - 连通域数必须等于设计孔数。
#
# V-B 平移与重影角
# - 单个非中心孔的峰值角位置与 r_i / f 的误差 < 2%；
# - 一阶相邻孔重影角与 p / f 的误差 < 5%，加密后争取 < 2%；
# - x、y 两个方向分别检查，不能只看径向距离。
#
# V-C 非相干叠加
# - 两个相同孔的峰值位置各自保持，总图在峰区不产生周期干涉条纹；
# - 将孔间相位随机化后结果不变，证明没有读取复振幅互相关；
# - 强度叠加的总能量等于各孔贡献之和。
#
# V-D 单孔退化
# - 只保留中央一个孔时，第三步点源结果必须与第一步 PSF 一致；
# - 只保留中央一个孔并加入太阳盘时，结果必须与太阳盘卷积单孔 PSF 一致；
# - 与第二阶段的 E 字卷积在相同 PSF 下保持一致的中心和总能量。
#
# V-E 双孔解析
# - 对称双孔应得到关于主像中心对称的一对等亮重影；
# - 平移其中一个孔后，两个峰的位置应按 r / f 变化；
# - 其中一个孔权重减半后，对应峰值也应减半。
#
# V-F 能量
# - 每一组孔权重的归一化总贡献等于 1；
# - 平移、重采样和太阳盘卷积后总能量偏差 < 0.1%；
# - 不允许用图像裁剪悄悄丢掉超出视场的重影而不报告。
#
# V-G 太阳源收敛
# - 标准采样与加密采样的主峰、总能量和重影谷值差异 < 2%；
# - 太阳圆盘面积权重总和等于 1，误差 < 1e-6；
# - 解析圆盘卷积与离散太阳样本卷积在轴对称测试中一致。
#
# V-H 网格无关性
# - 选三个代表设计，把 overview 和 local 网格分别加密到 2 倍；
# - 重影位置、峰比和谷值的差异 < 2%；
# - 若变化较大，先排查太阳盘积分和重采样，不要先改结论。
#
# V-I 排布 QC
# - 最近邻距离不小于配置 p，容差单独定义并报告；
# - square_packing 的内部最近邻数约为 4；
# - triangular_packing 的内部最近邻数约为 6，行距为 p * sqrt(3) / 2；
# - hexagonal_packing 的内部最近邻数约为 3，三个方向约为 120°；
# - concentric_hexagonal_rings 的第 r 圈必须恰有 6r 个孔；
# - ring_count 增加时，最大环半径按配置的 ring_radial_pitch 增长；
# - concentric_circular_rings 每圈孔数与配置列表完全一致；
# - 随机布局的报告必须包含 seed 稳定性，而不是只给一个样本。
#
# V-J p-d 耦合
# - edge_clearance = p - d 必须不小于配置值；
# - 固定 d 时，p 增大应使解析重影角 p / f 单调增大；
# - 固定 p 时，d 增大应使解析面积分数单调增大，同时几何余量下降；
# - 固定 d / p 时，若 array_pattern_kind 不变，解析 T 应基本保持不变；
# - 所有趋势都要使用同一个 f、同一个 p、同一个 d 和同一个 array_pattern_kind；
# - 对至少三个点手算 T、p / f 和 edge_clearance 与 CSV 对照。
#
# V-K 前两阶段孔径结论复用
# - 每个 M 的主锚点必须来自第二步 retinal_optotype 结果；
# - 第一阶段 D50 锚点与第二步锚点的差异必须写入 CSV，不能静默选择其一；
# - 对至少三个代表 (M, λ) 组合重算 PSF，确认复用值和现算值一致；
# - 缓存命中的组合不得重复调用 compute_psf；
# - 如果 coarse check 在远离 d_opt 的区域提示更强重影或更小模糊，
#   必须把该点提升为 refined 候选，而不是忽略。
#
# ============================================================================
# 11. 测试计划：先写小尺寸测试，再跑正式仿真
# ============================================================================
#
# 建议新增 tests/test_pinhole_array_ghost.py，至少包含：
# 1. square_packing 的行距、列距和 4 个最近邻测试；
# 2. triangular_packing 的行距、行偏移和 6 个最近邻测试；
# 3. hexagonal_packing 的 3 个最近邻和约 120° 角分布测试；
# 4. concentric_hexagonal_rings 每圈 6r 个孔测试；
# 5. concentric_circular_rings 的孔数和角步进测试；
# 6. 所有排布的最小间距和 edge_clearance 测试；
# 7. 圆孔覆盖度圆心为 1、外部为 0、边缘连续测试；
# 8. 两个孔的合成峰位置和 p / f 理论一致；
# 9. 非中心单孔平移后峰值位置正确；
# 10. 平移前后总能量守恒；
# 11. 非相干累加不产生复振幅干涉；
# 12. 太阳盘面积权重归一化；
# 13. 单孔退化到第一步 PSF；
# 14. 太阳盘卷积与逐角样本求和一致；
# 15. 固定 d / p 时解析面积分数守恒；
# 16. 固定 d 时 p 增大的重影角单调性；
# 17. ring_count 增加时重影数量按几何增长；
# 18. previous_diameter_anchors 能按 M 和 λ 映射到正确 d_opt；
# 19. 缓存命中时不重复计算 PSF，缺失组合才进入重算分支；
# 20. 配置非法分支全部抛出明确异常。
#
# 小尺寸测试使用 N = 64 或 128、少量孔和短向量，避免每次测试加载大 GPU；
# 正式大图测试放在 manifest 中，并明确标注为集成测试。不要用“图看起来对”
# 替代数值断言。
#
# ============================================================================
# 12. 性能与资源纪律
# ============================================================================
#
# - 单孔 PSF 只算一次，按 (d, M, wavelength) 缓存；不要在孔循环内重算；
# - 启动时先读取 output/single_hole_PSF 和 output/single_hole_retinal_image
#   的 CSV 与配置，把已有最优孔径和指标映射成缓存；
# - 只有缓存中缺失的 (d, M, λ) 才调用 compute_psf；
# - 孔径排布和抗锯齿掩膜在 CPU 上生成，几何中心再传到 GPU；
# - 平移叠加使用 CuPy；多个 seed 或多个波长可批处理，但先测单批显存；
# - 不要同时保留所有 PSF、所有太阳样本、所有大图和所有 seed 的结果；
# - 每个大数组用完立即删除并同步检查显存；
# - 需要计时时先调用 CUDA Stream synchronize，否则测到的是提交时间；
# - 周期密铺和有界环阵列的孔数差异必须显式报告；
# - 比较不同 pattern 时优先固定 N 或固定 T，不能同时乘以不同孔面积；
# - 宽视场总览和局部高分辨分开计算，不把局部细节强行塞进 2048 网格；
# - 若单次太阳盘积分太慢，先降低太阳方向数做粗筛，再用标准档复核；
# - 任何性能优化后都要重跑 V-A 到 V-K，而不是只重跑一张展示图。
#
# ============================================================================
# 13. 可复现与报告纪律
# ============================================================================
#
# - 使用 Path(__file__).resolve().parent 定位脚本目录；
# - 输出根目录固定为 output，不写绝对盘符；
# - 配置 JSON 记录所有参数，不能只保存“当前默认值”；
# - 记录 single_hole_PSF.py、single_hole_retinal_image.py、本脚本、
#   test 文件的 SHA-256；
# - 记录 GPU 名称、CuPy 版本、Python 版本和随机 seed；
# - 图像标题和图注必须写清波长、M、d、p、瞳孔直径、布局和光源类型；
# - 单色代理、几何近似、无渐晕、无绝对照度等限制必须写在图内；
# - 不要在结论图中只写“效果好/差”，要给出重影角、峰比、谷值或风险标签；
# - 对失败点保留 CSV 和 fail 标签，不要为了图美观删掉缺失值；
# - 对规则布局中的栅格峰和随机布局的蓝噪声进行单独说明；
# - 报告中区分“太阳照明结果”“激光相干对照”和“纯几何解析结果”。
#
# ============================================================================
# 14. 建议的函数拆分顺序
# ============================================================================
#
# 1. validate_config(config)：只做配置和路径校验；
# 2. build_myopia_values 与已有网格适配：复用第一步；
# 3. load_previous_diameter_anchors(config)：读取前两阶段最优孔径；
# 4. build_diameter_anchor_grid(config)：生成 d_opt 邻域和 coarse check；
# 5. generate_square_packing(config)：生成正方形密铺孔心；
# 6. generate_triangular_packing(config)：生成三角形密铺孔心；
# 7. generate_hexagonal_packing(config)：生成蜂窝六边形孔心；
# 8. generate_hexagonal_rings(config)：生成中心孔加同心环；
# 9. generate_circular_rings(config)：生成固定或可变每圈孔数的环；
# 10. validate_layout(centers, config)：间距、edge_clearance、孔数和近邻；
# 11. build_diameter_pitch_grid(config)：只生成合法 refined 和 coarse 组合；
# 12. rasterize_mask(centers, config)：抗锯齿掩膜；
# 13. validate_mask(mask, centers, config)：面积、连通域、regionprops；
# 14. build_solar_directions(config)：太阳盘的二维角样本与权重；
# 15. build_solar_spectrum_weights(config)：波长权重；
# 16. compute_single_hole_psf_cached(config, ...)：包装第一步函数；
# 17. place_shifted_psf(local_grid, psf, shift, local_weight)：中心对齐平移；
# 18. build_array_psf(local_grid, all_shifts, all_weights)：非相干累加；
# 19. integrate_solar_source(point_source_psf, solar_directions, weights)：扩展源；
# 20. extract_ghost_peaks(image, geometry, config)：峰值检测和分组；
# 21. compute_ghost_metrics(...)：分离、亮度、融合和风险标签；
# 22. compute_layout_regularity(...)：自相关和频谱特征；
# 23. compute_pitch_diameter_relations(...)：T、edge_clearance、R 和 Pareto；
# 24. run_scan(...)：分层扫描，不把全组合一次性展开；
# 25. build_validation_rows(...)：V-A 到 V-K；
# 26. write_csv、write_config、save_*_figure：只在最后写产物；
# 27. main()：只负责编排和打印进度，不放物理公式。
#
# ============================================================================
# 15. 前两步已经踩过的坑，第三步必须逐条规避
# ============================================================================
#
# - 覆盖度定义错误：矩形/圆形内部必须是 1，外部是 0；
# - PSF 中心错误：FFT 和卷积前不要重复 ifftshift；
# - 卷积裁切错误：full 模式后裁切必须从 kernel 中心索引开始；
# - 重采样偏移错误：显式角坐标映射，保证中心像素严格对应；
# - 显示窗口错误：总览可以宽，局部定量图必须围绕主像和重影裁切；
# - 能量错误：平移和卷积后检查总能量，不要只看峰值；
# - 配置硬编码：物理量、采样量、网格大小、阈值、图尺寸、字号、颜色
#   和路径全部集中配置，函数体内不出现任何魔法数字；
# - 浮点相等：扫描键和容差判定不能依赖裸相等；
# - 字体问题：使用 Agg 后端并配置中文字体，负号显示要正常；
# - 路径问题：输出必须落在仓库 output 下，不使用外部绝对目录；
# - 导入副作用：第一步脚本导入时会设置 CuPy 环境，本阶段不要叠加冲突设置；
# - 缺失值处理：无法在扫描范围内达到阈值的点要保留并标 censored；
# - 无效配置分支：不要写永远不可达的选项，配置校验要覆盖实际分支；
# - 大图内存：不保存全部阵列 PSF，不在孔循环里重复创建 GPU 数组；
# - 单元测试：先在 64/128 小网格上验证，再进入 2048 级正式图。
#
# ============================================================================
# 16. 完成定义
# ============================================================================
#
# 本阶段只有在以下条件全部满足后才算完成：
# - 能定量说明每个孔的像移位置和相邻重影角；
# - 能区分点源离散重影和太阳扩展源融合结果；
# - 能给出主像、重影的峰值与能量占比；
# - 能比较正方形密铺、三角形密铺、六边形密铺三种周期孔心；
# - 能比较中心孔加一环、两环、三环、四环的重影演化；
# - 能区分“增加环数”和“增加每圈孔数”两个效应；
# - 能说明六边形密铺采用蜂窝孔心定义，不与三角密铺重复；
# - 能对不同 d、p、瞳孔直径和 M 给出风险标签；
# - 能分别解释 d、p 和 p/d 对重影数量、位置、峰比和融合的独立影响；
# - 能给出 d-p 二维合法区域、固定变量切片和 Pareto 前沿；
# - 能证明 d_opt 邻域直接继承前两阶段结论，只有缺失组合才重新计算；
# - V-A 到 V-K 有 CSV 结果，所有 fail 项保留；
# - 所有图、CSV、JSON 可由同一配置和固定 seed 复现；
# - README 明确列出第三阶段的假设、缺失物理和不能外推的结论。


# ============================================================================
# 实现部分 1/5：配置、校验与输出基础设施
# ============================================================================

import csv
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import cupy as cp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree
from cupyx.scipy.ndimage import map_coordinates
from cupyx.scipy.signal import fftconvolve

from single_hole_PSF import (
    ARCMINUTES_PER_DEGREE,
    ARCMINUTES_PER_RADIAN,
    METRES_PER_MILLIMETRE,
    METRES_PER_NANOMETRE,
    SimulationConfig as PinholePSFSimulationConfig,
    build_gpu_grids,
    build_sampling,
    build_soft_aperture,
    compute_psf,
    configure_bilingual_plot_font,
    enclosing_diameter_arcmin,
    radial_bin_sum,
    theoretical_optimal_diameter_mm,
)


# 五种主排布的名称契约。它们是字符串标识，不是可调物理量；
# 阈值、尺寸和采样全部放在 ArrayGhostSimulationConfig 中。
ARRAY_PATTERN_SQUARE_PACKING = "square_packing"
ARRAY_PATTERN_TRIANGULAR_PACKING = "triangular_packing"
ARRAY_PATTERN_HEXAGONAL_PACKING = "hexagonal_packing"
ARRAY_PATTERN_HEXAGONAL_RINGS = "concentric_hexagonal_rings"
ARRAY_PATTERN_CIRCULAR_RINGS = "concentric_circular_rings"
ARRAY_PATTERN_KINDS: tuple[str, ...] = (
    ARRAY_PATTERN_SQUARE_PACKING,
    ARRAY_PATTERN_TRIANGULAR_PACKING,
    ARRAY_PATTERN_HEXAGONAL_PACKING,
    ARRAY_PATTERN_HEXAGONAL_RINGS,
    ARRAY_PATTERN_CIRCULAR_RINGS,
)

# 随机对照的两种布局只作可选比较，不进入主排名。
ARRAY_PATTERN_JITTERED_SQUARE = "jittered_square"
ARRAY_PATTERN_POISSON_DISK = "poisson_disk"
ARRAY_PATTERN_OPTIONAL_KINDS: tuple[str, ...] = (
    ARRAY_PATTERN_JITTERED_SQUARE,
    ARRAY_PATTERN_POISSON_DISK,
)

# 光源与风险标签的字符串契约。
SOURCE_KIND_POINT = "point"
SOURCE_KIND_SOLAR_DISK = "solar_disk"
RISK_GHOST_RESOLVED = "GhostResolved"
RISK_GHOST_MERGED = "GhostMerged"
RISK_GHOST_WEAK = "GhostWeak"
RISK_DEAD_ZONE = "DeadZoneRisk"
RISK_PUPIL_VIGNETTED = "PupilVignetted"
RISK_INVALID_GEOMETRY = "InvalidGeometry"


@dataclass(frozen=True)
class ArrayGhostSimulationConfig:
    """第三阶段多针孔阵列重影仿真的全部输入、输出与绘图参数。

    方案第 1 节要求：物理量、扫描量、排布量、光源量、网格大小、指标阈值、
    绘图尺寸、字号、颜色和路径全部集中在这里，函数体内不得出现散落字面量。
    """

    # ---------------------------------------------------------------- 复用项
    # 单孔 PSF 的物理与数值口径直接复用第一阶段配置，
    # 不在这里重复定义波长采样、圆孔软边、离焦波前或 FFT 网格。
    psf_config: PinholePSFSimulationConfig = field(
        default_factory=PinholePSFSimulationConfig
    )

    # ------------------------------------------------------------ GPU 与精度
    gpu_device_id: int = 0
    real_dtype: Any = cp.float32
    accumulator_dtype: Any = cp.float64
    floating_comparison_tolerance: float = 1.0e-9

    # ------------------------------------------------------------ 物理输入
    focal_length_mm: float = 25.0
    primary_wavelength_nm: float = 550.0
    myopia_values_d: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0)
    representative_myopia_d: float = 3.0
    pupil_diameter_values_mm: tuple[float, ...] = (2.0, 4.0, 8.0)
    array_extent_radius_mm: float = 12.0
    minimum_edge_clearance_mm: float = 0.2

    # 瞳孔渐晕：没有镜片到瞳孔距离时，权重取 1 并标记为几何上限。
    enable_pupil_vignetting: bool = False
    pupil_plane_distance_mm: float | None = None

    # --------------------------------------------------- 前两阶段结论复用
    reuse_previous_metrics: bool = True
    reuse_debug_label: str | None = None
    retinal_metrics_relative_path: str = (
        "output/single_hole_retinal_image/optotype_metrics.csv"
    )
    psf_metrics_relative_path_template: str = (
        "output/single_hole_PSF/PSF_{wavelength_nm:g}nm/single_hole_metrics.csv"
    )
    retinal_threshold_column: str = "threshold_log_mar"
    retinal_diameter_column: str = "diameter_mm"
    retinal_myopia_column: str = "myopia_d"
    retinal_wavelength_column: str = "wavelength_nm"
    psf_metric_column: str = "log_mar"
    psf_diameter_column: str = "diameter_mm"
    psf_myopia_column: str = "myopia_d"
    psf_wavelength_column: str = "wavelength_nm"
    reference_anchor_wavelength_nm: float = 550.0
    exclude_zero_myopia_from_anchors: bool = True
    crosscheck_anchor_myopia_values_d: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 6.0)
    diameter_anchor_factors: tuple[float, ...] = (0.8, 0.9, 1.0, 1.1, 1.2)
    coarse_diameter_check_factors: tuple[float, ...] = (0.5, 0.7, 1.5, 2.0)
    diameter_deduplication_tolerance_mm: float = 1.0e-6
    coarse_check_pitch_values_mm: tuple[float, ...] = (2.0, 4.0, 8.0)

    # ------------------------------------------------------------ 排布参数
    square_row_count: int = 9
    square_column_count: int = 9
    triangular_row_count: int = 9
    triangular_column_count: int = 9
    honeycomb_row_count: int = 7
    honeycomb_column_count: int = 7
    ring_count_values: tuple[int, ...] = (1, 2, 3, 4)
    hexagonal_ring_radial_pitch_mm: float = 2.0
    hexagonal_ring_rotation_step_deg: float = 30.0
    circular_ring_holes_per_ring: tuple[int, ...] = (6, 12, 18, 24)
    circular_ring_radial_pitch_mm: float = 2.0
    circular_ring_rotation_step_deg: float = 15.0
    pitch_reference_values_mm: tuple[float, ...] = (2.0, 2.5, 4.0, 8.0)
    minimum_pitch_mm: float = 1.5
    maximum_pitch_mm: float = 8.0
    pitch_log_count: int = 9
    pitch_to_diameter_ratio_values: tuple[float, ...] = (
        2.0,
        2.5,
        3.0,
        4.0,
        5.0,
        8.0,
    )
    diameter_to_pitch_ratio_values: tuple[float, ...] = (
        0.15,
        0.20,
        0.25,
        0.30,
        0.40,
    )
    design_point_diameter_mm: tuple[float, ...] = (1.2, 1.0, 0.7)
    design_point_pitch_mm: tuple[float, ...] = (2.0, 2.5, 2.0)
    design_point_labels: tuple[str, ...] = ("A", "B", "C")
    jitter_fraction: float = 0.05
    poisson_disk_minimum_factor: float = 1.0
    random_seed: int = 20260921
    random_control_seed_count: int = 10

    # ------------------------------------------------------------ 光源参数
    source_kind: str = SOURCE_KIND_SOLAR_DISK
    solar_angular_diameter_deg: float = 0.53
    # 方向积分只用于 V-G 校验。实测峰值随采样数约按 1/n 收敛：
    # 8x16 时峰高偏离 35%，32x64 偏离 2.8%，96x192 偏离 0.26%。
    # 因此主路径固定为面积加权的圆盘卷积，方向积分用稠密档复核。
    solar_radial_ring_count: int = 32
    solar_azimuth_sample_count: int = 64
    solar_radial_ring_dense_count: int = 96
    solar_azimuth_dense_count: int = 192
    solar_weight_sum_tolerance: float = 1.0e-6
    solar_convergence_peak_tolerance_percent: float = 2.0
    solar_convergence_width_tolerance_percent: float = 2.0
    spectral_wavelengths_nm: tuple[float, ...] = (
        400.0,
        450.0,
        500.0,
        550.0,
        600.0,
        650.0,
        700.0,
    )

    # ------------------------------------------------------- 网格与重采样
    analysis_grid_size: int = 1024
    analysis_pixel_arcmin: float = 0.25
    solar_kernel_half_width_arcmin: float = 64.0
    field_pixel_arcmin: float = 0.25
    field_grid_max_size: int = 2048
    field_min_pixels_per_image_width: float = 6.0
    field_margin_factor: float = 1.05
    field_extra_margin_arcmin: float = 32.0
    psf_resampling_order: int = 1
    convolution_mode: str = "full"
    aperture_anti_alias_subsamples: int = 4
    mask_pixel_mm: float = 0.02
    mask_margin_mm: float = 0.5
    mask_binary_threshold: float = 0.5
    mask_minimum_pixels_per_diameter: float = 12.0
    mask_minimum_circularity: float = 0.85

    # ------------------------------------------------------------ 重影指标
    visibility_peak_threshold: float = 1.0e-3
    merge_radius_factor: float = 0.5
    separation_ratio_threshold: float = 2.0
    ghost_peak_ratio_threshold: float = 0.05
    valley_numeric_max_separation_arcmin: float = 240.0
    valley_profile_sample_count: int = 512
    sun_width_energy_fraction: float = 0.50
    direction_histogram_bin_count: int = 36
    autocorrelation_min_lag_arcmin: float = 1.0
    autocorrelation_max_lag_arcmin: float = 120.0
    main_window_radius_factor: float = 1.0
    core_neighbor_radius_factor: float = 1.05
    pupil_reference_weight_threshold: float = 0.5

    # -------------------------------------------------------- 验证判据
    mask_area_relative_tolerance: float = 0.01
    mask_centroid_tolerance_pixel: float = 0.75
    shift_angle_tolerance_percent: float = 2.0
    first_order_angle_tolerance_percent: float = 5.0
    energy_relative_tolerance_percent: float = 0.1
    solar_convergence_tolerance_percent: float = 2.0
    grid_convergence_tolerance_percent: float = 2.0
    grid_convergence_dense_factor: int = 2
    coarse_check_degradation_tolerance_percent: float = 5.0

    # ------------------------------------------------------------ 绘图参数
    figure_dpi: int = 300
    pattern_figure_size_inches: tuple[float, float] = (28.0, 20.0)
    mask_figure_size_inches: tuple[float, float] = (28.0, 20.0)
    point_source_figure_size_inches: tuple[float, float] = (30.0, 22.0)
    solar_comparison_figure_size_inches: tuple[float, float] = (30.0, 23.0)
    ring_comparison_figure_size_inches: tuple[float, float] = (30.0, 22.0)
    ghost_heatmap_figure_size_inches: tuple[float, float] = (30.0, 20.0)
    diameter_pitch_figure_size_inches: tuple[float, float] = (30.0, 22.0)
    slice_figure_size_inches: tuple[float, float] = (28.0, 18.0)
    design_figure_size_inches: tuple[float, float] = (28.0, 18.0)
    validation_figure_size_inches: tuple[float, float] = (24.0, 14.0)
    suptitle_font_size: float = 17.0
    axis_title_font_size: float = 12.0
    axis_label_font_size: float = 11.0
    legend_font_size: float = 8.0
    panel_title_font_size: float = 9.0
    tick_font_size: float = 8.0
    conclusion_font_size: float = 11.0
    table_font_size: float = 9.0
    image_log_dynamic_range: float = 3.0
    image_colormap_name: str = "inferno"
    pattern_colormap_name: str = "viridis"
    heatmap_colormap_name: str = "magma"
    ghost_heatmap_vmin: float = 0.0
    ghost_heatmap_vmax: float = 4.0
    pattern_marker_size: float = 12.0
    pattern_schematic_pitch_mm: float = 2.0
    pattern_schematic_diameter_mm: float = 0.7
    pattern_schematic_ring_count: int = 3
    pattern_schematic_half_extent_mm: float = 9.0

    # ------------------------------------------------------- 输出与命名
    output_root_directory_name: str = "output"
    simulation_directory_name: str = "pinhole_array_ghost"
    config_filename: str = "array_ghost_config.json"
    previous_anchor_filename: str = "previous_diameter_anchors.csv"
    refined_grid_filename: str = "refined_diameter_pitch_grid.csv"
    coarse_grid_filename: str = "coarse_diameter_pitch_grid.csv"
    geometry_filename: str = "array_geometry.csv"
    ghost_metrics_filename: str = "ghost_metrics.csv"
    ring_geometry_filename: str = "ring_geometry.csv"
    ghost_peak_filename: str = "ghost_peak_rows.csv"
    solar_convergence_filename: str = "solar_convergence.csv"
    validation_filename: str = "validation_anchors.csv"
    pareto_filename: str = "pareto_front_diameter_pitch.csv"
    pattern_figure_filename: str = "array_patterns.png"
    mask_figure_filename: str = "array_masks.png"
    point_source_figure_filename: str = "point_source_array_psf.png"
    solar_comparison_figure_filename: str = "solar_ghost_comparison.png"
    ring_comparison_figure_filename: str = "ring_count_comparison.png"
    ghost_heatmap_figure_filename: str = "ghost_angle_heatmap.png"
    diameter_pitch_figure_filename: str = "ghost_diameter_pitch_relation.png"
    slice_figure_filename: str = "pitch_diameter_slices.png"
    design_figure_filename: str = "design_comparison.png"
    validation_figure_filename: str = "validation_table.png"

    # ------------------------------------------------------- 运行模式
    run_mode: str = "full"
    quick_row_count: int = 5
    quick_column_count: int = 5
    quick_ring_count_values: tuple[int, ...] = (1, 2)
    quick_analysis_grid_size: int = 512
    quick_field_grid_max_size: int = 768
    quick_pitch_values_mm: tuple[float, ...] = (2.0, 4.0)
    quick_myopia_values_d: tuple[float, ...] = (3.0,)
    quick_diameter_factors: tuple[float, ...] = (1.0,)
    quick_solar_radial_ring_count: int = 4
    quick_solar_azimuth_sample_count: int = 8

    # ------------------------------------------------------------ 派生量
    @property
    def solar_angular_radius_deg(self) -> float:
        return 0.5 * self.solar_angular_diameter_deg

    @property
    def solar_angular_radius_arcmin(self) -> float:
        return self.solar_angular_radius_deg * ARCMINUTES_PER_DEGREE

    @property
    def valid_pattern_kinds(self) -> tuple[str, ...]:
        return ARRAY_PATTERN_KINDS + ARRAY_PATTERN_OPTIONAL_KINDS

    @property
    def quick_mode(self) -> bool:
        return self.run_mode == "quick"

    def effective_row_count(self) -> int:
        return self.quick_row_count if self.quick_mode else self.square_row_count

    def effective_column_count(self) -> int:
        return (
            self.quick_column_count if self.quick_mode else self.square_column_count
        )

    def effective_analysis_grid_size(self) -> int:
        return (
            self.quick_analysis_grid_size
            if self.quick_mode
            else self.analysis_grid_size
        )

    def effective_field_grid_max_size(self) -> int:
        return (
            self.quick_field_grid_max_size
            if self.quick_mode
            else self.field_grid_max_size
        )

    def effective_ring_count_values(self) -> tuple[int, ...]:
        return (
            self.quick_ring_count_values if self.quick_mode else self.ring_count_values
        )

    def effective_pitch_values_mm(self) -> tuple[float, ...]:
        return self.quick_pitch_values_mm if self.quick_mode else self.pitch_reference_values_mm

    def effective_myopia_values_d(self) -> tuple[float, ...]:
        return self.quick_myopia_values_d if self.quick_mode else self.myopia_values_d

    def effective_diameter_anchor_factors(self) -> tuple[float, ...]:
        return (
            self.quick_diameter_factors
            if self.quick_mode
            else self.diameter_anchor_factors
        )

    def effective_solar_radial_ring_count(self) -> int:
        return (
            self.quick_solar_radial_ring_count
            if self.quick_mode
            else self.solar_radial_ring_count
        )

    def effective_solar_azimuth_sample_count(self) -> int:
        return (
            self.quick_solar_azimuth_sample_count
            if self.quick_mode
            else self.solar_azimuth_sample_count
        )

    # ------------------------------------------------------------ 自校验
    def validate(self) -> None:
        validate_config(self)

    def quick_variant(self) -> "ArrayGhostSimulationConfig":
        return replace(self, run_mode="quick")


def validate_output_component(value: str, field_name: str) -> str:
    """输出目录名与文件名必须是单个安全路径分量。"""
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    if value != Path(value).name:
        raise ValueError(f"{field_name} must be a single path component: {value!r}")
    if value in {".", ".."}:
        raise ValueError(f"{field_name} must not be a relative path token: {value!r}")
    return value


def validate_config(config: ArrayGhostSimulationConfig) -> None:
    """方案 1.5 节：在任何 GPU 分配之前拒绝非法配置。"""
    if config.real_dtype not in (cp.float32, cp.float64):
        raise ValueError("real_dtype must be cp.float32 or cp.float64")
    if config.accumulator_dtype not in (cp.float32, cp.float64):
        raise ValueError("accumulator_dtype must be cp.float32 or cp.float64")
    if config.gpu_device_id < 0:
        raise ValueError("gpu_device_id must be non-negative")

    positive_scalars = {
        "focal_length_mm": config.focal_length_mm,
        "primary_wavelength_nm": config.primary_wavelength_nm,
        "representative_myopia_d": config.representative_myopia_d,
        "array_extent_radius_mm": config.array_extent_radius_mm,
        "analysis_pixel_arcmin": config.analysis_pixel_arcmin,
        "field_pixel_arcmin": config.field_pixel_arcmin,
        "solar_kernel_half_width_arcmin": config.solar_kernel_half_width_arcmin,
        "solar_angular_diameter_deg": config.solar_angular_diameter_deg,
        "hexagonal_ring_radial_pitch_mm": config.hexagonal_ring_radial_pitch_mm,
        "circular_ring_radial_pitch_mm": config.circular_ring_radial_pitch_mm,
        "minimum_pitch_mm": config.minimum_pitch_mm,
        "maximum_pitch_mm": config.maximum_pitch_mm,
        "merge_radius_factor": config.merge_radius_factor,
        "separation_ratio_threshold": config.separation_ratio_threshold,
        "sun_width_energy_fraction": config.sun_width_energy_fraction,
        "field_min_pixels_per_image_width": config.field_min_pixels_per_image_width,
        "figure_dpi": float(config.figure_dpi),
    }
    for name, value in positive_scalars.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive, got {value!r}")

    if config.minimum_edge_clearance_mm < 0.0:
        raise ValueError("minimum_edge_clearance_mm must be non-negative")
    if config.maximum_pitch_mm <= config.minimum_pitch_mm:
        raise ValueError("maximum_pitch_mm must exceed minimum_pitch_mm")
    if not 0.0 < config.sun_width_energy_fraction < 1.0:
        raise ValueError("sun_width_energy_fraction must lie in (0, 1)")
    if config.field_margin_factor < 1.0:
        raise ValueError("field_margin_factor must be at least 1")
    if config.field_extra_margin_arcmin < config.solar_angular_radius_arcmin:
        raise ValueError(
            "field_extra_margin_arcmin must cover at least the solar radius"
        )
    if config.visibility_peak_threshold <= 0.0:
        raise ValueError("visibility_peak_threshold must be positive")
    for name in (
        "solar_convergence_peak_tolerance_percent",
        "solar_convergence_width_tolerance_percent",
    ):
        if getattr(config, name) <= 0.0:
            raise ValueError(f"{name} must be positive")
    if config.field_min_pixels_per_image_width < 2.0:
        raise ValueError(
            "field_min_pixels_per_image_width must be at least 2 so that a "
            "single-hole image is not collapsed into one pixel"
        )
    if config.autocorrelation_max_lag_arcmin <= config.autocorrelation_min_lag_arcmin:
        raise ValueError(
            "autocorrelation_max_lag_arcmin must exceed "
            "autocorrelation_min_lag_arcmin"
        )
    if not 0.0 <= config.pupil_reference_weight_threshold <= 1.0:
        raise ValueError("pupil_reference_weight_threshold must lie in [0, 1]")
    if config.quick_mode and config.reuse_previous_metrics is False:
        # quick 模式也必须在正式结论中复用前两阶段锚点。
        raise ValueError("quick mode still requires reuse_previous_metrics")
    if not config.reuse_previous_metrics:
        if not config.reuse_debug_label:
            raise ValueError(
                "reuse_previous_metrics=False requires a debug label; "
                "formal conclusions may not be produced this way"
            )

    if config.source_kind not in (SOURCE_KIND_POINT, SOURCE_KIND_SOLAR_DISK):
        raise ValueError(f"unsupported source_kind: {config.source_kind!r}")

    integer_fields = {
        "analysis_grid_size": config.analysis_grid_size,
        "field_grid_max_size": config.field_grid_max_size,
        "square_row_count": config.square_row_count,
        "square_column_count": config.square_column_count,
        "triangular_row_count": config.triangular_row_count,
        "triangular_column_count": config.triangular_column_count,
        "honeycomb_row_count": config.honeycomb_row_count,
        "honeycomb_column_count": config.honeycomb_column_count,
        "solar_radial_ring_count": config.solar_radial_ring_count,
        "solar_azimuth_sample_count": config.solar_azimuth_sample_count,
        "solar_radial_ring_dense_count": config.solar_radial_ring_dense_count,
        "solar_azimuth_dense_count": config.solar_azimuth_dense_count,
        "direction_histogram_bin_count": config.direction_histogram_bin_count,
        "valley_profile_sample_count": config.valley_profile_sample_count,
        "aperture_anti_alias_subsamples": config.aperture_anti_alias_subsamples,
    }
    for name, value in integer_fields.items():
        if int(value) < 1:
            raise ValueError(f"{name} must be a positive integer")
    if config.analysis_grid_size % 2 != 0:
        raise ValueError("analysis_grid_size must be even")
    if config.field_grid_max_size % 2 != 0:
        raise ValueError("field_grid_max_size must be even")
    if config.direction_histogram_bin_count < 4:
        raise ValueError("direction_histogram_bin_count must be at least 4")
    if config.solar_radial_ring_dense_count < config.solar_radial_ring_count:
        raise ValueError(
            "solar_radial_ring_dense_count must be at least solar_radial_ring_count"
        )
    if config.solar_azimuth_dense_count < config.solar_azimuth_sample_count:
        raise ValueError(
            "solar_azimuth_dense_count must be at least solar_azimuth_sample_count"
        )
    if config.grid_convergence_dense_factor < 2:
        raise ValueError("grid_convergence_dense_factor must be at least 2")

    non_empty_tuples = {
        "myopia_values_d": config.myopia_values_d,
        "pupil_diameter_values_mm": config.pupil_diameter_values_mm,
        "ring_count_values": config.ring_count_values,
        "circular_ring_holes_per_ring": config.circular_ring_holes_per_ring,
        "pitch_to_diameter_ratio_values": config.pitch_to_diameter_ratio_values,
        "diameter_to_pitch_ratio_values": config.diameter_to_pitch_ratio_values,
        "diameter_anchor_factors": config.diameter_anchor_factors,
        "coarse_diameter_check_factors": config.coarse_diameter_check_factors,
        "spectral_wavelengths_nm": config.spectral_wavelengths_nm,
        "design_point_diameter_mm": config.design_point_diameter_mm,
        "design_point_pitch_mm": config.design_point_pitch_mm,
        "design_point_labels": config.design_point_labels,
    }
    for name, values in non_empty_tuples.items():
        if not values:
            raise ValueError(f"{name} must not be empty")
    if any(value <= 0.0 for value in config.myopia_values_d):
        raise ValueError("myopia_values_d must be positive (0 D has no finite d*)")
    for value in config.diameter_anchor_factors:
        if value <= 0.0:
            raise ValueError("diameter_anchor_factors must be positive")
    for value in config.ring_count_values:
        if value < 1:
            raise ValueError("ring_count_values must be at least 1")
    for value in config.circular_ring_holes_per_ring:
        if value < 3:
            raise ValueError("each circular ring needs at least 3 holes")
    for value in config.pitch_to_diameter_ratio_values:
        if value <= 1.0:
            raise ValueError("pitch_to_diameter_ratio_values must exceed 1")
    for value in config.diameter_to_pitch_ratio_values:
        if not 0.0 < value < 1.0:
            raise ValueError("diameter_to_pitch_ratio_values must lie in (0, 1)")
    if len(config.design_point_diameter_mm) != len(config.design_point_pitch_mm):
        raise ValueError("design point diameters and pitches must have equal length")
    if len(config.design_point_labels) != len(config.design_point_diameter_mm):
        raise ValueError("design point labels must match design point count")
    for diameter_mm, pitch_mm in zip(
        config.design_point_diameter_mm, config.design_point_pitch_mm
    ):
        if pitch_mm - diameter_mm < config.minimum_edge_clearance_mm:
            raise ValueError(
                "design point violates minimum_edge_clearance_mm: "
                f"d={diameter_mm}, p={pitch_mm}"
            )

    # 排布规模必须足以体现周期规律（至少 3x3 内部孔）。
    for name, count in {
        "square_row_count": config.square_row_count,
        "square_column_count": config.square_column_count,
        "triangular_row_count": config.triangular_row_count,
        "triangular_column_count": config.triangular_column_count,
        "honeycomb_row_count": config.honeycomb_row_count,
        "honeycomb_column_count": config.honeycomb_column_count,
    }.items():
        if count < 3:
            raise ValueError(f"{name} must be at least 3")

    if config.enable_pupil_vignetting:
        if config.pupil_plane_distance_mm is None:
            raise ValueError(
                "enable_pupil_vignetting requires pupil_plane_distance_mm; "
                "without it the off-axis geometry is undefined"
            )
        if config.pupil_plane_distance_mm < 0.0:
            raise ValueError("pupil_plane_distance_mm must be non-negative")

    if config.psf_config.grid_size != config.psf_config.grid_size // 2 * 2:
        raise ValueError("psf_config.grid_size must be even")
    if config.psf_config.aperture_diameter_pixels <= 0.0:
        raise ValueError("psf_config.aperture_diameter_pixels must be positive")
    if config.mask_pixel_mm <= 0.0:
        raise ValueError("mask_pixel_mm must be positive")
    if config.mask_margin_mm < 0.0:
        raise ValueError("mask_margin_mm must be non-negative")
    if not 0.0 < config.mask_binary_threshold < 1.0:
        raise ValueError("mask_binary_threshold must lie in (0, 1)")
    if config.mask_minimum_pixels_per_diameter < 4.0:
        raise ValueError(
            "mask_minimum_pixels_per_diameter must be at least 4 so that "
            "the circle edge stays resolvable"
        )
    if not 0.0 < config.mask_minimum_circularity <= 1.0:
        raise ValueError("mask_minimum_circularity must lie in (0, 1]")

    for field_name in (
        "output_root_directory_name",
        "simulation_directory_name",
        "config_filename",
        "previous_anchor_filename",
        "refined_grid_filename",
        "coarse_grid_filename",
        "geometry_filename",
        "ghost_metrics_filename",
        "ring_geometry_filename",
        "ghost_peak_filename",
        "solar_convergence_filename",
        "validation_filename",
        "pareto_filename",
        "pattern_figure_filename",
        "mask_figure_filename",
        "point_source_figure_filename",
        "solar_comparison_figure_filename",
        "ring_comparison_figure_filename",
        "ghost_heatmap_figure_filename",
        "diameter_pitch_figure_filename",
        "slice_figure_filename",
        "design_figure_filename",
        "validation_figure_filename",
    ):
        validate_output_component(getattr(config, field_name), field_name)


def resolve_simulation_directory(config: ArrayGhostSimulationConfig) -> Path:
    """输出根目录固定在仓库下的 output，不使用外部绝对盘符。"""
    repository_root = Path(__file__).resolve().parent
    directory = (
        repository_root
        / config.output_root_directory_name
        / config.simulation_directory_name
    )
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def print_progress(
    completed: int,
    total: int,
    started_at: float,
    label: str,
) -> None:
    elapsed = time.perf_counter() - started_at
    fraction = completed / total if total else 0.0
    print(
        f"[{label}] {completed}/{total} ({fraction:6.2%}) "
        f"elapsed {elapsed:7.1f}s",
        flush=True,
    )


# ============================================================================
# 实现部分 2/5：五种孔心排布与排布质检
# ============================================================================
#
# 排布函数只负责生成孔心坐标和几何元数据，不画图、不写文件、不做光学计算。
# 孔心在 CPU 上用 NumPy 生成（方案第 12 节：掩膜与几何在 CPU，光学在 GPU），
# 进入光学计算前再显式上传。


@dataclass(frozen=True)
class ArrayLayout:
    """一次排布生成的全部几何结果。"""

    pattern_kind: str
    centers_mm: Any
    pitch_nominal_mm: float
    diameter_mm: float
    row_count: int
    column_count: int
    ring_count: int
    holes_per_ring: tuple[int, ...]
    ring_radii_mm: tuple[float, ...]
    ring_radial_pitch_mm: float
    analytic_area_fraction: float
    notes: str

    @property
    def hole_count(self) -> int:
        return int(self.centers_mm.shape[0])


def analytic_area_fraction(
    pattern_kind: str,
    diameter_mm: float,
    pitch_mm: float,
) -> float:
    """方案 2.2 到 2.6 节给出的解析面积分数。

    蜂窝与环阵列没有单一解析式，返回 NaN，由掩膜数值统计给出面积分数。
    """
    if pitch_mm <= 0.0:
        raise ValueError("pitch_mm must be positive")
    unit_cell_factor = {
        ARRAY_PATTERN_SQUARE_PACKING: math.pi / 4.0,
        ARRAY_PATTERN_TRIANGULAR_PACKING: math.pi / (2.0 * math.sqrt(3.0)),
    }
    factor = unit_cell_factor.get(pattern_kind)
    if factor is None:
        return math.nan
    return factor * (diameter_mm / pitch_mm) ** 2


def build_pitch_grid(
    config: ArrayGhostSimulationConfig,
    diameter_mm: float,
) -> tuple[float, ...]:
    """对数采样孔距，并保证下界同时满足瞳孔和加工约束。"""
    lower_bound = max(
        config.minimum_pitch_mm,
        diameter_mm + config.minimum_edge_clearance_mm,
    )
    upper_bound = max(config.maximum_pitch_mm, lower_bound)
    if config.pitch_log_count == 1:
        values = [lower_bound]
    else:
        log_min = math.log(lower_bound)
        log_max = math.log(upper_bound)
        step = (log_max - log_min) / (config.pitch_log_count - 1)
        values = [math.exp(log_min + index * step) for index in range(config.pitch_log_count)]
    return tuple(values)


def _centered_index_range(count: int) -> np.ndarray:
    """对称整数索引，且保证 0 在范围内，使参考主孔落在原点。"""
    return np.arange(count) - count // 2


def generate_square_packing(
    config: ArrayGhostSimulationConfig,
    pitch_mm: float,
    diameter_mm: float,
) -> ArrayLayout:
    """方案 2.2 节：正方形密铺，孔心位于正方形单元中心。"""

    row_count = config.effective_row_count()
    column_count = config.effective_column_count()
    x_offsets = _centered_index_range(column_count) * pitch_mm
    y_offsets = _centered_index_range(row_count) * pitch_mm
    grid_x, grid_y = np.meshgrid(x_offsets, y_offsets, indexing="xy")
    centers = np.stack((grid_x.ravel(), grid_y.ravel()), axis=1)
    centers = _truncate_to_extent(centers, config.array_extent_radius_mm)

    return ArrayLayout(
        pattern_kind=ARRAY_PATTERN_SQUARE_PACKING,
        centers_mm=centers,
        pitch_nominal_mm=pitch_mm,
        diameter_mm=diameter_mm,
        row_count=row_count,
        column_count=column_count,
        ring_count=0,
        holes_per_ring=(),
        ring_radii_mm=(),
        ring_radial_pitch_mm=pitch_mm,
        analytic_area_fraction=analytic_area_fraction(
            ARRAY_PATTERN_SQUARE_PACKING, diameter_mm, pitch_mm
        ),
        notes="square unit cell, four nearest neighbours along x and y",
    )


def generate_triangular_packing(
    config: ArrayGhostSimulationConfig,
    pitch_mm: float,
    diameter_mm: float,
) -> ArrayLayout:
    """方案 2.3 节：三角形密铺，同时也是 PDF 的'六边形密排'。"""

    row_count = (
        config.quick_row_count if config.quick_mode else config.triangular_row_count
    )
    column_count = (
        config.quick_column_count
        if config.quick_mode
        else config.triangular_column_count
    )
    row_spacing_mm = pitch_mm * math.sqrt(3.0) / 2.0
    centers: list[tuple[float, float]] = []
    for row_index in range(row_count):
        row_offset = 0.5 * pitch_mm if row_index % 2 else 0.0
        y_mm = (row_index - (row_count - 1) / 2.0) * row_spacing_mm
        for column_index in range(column_count):
            x_mm = (
                column_index - (column_count - 1) / 2.0
            ) * pitch_mm + row_offset
            centers.append((x_mm, y_mm))
    centers_array = np.asarray(centers, dtype=np.float64)
    centers_array = _truncate_to_extent(
        centers_array, config.array_extent_radius_mm
    )

    return ArrayLayout(
        pattern_kind=ARRAY_PATTERN_TRIANGULAR_PACKING,
        centers_mm=centers_array,
        pitch_nominal_mm=pitch_mm,
        diameter_mm=diameter_mm,
        row_count=row_count,
        column_count=column_count,
        ring_count=0,
        holes_per_ring=(),
        ring_radii_mm=(),
        ring_radial_pitch_mm=row_spacing_mm,
        analytic_area_fraction=analytic_area_fraction(
            ARRAY_PATTERN_TRIANGULAR_PACKING, diameter_mm, pitch_mm
        ),
        notes=(
            "triangular lattice, six nearest neighbours; this is the "
            "PDF hexagonal close packing"
        ),
    )


def generate_hexagonal_packing(
    config: ArrayGhostSimulationConfig,
    pitch_mm: float,
    diameter_mm: float,
) -> ArrayLayout:
    """方案 2.4 节：蜂窝顶点型六边形密铺，三个最近邻。

    蜂窝是带双原子基元的三角晶格：
        基矢 a1 = (sqrt(3) p, 0)，a2 = (sqrt(3) p / 2, 3 p / 2)
        基元 A = (0, 0)，B = (0, p)
    这样同一子晶格内的最近邻距离为 p，每个内部孔恰好 3 个最近邻。
    """

    row_count = (
        config.quick_row_count if config.quick_mode else config.honeycomb_row_count
    )
    column_count = (
        config.quick_column_count
        if config.quick_mode
        else config.honeycomb_column_count
    )
    a1 = np.array([math.sqrt(3.0) * pitch_mm, 0.0])
    a2 = np.array([0.5 * math.sqrt(3.0) * pitch_mm, 1.5 * pitch_mm])
    basis = (np.array([0.0, 0.0]), np.array([0.0, pitch_mm]))

    row_offsets = _centered_index_range(row_count)
    column_offsets = _centered_index_range(column_count)
    centers: list[tuple[float, float]] = []
    for row_index in row_offsets:
        for column_index in column_offsets:
            origin = row_index * a2 + column_index * a1
            for offset in basis:
                point = origin + offset
                centers.append((float(point[0]), float(point[1])))
    centers_array = np.asarray(centers, dtype=np.float64)
    centers_array = _truncate_to_extent(
        centers_array, config.array_extent_radius_mm
    )

    return ArrayLayout(
        pattern_kind=ARRAY_PATTERN_HEXAGONAL_PACKING,
        centers_mm=centers_array,
        pitch_nominal_mm=pitch_mm,
        diameter_mm=diameter_mm,
        row_count=row_count,
        column_count=column_count,
        ring_count=0,
        holes_per_ring=(),
        ring_radii_mm=(),
        ring_radial_pitch_mm=math.sqrt(3.0) * pitch_mm,
        analytic_area_fraction=math.nan,
        notes=(
            "honeycomb vertices, three nearest neighbours at 120 degrees; "
            "area fraction must come from the rasterised mask"
        ),
    )


def _truncate_to_extent(centers: Any, extent_radius_mm: float) -> Any:

    radius = np.sqrt((centers**2).sum(axis=1))
    return centers[radius <= extent_radius_mm]


def minimum_ring_radius_mm(hole_count: int, pitch_mm: float) -> float:
    """同一圈内相邻孔的**弦长**不小于孔距所需的最小半径。

    踩坑提示：用弧长约束 2*pi*R/n >= p 是错的。孔心之间的真实距离是弦长
    2*R*sin(pi/n)，它比弧长小；按弧长算出的半径会让 12 孔一圈的实际间距
    只剩约 0.989 p，几何质检会判为孔太近。
    """
    if hole_count < 2:
        raise ValueError("hole_count must be at least 2")
    half_step_rad = math.pi / hole_count
    return pitch_mm / (2.0 * math.sin(half_step_rad))


def generate_hexagonal_rings(
    config: ArrayGhostSimulationConfig,
    pitch_mm: float,
    diameter_mm: float,
    ring_count: int,
) -> ArrayLayout:
    """方案 2.5 节：中心孔加同心六边形环，第 r 圈 6r 个孔。"""

    if ring_count < 1:
        raise ValueError("ring_count must be at least 1")
    ring_radial_pitch_mm = max(config.hexagonal_ring_radial_pitch_mm, pitch_mm)
    centers: list[tuple[float, float]] = [(0.0, 0.0)]
    holes_per_ring: list[int] = []
    ring_radii_mm: list[float] = []
    for ring_index in range(1, ring_count + 1):
        hole_count = 6 * ring_index
        radius_mm = max(
            ring_index * ring_radial_pitch_mm,
            minimum_ring_radius_mm(hole_count, pitch_mm),
        )
        rotation_deg = (
            config.hexagonal_ring_rotation_step_deg
            * (ring_index - 1)
            / ring_index
        )
        for hole_index in range(hole_count):
            angle_deg = 360.0 * hole_index / hole_count + rotation_deg
            angle_rad = math.radians(angle_deg)
            centers.append(
                (
                    radius_mm * math.cos(angle_rad),
                    radius_mm * math.sin(angle_rad),
                )
            )
        holes_per_ring.append(hole_count)
        ring_radii_mm.append(radius_mm)
    centers_array = np.asarray(centers, dtype=np.float64)
    centers_array = _truncate_to_extent(
        centers_array, config.array_extent_radius_mm
    )

    return ArrayLayout(
        pattern_kind=ARRAY_PATTERN_HEXAGONAL_RINGS,
        centers_mm=centers_array,
        pitch_nominal_mm=pitch_mm,
        diameter_mm=diameter_mm,
        row_count=0,
        column_count=0,
        ring_count=ring_count,
        holes_per_ring=tuple(holes_per_ring),
        ring_radii_mm=tuple(ring_radii_mm),
        ring_radial_pitch_mm=ring_radial_pitch_mm,
        analytic_area_fraction=math.nan,
        notes=(
            "central hole plus rings of 6r holes; ring radial pitch is "
            "max(configured value, pitch)"
        ),
    )


def generate_circular_rings(
    config: ArrayGhostSimulationConfig,
    pitch_mm: float,
    diameter_mm: float,
    holes_per_ring: Sequence[int] | None = None,
) -> ArrayLayout:
    """方案 2.6 节：中心孔加等孔数或逐圈递增的圆环。"""

    counts = tuple(holes_per_ring or config.circular_ring_holes_per_ring)
    if not counts:
        raise ValueError("holes_per_ring must not be empty")
    if any(count < 3 for count in counts):
        raise ValueError("each ring needs at least 3 holes")
    ring_radial_pitch_mm = max(config.circular_ring_radial_pitch_mm, pitch_mm)
    radius_mm = ring_radial_pitch_mm
    centers: list[tuple[float, float]] = [(0.0, 0.0)]
    holes_per_ring_list: list[int] = []
    ring_radii_mm: list[float] = []
    for ring_index, hole_count in enumerate(counts, start=1):
        # 同一圈内相邻孔的弦长必须不小于 p（见 minimum_ring_radius_mm）。
        minimum_radius_mm = minimum_ring_radius_mm(hole_count, pitch_mm)
        if ring_index == 1:
            radius_mm = max(radius_mm, minimum_radius_mm)
        else:
            radius_mm = max(
                radius_mm + ring_radial_pitch_mm,
                minimum_radius_mm,
            )
        rotation_deg = (
            config.circular_ring_rotation_step_deg * (ring_index - 1) / ring_index
        )
        for hole_index in range(hole_count):
            angle_deg = 360.0 * hole_index / hole_count + rotation_deg
            angle_rad = math.radians(angle_deg)
            centers.append(
                (
                    radius_mm * math.cos(angle_rad),
                    radius_mm * math.sin(angle_rad),
                )
            )
        holes_per_ring_list.append(hole_count)
        ring_radii_mm.append(radius_mm)
    centers_array = np.asarray(centers, dtype=np.float64)
    centers_array = _truncate_to_extent(
        centers_array, config.array_extent_radius_mm
    )

    return ArrayLayout(
        pattern_kind=ARRAY_PATTERN_CIRCULAR_RINGS,
        centers_mm=centers_array,
        pitch_nominal_mm=pitch_mm,
        diameter_mm=diameter_mm,
        row_count=0,
        column_count=0,
        ring_count=len(counts),
        holes_per_ring=tuple(holes_per_ring_list),
        ring_radii_mm=tuple(ring_radii_mm),
        ring_radial_pitch_mm=ring_radial_pitch_mm,
        analytic_area_fraction=math.nan,
        notes=(
            "central hole plus circular rings with configured hole counts; "
            "arc spacing is reported separately"
        ),
    )


def generate_jittered_square(
    config: ArrayGhostSimulationConfig,
    pitch_mm: float,
    diameter_mm: float,
    seed: int,
) -> ArrayLayout:
    """方案 2.7 节：可选随机对照，抖动不得破坏最小孔距。"""

    base = generate_square_packing(config, pitch_mm, diameter_mm)
    rng = np.random.default_rng(seed)
    max_offset_mm = config.jitter_fraction * pitch_mm
    offsets = rng.uniform(
        -max_offset_mm,
        max_offset_mm,
        size=base.centers_mm.shape,
    )
    centers = base.centers_mm + offsets
    return replace(
        base,
        centers_mm=centers,
        notes=f"optional jittered square control, seed={seed}",
    )


def nearest_neighbor_statistics(
    centers_mm: Any,
    core_radius_mm: float,
) -> dict[str, Any]:
    """用 CPU 上的 cKDTree 统计最近邻距离与近邻数。"""

    centers = np.asarray(centers_mm, dtype=np.float64)
    if centers.shape[0] < 2:
        return {
            "nearest_neighbor_min_mm": math.nan,
            "nearest_neighbor_mean_mm": math.nan,
            "core_neighbor_count_mean": 0.0,
            "core_neighbor_count_min": 0,
            "core_neighbor_count_max": 0,
        }
    tree = cKDTree(centers)
    distances, _ = tree.query(centers, k=2)
    nearest = distances[:, 1]
    neighbor_lists = tree.query_ball_point(centers, core_radius_mm)
    # 去掉自身，只统计核心半径内的真实近邻数。
    core_counts = np.array([len(entries) - 1 for entries in neighbor_lists])
    return {
        "nearest_neighbor_min_mm": float(nearest.min()),
        "nearest_neighbor_mean_mm": float(nearest.mean()),
        "core_neighbor_count_mean": float(core_counts.mean()),
        "core_neighbor_count_min": int(core_counts.min()),
        "core_neighbor_count_max": int(core_counts.max()),
    }


def validate_layout(
    layout: ArrayLayout,
    config: ArrayGhostSimulationConfig,
) -> dict[str, Any]:
    """方案 3.2 与 V-I：孔数、最小间距、边缘余量与近邻拓扑。"""

    centers = np.asarray(layout.centers_mm, dtype=np.float64)
    if centers.shape[0] == 0:
        raise ValueError("layout contains no holes")
    core_radius_mm = config.core_neighbor_radius_factor * layout.pitch_nominal_mm
    statistics = nearest_neighbor_statistics(centers, core_radius_mm)
    edge_clearance_mm = (
        statistics["nearest_neighbor_min_mm"] - layout.diameter_mm
        if math.isfinite(statistics["nearest_neighbor_min_mm"])
        else math.nan
    )

    geometry_ok = True
    if math.isfinite(edge_clearance_mm):
        if edge_clearance_mm < config.minimum_edge_clearance_mm - config.floating_comparison_tolerance:
            geometry_ok = False
    else:
        geometry_ok = False

    expected_neighbors = {
        ARRAY_PATTERN_SQUARE_PACKING: 4,
        ARRAY_PATTERN_TRIANGULAR_PACKING: 6,
        ARRAY_PATTERN_HEXAGONAL_PACKING: 3,
    }
    expected = expected_neighbors.get(layout.pattern_kind)
    topology_ok = True
    if expected is not None:
        observed = statistics["core_neighbor_count_max"]
        topology_ok = observed == expected

    return {
        "pattern_kind": layout.pattern_kind,
        "hole_count": layout.hole_count,
        "pitch_nominal_mm": layout.pitch_nominal_mm,
        "pitch_to_diameter_ratio": layout.pitch_nominal_mm / layout.diameter_mm,
        "diameter_mm": layout.diameter_mm,
        "nearest_neighbor_min_mm": statistics["nearest_neighbor_min_mm"],
        "nearest_neighbor_mean_mm": statistics["nearest_neighbor_mean_mm"],
        "core_neighbor_count_mean": statistics["core_neighbor_count_mean"],
        "core_neighbor_count_max": statistics["core_neighbor_count_max"],
        "expected_core_neighbors": expected if expected is not None else "",
        "edge_clearance_mm": edge_clearance_mm,
        "analytic_area_fraction": layout.analytic_area_fraction,
        "ring_count": layout.ring_count,
        "holes_per_ring": "|".join(str(value) for value in layout.holes_per_ring),
        "ring_radii_mm": "|".join(
            f"{value:.6f}" for value in layout.ring_radii_mm
        ),
        "geometry_ok": geometry_ok,
        "topology_ok": topology_ok,
        "notes": layout.notes,
    }


# ============================================================================
# 实现部分 3/5：抗锯齿掩膜与掩膜质检
# ============================================================================
#
# 掩膜只服务几何、透过率与图形质量，不作为整块镜片的波动传播输入。
# 第二步曾出现'矩形覆盖度内部不是 1、外部不是 0'的错误，因此这里的圆孔
# 覆盖度必须测试圆心为 1、远处为 0、边缘落在 0 到 1 之间。


@dataclass(frozen=True)
class MaskGeometry:
    """掩膜数组与它的物理坐标窗口。"""

    coverage: np.ndarray
    pixel_mm: float
    x_min_mm: float
    x_max_mm: float
    y_min_mm: float
    y_max_mm: float

    @property
    def pixel_count(self) -> int:
        return int(self.coverage.size)

    @property
    def window_area_mm2(self) -> float:
        return (self.x_max_mm - self.x_min_mm) * (self.y_max_mm - self.y_min_mm)

    def axes_mm(self) -> tuple[np.ndarray, np.ndarray]:
        x_axis = np.linspace(
            self.x_min_mm + 0.5 * self.pixel_mm,
            self.x_max_mm - 0.5 * self.pixel_mm,
            self.coverage.shape[1],
        )
        y_axis = np.linspace(
            self.y_min_mm + 0.5 * self.pixel_mm,
            self.y_max_mm - 0.5 * self.pixel_mm,
            self.coverage.shape[0],
        )
        return x_axis, y_axis


def _mask_grid_parameters(
    layout: ArrayLayout,
    config: ArrayGhostSimulationConfig,
) -> tuple[int, float, float]:
    """由孔心范围与孔半径决定掩膜窗口半宽、像素数与像素尺寸。"""
    max_radius_mm = float(np.sqrt((layout.centers_mm**2).sum(axis=1)).max())
    half_extent_mm = max_radius_mm + 0.5 * layout.diameter_mm + config.mask_margin_mm
    pixel_count = int(math.ceil(2.0 * half_extent_mm / config.mask_pixel_mm))
    if pixel_count % 2 != 0:
        pixel_count += 1
    pixel_count = max(pixel_count, 8)
    pixel_mm = 2.0 * half_extent_mm / pixel_count
    return pixel_count, pixel_mm, half_extent_mm


def rasterize_mask(
    layout: ArrayLayout,
    config: ArrayGhostSimulationConfig,
) -> MaskGeometry:
    """在 4x4 子网格上画圆孔再块平均，得到 [0, 1] 覆盖度。

    每个孔只在自己的包围盒内计算，避免 O(像素数 x 孔数) 的全图距离运算。
    """
    pixel_count, pixel_mm, half_extent_mm = _mask_grid_parameters(layout, config)
    pixels_per_diameter = layout.diameter_mm / pixel_mm
    if pixels_per_diameter < config.mask_minimum_pixels_per_diameter:
        raise ValueError(
            "mask resolution too coarse: "
            f"diameter {layout.diameter_mm} mm spans only "
            f"{pixels_per_diameter:.2f} pixels"
        )

    coverage = np.zeros((pixel_count, pixel_count), dtype=np.float64)
    subsample_count = config.aperture_anti_alias_subsamples
    if subsample_count < 1:
        raise ValueError("aperture_anti_alias_subsamples must be positive")

    # 子采样点位于像素内的等距格点中心。
    subsample_offsets = (
        np.arange(subsample_count, dtype=np.float64) + 0.5
    ) / subsample_count - 0.5
    radius_mm = 0.5 * layout.diameter_mm
    axis_origin_mm = -half_extent_mm
    pixel_area_subweight = 1.0 / float(subsample_count * subsample_count)

    for center_x_mm, center_y_mm in layout.centers_mm:
        x_low_mm = center_x_mm - radius_mm - config.mask_margin_mm
        x_high_mm = center_x_mm + radius_mm + config.mask_margin_mm
        y_low_mm = center_y_mm - radius_mm - config.mask_margin_mm
        y_high_mm = center_y_mm + radius_mm + config.mask_margin_mm

        x_start = max(0, int(math.floor((x_low_mm - axis_origin_mm) / pixel_mm)))
        x_stop = min(
            pixel_count,
            int(math.ceil((x_high_mm - axis_origin_mm) / pixel_mm)) + 1,
        )
        y_start = max(0, int(math.floor((y_low_mm - axis_origin_mm) / pixel_mm)))
        y_stop = min(
            pixel_count,
            int(math.ceil((y_high_mm - axis_origin_mm) / pixel_mm)) + 1,
        )
        if x_stop <= x_start or y_stop <= y_start:
            continue

        x_centers_mm = (
            axis_origin_mm + (np.arange(x_start, x_stop) + 0.5) * pixel_mm
        )
        y_centers_mm = (
            axis_origin_mm + (np.arange(y_start, y_stop) + 0.5) * pixel_mm
        )
        local = np.zeros((y_stop - y_start, x_stop - x_start), dtype=np.float64)
        for offset_y in subsample_offsets:
            for offset_x in subsample_offsets:
                distance_squared = (
                    (x_centers_mm[None, :] + offset_x * pixel_mm - center_x_mm) ** 2
                    + (y_centers_mm[:, None] + offset_y * pixel_mm - center_y_mm) ** 2
                )
                local += (distance_squared <= radius_mm * radius_mm).astype(
                    np.float64
                )
        coverage[y_start:y_stop, x_start:x_stop] += (
            local * pixel_area_subweight
        )

    coverage = np.clip(coverage, 0.0, 1.0)
    return MaskGeometry(
        coverage=coverage,
        pixel_mm=pixel_mm,
        x_min_mm=-half_extent_mm,
        x_max_mm=half_extent_mm,
        y_min_mm=-half_extent_mm,
        y_max_mm=half_extent_mm,
    )


def region_equivalent_diameter_px(region: Any) -> float:
    """兼容 skimage 版本：0.26 起 equivalent_diameter 改名为
    equivalent_diameter_area，两者都表示由面积反推的等效直径。"""
    if hasattr(region, "equivalent_diameter_area"):
        return float(region.equivalent_diameter_area)
    return float(region.equivalent_diameter)


def lattice_unit_cell_area_mm2(pattern_kind: str, pitch_mm: float) -> float:
    """每种晶格分摊到单个孔的单元面积，用于解析填充率。"""
    if pattern_kind == ARRAY_PATTERN_SQUARE_PACKING:
        return pitch_mm * pitch_mm
    if pattern_kind == ARRAY_PATTERN_TRIANGULAR_PACKING:
        return math.sqrt(3.0) / 2.0 * pitch_mm * pitch_mm
    if pattern_kind == ARRAY_PATTERN_HEXAGONAL_PACKING:
        # 蜂窝基元含 2 个孔，晶胞面积 |a1 x a2| = 1.5*sqrt(3)*p^2。
        return 0.75 * math.sqrt(3.0) * pitch_mm * pitch_mm
    return math.nan


def validate_mask(
    mask: MaskGeometry,
    layout: ArrayLayout,
    config: ArrayGhostSimulationConfig,
) -> dict[str, Any]:
    """方案 3.2 与 V-A：面积、连通域、等效直径、圆度与质心。"""
    from scipy import ndimage
    from skimage.measure import regionprops

    binary = mask.coverage >= config.mask_binary_threshold
    labeled, label_count = ndimage.label(binary)
    regions = regionprops(labeled)

    pixel_area_mm2 = mask.pixel_mm * mask.pixel_mm
    numeric_area_mm2 = float(mask.coverage.sum() * pixel_area_mm2)
    analytic_area_mm2 = layout.hole_count * math.pi * (0.5 * layout.diameter_mm) ** 2
    relative_area_error = (
        abs(numeric_area_mm2 - analytic_area_mm2) / analytic_area_mm2
        if analytic_area_mm2 > 0.0
        else math.nan
    )

    equivalent_diameters_mm = [
        region_equivalent_diameter_px(region) * mask.pixel_mm
        for region in regions
    ]
    circularity_values = []
    for region in regions:
        perimeter = float(region.perimeter) * mask.pixel_mm
        area = float(region.area) * pixel_area_mm2
        if perimeter > 0.0:
            circularity_values.append(4.0 * math.pi * area / (perimeter * perimeter))

    # 质心偏差：每个设计孔心去最近的设计位置比对。
    max_centroid_offset_mm = 0.0
    if regions:
        designed_centers = np.asarray(layout.centers_mm, dtype=np.float64)
        for region in regions:
            centroid_y_px, centroid_x_px = region.centroid
            centroid_x_mm = mask.x_min_mm + (centroid_x_px + 0.5) * mask.pixel_mm
            centroid_y_mm = mask.y_min_mm + (centroid_y_px + 0.5) * mask.pixel_mm
            offsets = np.sqrt(
                (designed_centers[:, 0] - centroid_x_mm) ** 2
                + (designed_centers[:, 1] - centroid_y_mm) ** 2
            )
            max_centroid_offset_mm = max(
                max_centroid_offset_mm, float(offsets.min())
            )

    unit_cell_area_mm2 = lattice_unit_cell_area_mm2(
        layout.pattern_kind, layout.pitch_nominal_mm
    )
    lattice_filling_fraction = (
        math.pi * (0.5 * layout.diameter_mm) ** 2 / unit_cell_area_mm2
        if math.isfinite(unit_cell_area_mm2) and unit_cell_area_mm2 > 0.0
        else math.nan
    )

    centroid_tolerance_mm = (
        config.mask_centroid_tolerance_pixel * mask.pixel_mm
    )
    fail_reasons: list[str] = []
    if not math.isfinite(relative_area_error):
        fail_reasons.append("area_error_not_finite")
    elif relative_area_error > config.mask_area_relative_tolerance:
        fail_reasons.append("area_error_exceeds_tolerance")
    if label_count != layout.hole_count:
        fail_reasons.append("connected_component_count_mismatch")
    if max_centroid_offset_mm > centroid_tolerance_mm:
        fail_reasons.append("centroid_offset_exceeds_tolerance")
    if circularity_values and min(circularity_values) < config.mask_minimum_circularity:
        fail_reasons.append("circularity_below_tolerance")

    return {
        "pattern_kind": layout.pattern_kind,
        "designed_hole_count": layout.hole_count,
        "labeled_hole_count": int(label_count),
        "numeric_hole_area_mm2": numeric_area_mm2,
        "analytic_hole_area_mm2": analytic_area_mm2,
        "relative_area_error": relative_area_error,
        "window_area_fraction": numeric_area_mm2 / mask.window_area_mm2,
        "lattice_filling_fraction": lattice_filling_fraction,
        "mean_equivalent_diameter_mm": (
            float(np.mean(equivalent_diameters_mm))
            if equivalent_diameters_mm
            else math.nan
        ),
        "min_circularity": (
            float(min(circularity_values)) if circularity_values else math.nan
        ),
        "max_centroid_offset_mm": max_centroid_offset_mm,
        "centroid_tolerance_mm": centroid_tolerance_mm,
        "mask_pixel_mm": mask.pixel_mm,
        "mask_pixel_count": mask.pixel_count,
        "mask_ok": not fail_reasons,
        "fail_reasons": "|".join(fail_reasons),
    }


# ============================================================================
# 实现部分 4/5：光学链路（锚点复用、PSF 缓存、太阳盘、平移叠加）
# ============================================================================
#
# 本节的物理模型完全继承第一阶段：
#   - 单孔 PSF 由 compute_psf 给出，不重新实现圆孔、离焦波前或 FFT；
#   - 孔间按强度叠加，不建立任何复振幅互相关；
#   - 太阳盘积分用面积权重，等价于与离散化圆盘核做卷积，
#     但两种实现必须互相复算（V-G）。


def _read_metric_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _optimal_diameter_from_rows(
    rows: Iterable[dict[str, str]],
    wavelength_column: str,
    myopia_column: str,
    diameter_column: str,
    metric_column: str,
    wavelength_nm: float,
) -> tuple[dict[float, float], dict[float, float]]:
    """按 (波长, 近视) 分组，取指标最小的一行作为该组最优孔径。"""
    best_metric: dict[float, float] = {}
    best_diameter: dict[float, float] = {}
    for row in rows:
        try:
            row_wavelength_nm = float(row[wavelength_column])
            myopia_d = float(row[myopia_column])
            diameter_mm = float(row[diameter_column])
            metric_value = float(row[metric_column])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"cannot parse previous-stage row for {metric_column}: {row!r}"
            ) from error
        if not math.isclose(
            row_wavelength_nm, wavelength_nm, abs_tol=1.0e-6
        ):
            continue
        if not math.isfinite(metric_value):
            continue
        if myopia_d <= 0.0:
            # 0 D 没有离焦，理论最优孔径无有限解，不作为第三步锚点。
            continue
        if myopia_d not in best_metric or metric_value < best_metric[myopia_d]:
            best_metric[myopia_d] = metric_value
            best_diameter[myopia_d] = diameter_mm
    return best_diameter, best_metric


def load_previous_diameter_anchors(
    config: ArrayGhostSimulationConfig,
) -> dict[str, Any]:
    """方案 0.4 与 4.1：读取前两阶段最优孔径，作为本阶段的主锚点。

    正式结论要求 reuse_previous_metrics=True。只有在显式给出调试标签时
    才允许缺锚点，并在后续用理论最优孔径兜底。
    """
    repository_root = Path(__file__).resolve().parent
    retinal_path = repository_root / config.retinal_metrics_relative_path
    psf_path = repository_root / (
        config.psf_metrics_relative_path_template.format(
            wavelength_nm=config.reference_anchor_wavelength_nm
        )
    )

    anchors: dict[str, Any] = {
        "retinal_source": str(config.retinal_metrics_relative_path),
        "psf_source": str(
            config.psf_metrics_relative_path_template.format(
                wavelength_nm=config.reference_anchor_wavelength_nm
            )
        ),
        "retinal_optimal_mm": {},
        "retinal_metric": {},
        "psf_optimal_mm": {},
        "psf_metric": {},
    }

    if retinal_path.exists():
        rows = _read_metric_rows(retinal_path)
        diameters, metrics = _optimal_diameter_from_rows(
            rows,
            config.retinal_wavelength_column,
            config.retinal_myopia_column,
            config.retinal_diameter_column,
            config.retinal_threshold_column,
            config.reference_anchor_wavelength_nm,
        )
        anchors["retinal_optimal_mm"] = diameters
        anchors["retinal_metric"] = metrics
        anchors["retinal_row_count"] = len(rows)
    elif config.reuse_previous_metrics:
        raise FileNotFoundError(
            "second-stage metrics are required for anchor reuse: "
            f"{retinal_path}"
        )

    if psf_path.exists():
        rows = _read_metric_rows(psf_path)
        diameters, metrics = _optimal_diameter_from_rows(
            rows,
            config.psf_wavelength_column,
            config.psf_myopia_column,
            config.psf_diameter_column,
            config.psf_metric_column,
            config.reference_anchor_wavelength_nm,
        )
        anchors["psf_optimal_mm"] = diameters
        anchors["psf_metric"] = metrics
        anchors["psf_row_count"] = len(rows)
    elif config.reuse_previous_metrics:
        raise FileNotFoundError(
            "first-stage metrics are required for cross-check anchors: "
            f"{psf_path}"
        )

    return anchors


def build_anchor_rows(
    config: ArrayGhostSimulationConfig,
    anchors: dict[str, Any],
) -> list[dict[str, Any]]:
    """方案第 9 节：previous_diameter_anchors.csv 的每一行。"""
    rows: list[dict[str, Any]] = []
    myopia_values = sorted(
        set(anchors["retinal_optimal_mm"]) | set(anchors["psf_optimal_mm"])
    )
    for myopia_d in myopia_values:
        retinal_diameter = anchors["retinal_optimal_mm"].get(myopia_d)
        psf_diameter = anchors["psf_optimal_mm"].get(myopia_d)
        if retinal_diameter is not None and psf_diameter is not None:
            relative_difference_percent = (
                100.0 * (retinal_diameter - psf_diameter) / psf_diameter
            )
            reuse_status = "retinal_primary_with_psf_crosscheck"
        elif retinal_diameter is not None:
            relative_difference_percent = math.nan
            reuse_status = "retinal_primary_only"
        elif psf_diameter is not None:
            relative_difference_percent = math.nan
            reuse_status = "psf_crosscheck_only"
        else:
            relative_difference_percent = math.nan
            reuse_status = "missing"
        rows.append(
            {
                "wavelength_nm": config.reference_anchor_wavelength_nm,
                "myopia_d": myopia_d,
                "retinal_d_opt_mm": (
                    retinal_diameter if retinal_diameter is not None else math.nan
                ),
                "retinal_metric": anchors["retinal_metric"].get(myopia_d, math.nan),
                "psf_d50_d_opt_mm": (
                    psf_diameter if psf_diameter is not None else math.nan
                ),
                "psf_metric": anchors["psf_metric"].get(myopia_d, math.nan),
                "relative_difference_percent": relative_difference_percent,
                "reuse_status": reuse_status,
                "reuse_previous_metrics": config.reuse_previous_metrics,
            }
        )
    return rows


def reference_diameter_mm(
    config: ArrayGhostSimulationConfig,
    anchors: dict[str, Any],
    myopia_d: float,
    wavelength_nm: float,
) -> tuple[float, str]:
    """给出该近视度数的主锚点孔径，并说明来源。"""
    if math.isclose(
        wavelength_nm,
        config.reference_anchor_wavelength_nm,
        abs_tol=1.0e-6,
    ):
        retinal_diameter = anchors["retinal_optimal_mm"].get(myopia_d)
        if retinal_diameter is not None:
            return retinal_diameter, "retinal_optotype_primary"
        psf_diameter = anchors["psf_optimal_mm"].get(myopia_d)
        if psf_diameter is not None:
            return psf_diameter, "psf_d50_crosscheck"
    if not config.reuse_previous_metrics:
        return (
            theoretical_optimal_diameter_mm(config.psf_config, wavelength_nm, myopia_d),
            "theoretical_fallback_debug",
        )
    raise KeyError(
        "no previous-stage anchor for "
        f"M={myopia_d} D at wavelength {wavelength_nm} nm; "
        "extend the anchor table instead of silently scanning the full range"
    )


def build_diameter_anchor_grid(
    config: ArrayGhostSimulationConfig,
    anchors: dict[str, Any],
    myopia_d: float,
    wavelength_nm: float,
) -> tuple[tuple[float, ...], str]:
    """方案 0.4 与 2.9 节：d_opt 邻域，而非全范围重扫孔径。"""
    reference_mm, source = reference_diameter_mm(
        config, anchors, myopia_d, wavelength_nm
    )
    values: list[float] = []
    for factor in config.effective_diameter_anchor_factors():
        values.append(reference_mm * factor)
    deduplicated: list[float] = []
    for value in sorted(values):
        if (
            not deduplicated
            or abs(value - deduplicated[-1])
            > config.diameter_deduplication_tolerance_mm
        ):
            deduplicated.append(value)
    return tuple(deduplicated), source


@dataclass
class PinholePsfCache:
    """按 (d, M, lambda) 缓存单孔 PSF，避免重复 FFT。"""

    aperture: Any
    normalized_radius_squared: Any
    psf_by_key: dict[tuple[float, float, float], Any] = field(default_factory=dict)
    sampling_by_key: dict[tuple[float, float, float], Any] = field(
        default_factory=dict
    )
    compute_count: int = 0
    hit_count: int = 0

    def key(
        self,
        diameter_mm: float,
        myopia_d: float,
        wavelength_nm: float,
    ) -> tuple[float, float, float]:
        # 浮点扫描键统一按固定小数位取整，禁止裸浮点相等。
        return (
            round(diameter_mm, 6),
            round(myopia_d, 6),
            round(wavelength_nm, 6),
        )

    def get(
        self,
        config: ArrayGhostSimulationConfig,
        diameter_mm: float,
        myopia_d: float,
        wavelength_nm: float,
    ) -> tuple[Any, Any]:
        key = self.key(diameter_mm, myopia_d, wavelength_nm)
        if key in self.psf_by_key:
            self.hit_count += 1
            return self.psf_by_key[key], self.sampling_by_key[key]
        sampling = build_sampling(config.psf_config, wavelength_nm, diameter_mm)
        psf = compute_psf(
            config.psf_config,
            self.aperture,
            self.normalized_radius_squared,
            wavelength_nm,
            diameter_mm,
            myopia_d,
        )
        self.compute_count += 1
        self.psf_by_key[key] = psf
        self.sampling_by_key[key] = sampling
        return psf, sampling


def build_psf_cache(config: ArrayGhostSimulationConfig) -> PinholePsfCache:
    """在 GPU 上一次性建立圆孔与归一化半径网格，供全部扫描点复用。"""
    with cp.cuda.Device(config.gpu_device_id):
        (
            normalized_radius,
            normalized_radius_squared,
            _radial_bin_index,
            _radial_pixel_count,
        ) = build_gpu_grids(config.psf_config)
        aperture = build_soft_aperture(config.psf_config, normalized_radius)
    return PinholePsfCache(
        aperture=aperture,
        normalized_radius_squared=normalized_radius_squared,
    )


def resample_psf_to_grid(
    values: Any,
    source_pixel_arcmin: float,
    target_size: int,
    target_pixel_arcmin: float,
    config: ArrayGhostSimulationConfig,
) -> Any:
    """把任意以角坐标为单位的阵列重采样到目标网格。

    中心约定沿用第二阶段：显式角坐标映射，中心像素严格对应中心像素，
    不做任何预先 ifftshift；重采样后按总能量归一化。
    """
    source_size = int(values.shape[0])
    centre = source_size // 2
    target_axis = (
        cp.arange(target_size, dtype=cp.float32) - target_size // 2
    ) * (target_pixel_arcmin / source_pixel_arcmin)
    source_axis = target_axis + centre
    source_y, source_x = cp.meshgrid(source_axis, source_axis, indexing="ij")
    resampled = map_coordinates(
        values,
        cp.stack((source_y, source_x), axis=0),
        order=config.psf_resampling_order,
        mode="constant",
        cval=0.0,
    )
    energy = resampled.sum(dtype=config.accumulator_dtype)
    if float(energy) > 0.0:
        resampled = resampled / energy
    return resampled.astype(config.real_dtype)


def build_solar_disk_kernel(
    config: ArrayGhostSimulationConfig,
    grid_size: int,
    pixel_arcmin: float,
) -> tuple[Any, dict[str, Any]]:
    """在目标网格上离散化太阳圆盘，得到面积权重核。

    每个像素的权重是该像素被太阳圆盘覆盖的面积占比；子采样密度由
    aperture_anti_alias_subsamples 控制。归一化后总和为 1。
    """
    radius_arcmin = config.solar_angular_radius_arcmin
    if radius_arcmin > config.solar_kernel_half_width_arcmin:
        raise ValueError(
            "solar_kernel_half_width_arcmin must cover the solar radius"
        )
    half_width_arcmin = 0.5 * grid_size * pixel_arcmin
    if radius_arcmin + pixel_arcmin > half_width_arcmin:
        raise ValueError(
            "analysis grid is too small for the solar disk: "
            f"half width {half_width_arcmin:.3f} arcmin, "
            f"solar radius {radius_arcmin:.3f} arcmin"
        )

    axis_arcmin = (
        cp.arange(grid_size, dtype=cp.float32) - grid_size // 2
    ) * pixel_arcmin
    grid_x, grid_y = cp.meshgrid(axis_arcmin, axis_arcmin, indexing="xy")
    subsample_count = config.aperture_anti_alias_subsamples
    offsets = (
        cp.arange(subsample_count, dtype=cp.float32) + 0.5
    ) / subsample_count - 0.5
    coverage = cp.zeros_like(grid_x)
    radius_squared = radius_arcmin * radius_arcmin
    for offset_y in offsets:
        for offset_x in offsets:
            distance_squared = (
                (grid_x + offset_x * pixel_arcmin) ** 2
                + (grid_y + offset_y * pixel_arcmin) ** 2
            )
            coverage += (distance_squared <= radius_squared).astype(
                config.real_dtype
            )
    coverage /= float(subsample_count * subsample_count)
    total = coverage.sum(dtype=config.accumulator_dtype)
    if float(total) <= 0.0:
        raise ValueError("solar disk kernel is empty; check the angular radius")
    kernel = (coverage / total).astype(config.real_dtype)

    # 解析圆盘面积占比：pi R^2 / 网格窗口面积，用于交叉校验。
    analytic_fraction = (
        math.pi * radius_arcmin**2 / (grid_size * pixel_arcmin) ** 2
    )
    metadata = {
        "weight_sum": float(kernel.sum(dtype=config.accumulator_dtype)),
        "nonzero_pixel_count": int((kernel > 0).sum()),
        "analytic_disk_fraction_of_window": analytic_fraction,
        "numeric_disk_fraction_of_window": float(total) / float(grid_size * grid_size),
        "pixel_arcmin": pixel_arcmin,
        "grid_size": grid_size,
    }
    return kernel, metadata


def build_solar_directions(
    config: ArrayGhostSimulationConfig,
    radial_count: int | None = None,
    azimuth_count: int | None = None,
) -> tuple[Any, Any]:
    """等面积圆环 + 方位采样的太阳盘方向与权重（V-G 用）。"""
    radial_samples = (
        radial_count
        if radial_count is not None
        else config.effective_solar_radial_ring_count()
    )
    azimuth_samples = (
        azimuth_count
        if azimuth_count is not None
        else config.effective_solar_azimuth_sample_count()
    )
    if radial_samples < 1 or azimuth_samples < 1:
        raise ValueError("solar sampling counts must be positive")
    radius_arcmin = config.solar_angular_radius_arcmin
    # 等面积环：每个环的代表半径为 sqrt((k+0.5)/K) * R，权重相同。
    radial_indices = cp.arange(radial_samples, dtype=cp.float32)
    ring_radii = radius_arcmin * cp.sqrt(
        (radial_indices + 0.5) / radial_samples
    )
    azimuth_angles = (
        2.0 * math.pi * cp.arange(azimuth_samples, dtype=cp.float32)
        / azimuth_samples
    )
    radii_grid, angles_grid = cp.meshgrid(
        ring_radii, azimuth_angles, indexing="ij"
    )
    directions = cp.stack(
        (
            radii_grid.ravel() * cp.cos(angles_grid.ravel()),
            radii_grid.ravel() * cp.sin(angles_grid.ravel()),
        ),
        axis=1,
    )
    weight = 1.0 / float(radial_samples * azimuth_samples)
    weights = cp.full(directions.shape[0], weight, dtype=config.accumulator_dtype)
    return directions.astype(config.real_dtype), weights


def convolve_with_solar_disk(
    psf_grid: Any,
    disk_kernel: Any,
    config: ArrayGhostSimulationConfig,
) -> tuple[Any, dict[str, Any]]:
    """与太阳盘核做卷积。

    沿用第二阶段的中心约定：不预先 ifftshift，full 卷积后从
    kernel.shape // 2 开始裁切。
    """
    energy_before = float(psf_grid.sum(dtype=config.accumulator_dtype))
    convolved = fftconvolve(psf_grid, disk_kernel, mode=config.convolution_mode)
    start_y = disk_kernel.shape[0] // 2
    start_x = disk_kernel.shape[1] // 2
    convolved = convolved[
        start_y : start_y + psf_grid.shape[0],
        start_x : start_x + psf_grid.shape[1],
    ]
    energy_after = float(convolved.sum(dtype=config.accumulator_dtype))
    if energy_after > 0.0:
        convolved = convolved / energy_after
    metadata = {
        "energy_before": energy_before,
        "energy_after": energy_after,
        "energy_lost_fraction": (
            abs(energy_before - energy_after) / energy_before
            if energy_before > 0.0
            else math.nan
        ),
    }
    return convolved.astype(config.real_dtype), metadata


def integrate_solar_source_direct(
    grid: Any,
    directions_arcmin: Any,
    weights: Any,
    pixel_arcmin: float,
    config: ArrayGhostSimulationConfig,
) -> Any:
    """按方向逐项求和实现太阳盘积分，用于与卷积实现互相复算。

    这条路径是平移不变的直接定义：
        I(theta) = sum_s w_s * grid(theta - theta_s)
    """
    grid_size = int(grid.shape[0])
    axis = (
        cp.arange(grid_size, dtype=cp.float32) - grid_size // 2
    )
    grid_y, grid_x = cp.meshgrid(axis, axis, indexing="ij")
    # 采样索引必须把'居中坐标'换算回'数组下标'，即加上中心偏移；
    # 漏掉这一步会把核采到窗口外，总能量会凭空损失。
    centre_index = grid_size // 2
    accumulated = cp.zeros_like(grid, dtype=config.accumulator_dtype)
    for index in range(int(directions_arcmin.shape[0])):
        shift_x = float(directions_arcmin[index, 0]) / pixel_arcmin
        shift_y = float(directions_arcmin[index, 1]) / pixel_arcmin
        sampled = map_coordinates(
            grid,
            cp.stack(
                (
                    grid_y - shift_y + centre_index,
                    grid_x - shift_x + centre_index,
                ),
                axis=0,
            ),
            order=config.psf_resampling_order,
            mode="constant",
            cval=0.0,
        )
        accumulated += float(weights[index]) * sampled
    total = accumulated.sum(dtype=config.accumulator_dtype)
    if float(total) > 0.0:
        accumulated = accumulated / total
    return accumulated.astype(config.real_dtype)


def measure_image_width_arcmin(
    grid: Any,
    pixel_arcmin: float,
    energy_fraction: float,
    config: ArrayGhostSimulationConfig,
) -> float:
    """用径向包围能量（默认 50%）测量单孔像的等效宽度。"""
    grid_size = int(grid.shape[0])
    coordinate = (
        cp.arange(grid_size, dtype=cp.float32) - grid_size // 2
    )
    grid_x, grid_y = cp.meshgrid(coordinate, coordinate, indexing="xy")
    radius_px = cp.sqrt(grid_x * grid_x + grid_y * grid_y)
    radial_bin_index = cp.floor(radius_px).astype(cp.int32)
    radial_bin_count = int(radial_bin_index.max().item()) + 1
    radial_energy = radial_bin_sum(
        grid,
        radial_bin_index,
        radial_bin_count,
        config.accumulator_dtype,
    )
    cumulative = cp.cumsum(radial_energy)
    total = cumulative[-1]
    target = total * energy_fraction
    upper_index = int(cp.searchsorted(cumulative, target).item())
    if upper_index <= 0:
        radius_px_value = 0.0
    else:
        lower = cumulative[upper_index - 1]
        upper = cumulative[upper_index]
        if float(upper - lower) <= 0.0:
            radius_px_value = float(upper_index)
        else:
            fraction = float((target - lower) / (upper - lower))
            radius_px_value = upper_index - 1.0 + fraction
    return 2.0 * radius_px_value * pixel_arcmin


def plan_field_grid(
    shifts_arcmin: Any,
    image_width_arcmin: float,
    config: ArrayGhostSimulationConfig,
    kernel_half_width_arcmin: float | None = None,
) -> dict[str, Any]:
    """随视场大小和单孔像宽度自适应选择视场网格。

    方案第 5 节要求宽视场总览与局部高分辨分开。这里让像素尺寸在
    '解析单孔像宽度'与'装得下全部重影'之间取折中：先按单孔像宽度要求
    细分，若网格超出上限再按上限粗化，并记录每个像宽占多少像素。

    视场半宽必须同时容纳'最大孔心位移'和'单孔像核的半宽'。踩坑提示：
    只按位移留边会把最外圈孔的核尾部裁掉，实测单孔能量会损失约 0.15%，
    核半宽越大损失越明显。
    """
    if shifts_arcmin.size == 0:
        max_shift_arcmin = 0.0
    else:
        max_shift_arcmin = float(
            cp.sqrt((shifts_arcmin**2).sum(axis=1)).max()
        )
    if kernel_half_width_arcmin is None:
        kernel_half_width_arcmin = config.solar_kernel_half_width_arcmin
    half_width_arcmin = (
        max_shift_arcmin * config.field_margin_factor
        + kernel_half_width_arcmin
        + config.field_extra_margin_arcmin
    )
    required_pixel_arcmin = (
        image_width_arcmin / config.field_min_pixels_per_image_width
        if image_width_arcmin > 0.0
        else config.field_pixel_arcmin
    )
    pixel_arcmin = min(config.field_pixel_arcmin, required_pixel_arcmin)
    grid_size = int(math.ceil(2.0 * half_width_arcmin / pixel_arcmin))
    if grid_size % 2 != 0:
        grid_size += 1
    max_size = config.effective_field_grid_max_size()
    coarsened = False
    if grid_size > max_size:
        grid_size = max_size
        pixel_arcmin = 2.0 * half_width_arcmin / grid_size
        coarsened = True
    grid_size = max(grid_size, 16)
    pixels_per_width = (
        image_width_arcmin / pixel_arcmin if image_width_arcmin > 0.0 else math.nan
    )
    return {
        "grid_size": grid_size,
        "pixel_arcmin": pixel_arcmin,
        "half_width_arcmin": half_width_arcmin,
        "max_shift_arcmin": max_shift_arcmin,
        "kernel_half_width_arcmin": kernel_half_width_arcmin,
        "coarsened": coarsened,
        "pixels_per_image_width": pixels_per_width,
        "image_width_arcmin": image_width_arcmin,
    }


def circle_overlap_area_mm2(
    centre_distance_mm: float,
    radius_a_mm: float,
    radius_b_mm: float,
) -> float:
    """两个圆的交叠面积（解析式），用于瞳孔通光权重。"""
    if radius_a_mm <= 0.0 or radius_b_mm <= 0.0:
        return 0.0
    if centre_distance_mm >= radius_a_mm + radius_b_mm:
        return 0.0
    if centre_distance_mm <= abs(radius_a_mm - radius_b_mm):
        return math.pi * min(radius_a_mm, radius_b_mm) ** 2
    distance = centre_distance_mm
    alpha = math.acos(
        (distance * distance + radius_a_mm * radius_a_mm - radius_b_mm * radius_b_mm)
        / (2.0 * distance * radius_a_mm)
    )
    beta = math.acos(
        (distance * distance + radius_b_mm * radius_b_mm - radius_a_mm * radius_a_mm)
        / (2.0 * distance * radius_b_mm)
    )
    return (
        radius_a_mm * radius_a_mm * alpha
        + radius_b_mm * radius_b_mm * beta
        - 0.5
        * math.sqrt(
            max(
                0.0,
                (-distance + radius_a_mm + radius_b_mm)
                * (distance + radius_a_mm - radius_b_mm)
                * (distance - radius_a_mm + radius_b_mm)
                * (distance + radius_a_mm + radius_b_mm),
            )
        )
    )


def compute_hole_weights(
    layout: ArrayLayout,
    pupil_diameter_mm: float | None,
    config: ArrayGhostSimulationConfig,
) -> tuple[np.ndarray, str]:
    """每个孔的相对通光权重。

    方案 5.4 节的权重来源。当前实现取'轴上视角的侧向瞳孔接受度'：
    孔径为 d 的孔，其平行光束在瞳孔面上的落点半径约为 |x_i|，
    通光比例等于孔圆与瞳孔圆的交叠面积除以孔面积。

    限制：随视角变化的项（镜片到瞳孔距离乘以入射角）尚未计入，
    因此结果标记为 quasi_static_pupil；未开启时权重为 1，
    结果标记为 infinite_pupil_upper_bound（几何上限）。
    """
    hole_count = layout.hole_count
    if not config.enable_pupil_vignetting or pupil_diameter_mm is None:
        return np.ones(hole_count, dtype=np.float64), "infinite_pupil_upper_bound"

    hole_radius_mm = 0.5 * layout.diameter_mm
    pupil_radius_mm = 0.5 * pupil_diameter_mm
    hole_area_mm2 = math.pi * hole_radius_mm * hole_radius_mm
    offsets_mm = np.sqrt((layout.centers_mm**2).sum(axis=1))
    weights = np.array(
        [
            circle_overlap_area_mm2(
                float(offset_mm), hole_radius_mm, pupil_radius_mm
            )
            / hole_area_mm2
            for offset_mm in offsets_mm
        ],
        dtype=np.float64,
    )
    return weights, "quasi_static_pupil"


def accumulate_shifted_kernels(
    kernel: Any,
    kernel_pixel_arcmin: float,
    shifts_arcmin: Any,
    weights: Any,
    field: dict[str, Any],
    config: ArrayGhostSimulationConfig,
) -> dict[str, Any]:
    """把同一个单孔像核平移到每个孔位置并按强度累加。

    这是方案 0.1 节的强度叠加：PSF_array = sum_i w_i * PSF_single(r - r_i/f)，
    绝不出现复振幅求和。
    """
    field_size = int(field["grid_size"])
    field_pixel_arcmin = float(field["pixel_arcmin"])
    resampled_kernel = resample_psf_to_grid(
        kernel,
        kernel_pixel_arcmin,
        field_size,
        field_pixel_arcmin,
        config,
    )
    axis = cp.arange(field_size, dtype=cp.float32) - field_size // 2
    field_y, field_x = cp.meshgrid(axis, axis, indexing="ij")
    # 居中坐标 -> 数组下标：加中心偏移。平移量为负号方向，保持与
    # PSF_array(theta) = sum_i PSF_single(theta - r_i / f) 的定义一致。
    centre_index = field_size // 2
    accumulated = cp.zeros(
        (field_size, field_size), dtype=config.accumulator_dtype
    )
    normalized_kernel = resampled_kernel / resampled_kernel.sum(
        dtype=config.accumulator_dtype
    )
    for index in range(int(shifts_arcmin.shape[0])):
        shift_x_px = float(shifts_arcmin[index, 0]) / field_pixel_arcmin
        shift_y_px = float(shifts_arcmin[index, 1]) / field_pixel_arcmin
        sampled = map_coordinates(
            normalized_kernel,
            cp.stack(
                (
                    field_y - shift_y_px + centre_index,
                    field_x - shift_x_px + centre_index,
                ),
                axis=0,
            ),
            order=config.psf_resampling_order,
            mode="constant",
            cval=0.0,
        )
        accumulated += float(weights[index]) * sampled
    total_energy = float(accumulated.sum(dtype=config.accumulator_dtype))
    reference_weight = float(weights[0]) if weights.size else 1.0
    peak = float(accumulated.max())
    return {
        "image": accumulated.astype(config.real_dtype),
        "total_energy": total_energy,
        "expected_energy": float(weights.sum(dtype=config.accumulator_dtype)),
        "main_peak": peak,
        "reference_weight": reference_weight,
        "kernel_peak": float(normalized_kernel.max()),
        "pixel_arcmin": field_pixel_arcmin,
        "grid_size": field_size,
    }


# ============================================================================
# 实现部分 5/5：重影指标、风险标签与规则性
# ============================================================================


def cluster_ghost_positions(
    shifts_arcmin: np.ndarray,
    weights: np.ndarray,
    merge_radius_arcmin: float,
) -> list[dict[str, Any]]:
    """把落在同一个像宽内的孔心位移合并成一个重影簇。

    合并半径取 merge_radius_factor * W（W 为太阳盘卷积后的像宽）。
    为了不引入位置偏差，簇心取成员位移的加权平均，而不是取某个孔。
    """
    if shifts_arcmin.shape[0] == 0:
        return []
    radius = max(merge_radius_arcmin, 1.0e-9)
    tree = cKDTree(shifts_arcmin)
    remaining = set(range(shifts_arcmin.shape[0]))
    clusters: list[dict[str, Any]] = []
    while remaining:
        seed = remaining.pop()
        members = {seed}
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            for neighbour in tree.query_ball_point(shifts_arcmin[current], radius):
                if neighbour in remaining:
                    remaining.discard(neighbour)
                    members.add(neighbour)
                    frontier.append(neighbour)
        member_indices = sorted(members)
        member_shifts = shifts_arcmin[member_indices]
        member_weights = weights[member_indices]
        weight_sum = float(member_weights.sum())
        if weight_sum > 0.0:
            centre = (
                member_shifts * member_weights[:, None]
            ).sum(axis=0) / weight_sum
        else:
            centre = member_shifts.mean(axis=0)
        clusters.append(
            {
                "centre_arcmin": centre,
                "member_indices": member_indices,
                "multiplicity": len(member_indices),
                "total_weight": weight_sum,
                "radius_arcmin": float(
                    math.hypot(float(centre[0]), float(centre[1]))
                ),
            }
        )
    clusters.sort(key=lambda cluster: cluster["radius_arcmin"])
    return clusters


def local_peak_arcmin(
    image: Any,
    field_pixel_arcmin: float,
    centre_arcmin: np.ndarray,
    half_width_arcmin: float,
) -> float:
    """取指定角位置附近的峰值，并做抛物线亚像素修正。

    踩坑提示：直接取离散最大值会因亚像素落点产生可达 ±25% 的峰高抖动，
    足以把'等亮重影'误判成'亮度不同的重影'。这里在离散峰周围做二维
    抛物线拟合，修正到真实峰顶。
    """
    grid_size = int(image.shape[0])
    centre_index = grid_size // 2
    centre_x = int(round(centre_index + float(centre_arcmin[0]) / field_pixel_arcmin))
    centre_y = int(round(centre_index + float(centre_arcmin[1]) / field_pixel_arcmin))
    half_width_px = max(1, int(math.ceil(half_width_arcmin / field_pixel_arcmin)))
    x_start = max(0, centre_x - half_width_px)
    x_stop = min(grid_size, centre_x + half_width_px + 1)
    y_start = max(0, centre_y - half_width_px)
    y_stop = min(grid_size, centre_y + half_width_px + 1)
    if x_stop <= x_start or y_stop <= y_start:
        return 0.0
    window = image[y_start:y_stop, x_start:x_stop]
    if window.size == 0:
        return 0.0
    peak_index = int(cp.argmax(window).item())
    peak_y, peak_x = divmod(peak_index, int(window.shape[1]))
    peak_value = float(window[peak_y, peak_x])
    # 只有峰不在局部窗口边界时才能做抛物线修正。
    if (
        0 < peak_y < int(window.shape[0]) - 1
        and 0 < peak_x < int(window.shape[1]) - 1
    ):
        left = float(window[peak_y, peak_x - 1])
        right = float(window[peak_y, peak_x + 1])
        down = float(window[peak_y - 1, peak_x])
        up = float(window[peak_y + 1, peak_x])
        curvature_x = left - 2.0 * peak_value + right
        curvature_y = down - 2.0 * peak_value + up
        if curvature_x < 0.0:
            # 抛物线顶点修正：delta = -(right-left)^2 / (8 * curvature)。
            peak_value -= 0.125 * (left - right) ** 2 / curvature_x
        if curvature_y < 0.0:
            peak_value -= 0.125 * (down - up) ** 2 / curvature_y
    return peak_value


def extract_ghost_peaks(
    image: Any,
    field: dict[str, Any],
    clusters: Sequence[dict[str, Any]],
    main_peak: float,
    image_width_arcmin: float,
    config: ArrayGhostSimulationConfig,
) -> list[dict[str, Any]]:
    """每个重影簇一行：位置、重数、权重、实测峰与峰比。"""
    rows: list[dict[str, Any]] = []
    search_half_width_arcmin = max(
        config.merge_radius_factor * image_width_arcmin,
        field["pixel_arcmin"],
    )
    for index, cluster in enumerate(clusters):
        if cluster["radius_arcmin"] <= config.floating_comparison_tolerance:
            continue
        centre = cluster["centre_arcmin"]
        peak = local_peak_arcmin(
            image,
            field["pixel_arcmin"],
            centre,
            search_half_width_arcmin,
        )
        rows.append(
            {
                "ghost_index": index,
                "ghost_x_arcmin": float(centre[0]),
                "ghost_y_arcmin": float(centre[1]),
                "ghost_radius_arcmin": cluster["radius_arcmin"],
                "ghost_azimuth_deg": math.degrees(
                    math.atan2(float(centre[1]), float(centre[0]))
                )
                % 360.0,
                "multiplicity": cluster["multiplicity"],
                "total_weight": cluster["total_weight"],
                "measured_peak": peak,
                "peak_ratio": (
                    peak / main_peak if main_peak > 0.0 else math.nan
                ),
                "above_visibility_threshold": (
                    peak / main_peak >= config.visibility_peak_threshold
                    if main_peak > 0.0
                    else False
                ),
            }
        )
    return rows


def valley_visibility_between_peaks(
    image: Any,
    field_pixel_arcmin: float,
    first_center_arcmin: np.ndarray,
    second_center_arcmin: np.ndarray,
    config: ArrayGhostSimulationConfig,
) -> tuple[float, str]:
    """方案 7.2 节：两峰之间谷值的可见度。

        visibility = (I_low - I_valley) / (I_low + I_valley)
    完全分离时趋向 1，完全融合时趋向 0。
    """
    separation_arcmin = float(
        math.hypot(
            float(second_center_arcmin[0] - first_center_arcmin[0]),
            float(second_center_arcmin[1] - first_center_arcmin[1]),
        )
    )
    if separation_arcmin > config.valley_numeric_max_separation_arcmin:
        return 1.0, "not_evaluated_well_separated"
    sample_count = config.valley_profile_sample_count
    fractions = cp.linspace(0.0, 1.0, sample_count, dtype=cp.float32)
    x_arcmin = (
        float(first_center_arcmin[0])
        + fractions * float(second_center_arcmin[0] - first_center_arcmin[0])
    )
    y_arcmin = (
        float(first_center_arcmin[1])
        + fractions * float(second_center_arcmin[1] - first_center_arcmin[1])
    )
    grid_size = int(image.shape[0])
    centre_index = grid_size // 2
    sample_y = cp.asarray(y_arcmin) / field_pixel_arcmin + centre_index
    sample_x = cp.asarray(x_arcmin) / field_pixel_arcmin + centre_index
    profile = map_coordinates(
        image,
        cp.stack((sample_y, sample_x), axis=0),
        order=config.psf_resampling_order,
        mode="constant",
        cval=0.0,
    )
    # 峰之间只比较内部区段，避免把峰本身当成谷值。
    interior_start = max(1, int(0.15 * sample_count))
    interior_stop = min(sample_count - 1, int(0.85 * sample_count))
    if interior_stop <= interior_start:
        return math.nan, "profile_too_short"
    interior = profile[interior_start:interior_stop]
    peak_values = cp.array(
        [float(profile[0]), float(profile[-1])], dtype=config.real_dtype
    )
    lower_peak = float(peak_values.min())
    valley = float(interior.min())
    if lower_peak + valley <= 0.0:
        return math.nan, "zero_signal"
    return (lower_peak - valley) / (lower_peak + valley), "numeric"


def compute_image_autocorrelation_metrics(
    image: Any,
    field_pixel_arcmin: float,
    config: ArrayGhostSimulationConfig,
) -> dict[str, Any]:
    """方案 7.3 节：用自相关旁瓣衡量阵列的规则性。

    规则晶格会产生强旁瓣，随机排布更平滑。这里取中心以外、给定
    滞后半径内的最大旁瓣与中位背景之比。
    """
    grid_size = int(image.shape[0])
    centered = image - image.mean()
    spectrum = cp.fft.fft2(centered)
    autocorrelation = cp.fft.fftshift(
        cp.real(cp.fft.ifft2(spectrum * cp.conj(spectrum)))
    )
    centre_index = grid_size // 2
    peak = float(autocorrelation[centre_index, centre_index])
    if peak <= 0.0:
        return {
            "autocorrelation_peak_to_median": math.nan,
            "autocorrelation_max_offcentre_radius_arcmin": math.nan,
        }
    coordinate = cp.arange(grid_size, dtype=cp.float32) - centre_index
    grid_x, grid_y = cp.meshgrid(coordinate, coordinate, indexing="xy")
    radius_px = cp.sqrt(grid_x * grid_x + grid_y * grid_y)
    min_lag_px = config.autocorrelation_min_lag_arcmin / field_pixel_arcmin
    max_lag_px = config.autocorrelation_max_lag_arcmin / field_pixel_arcmin
    ring_mask = (radius_px >= min_lag_px) & (radius_px <= max_lag_px)
    if not bool(ring_mask.any()):
        return {
            "autocorrelation_peak_to_median": math.nan,
            "autocorrelation_max_offcentre_radius_arcmin": math.nan,
        }
    ring_values = autocorrelation[ring_mask]
    median = float(cp.median(ring_values))
    ring_normalised = ring_values / peak
    max_index = int(cp.argmax(ring_normalised).item())
    max_value = float(ring_normalised[max_index])
    ring_radii_px = radius_px[ring_mask]
    max_radius_px = float(ring_radii_px[max_index])
    return {
        "autocorrelation_peak_to_median": (
            max_value / (median / peak) if median > 0.0 else math.nan
        ),
        "autocorrelation_max_offcentre_value": max_value,
        "autocorrelation_max_offcentre_radius_arcmin": (
            max_radius_px * field_pixel_arcmin
        ),
    }


def ghost_direction_histogram(
    ghost_rows: Sequence[dict[str, Any]],
    config: ArrayGhostSimulationConfig,
) -> dict[str, Any]:
    """方案 7.3 节：重影方位角直方图，用于区分各向异性排布。"""
    bin_count = config.direction_histogram_bin_count
    counts = np.zeros(bin_count, dtype=np.float64)
    for row in ghost_rows:
        if not row["above_visibility_threshold"]:
            continue
        index = int(row["ghost_azimuth_deg"] / 360.0 * bin_count) % bin_count
        counts[index] += 1.0
    total = counts.sum()
    fractions = counts / total if total > 0.0 else counts
    nonzero = counts[counts > 0.0]
    anisotropy = (
        float(nonzero.max() / nonzero.mean()) if nonzero.size > 0 else math.nan
    )
    return {
        "direction_histogram": "|".join(f"{value:.6f}" for value in fractions),
        "direction_anisotropy": anisotropy,
        "direction_occupied_bin_count": int((counts > 0.0).sum()),
    }


def classify_risk_labels(
    layout_qc: dict[str, Any],
    metrics: dict[str, Any],
    pupil_diameter_mm: float,
    reference_weight: float,
    config: ArrayGhostSimulationConfig,
) -> tuple[str, str]:
    """方案 7.4 节：每个参数点至少一个离散风险标签。"""
    labels: list[str] = []
    if not layout_qc["geometry_ok"] or not layout_qc["topology_ok"]:
        labels.append(RISK_INVALID_GEOMETRY)
    if layout_qc["pitch_nominal_mm"] >= pupil_diameter_mm:
        # 经典 Scheiner 条件：孔距不小于瞳孔直径时，瞳孔内最多一个孔，
        # 视线落在孔间就没有主像，形成死区。
        labels.append(RISK_DEAD_ZONE)
    if reference_weight < config.pupil_reference_weight_threshold:
        labels.append(RISK_PUPIL_VIGNETTED)
    if metrics["separation_to_width_ratio"] < config.separation_ratio_threshold:
        labels.append(RISK_GHOST_MERGED)
    if metrics["max_ghost_peak_ratio"] < config.ghost_peak_ratio_threshold:
        labels.append(RISK_GHOST_WEAK)
    if (
        metrics["separation_to_width_ratio"] >= config.separation_ratio_threshold
        and metrics["max_ghost_peak_ratio"] >= config.ghost_peak_ratio_threshold
    ):
        labels.append(RISK_GHOST_RESOLVED)
    if not labels:
        labels.append(RISK_GHOST_WEAK)

    # 主标签的优先级：几何非法 > 参考孔被挡 > 重影融合 > 死区 > 重影可分辨。
    # 死区排在'可分辨'之前，是因为孔距不小于瞳孔直径时瞳孔内最多只有一个
    # 孔，参数点即使重影分得很开，也会在视线落在孔间时丢失主像。
    if RISK_INVALID_GEOMETRY in labels:
        primary = RISK_INVALID_GEOMETRY
    elif RISK_PUPIL_VIGNETTED in labels:
        primary = RISK_PUPIL_VIGNETTED
    elif RISK_GHOST_MERGED in labels:
        primary = RISK_GHOST_MERGED
    elif RISK_DEAD_ZONE in labels:
        primary = RISK_DEAD_ZONE
    elif RISK_GHOST_RESOLVED in labels:
        primary = RISK_GHOST_RESOLVED
    else:
        primary = RISK_GHOST_WEAK
    return primary, "|".join(labels)


def compute_ghost_metrics(
    image_result: dict[str, Any],
    field: dict[str, Any],
    shifts_arcmin: np.ndarray,
    weights: np.ndarray,
    image_width_arcmin: float,
    theory_first_order_angle_arcmin: float,
    config: ArrayGhostSimulationConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """方案第 7 节的最小指标集合。"""
    image = image_result["image"]
    main_peak = local_peak_arcmin(
        image,
        field["pixel_arcmin"],
        np.zeros(2, dtype=np.float64),
        max(
            config.main_window_radius_factor * image_width_arcmin,
            field["pixel_arcmin"],
        ),
    )
    merge_radius_arcmin = max(
        config.merge_radius_factor * image_width_arcmin,
        field["pixel_arcmin"],
    )
    clusters = cluster_ghost_positions(shifts_arcmin, weights, merge_radius_arcmin)
    ghost_rows = extract_ghost_peaks(
        image,
        field,
        clusters,
        main_peak,
        image_width_arcmin,
        config,
    )
    visible_rows = [
        row for row in ghost_rows if row["above_visibility_threshold"]
    ]

    first_order_angle_arcmin = (
        min((row["ghost_radius_arcmin"] for row in ghost_rows), default=math.nan)
    )
    max_peak_ratio = max(
        (row["peak_ratio"] for row in ghost_rows), default=0.0
    )
    separation_to_width_ratio = (
        first_order_angle_arcmin / image_width_arcmin
        if image_width_arcmin > 0.0 and math.isfinite(first_order_angle_arcmin)
        else math.nan
    )

    # 主像窗口外的能量占比。
    grid_size = int(image.shape[0])
    centre_index = grid_size // 2
    coordinate = cp.arange(grid_size, dtype=cp.float32) - centre_index
    grid_x, grid_y = cp.meshgrid(coordinate, coordinate, indexing="xy")
    window_radius_px = (
        config.main_window_radius_factor * image_width_arcmin / field["pixel_arcmin"]
    )
    outside_mask = cp.sqrt(grid_x**2 + grid_y**2) > window_radius_px
    total_energy = float(image.sum(dtype=config.accumulator_dtype))
    outside_energy = float(image[outside_mask].sum(dtype=config.accumulator_dtype))
    ghost_integrated_fraction = (
        outside_energy / total_energy if total_energy > 0.0 else math.nan
    )

    visible_ghosts = [
        row for row in ghost_rows if row["above_visibility_threshold"]
    ]
    if visible_ghosts:
        nearest = min(
            visible_ghosts, key=lambda row: row["ghost_radius_arcmin"]
        )
        valley_visibility, valley_status = valley_visibility_between_peaks(
            image,
            field["pixel_arcmin"],
            np.zeros(2, dtype=np.float64),
            np.array(
                [nearest["ghost_x_arcmin"], nearest["ghost_y_arcmin"]],
                dtype=np.float64,
            ),
            config,
        )
    else:
        valley_visibility = math.nan
        valley_status = "no_visible_ghost"

    autocorrelation = compute_image_autocorrelation_metrics(
        image, field["pixel_arcmin"], config
    )
    direction = ghost_direction_histogram(ghost_rows, config)
    multiplicity_values = [row["multiplicity"] for row in ghost_rows]
    pattern_signature = "|".join(
        f"{row['ghost_azimuth_deg']:.3f}@{row['peak_ratio']:.4f}"
        for row in visible_rows
    )

    metrics = {
        "ghost_count": len(visible_rows),
        "ghost_cluster_count": len(ghost_rows),
        "unique_overlap_count": sum(
            1 for value in multiplicity_values if value > 1
        ),
        "max_ghost_multiplicity": max(multiplicity_values, default=0),
        "first_order_angle_arcmin": first_order_angle_arcmin,
        "theory_first_order_angle_arcmin": theory_first_order_angle_arcmin,
        "first_order_angle_error_percent": (
            100.0
            * abs(first_order_angle_arcmin - theory_first_order_angle_arcmin)
            / theory_first_order_angle_arcmin
            if theory_first_order_angle_arcmin > 0.0
            and math.isfinite(first_order_angle_arcmin)
            else math.nan
        ),
        "max_ghost_peak_ratio": max_peak_ratio,
        "ghost_integrated_fraction": ghost_integrated_fraction,
        "sun_image_width_arcmin": image_width_arcmin,
        "separation_to_width_ratio": separation_to_width_ratio,
        "valley_visibility": valley_visibility,
        "valley_status": valley_status,
        "main_peak": main_peak,
        "image_total_energy": total_energy,
        "energy_relative_error_percent": (
            100.0
            * abs(total_energy - image_result["expected_energy"])
            / image_result["expected_energy"]
            if image_result["expected_energy"] > 0.0
            else math.nan
        ),
        "field_grid_size": field["grid_size"],
        "field_pixel_arcmin": field["pixel_arcmin"],
        "field_pixels_per_image_width": field["pixels_per_image_width"],
        "field_grid_coarsened": field["coarsened"],
        "ghost_positions": "|".join(
            f"{row['ghost_x_arcmin']:.3f}:{row['ghost_y_arcmin']:.3f}"
            for row in visible_rows
        ),
        "pattern_signature": pattern_signature,
    }
    metrics.update(autocorrelation)
    metrics.update(direction)
    return metrics, ghost_rows, clusters
#
# ============================================================================
# 17. Git 提交与推送规范：小步提交，多角度详细提交信息
# ============================================================================
#
# 本节约束编码阶段的版本控制行为，与第 1 节的硬编码禁令同级，必须执行。
# 远端为 origin（GitHub: BassttElSevic/IYPT2027-1），主分支为 main。
#
# 17.1 提交节奏：写一点就 commit，commit 完就 push
# - 每完成一个可独立说明的小单元立即提交，禁止攒到阶段结束一次性提交：
#   一个排布生成函数连同其测试、一个验证锚点、一组配置字段、
#   一张主结论图的绘图函数、一节 README，都各自构成一次提交；
# - 工作区一旦处于“测试通过、能说明白做了什么”的状态就提交；
# - 每次 commit 之后立即 push 到 origin main，不积累未推送的本地提交；
#   若 push 因网络等原因失败，恢复后第一时间补推，不让未推送提交
#   堆积超过一个工作单元；
# - 单次提交只装一个主题：排布、验证、绘图、文档分开提交，
#   不混成一笔同时改十几个文件的大杂烩；
# - 带着失败测试的提交原则上不允许；确属有意保留的 fail 验证项，
#   必须在提交信息中说明原因和对应验证编号。
#
# 17.2 提交信息：一句话标题加多角度详细正文
# - 沿用仓库现有类型前缀：feat、fix、docs、test、refactor、perf、chore；
# - 标题一行写清做了什么，不超过 72 个字符；
# - 正文必须分段从多个角度展开，至少覆盖以下六个角度：
#   1. 动机与物理依据：为什么做，对应本方案哪一节、哪个物理口径；
#   2. 实现方式：关键算法，复用了前两阶段的哪些函数，新引入了什么；
#   3. 参数与配置：新增或修改了哪些配置字段，默认值和取值依据；
#   4. 验证情况：跑了哪些测试和锚点（用 V-A 到 V-K 编号），误差量级，
#      通过项与保留的 fail 项；
#   5. 影响与限制：对既有结果的影响，尚未覆盖的物理，后续待办；
#   6. 产物清单：本次生成或更新了哪些 CSV、图、JSON、文档；
# - 参考 fa02130 的写法（方法修正加扫描规模加测试与报告），
#   但第三阶段要求每个角度单独成段，不再压缩成一句话摘要；
# - 反例：只写 update、fix bug、wip、完成布局 等一律不允许；
#   只写标题不写正文也不允许。
#
# 17.3 提交内容边界
# - 沿用前两阶段惯例：代表性的结果图、指标 CSV、配置 JSON 随里程碑
#   提交入库作为结果留档（前两阶段分别有 25 和 7 个产物文件在库）；
# - 不入库的内容：逐点扫描的海量中间数组、GPU 缓存、临时调试脚本、
#   __pycache__ 和 .venv（继续由 .gitignore 覆盖）；若第三阶段某类产物
#   体积过大，先在 .gitignore 增加规则并在提交信息中说明，
#   不要先提交再删除；
# - 逐个 git add 具名文件，不用 git add -A 把无关改动顺手带入；
# - 不顺手修改前两阶段文件；确需修改时单独提交，正文说明回归验证；
# - 提交前用 git status 和 git diff --staged 自查，确认没有临时打印、
#   注释掉的死代码和调试开关混入。
#
# 17.4 里程碑提交清单
# 第三阶段至少形成以下独立提交，各自一次或多次：
# 1. 本方案注释文件；
# 2. 配置骨架与校验（第 1 节）；
# 3. 五种排布生成与排布测试（第 2 节、第 11 节前半）；
# 4. 掩膜与 QC（第 3 节）；
# 5. 平移叠加与太阳盘积分（第 5、6 节）；
# 6. 重影指标与风险标签（第 7 节）；
# 7. 分层扫描与 d-p 关系产出（第 8、2.9 节）；
# 8. 验证锚点汇总（V-A 到 V-K）；
# 9. 第三阶段 README 报告与产物清单。
# 每个里程碑提交正文末尾附当前 V-A 到 V-K 的通过状态摘要，
# 让提交历史本身可以还原验证进度。
