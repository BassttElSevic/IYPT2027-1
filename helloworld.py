"""
verify_env.py
IYPT2027-1 环境自检 + CPU/GPU 性能对比
目标硬件: RTX 5070 Ti (Blackwell, sm_120) + CUDA 12.8
"""
import os
import warnings
warnings.filterwarnings("ignore", message="CUDA path could not be detected")
# 强制指定 CuPy 的缓存和工作目录，绕开系统 TEMP 的干扰
os.environ["CUPY_CACHE_DIR"] = r"C:\Temp\cupy_cache"
os.environ["TEMP"] = r"C:\Temp"
os.environ["TMP"] = r"C:\Temp"

# 确保目录存在
os.makedirs(r"C:\Temp\cupy_cache", exist_ok=True)
import sys
import time
import warnings

# ============================================================
# 0. 快速环境变量自检（在你那个 warning 出现之前就修好它）
# ============================================================
print("=" * 70)
print(" [0] 环境变量自检")
print("=" * 70)

cuda_path = os.environ.get("CUDA_PATH", "<未设置>")
print(f"CUDA_PATH = {cuda_path}")
if cuda_path == "<未设置>":
    print("Nyah~~~")


# ============================================================
# 1. 依赖版本检查
# ============================================================
print("\n" + "=" * 70)
print(" [1] 依赖版本检查")
print("=" * 70)

from importlib.metadata import version, PackageNotFoundError

REQUIRED = {
    "numpy":         "1.26",
    "scipy":         "1.11",
    "matplotlib":    "3.8",
    "pandas":        "2.0",
    "scikit-image":  "0.22",
    "cupy-cuda12x":  "13.4.0",   # 红线: 必须 >= 13.4.0
}

def parse_ver(v):
    """把 '1.26.4' / '13.4.0rc1' 解析成可比较的元组"""
    core = v.split("+")[0].split("rc")[0].split("a")[0].split("b")[0]
    return tuple(int(x) for x in core.split(".") if x.isdigit())

ok_all = True
for pkg, min_ver in REQUIRED.items():
    try:
        inst = version(pkg)
        ok = parse_ver(inst) >= parse_ver(min_ver)
        flag = "✅" if ok else "❌"
        if not ok:
            ok_all = False
        print(f" {flag} {pkg:<15} {inst:<15} (需要 >= {min_ver})")
    except PackageNotFoundError:
        ok_all = False
        print(f" ❌ {pkg:<15} <未安装>       (需要 >= {min_ver})")

if not ok_all:
    print("\n⚠️  有依赖缺失或版本过低，请先修复再继续。")
    sys.exit(1)


# ============================================================
# 2. CuPy 与 GPU 能力检查（关键！）
# ============================================================
print("\n" + "=" * 70)
print(" [2] CuPy / GPU 检查")
print("=" * 70)

try:
    import cupy as cp
except ImportError as e:
    print(f"❌ 无法导入 cupy: {e}")
    sys.exit(1)

print(f"CuPy 版本: {cp.__version__}")

try:
    n_gpu = cp.cuda.runtime.getDeviceCount()
    if n_gpu == 0:
        raise RuntimeError("未检测到 CUDA 设备")
    print(f"检测到 {n_gpu} 个 GPU 设备")
except Exception as e:
    print(f"❌ CUDA 运行时初始化失败: {e}")
    print("   请检查: 驱动版本 >= 570, CUDA 12.8 runtime 是否可用")
    sys.exit(1)

# 打印设备详细信息
for dev_id in range(n_gpu):
    props = cp.cuda.runtime.getDeviceProperties(dev_id)
    name       = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
    cc_major   = props["major"]
    cc_minor   = props["minor"]
    mem_gb     = props["totalGlobalMem"] / 1024**3
    print(f"\n  GPU {dev_id}: {name}")
    print(f"    计算能力 (CC): sm_{cc_major}{cc_minor}")
    print(f"    显存: {mem_gb:.2f} GB")

    # Blackwell (RTX 50 系) 应该是 sm_120
    if cc_major == 12:
        print(f"    → ✅ 识别为 Blackwell 架构 (sm_{cc_major}{cc_minor})")
    elif cc_major >= 9:
        print(f"    → ⚠️  非预期架构 (预期 sm_120)，但可能仍可用")
    else:
        print(f"    → ⚠️  较旧架构")

# 冒烟测试：确保真的能跑 kernel（而不是回退）
try:
    a = cp.array([1.0, 2.0, 3.0])
    b = a * 2 + 1
    cp.cuda.Stream.null.synchronize()
    result = cp.asnumpy(b)
    assert (result == [3.0, 5.0, 7.0]).all()
    print("\n  ✅ CUDA kernel 冒烟测试通过")
except Exception as e:
    print(f"\n  ❌ CUDA kernel 执行失败: {e}")
    print("    如果报 'no kernel image is available'，说明 CuPy 编译时未包含 sm_120，")
    print("    请升级到 cupy-cuda12x >= 13.4.0")
    sys.exit(1)


# ============================================================
# 3. CPU vs GPU 速度对比（BL-ASM 风格 FFT 传播）
# ============================================================
print("\n" + "=" * 70)
print(" [3] CPU vs GPU 性能对比 (BL-ASM 风格角谱传播)")
print("=" * 70)

import numpy as np
import scipy.fft as sfft

def blasm_cpu(N, wavelength, dx, z, n_iter=5):
    """CPU 版带限角谱法 (Band-Limited Angular Spectrum Method)"""
    k = 2 * np.pi / wavelength
    fx = np.fft.fftfreq(N, d=dx)
    FX, FY = np.meshgrid(fx, fx, indexing="xy")
    # 倏逝波掩膜
    arg = 1.0 - (wavelength * FX)**2 - (wavelength * FY)**2
    H = np.exp(1j * k * z * np.sqrt(np.maximum(arg, 0))) * (arg > 0)

    # 用 mask（带限）作为输入，模拟真实的 IYPT 场景
    x = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, x, indexing="xy")
    mask = ((X**2 + Y**2) < 0.5).astype(np.complex64)

    t0 = time.perf_counter()
    U = mask
    for _ in range(n_iter):
        U = sfft.ifft2(sfft.fft2(U) * H)
    _ = U.sum()
    return time.perf_counter() - t0


def blasm_gpu(N, wavelength, dx, z, n_iter=5):
    """GPU 版 BL-ASM"""
    k = 2 * np.pi / wavelength
    fx = cp.fft.fftfreq(N, d=dx)
    FX, FY = cp.meshgrid(fx, fx, indexing="xy")
    arg = 1.0 - (wavelength * FX)**2 - (wavelength * FY)**2
    H = cp.exp(1j * k * z * cp.sqrt(cp.maximum(arg, 0))) * (arg > 0)

    x = cp.linspace(-1, 1, N)
    X, Y = cp.meshgrid(x, x, indexing="xy")
    mask = ((X**2 + Y**2) < 0.5).astype(cp.complex64)

    # 预热（含 JIT、分配缓存等）
    _ = cp.fft.ifft2(cp.fft.fft2(mask) * H).sum()
    cp.cuda.Stream.null.synchronize()

    t0 = time.perf_counter()
    U = mask
    for _ in range(n_iter):
        U = cp.fft.ifft2(cp.fft.fft2(U) * H)
    _ = U.sum()
    cp.cuda.Stream.null.synchronize()   # 必须！否则计时只是 launch 时间
    return time.perf_counter() - t0


# ---- 参数（典型 IYPT BL-ASM 场景，N=2048 已能压满 GPU）----
N         = 8192
wavelength = 532e-9      # 绿光 532nm
dx        = 5e-6         # 采样 5um
z         = 0.5          # 传播 0.5m
n_iter    = 10

print(f"\n参数: N={N}, λ={wavelength*1e9:.0f}nm, dx={dx*1e6:.1f}μm, z={z}m, iter={n_iter}")
print("-" * 70)

# --- CPU 计时 ---
print("[CPU] 运行中...", end=" ", flush=True)
t_cpu = blasm_cpu(N, wavelength, dx, z, n_iter)
print(f"完成  ->  {t_cpu*1000:8.2f} ms")

# --- GPU 计时 ---
print("[GPU] 运行中...", end=" ", flush=True)
t_gpu = blasm_gpu(N, wavelength, dx, z, n_iter)
print(f"完成  ->  {t_gpu*1000:8.2f} ms")

# --- 结果 ---
print("-" * 70)
speedup = t_cpu / t_gpu
print(f"加速比: {speedup:.2f}x")
if speedup > 5:
    print("✅ GPU 显著加速，CuPy 正常工作")
elif speedup > 1.5:
    print("⚠️  GPU 略快，但可能未充分发挥（检查是否选对了 GPU）")
else:
    print("❌ GPU 未生效！CuPy 可能回退到了 CPU，请检查 CUDA 安装")

# 再报一个纯 FFT 的微基准（更能反映硬件差距）
print("\n--- 附: 纯 FFT 微基准 (N=4096, 20 次) ---")
a_cpu = np.random.rand(4096, 4096).astype(np.complex64)
a_gpu = cp.asarray(a_cpu)

t0 = time.perf_counter()
for _ in range(20):
    _ = sfft.fft2(a_cpu)
t_fft_cpu = time.perf_counter() - t0

_ = cp.fft.fft2(a_gpu); cp.cuda.Stream.null.synchronize()
t0 = time.perf_counter()
for _ in range(20):
    _ = cp.fft.fft2(a_gpu)
cp.cuda.Stream.null.synchronize()
t_fft_gpu = time.perf_counter() - t0

print(f"CPU FFT2: {t_fft_cpu*1000:8.2f} ms  ({t_fft_cpu/20*1000:.2f} ms/call)")
print(f"GPU FFT2: {t_fft_gpu*1000:8.2f} ms  ({t_fft_gpu/20*1000:.2f} ms/call)")
print(f"FFT 加速比: {t_fft_cpu/t_fft_gpu:.2f}x")

print("\n" + "=" * 70)
print(" 自检结束")
print("=" * 70)