"""Представление речевого сигнала. Запуск: python speech_pipeline.py --help"""
import argparse
from math import gcd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.fft import dct
from scipy.io import wavfile
from scipy.signal import resample_poly

SR, N_FFT, N_MFCC = 16_000, 512, 13
BLUE, ORANGE, TEAL = "#28618D", "#BB541E", "#087969"


def load_or_generate(path, seed=42):
    if path is None:
        t = np.arange(2 * SR) / SR
        x = sum(np.sin(2 * np.pi * 140 * k * t) / k for k in range(1, 12))
        x *= 0.5 * (1 - np.cos(np.pi * t)) + 0.2
        mask = (t > 1.4) & (t < 1.6)
        noise = np.random.default_rng(seed).normal(0, 0.15, len(t))
        x[mask] += noise[mask]
    else:
        rate, raw = wavfile.read(path)
        x = raw.astype(float)
        if raw.dtype == np.uint8:
            x -= 128  # ноль беззнакового 8-битного PCM
        if x.ndim == 2:
            x = x.mean(axis=1)
        if rate != SR:
            divisor = gcd(SR, rate)
            x = resample_poly(x, SR // divisor, rate // divisor)
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError("Аудиозапись пуста или содержит неконечные значения.")
    peak = np.max(np.abs(x))
    return (x / peak if peak else x).astype(np.float32)


def quantize(x, bits):
    step = 2.0 / (2 ** bits)
    return np.clip(step * np.rint(x / step), -1, 1 - step)


def mel_filterbank(freqs, count):
    top_mel = 2595 * np.log10(1 + freqs[-1] / 700)
    points = 700 * (10 ** (np.linspace(0, top_mel, count + 2) / 2595) - 1)
    left, center, right = points[:-2, None], points[1:-1, None], points[2:, None]
    rise = (freqs - left) / (center - left)
    fall = (right - freqs) / (right - center)
    bank = np.maximum(0, np.minimum(rise, fall))
    if not np.all(bank.max(axis=1) > 0):
        raise ValueError("Есть пустая мел-полоса. Уменьшите число фильтров.")
    return points, bank


def delta(feat, radius=2):
    padded = np.pad(feat, ((0, 0), (radius, radius)), mode="edge")
    length = feat.shape[1]
    out = np.zeros_like(feat)
    for offset in range(1, radius + 1):
        left, right = radius - offset, radius + offset
        out += offset * (padded[:, right:right+length] - padded[:, left:left+length])
    return out / (2 * sum(d * d for d in range(1, radius + 1)))


def analyze(x, frame_ms=25, hop_ms=10, n_mels=26):
    n, h = round(SR * frame_ms / 1000), round(SR * hop_ms / 1000)
    if not (2 <= n <= N_FFT and 1 <= h <= n and len(x) >= n):
        raise ValueError("Нужен полный кадр: 2 ≤ длина ≤ 512, 1 ≤ шаг ≤ длина.")
    if not N_MFCC <= n_mels <= N_FFT // 2:
        raise ValueError("Число мел-полос должно быть между 13 и 256.")
    frames = sliding_window_view(x, n)[::h]
    window = np.hamming(n)  # симметричное окно с делителем N - 1
    freqs = np.fft.rfftfreq(N_FFT, 1 / SR)
    stft = np.fft.rfft(frames * window, n=N_FFT, axis=1).T
    power = np.abs(stft) ** 2
    points, bank = mel_filterbank(freqs, n_mels)
    mel = bank @ power
    log_mel = np.log(mel + 1e-10)
    mfcc = dct(log_mel, type=2, norm="ortho", axis=0)[:N_MFCC]
    d1 = delta(mfcc)
    d2 = delta(d1)
    features = np.vstack((mfcc, d1, d2)).astype(np.float32)
    centers = (np.arange(len(frames)) * h + (n - 1) / 2) / SR
    return dict(n=n, h=h, frames=frames, window=window, freqs=freqs,
                stft=stft, power=power, points=points, bank=bank, mel=mel,
                log_mel=log_mel, mfcc=mfcc, d1=d1, d2=d2,
                features=features, centers=centers)


def save(fig, folder, name):
    fig.savefig(folder / (name + ".png"), dpi=155, facecolor="white")
    plt.close(fig)


def axes(rows=1, height=3.0, sharex=False):
    return plt.subplots(rows, 1, figsize=(8.8, height), sharex=sharex,
                        constrained_layout=True)


def heat(ax, data, extent, cmap, scale, label, **kwargs):
    im = ax.imshow(data, origin="lower", aspect="auto", extent=extent,
                   interpolation="nearest", cmap=cmap, **kwargs)
    ax.figure.colorbar(im, ax=ax, pad=0.02, label=scale)
    ax.set(xlabel="Время, с", ylabel=label)


def figures(x, a, folder):
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "lines.linewidth": 1.2})
    t = np.arange(len(x)) / SR
    n, h = a["n"], a["h"]
    freqs, centers = a["freqs"], a["centers"]
    times = [centers[0] - h / (2 * SR), centers[-1] + h / (2 * SR)]
    df = SR / N_FFT
    frame, win = a["frames"][0], a["window"]

    fig, (ax, sub) = axes(2, 4.2)
    ax.plot(t, x, lw=0.65, color=BLUE)
    ax.set(xlabel="Время, с", ylabel="Амплитуда", title="Осциллограмма")
    start = min(1000, max(len(x) - 240, 0))
    sel = slice(start, start + 240)
    sub.plot(1000 * t[sel], x[sel], ".-", lw=0.65, ms=3, color=BLUE)
    sub.set(xlabel="Время, мс", ylabel="Амплитуда", title="Отдельные отсчёты")
    save(fig, folder, "01_sampling")

    fig, ax = axes()
    start = min(2000, max(len(x) - 200, 0))
    sel = slice(start, start + 200)
    q = quantize(x[sel], 3)
    for level in np.arange(-1, 1, 0.25):
        ax.axhline(level, color="0.80", lw=0.7, ls=":")
    ax.plot(1000 * t[sel], x[sel], color=BLUE, label="Исходный сигнал")
    ax.step(1000 * t[sel], q, where="mid", color=ORANGE, ls="--", label="3 бита")
    ax.set(xlabel="Время, мс", ylabel="Амплитуда", title="Квантование, шаг 0,25")
    ax.legend(loc="upper right")
    save(fig, folder, "02_quantization")

    fig, (ax, sub) = axes(2, 3.9, sharex=True)
    count = min(6, len(a["frames"]))
    end = (count - 1) * h + n
    ax.plot(1000 * t[:end], x[:end], color=BLUE)
    ax.set(ylabel="Амплитуда", title=f"Кадры с перекрытием {100*(1-h/n):.1f}%")
    for k in range(count):
        sub.barh(k, 1000 * n / SR, left=1000 * k * h / SR, color=plt.cm.tab10(k), edgecolor="0.25")
    sub.set(xlabel="Время, мс", ylabel="Кадр", yticks=np.arange(count))
    sub.invert_yaxis()
    save(fig, folder, "03_framing")

    fig, (ax, sub) = axes(2, 3.9, sharex=True)
    ax.plot(win, color=BLUE)
    ax.set(ylabel="Вес", title="Симметричное окно Хэмминга")
    sub.plot(frame, color=BLUE, label="Исходный кадр")
    sub.plot(frame * win, color=ORANGE, label="После окна")
    sub.set(xlabel="Отсчёт в кадре", ylabel="Амплитуда")
    sub.legend(ncol=2)
    save(fig, folder, "04_window")

    fig, (ax, sub) = axes(2, 4.2)
    for weights, style, label in [(np.ones(n), "--", "Без окна"), (win, "-", "Хэмминг")]:
        mag = np.abs(np.fft.rfft(frame * weights, n=N_FFT))
        ax.plot(freqs, mag, color=BLUE if style == "--" else ORANGE, ls=style, label=label)
        db = 20 * np.log10(np.maximum(mag / max(mag.max(), 1e-12), 1e-6))
        sub.plot(freqs, db, color=BLUE if style == "--" else ORANGE, ls=style, label=label)
    ax.set(xlim=(0, 2000), ylabel="Модуль БПФ", xlabel="Частота, Гц",
           title="Первый кадр до и после применения окна")
    ax.legend(ncol=2)
    sub.set(xlim=(0, SR/2), xlabel="Частота, Гц", ylabel="дБ от своего пика")
    save(fig, folder, "05_fft_frame")

    fig, ax = axes()
    power = a["power"]
    rel = power / max(power.max(), 1e-20)
    heat(ax, 10*np.log10(np.maximum(rel, 1e-8)), times + [-df/2, SR/2+df/2],
         "magma", "дБ от максимума", "Частота, Гц", vmin=-80, vmax=0)
    ax.set(ylim=(0, SR/2), title="Спектрограмма мощности")
    save(fig, folder, "06_spectrogram")

    fig, ax = axes()
    for k, curve in enumerate(a["bank"]):
        ax.plot(freqs, curve, color=(BLUE, ORANGE, TEAL)[k % 3], lw=1.0)
    ax.set(title=f"Мел-банк, {len(a['bank'])} полос", xlabel="Частота, Гц",
           ylabel="Вес", xlim=(0, SR/2))
    save(fig, folder, "07_mel_filterbank")

    for key, title, label, name, cmap in [
        ("log_mel", "Логарифмы энергий мел-полос", "Мел-полоса", "08_mel_spectrogram", "viridis"),
        ("mfcc", "Мел-частотные кепстральные коэффициенты", "Индекс MFCC", "09_mfcc", "RdBu_r"),
    ]:
        fig, ax = axes()
        data = a[key]
        limit = np.max(np.abs(data)) or 1.0
        bounds = dict(vmin=-limit, vmax=limit) if key == "mfcc" else {}
        heat(ax, data, times + [-0.5, len(data)-0.5], cmap,
             "Значение" if key == "mfcc" else "ln(энергия)", label, **bounds)
        ax.set_title(title)
        save(fig, folder, name)

    fig, axs = axes(3, 4.6, sharex=True)
    for i, ax in enumerate(axs):
        ax.plot(np.arange(len(centers)), a["mfcc"][i], color=(BLUE, ORANGE, TEAL)[i])
        ax.set_ylabel(f"c{i}")
    axs[0].set_title("Динамика первых трёх MFCC")
    axs[-1].set_xlabel("Номер кадра (с нуля)")
    save(fig, folder, "10_mfcc_dynamics")

    fig, axs = axes(2, 4.3, sharex=True)
    for ax, key, title in zip(axs, ("d1", "d2"), ("Первая дельта", "Вторая дельта")):
        data = a[key]
        limit = np.max(np.abs(data)) or 1.0
        heat(ax, data, times + [-0.5, N_MFCC-0.5], "RdBu_r", "Значение",
             "Индекс MFCC", vmin=-limit, vmax=limit)
        ax.set_title(title)
    save(fig, folder, "11_deltas")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", type=Path, help="WAV с речью")
    parser.add_argument("--synthetic", action="store_true", help="Учебный сигнал")
    parser.add_argument("--frame-ms", type=float, default=25)
    parser.add_argument("--hop-ms", type=float, default=10)
    parser.add_argument("--n-mels", type=int, default=26)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    nearby = Path(__file__).with_name("speech.wav")
    source = args.wav or (nearby if nearby.exists() else None)
    if args.synthetic and args.wav:
        parser.error("Выберите один источник: --synthetic или --wav.")
    if args.synthetic:
        source = None
    x = load_or_generate(source)
    result = analyze(x, args.frame_ms, args.hop_ms, args.n_mels)
    figures(x, result, args.out)
    np.save(args.out / "features.npy", result["features"])
    print("Источник:", source or "учебный генератор, seed=42")
    print(f"{len(x)} отсчётов, {len(x)/SR:.3f} с, {SR} Гц")
    print(f"Кадр {result['n']}, шаг {result['h']}, кадров {len(result['centers'])}")
    for key in ("stft", "mel", "mfcc", "features"):
        print(f"{key}: {result[key].shape}")
    print("Графики 01–11 и features.npy:", args.out)


if __name__ == "__main__":
    main()
