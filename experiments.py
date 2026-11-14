"""Опыты и дополнительные задания. python experiments.py [--voice мой_голос.wav]"""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import correlate, find_peaks, lfilter

from speech_pipeline import BLUE, ORANGE, TEAL, SR, N_FFT, analyze, axes, load_or_generate, plt, quantize, save


def write_csv(folder, name, heading, rows):
    with (folder / name).open("w", encoding="utf-8-sig", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(heading)
        writer.writerows(rows)


def formant_demo(path, folder):
    x = load_or_generate(path)
    a = analyze(x)
    # LPC на предыскаженной речи. Пропускаем тихие и непериодические кадры.
    data = lfilter([1, -0.97], [1], x)
    tracks = np.full((len(a["frames"]), 3), np.nan)
    rms = np.sqrt(np.mean(a["frames"] ** 2, axis=1))
    for i, frame in enumerate(a["frames"]):
        if rms[i] < max(rms.max() * 0.09, 1e-8):
            continue
        raw_ac = correlate(frame - frame.mean(), frame - frame.mean(), mode="full")[len(frame)-1:]
        if np.max(raw_ac[SR//400:SR//75]) < 0.48 * raw_ac[0]:
            continue
        start = i * a["h"]
        seg = data[start:start+a["n"]] * a["window"]
        ac = correlate(seg, seg, mode="full")[len(seg)-1:]
        order = 16
        ac[0] *= 1.000001
        coef = solve_toeplitz(ac[:order], -ac[1:order+1])
        roots = np.roots(np.r_[1, coef])
        roots = roots[roots.imag > 0]
        f = np.angle(roots) * SR / (2*np.pi)
        bw = -SR / np.pi * np.log(np.abs(roots))
        poles = sorted(f[(f > 220) & (f < 4200) & (bw > 0) & (bw < 420)])
        if len(poles) >= 3 and poles[0] < 1100 and poles[1] < 3300:
            tracks[i] = poles[:3]
    fig, ax = axes(height=4.0)
    p = a["power"]
    db = 10*np.log10(np.maximum(p / max(p.max(), 1e-20), 1e-7))
    ax.pcolormesh(a["centers"], a["freqs"], db, shading="nearest",
                  cmap="gray_r", vmin=-65, vmax=0, rasterized=True)
    for i, marker in enumerate(("o", "^", "s")):
        ax.plot(a["centers"], tracks[:, i], marker, ls="none", ms=3.4,
                mfc=(BLUE, ORANGE, TEAL)[i], mec="white", mew=0.45, label=f"F{i+1}")
    ax.set(ylim=(0, 4800), xlabel="Время, с", ylabel="Частота, Гц",
           title="Оценки формант в озвученных кадрах")
    ax.legend(ncol=3, loc="upper right")
    save(fig, folder, "formants")
    write_csv(folder, "formants.csv", ["t_s", "F1_Hz", "F2_Hz", "F3_Hz"],
              zip(a["centers"], *tracks.T))
    return a, tracks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice", type=Path, default=Path(__file__).with_name("voice.wav"))
    parser.add_argument("--out", type=Path, default=Path("experiments"))
    args = parser.parse_args()
    folder = args.out
    folder.mkdir(parents=True, exist_ok=True)
    x = load_or_generate(None).astype(float)
    configs = [(25, 10, 26), (25, 10, 40), (30, 10, 26), (25, 15, 26)]
    runs = [analyze(x, *config) for config in configs]
    base = runs[0]
    rows = []
    for (frame, hop, m), a in zip(configs, runs):
        rows.append([frame, hop, m, a["n"], a["h"], len(a["centers"]),
                     round(100 * (1 - a["h"] / a["n"]), 3), a["features"].nbytes])
    write_csv(folder, "parameters.csv", ["frame_ms", "hop_ms", "mel_filters", "N", "H",
              "frames", "overlap_percent", "features_bytes"], rows)

    fig, axs = plt.subplots(2, 2, figsize=(9.2, 5.0), constrained_layout=True,
                            sharex=True)
    for ax, (frame, hop, m), a in zip(axs.flat, configs, runs):
        # Единая шкала логарифмов и одна временная область во всех опытах.
        im = ax.pcolormesh(a["centers"], np.arange(m), a["log_mel"],
                           shading="nearest", cmap="viridis", vmin=-12, vmax=9)
        ax.set(xlim=(1.34, 1.66), title=f"{frame} мс  /  {hop} мс  /  {m} полос",
               ylabel="Мел-полоса")
    for ax in axs[-1]:
        ax.set_xlabel("Время, с")
    fig.colorbar(im, ax=axs, label="ln(энергия)")
    save(fig, folder, "parameter_comparison")

    bits = np.array([4, 8, 12, 16])
    measured = [10*np.log10(np.mean(x*x)/np.mean((x-quantize(x, b))**2)) for b in bits]
    theory = 6.02 * bits + 1.76
    fig, ax = axes()
    ax.plot(bits, theory, "o--", color=BLUE, label="Полномасштабная синусоида, теория")
    ax.plot(bits, measured, "s-", color=ORANGE, label="Тестовый сигнал, измерение")
    ax.set(xticks=bits, xlabel="Разрядность, бит", ylabel="SNR, дБ", title="Влияние разрядности")
    ax.legend()
    save(fig, folder, "snr")
    write_csv(folder, "snr.csv", ["bits", "measured_dB", "theory_dB"], zip(bits, measured, theory))

    fig, axs = plt.subplots(2, 2, figsize=(9, 4.7), constrained_layout=True,
                            sharex=True, sharey=True)
    for ax, m in zip(axs.flat, (13, 26, 40, 80)):
        a = analyze(x, n_mels=m)
        im = ax.pcolormesh(a["centers"], np.arange(13), a["mfcc"], shading="nearest",
                           cmap="RdBu_r", vmin=-60, vmax=60)
        ax.set(title=f"{m} мел-фильтров", ylabel="Индекс MFCC")
    for ax in axs[-1]:
        ax.set_xlabel("Время, с")
    fig.colorbar(im, ax=axs, label="MFCC (шкала ±60)")
    save(fig, folder, "mfcc_by_filters")

    n = base["n"]
    fig, ax = axes()
    response = []
    for label, w in [("Прямоугольное", np.ones(n)), ("Хэмминга", base["window"])]:
        spec = np.abs(np.fft.rfft(w, n=65536))
        db = 20 * np.log10(np.maximum(spec / spec[0], 1e-8))
        minima = find_peaks(-db)[0]
        highest_side = db[minima[0]:].max()
        response.append([label, minima[0]*SR/65536, highest_side])
        ax.plot(np.fft.rfftfreq(65536, 1/SR), db, color=BLUE if label == "Прямоугольное" else ORANGE,
                ls="--" if label == "Прямоугольное" else "-", label=label)
    ax.set(xlim=(0, 400), ylim=(-85, 2), xlabel="Смещение частоты, Гц",
           ylabel="дБ от максимума", title="Частотные характеристики окон, N = 400")
    ax.legend()
    save(fig, folder, "windows")
    write_csv(folder, "windows.csv", ["window", "first_min_Hz", "highest_sidelobe_dB"], response)
    formant_demo(args.voice, folder)


if __name__ == "__main__":
    main()
