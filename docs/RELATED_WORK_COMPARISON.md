# Comparison with the closest projects (including Hitachi's)

**Written:** 23 October 2026. **For:** the project team, to prepare the 50% review.
**How to trust this file:** our numbers come from our result files (path given). Hitachi's numbers are their published Table 3, typed into `acoustic/compare_hitachi.py`. Numbers for other projects come from `docs/LITERATURE_SURVEY.md`, which says for each paper whether I opened it or only saw a listing. Where I could not check a number, the table says so. **Nothing here ranks us above or below other projects**, because almost none of them use the same data, test or metric (explained in section 3).

---

## 1. The only fair comparison: Hitachi, same dataset (MIMII)

Hitachi is the one project that used exactly our sound data. AUC = chance that a faulty clip scores higher than a healthy one (0.5 = guessing, 1.0 = perfect). Each cell is the average over the same four machines (IDs 00, 02, 04, 06) of that machine type.

| Machine | Noise | Hitachi published AUC | Our copy of Hitachi's model (`baseline_vs_hitachi.md`) | Our final system: small model + calibration (`final_summary.md`) |
|---|---|---|---|---|
| Fan | 6 dB | 0.91 | 0.93 | 0.92 |
| Fan | 0 dB | 0.80 | 0.82 | 0.83 |
| Fan | −6 dB | 0.66 | 0.68 | 0.71 |
| Pump | 6 dB | 0.81 | 0.76 | 0.96 |
| Pump | 0 dB | 0.71 | 0.73 | 0.84 |
| Pump | −6 dB | 0.66 | 0.68 | 0.75 |
| Slider | 6 dB | 0.88 | 0.89 | 0.94 |
| Slider | 0 dB | 0.78 | 0.78 | 0.84 |
| Slider | −6 dB | 0.70 | 0.72 | 0.72 |
| Valve | 6 dB | 0.67 | 0.71 | 0.56 |
| Valve | 0 dB | 0.61 | 0.63 | 0.52 |
| Valve | −6 dB | 0.56 | 0.57 | 0.51 |

**What the table says, in plain words**
- Our copy of Hitachi's model matches their published numbers (correlation 0.97 over all 48 cases, mean difference +0.013). This shows our pipeline is correct.
- Our final system is **clearly better on pumps** (0.96 vs 0.81 at 6 dB) and **somewhat better on sliders** at 6 dB and 0 dB. Fans are about equal.
- **Valves are worse than Hitachi's** (0.56 vs 0.67 at 6 dB). We do not beat them here, and our valve model does not work.
- The gain at low noise levels for fans and sliders is small (a few points) and one run per case can move a number by that much (fan ID 00 at 6 dB gave 0.80, 0.81, 0.75 on three seeds).

## 2. Why this is still not a fully equal comparison

| | Hitachi | Ours |
|---|---|---|
| Model trained on | **that same machine's** own healthy recordings, per machine | the **other three machines** only, then adapted to the new machine |
| Anything learned from the test machine | its healthy recordings during training | 60 healthy clips (about 10 minutes): 30 for feature numbers, 30 for the alert level |
| Model size | 50,760 parameters | 2,408 to 6,856 parameters |
| Runs per case | average of 3 | 3 seeds for the final table |
| Metric used | AUC | AUC, precision, recall, F1, false-alarm rate |

So we answer a slightly different question ("a machine the model has never heard, after 10 minutes of listening") from Hitachi's ("a model trained on this machine"). Say this when you show the table.

---

## 3. Other projects: what they report (NOT directly comparable)

These projects use different data, tasks or metrics, so their numbers cannot be put in a ranking against ours. They show what is known and where we sit.

| Project | What it is | Score they report | Checked? | How it differs from us |
|---|---|---|---|---|
| **OutlierNets** (Abbasi et al., 2021) | Very small autoencoders for machine-sound anomaly detection | Smallest model 686 parameters, 2.7 KB; accuracy comparable to much larger published models; up to 21× lower CPU latency | Abstract read. Dataset and hardware not named on the abstract page | Closest to our "tiny model" result. No adaptation to a new machine, no vibration. Not tested on a low-cost chip according to the deck |
| **Kolok et al., 2025** (Sensors) | ESP32 + MPU6050 + INMP441 vibration monitoring | ROC-AUC 0.78, accuracy 0.73 (the paper also quotes 93.4%) | Opened. Paper is inconsistent | Vibration only (the microphone is a design idea); Isolation Forest on a laptop; one fault type (imbalance). The same hardware family as ours |
| **MIMII DUE** (Tanabe et al., 2021) | Same family of sound dataset with changes in machine conditions | "Large performance gaps" between source and target conditions | Abstract read; no numbers noted | Shows that changed conditions are a known, hard problem, which supports our need for calibration |
| **AdaFlow** (Yamaguchi et al., 2019) | Adapts a density model to a new data distribution using adaptive batch normalisation, aimed at limited compute | No numbers noted | Abstract (via search summary) | **Closest idea to our calibration.** Do not claim our method is new without reading it |
| **Few-shot adaptation** (Chen et al., 2022) | Adapts to new machine conditions from a few target samples (meta-learning) | No numbers noted | Abstract | Very close to "adapt with few healthy clips"; a heavier method |
| **Almaini et al., 2026** | MFCC plus small classifier on IoT sensors | 91% accuracy | Abstract | Uses **UrbanSound8K, which is environmental sound, not machine sound**. Not comparable at all |
| **Jakobsen, 2024** (HardwareX) | Low-cost sensor with MEMS microphone and accelerometers, LoRa/BLE | No detector score | Opened | Hardware paper; shows both sensors see bearing-defect frequencies. Supports our sensor choice |
| **"Modified Overcomplete Autoencoder … TinyML"** (IEEE 10684143) | USB-fan broken-blade faults on an Arduino | 99.23% accuracy, about 17 kB (per a search summary) | **Listing only, not opened** | Very close to our fan-on-a-small-chip work. Open it first before citing |

**Rows in our deck that I could not verify** (they are in the literature table on slides 5–7). Do not quote their numbers until you have opened the papers:

| Deck row | Number in the deck | Status |
|---|---|---|
| Arciniegas et al. 2025 (MPU6050 + ESP32-S3, TinyML) | 25 ms inference, about 300 ms alert | Not re-checked by me |
| Garay et al. 2026 (Syntiant NDP120, audio + vibration + thermal) | F1 = 0.98 | Not re-checked |
| Selis et al. 2025 (RP2040, domain shift) | AUC 0.65–0.78 | Not re-checked |
| Lo et al. 2024 (pruned autoencoder for MCU) | "favorable accuracy", no number | The deck itself says the AUC is not in the abstract |
| **J. Chen et al., IEEE TIM 2024, lightweight 1D-CNN** | 98.2% accuracy, under 50 KB, 18 ms | **I could not find this paper earlier.** The project notes say it was excluded. Check whether it is still on the slide |
| **H. Zhang et al., Sensors 2025, noise-robust autoencoder** | ROC-AUC 0.954 | **Could not be found earlier; excluded.** Same check |

---

## 4. What you can fairly say from this

- "On MIMII, our copy of Hitachi's model reproduces their published results. Our final system, tested on machines it has never heard, is better on pumps (0.96 vs 0.81 at 6 dB), somewhat better on sliders, about equal on fans, and worse on valves."
- "Our models are 7 to 21 times smaller than Hitachi's (2,408–6,856 vs 50,760 parameters)."
- "Tiny autoencoders for machine sound already exist (OutlierNets, 686 parameters). What we add is adapting to a new machine from about 10 minutes of healthy sound, and a leave-one-machine-out test that reports where it fails."

## 5. What you must not say

- "Better than other projects": we only share data and test with Hitachi, and not even the same question.
- "State of the art": not shown.
- Any number from the unverified deck rows, or from papers you have not opened.
- "First" or "novel method": AdaFlow and few-shot adaptation are close.

## 6. Before presenting, if time allows

1. Open the "Modified Overcomplete Autoencoder" paper and OutlierNets in full and note the dataset and the hardware each used.
2. Read AdaFlow's method and write one sentence on how ours differs.
3. Check the two deck rows marked above and remove them if the papers cannot be found.
