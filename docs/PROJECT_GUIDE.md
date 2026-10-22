# PROJECT GUIDE — read this first

**Project title (as in the proposal):** Multimodal Edge AI for Acoustic-Vibration-Based Fault Detection and Thermal Monitoring for Environmental Impact Assessment

**Team:** Nikitha D, Rajika Saha, Monalisa, Sunand P Chowdary — Dayananda Sagar University, Dept. of CSE. Guide: Dr. Meenakshi Malhotra.

**Status date:** 21 October 2026 (evening). Written for someone who is new to the project and wants to understand every part from scratch: what it is, why each piece exists, what each code file does, how we checked our work, what is wrong or unfinished, and what happens next.

**How to read this document.** Everything is marked as one of:
- **DONE** – finished and checked.
- **RUNNING** – a computation is in progress right now.
- **PLAN** – decided or proposed but not built.
- **PROBLEM** – something we found that is wrong or weak.

Where I give a number, it comes from a file in this folder (path given) unless it says "estimate".

---

## 1. The project in plain words

Small factories run fans, pumps, motors and slide rails. When one of these machines starts to fail, nobody notices until it breaks. The breakdown stops production, can start a fire or injure someone, and a failing machine wastes electricity and gives off extra noise and heat.

Our idea is a **small, cheap device that sits on a machine, listens to it and feels its vibration, learns what "normal" looks like for that particular machine, and raises an alert when the machine starts to behave abnormally.** It should work on its own, without the cloud, and it should also estimate the extra energy and noise a fault causes (the environmental part).

Why this matters for the funding proposal (KSCST, Karnataka): the device is aimed at small Indian factories (MSMEs), where cost and complexity are the main barriers to condition monitoring.

**The honest "novelty" in one paragraph.** We did not invent a new algorithm. Other researchers already showed pieces of this idea (sound-based fault detection, small neural networks on microcontrollers, learning each machine's normal behaviour). What we add is bringing the pieces together in one low-cost device that works on the machine itself, is tested on a real motor set-up, and also estimates the extra energy and noise caused by a fault. Do not call it "the first" anything.

---

## 2. The big picture (architecture)

```
   MICROPHONE (INMP441)                VIBRATION SENSOR (MPU6050)
        |  sound samples (16,000 per second)       |  vibration readings
        v                                          v
 +-------------------+                    +--------------------+
 | SOUND FEATURES    |                    | VIBRATION FEATURES |
 | 64 numbers every  |                    | 3 numbers (RMS)    |
 | 32 ms ("log-mel") |                    | every 32 ms        |
 +---------+---------+                    +----------+---------+
           |                                         |
           v                                         v
   +----------------+                       +----------------+
   | AUTOENCODER    |                       | (simple check) |
   | (small neural  |                       |                |
   |  network)      |                       |                |
   +-------+--------+                       +--------+-------+
           | "how strange does this sound?" score    | vibration score
           +------------------+----------------------+
                              v
                    COMBINE THE TWO SCORES   <-- open design question (section 10)
                              v
        COMPARE WITH THE MACHINE'S OWN ALERT LEVEL  <-- set by CALIBRATION (section 7)
                              v
                  GREEN LED (normal) / RED LED (alert)
                              +
      also record noise level and power use -> estimate extra energy/noise cost of a fault
```

Everything in the boxes is meant to run on an **ESP32-S3 microcontroller** (a small, cheap chip). Today **nothing runs on the chip** — we have no hardware yet. All work so far is on a laptop with real recordings.

Planned hardware: ESP32-S3, INMP441 microphone, MPU6050 vibration sensor, LEDs; for the environmental part also a power sensor (INA219 for DC, or an AC energy meter) and a reference sound-level meter to check noise readings. Thermal (temperature) sensing is **future work**; it is in the title but not in the build.

---

## 3. Concepts you need (explained from zero)

**Sound as numbers.** A microphone measures air pressure. A computer records it by measuring many times per second. Our recordings are 16,000 measurements per second ("16 kHz"), each stored as a whole number ("16-bit"). Ten seconds therefore equals 160,000 numbers per microphone.

**Channels.** The recording device that made our dataset had 8 microphones in a ring. All 8 recorded the same 10 seconds at the same time and are stored together in one file. We use only **channel 1**, because our device will have only one microphone.

**Spectrogram / log-mel features.** A raw waveform is hard to learn from. We cut the sound into short overlapping pieces of 1024 samples (64 milliseconds), moving forward 512 samples (32 ms) each time. For each piece we find how much energy there is at each pitch (using the Fourier transform), group the pitches into **64 bands** (the "mel" bands, narrow at low pitch and wide at high pitch), and take a logarithm so loud and quiet parts are comparable. The result: **64 numbers per 32 ms**. We then join **5 neighbouring pieces** into one input of **320 numbers**. One 10-second clip gives **309 such inputs**.

**Autoencoder (our "model").** A small neural network that squeezes its 320 input numbers down to just 8 and then tries to rebuild all 320. It is trained **only on healthy sound**, so it gets good at rebuilding normal sound. When something abnormal comes in, it rebuilds badly. The **reconstruction error** (how far the rebuilt numbers are from the real ones) is the "how strange is this?" score. Layer sizes: 320 → 64 → 64 → 8 → 64 → 64 → 320, which is **50,760 learned numbers** ("parameters").

**Why train only on healthy sound.** In a real factory you cannot record every possible fault in advance. Healthy sound is plentiful. This is called unsupervised (or one-class) anomaly detection.

**AUC.** A single number that says how well the score separates normal from abnormal clips. It equals the chance that a randomly chosen abnormal clip gets a higher "strange" score than a randomly chosen normal one. 0.5 means a coin flip, 1.0 means perfect, and below 0.5 means the model is backwards. It does not depend on where you put the alert level.

**Alert level (threshold) and the two kinds of mistake.** The device must choose a score above which it raises the alarm. A **false alarm** is flagging a healthy machine. A **missed fault** is staying quiet on a faulty one. We report the false-alarm rate (FPR) and the detection rate (TPR) at the device's own alert level. Our rule is: **alert level = average healthy score + 3 × the spread (standard deviation) of healthy scores**.

**Calibration.** A real factory has many different machines. Each has its own loudness and sound character. Calibration means: when the device is installed on a new machine, it listens to that machine while it is healthy for a few minutes, learns that machine's own normal levels, and sets its own alert level. Details and the problems we found are in section 7.

**Standardizing (z-score).** Subtract the average and divide by the spread, feature by feature. It re-expresses the numbers as "how many spreads away from this machine's usual value".

**Quantization (INT8).** The model's numbers are normally 32-bit decimals. Quantization stores them as 8-bit whole numbers. The model becomes about 4 times smaller and often faster on a chip, at some cost in accuracy. It is optional: the chip can run the model in the normal 32-bit form too.

**Edge vs cloud.** "Edge" means the processing happens on the device next to the machine. "Cloud" means data is sent over the internet to a remote computer. We want edge, with no cloud.

**Fusion.** Combining two sensors. *Early fusion* joins their features into one long input (64 sound + 3 vibration = 67 numbers). *Late fusion* scores each sensor separately and combines the decisions.

---

## 4. The data

### 4.1 MIMII (sound) — Hitachi's public dataset
- Four machine types: **fan, pump, slider (slide rail), valve**. Each has several individual machines called "IDs". The public release has IDs **00, 02, 04, 06** (the paper describes 7; the other three were not released).
- Each ID has **normal** and **abnormal** recordings, each 10 seconds long, 16 kHz, 16-bit, 8 channels.
- Real factory background noise was mixed in at three levels: **6 dB** (machine louder than noise, easiest), **0 dB** (equal), **−6 dB** (noise louder than machine, hardest). The *same clips* appear at all three levels with different noise mixed in, so never train on one level and test on another as if it were new data.
- Total: **54,057 WAV files** (18,019 per noise level: 14,719 normal and 3,300 abnormal). Our inventory counts match Hitachi's published table for all 16 type/ID combinations.
- Location on this laptop: `N:\dev\Sonar-acoustic-dataset\<snr>_dB_<type>\<type>\id_XX\{normal,abnormal}\*.wav`
- Extracted features: `N:\dev\mimii_features\<snr>_<type>_<id>_<label>.npy` (96 files).

### 4.2 CWRU (vibration) — bearing test rig
Four recordings only: `97.mat` (healthy), `105.mat` (inner-race fault), `118.mat` (ball fault), `130.mat` (outer-race fault), all sampled at 12 kHz, in `dataset/cwru/`. This is a thin dataset: one healthy recording (about 20 s), three 10 s fault recordings, one fault size and one load. See section 9 for the problems.

### 4.3 Dataset we plan to use for real sound+vibration fusion (not downloaded yet)
Jung et al., *Data in Brief* 2023 (Mendeley Data, DOI 10.17632/ztmf3m7h5x.6, licence CC BY 4.0): one rotating-machine rig with a microphone (values in pascals), vibration sensors, motor current and temperature recorded together, with bearing, misalignment and unbalance faults at several severities and three loads. Notes: the microphone data was reportedly recorded at no load only, and its total size is unknown until downloaded.

---

## 5. Code files, one by one

All paths are relative to `C:\dev\capstone`. The working environment is Python 3.13 with PyTorch (CPU), librosa 1.0, scikit-learn and NumPy.

### 5.1 Dataset inspection
**`dataset/inventory_mimii.py`** — DONE. Walks the MIMII folders, counts files per (noise level, machine type, ID, normal/abnormal), and opens every WAV header to check the format. Writes `dataset/mimii_inventory.csv`. Result: 54,057 files, all 16 kHz / 8-channel / 16-bit / 10.0 s, none corrupt, counts match the paper.

### 5.2 Feature extraction (sound)
**`acoustic/features.py`** — DONE. The heart of the sound pipeline. It contains:
- `load_channel0(path)`: reads a WAV and keeps only microphone 1 (the file is read as a table of 160,000 rows by 8 columns, and column 0 is kept).
- `logmel(y)`: turns the 160,000 samples into a 64 × 313 table (64 bands × 313 time steps). It calls one librosa function (`melspectrogram`) with Hitachi's settings: window 1024, hop 512, 64 bands, power 2, default Hann window; then takes `10·log10(energy + tiny)`.
- `stack_frames(lm)`: joins 5 neighbouring time steps into 320-number inputs, giving 309 inputs per clip.
- Constants and helpers for file paths.
We checked it by recomputing one frame by hand (window, FFT, mel table, log): it matches the library to about 4 millionths.

**`acoustic/extract_features.py`** — DONE. Runs `logmel` on every WAV in parallel (8 workers, about 30 minutes) and saves one `.npy` array per (noise level, type, ID, label), shape (clips, 64, 313). It skips files that already exist, so it can be re-run safely.

**`acoustic/plot_spectrograms.py`** — DONE. Makes the pictures in `acoustic/plots/`: normal vs abnormal spectrograms for each machine type, the same clip at three noise levels, and average spectrum comparisons. Used for understanding the data, not for results.

**`acoustic/export_csv.py`** — DONE. Exports one clip's features to CSV (`acoustic/csv_samples/`) so you can look at them in Excel. Usage: `python export_csv.py --snr 6 --type fan --id id_00 --label normal --index 0`.

### 5.3 The Hitachi baseline (our validation)
**`acoustic/baseline_ae.py`** — DONE. Defines the autoencoder (`build_model`), the train/evaluation split (`split_files`), training (`train_model`), scoring (`per_file_scores`) and `run_one`, which trains and evaluates one machine. Details of how it copies Hitachi are in section 6.

**`acoustic/run_baseline.py`** — DONE. Runs the whole grid: 4 types × 4 IDs × 3 noise levels = 48 models (3 in parallel), writing `acoustic/results/baseline_results.csv`. Resumable.

**`acoustic/compare_hitachi.py`** — DONE. Holds Hitachi's published AUC table for IDs 00/02/04/06, compares it with our results, and writes `acoustic/results/baseline_vs_hitachi.md` and `acoustic/plots/baseline_vs_hitachi.png`.

### 5.4 Calibration experiments
**`acoustic/calibration_experiment.py`** — DONE (version 1; has known problems, see section 7.3). Results in `acoustic/results/calibration_results.csv`.

**`acoustic/calibration_v2.py`** — DONE (version 2 with fixes and controls, section 7.4). Writes `acoustic/results/calibration_v2_results.csv`; log in `acoustic/results/calibration_v2_run.log`. All 48 machine/noise-level cases finished.

**`acoustic/summarize_calibration_v2.py`** — DONE (ready). Turns the v2 CSV into a plain table: `python summarize_calibration_v2.py 30 6` (30 calibration clips, 6 dB).

**`acoustic/model_selection.py`** — DONE (section 7B). Compares 12 model designs under the calibration set-up: stage `tune` (tuning machines only), stage `bands` (32/128 mel bands for each winner), stage `final` (one test on the sealed machine). Results: `acoustic/results/model_selection_tune.csv`, `model_selection_final.csv`.
**`acoustic/summarize_selection.py`** — DONE. Applies the pre-set selection rule and writes `model_selection_choice.csv` (stage 1) and `model_selection_choice_final.csv` (stage 2) plus readable summaries (`model_selection_stage1_summary.md`, `model_selection_stage2_summary.md`).
**`acoustic/demo.py`** — DONE. The live demo (a simulation on the laptop): trains the chosen small model on the other machines of a type, calibrates on the new machine's healthy clips, then replays shuffled normal and faulty clips from the original WAV files and prints a GREEN/RED verdict for each, with a summary and a picture in `acoustic/demo_output/`. Models are cached in `acoustic/demo_models/`. Example: `python demo.py --type fan --held id_06 --clips 16`; add `--play 3` to hear the first three clips on the speakers.
**`acoustic/demo_gui.py`** — DONE. A small window version of the demo (Python's built-in tkinter, no installs): choose the machine type and the machine the device is installed on, press "Calibrate device", then "Choose WAV file..." (or random healthy / faulty clip) to get a big GREEN/RED verdict, the score against the alert level, the clip's spectrogram, a history table and a "Play this clip" button. It warns if the chosen healthy clip was used for calibration. Run: `python demo_gui.py`.
**`acoustic/noise_mismatch.py`** and **`acoustic/summarize_noise_mismatch.py`** — DONE (section 7C). Calibrate at one noise level, test at another; results in `acoustic/results/noise_mismatch_results.csv` and a readable table in `acoustic/results/noise_mismatch_summary.md`.
**`acoustic/extract_bands.py`** — DONE. Extracts log-mel features with 32 or 128 bands (6 dB only) into `N:\dev\mimii_features_b32` / `_b128`.

### 5.5 Vibration
**`dataset/clean_cwru_dataset.py`** — LEGACY, has PROBLEMS (section 9.2); now carries a warning banner. Cleans the CWRU vibration signals (remove offset, band-pass 10–5000 Hz, remove spikes), cuts them into 32 ms frames and computes RMS, peak-to-peak, kurtosis and crest factor. Its cleaning functions are re-used by the new script.

**`dataset/cwru_vibration_v2.py`** — DONE (section 9.3). The clean replacement: only the real drive-end and fan-end channels, a time-ordered split, a low-bandwidth "MPU-like" sensor path, a sensor-noise test, and a no-network detector. Writes `dataset/cwru_vibration_v2_results.csv`, `dataset/cwru_vibration_v2_summary.md` and the picture `dataset/cwru/processed_v2/vibration_v2_overview.png`. Runs in about 11 seconds: `python cwru_vibration_v2.py` (from the `dataset` folder).

### 5.6 Older files — LEGACY, do not trust
**`train_edge_autoencoder.py`** and **`export_edge_model_c_header.py`** come from the first phase of the project, when the "sound" half of the data was **simulated** (random numbers shaped to look like sound). Those results were invalid and were deleted. These two scripts remain; they train a 67 → 32 → 16 → 32 → 67 model and export it as a C header. They depend on a pipeline that no longer exists, so they must be rebuilt before use.

### 5.7 Reference and documents
- `sampleandfulldataset/mimii_baseline-master/` — Hitachi's original baseline code (Keras). We read it and re-implemented it; we did not run it.
- `ppts/` — presentation PDFs, the MIMII paper (arXiv 1909.09347), the KSCST themes PDF, the literature survey (`Lit_Survey.pdf`), and the proposal versions (`... - v5.docx` is the latest).
- `docs/PROJECT_GUIDE.md` — this file.

---

## 6. How we copied the Hitachi plan, step by step

**Why copy it at all?** Hitachi published results for exactly this task. If our own pipeline gets similar numbers, we know our data handling, features and evaluation are correct. If it did not, we would have a bug. This is the single most valuable check in the project.

**What Hitachi did (paper + their `baseline.py` and `baseline.yaml`), and what we did:**

| Step | Hitachi | Ours |
|---|---|---|
| Channel | first microphone only | same (`load_channel0`) |
| Features | log-mel: 1024 window, 512 hop, 64 bands, power 2; `20/power·log10(mel+eps)` | same, same constants |
| Input | 5 frames joined → 320 numbers | same (`stack_frames`) |
| Model | dense 320-64-64-8-64-64-320, ReLU, linear output | same layers |
| Training | Adam, MSE loss, 50 epochs, batch 512, 10% validation (last 10%), final epoch kept | same; Adam learning rate 0.001, Xavier-uniform start with zero biases to mimic Keras defaults |
| One model per | machine type and ID | same |
| Split | sort the healthy clips; the first N (N = number of abnormal clips) go to testing, the rest to training | same rule |
| Test set | N healthy + all abnormal clips | same |
| Score of a clip | average reconstruction error over its inputs | same |
| Metric | AUC per type and ID | same |

**What is different (so we do not claim it is identical):** we use PyTorch, Hitachi used Keras/TensorFlow; librosa 1.0 vs their 0.6; their numbers are averages of 3 runs, ours is 1 run per case (3 runs only for one case); weight start-up values are an imitation of Keras. We never ran Hitachi's script.

**Result of the copy (48 cases; `acoustic/results/baseline_vs_hitachi.md`):**
- Across all 48 type/ID/noise cases, our AUCs and Hitachi's published ones have **correlation 0.97**, mean difference **+0.013**, mean absolute difference **0.028**.
- Average AUC over the 4 IDs, ours vs Hitachi's for the same 4 IDs:

| Type | 6 dB | 0 dB | −6 dB |
|---|---|---|---|
| Fan | 0.93 / 0.91 | 0.82 / 0.80 | 0.68 / 0.66 |
| Slider | 0.89 / 0.88 | 0.78 / 0.78 | 0.72 / 0.70 |
| Pump | 0.76 / 0.81 | 0.73 / 0.71 | 0.68 / 0.66 |
| Valve | 0.71 / 0.67 | 0.63 / 0.61 | 0.57 / 0.56 |

- Run-to-run spread: fan ID 00 at 6 dB gave 0.80, 0.81 and 0.75 for three random seeds (Hitachi: 0.75). So single-case gaps of a few points, even up to 0.1–0.16 for pump ID 06, are plausibly random variation. This explanation has not been tested for every case.

**What the baseline tells us about the project:** fans are easiest, then sliders, pumps, valves (valves are barely better than chance in noise). More noise lowers every score. The deck's ">0.90 AUC" target is only realistic for fans in moderate noise.

---

## 7. Calibration — the mechanism, the experiment, the problems

### 7.1 The problem calibration solves
Different machines of the same type sound different (loudness, pitch pattern). Suppose we train a model on three fans and install the device on a fourth. Even when the fourth fan is perfectly healthy, its sound differs from the training fans. The model reconstructs it badly, the "strange" score is high, and the device sounds the alarm constantly. Calibration is meant to fix that without retraining.

### 7.2 What calibration does, step by step (the mechanism)
When the device is installed on a new machine (assumed healthy at that moment):
1. **Listen** for a few minutes (we test 10, 30 and 100 ten-second clips).
2. **Learn this machine's normal feature levels:** for each of the 320 features, the average and the spread over those clips.
3. **Standardize** every future sound clip with those numbers (feature minus this machine's average, divided by this machine's spread). Now this machine's healthy sound sits near zero, like every other machine's healthy sound does after the same treatment.
4. **Set the alert level:** score some further healthy clips and use *average + 3 × spread* of those scores as the alert level.
5. From then on, a clip scoring above the alert level raises the alarm.

*Tiny illustration with made-up numbers (not real data):* a healthy fan's feature "energy in band 12" averages −30 with spread 2. A reading of −29 becomes (−29 − (−30)) / 2 = 0.5 (normal). A reading of −22 becomes 4.0 (very unusual). A louder, different fan whose normal level is −24 would look unusual to a model trained on the first fan — unless it is standardized with its own average (−24), which makes its normal readings come out near 0 too.

*A subtle point:* shifting the alert level alone never changes AUC, because AUC only looks at the ranking of scores. So we also measure false-alarm and detection rates at the alert level. Standardizing the features, on the other hand, *does* change the scores and the ranking.

### 7.3 Version 1 of the experiment (finished, flawed)
`acoustic/calibration_experiment.py`, results in `acoustic/results/calibration_results.csv`, 6 dB only, one random seed, 30 training rounds.

**Design ("leave one machine out"):** for each machine type and each ID in turn, hide that ID, train the model on the other three IDs, then treat the hidden ID as a brand-new machine. Its healthy clips are split into evaluation clips (Hitachi's split), calibration clips (never in the evaluation set) and so on. Three setups are compared on the new machine:
- **M0, no calibration:** features standardized with the training machines' pooled numbers; alert level from the training machines' healthy scores.
- **M1, threshold only:** same as M0, but the alert level comes from the new machine's own healthy clips.
- **M2, full calibration:** the model is trained with each machine standardized separately; the new machine's own healthy clips supply both its feature numbers and its alert level.

**Headline results of v1** (average over the 4 machines of each type, 30 calibration clips):

| Type | per-machine Hitachi-style AUC | M0 AUC | M2 AUC | M2 false alarms / detected |
|---|---|---|---|---|
| Fan | 0.93 | 0.55 | 0.95 | 4% / 70% |
| Pump | 0.76 | 0.46 | 0.95 | 42% / 89% |
| Slider | 0.89 | 0.78 | 0.97 | 5% / 74% |
| Valve | 0.71 | 0.43 | 0.55 | 5% / 4% |

**The three PROBLEMS with v1 (found by checking my own work):**
1. **A control was missing.** Hitachi's baseline does *not* standardize features and M2 does. The gain might come from standardizing, not from the neural network. To know, we need (a) a simple detector with no network (distance from the machine's own healthy numbers), and (b) a per-machine model that also standardizes.
2. **The alert level was set on the same clips used to standardize.** The model looks better on clips it was tuned to, so the alert level came out too low and the false-alarm rate was too high on some machines (pump ID 04: 99%, pump ID 06: 51%).
3. **The "no calibration" comparison is weak.** In 6 of 16 cases its AUC was below 0.5 (abnormal clips looked *more* normal than healthy ones). A weak baseline makes calibration look better than it is.
Also: one seed and one noise level only, and more calibration clips sometimes gave a *lower* AUC, which suggests random variation. Valves are not helped by calibration at all.

I reran one case (fan, ID 02) from scratch and got identical values to four decimals, so the numbers are real and reproducible. What is uncertain is the *explanation*, not the numbers.

### 7.4 Version 2 (DONE — final results below)
`acoustic/calibration_v2.py`, results in `acoustic/results/calibration_v2_results.csv`.

**Fixes:**
- Calibration clips are split: the first K clips give the feature numbers; the **next, different K clips** give the alert level.
- **Control S (simple, no neural network):** score = average squared standardized distance from the machine's own healthy numbers; same clips and same alert-level rule.
- **Control O (per-machine "best case"):** a model trained on the new machine's *own* healthy clips, standardized with its own numbers. A real device cannot train per machine, but it shows the ceiling.
- **Three seeds** for 6 dB (0, 1, 2), then one seed each at 0 dB and −6 dB.

**How to read the results when they arrive:**
- If M2 is clearly better than M0/M1 → calibration helps (as defined in the proposal: healthy numbers + alert level).
- If S (no network) is about as good as M2 → the gain is from the standardizing, and the neural network adds little. That would still be a valid finding but it changes how we describe the project.
- If M2 is close to O → calibration recovers most of what per-machine training would give. That would support the "adapts without retraining" claim.
- Valves: expect weak results everywhere; report them honestly.

**Status:** finished on 21 October 2026 (6 dB with 3 seeds, 0 dB and −6 dB with 1 seed; 48 cases in total).

**Final results** — AUC, average over the 4 machines of each type, 30 calibration clips (about 5 minutes of healthy sound). "none" = no calibration; "calib" = full calibration; "simple" = the no-network control; "best case" = the per-machine model.

| Type | Noise | none | calib | simple | best case |
|---|---|---|---|---|---|
| Fan | 6 dB | 0.53 | 0.95 | 0.88 | 0.95 |
| Fan | 0 dB | 0.47 | 0.86 | 0.73 | 0.83 |
| Fan | −6 dB | 0.47 | 0.73 | 0.61 | 0.70 |
| Pump | 6 dB | 0.47 | 0.95 | 0.98 | 0.74 |
| Pump | 0 dB | 0.47 | 0.88 | 0.92 | 0.63 |
| Pump | −6 dB | 0.51 | 0.79 | 0.76 | 0.63 |
| Slider | 6 dB | 0.80 | 0.97 | 0.88 | 0.89 |
| Slider | 0 dB | 0.76 | 0.86 | 0.78 | 0.80 |
| Slider | −6 dB | 0.68 | 0.77 | 0.67 | 0.74 |
| Valve | 6 dB | 0.45 | 0.55 | 0.47 | 0.66 |
| Valve | 0 dB | 0.42 | 0.50 | 0.49 | 0.58 |
| Valve | −6 dB | 0.41 | 0.47 | 0.49 | 0.55 |

(Source: `acoustic/results/calibration_v2_results.csv`. Seed-to-seed spread at 6 dB was only about ±0.00–0.02; 0 dB and −6 dB are single-seed.)

**What the results show:**
- Calibration (rescaling features with the new machine's own healthy numbers, then setting the alert level) is what makes detection work on a new machine. Without it the model is near chance on fans, pumps and valves.
- Setting only the alert level fixes false alarms but leaves detection very low (about 17–38% of faults found at 6 dB), because the model's scores are still dominated by machine-to-machine differences.
- **Fans and sliders:** full calibration is at or above the per-machine best case at all three noise levels and above the simple no-network control. The neural network adds real value here.
- **Pumps:** the simple no-network control is as good as or better than the network at 6 and 0 dB. The credit goes to the rescaling step.
- **Valves:** unsolved at every noise level (AUC about 0.5, best case 0.55–0.66). Valve sound is short, sparse clicks that an average over 10 seconds hides.
- **Noise lowers everything.** For fans at −6 dB the calibrated detector finds only 23% of faults (about 3% false alarms) at its alert level.
- **Calibration length:** at 6 dB, AUC hardly changes between 10, 30 and 100 calibration clips (fans 0.96 / 0.95 / 0.94). About 100 seconds of healthy sound is enough to rescale the features. The alert level is more sensitive: with 10 clips false alarms are higher (pumps 38%); with 100 clips they fall to about 3% but fewer faults are caught (fans 77% down to 59%). So calibration length trades false alarms against missed faults.
- Caveats: the "best case" model trains on fewer clips than the Hitachi baseline did, so it may be slightly understated; only one dataset (MIMII, channel 1); 0 dB and −6 dB use a single seed; not yet tested on the chip.

### 7.5 What we can and cannot claim from calibration
- **Can say:** "On a new machine, a few minutes of healthy sound is enough to make detection work for fans, pumps and sliders at all three noise levels, matching a per-machine model for fans and sliders; the benefit shrinks in heavy noise, and valves are not solved."
- **Cannot say:** that the neural network is essential for pumps; that it works for valves; that it works in heavy noise without loss; that it works on real hardware or other datasets.

---

## 7B. Model selection (DONE)

**Question.** The first baseline copied Hitachi's model (50,760 parameters, 5 stacked frames, 64 bands). A chip prefers a smaller model. Which design should each machine type use?

**Protocol (fixed before looking at results, so the answer cannot be tuned to the test).**
- Per machine type, IDs 00, 02 and 04 are the *tuning* machines; ID 06 is *sealed* and never used to choose anything.
- Tuning: each tuning machine is held out in turn as a "new machine"; the model trains on the other two tuning machines (each standardised with its own numbers) and is evaluated with full calibration (feature numbers from 30 healthy clips, alert level from the next 30 clips). 6 dB, one seed, 30 training rounds.
- 12 candidate designs: frames stacked 1, 3 or 5; deep network (hidden 64 or 32, bottleneck 8 or 16), a deck-style shallow network (32-16-32), and a tiny network (16-8-16).
- Selection rule: per machine type, the smallest model whose mean tuning AUC is within 0.02 of the best (0.02 is about the seed-to-seed spread seen earlier).
- Stage 2: the winner of each type rebuilt with 32 and 128 mel bands (same rule).
- Final: train on all three tuning machines, calibrate on the sealed machine's healthy clips, evaluate once (3 seeds, 6 dB), next to the Hitachi-style design for comparison.

**Findings from the tuning stage (mean AUC over the three tuning folds).**
- Model size barely matters for fans, pumps and sliders: every design is within about 0.02–0.05 of the best. A 2,408-parameter tiny network matches Hitachi's 50,760-parameter one on pumps.
- Stacking 5 frames is not needed: single-frame inputs do about as well.
- More mel bands gives a small gain (for example pumps 0.925 / 0.954 / 0.971 with 32 / 64 / 128 bands) but costs more parameters.
- Valves: no design works (best mean AUC 0.57), so the choice for valves carries no information.

**Final choice per machine type.**

| Type | Design | Mel bands | Frames | Parameters | Mean tuning AUC |
|---|---|---|---|---|---|
| Fan | 2 hidden layers of 32, bottleneck 8 | 64 | 1 | 6,856 | 0.891 |
| Pump | tiny 16-8-16 | 64 | 1 | 2,408 | 0.954 |
| Slider | shallow 32-16-32 | 32 | 1 | 3,184 | 0.938 |
| Valve | shallow 32-16-32 (arbitrary) | 64 | 1 | 5,264 | 0.571 |

**Sealed-machine test (ID 06, 6 dB, 3 seeds, 30 calibration clips).**

| Type | Hitachi-style design (50,760 params) | Chosen design |
|---|---|---|
| Fan | AUC 1.00, false alarms 11%, faults found 100% | AUC 1.00, false alarms 7%, found 100% (6,856 params) |
| Pump | AUC 0.98, false alarms 44%, found 100% | AUC 0.97, false alarms 42%, found 100% (2,408 params) |
| Slider | AUC 0.97, false alarms 2%, found 50% | AUC 1.00, false alarms 2%, found 97% (3,184 params) |
| Valve | AUC 0.61, false alarms 2%, found 3% | AUC 0.52, false alarms 1%, found 2% (5,264 params) |

**What this means.**
- The chosen small models match the large baseline on fans and pumps, and beat it on sliders, with 7 to 21 times fewer parameters. At 32-bit these are about 10–27 KB (about 2.4–6.9 KB at INT8).
- A single 64-band frame per 32 ms is enough, so the chip does not need the 5-frame buffer (the slider model needs only 32 bands).
- Valves remain unsolved; the "worse" valve result on the sealed machine is within what an arbitrary choice would give.

**Caveats (read these).**
- The sealed test uses ONE machine per type, so differences of a few points are within noise. Fan and pump are effectively tied with the baseline; the slider gain may be partly luck.
- **Pumps still have about 42% false alarms** at the device's alert level even though AUC is 0.97. That means the alert-level rule (average + 3 spreads from 30 clips) is wrong for pumps. Using 100 calibration clips lowered pump false alarms to about 3% earlier but also lowered detection. This is a concrete open problem.
- Only 6 dB was used for selection and the sealed test. 0 dB and −6 dB were not re-tested with the chosen designs.
- Honest note on "sealed": ID 06 was a held-out machine in the earlier calibration experiments, so its general behaviour was known to us, but no model choice used it.

---

## 7C. What if the noise changes after calibration? (DONE)

**Words first.** Three separate moments involve noise: (1) *training* — the model learns from healthy sound of other machines; (2) *calibration* — at installation the device learns this machine's normal levels in the noise present at that moment; (3) *monitoring* — every day afterwards, in whatever noise is present then. **Matched noise** means the noise at calibration and at monitoring is about the same; **mismatched** means they differ (for example calibrating on a quiet Sunday and monitoring on a loud Monday).

**Experiment** (`acoustic/noise_mismatch.py`): for each machine type and each held-out machine (4 per type), the device is calibrated at one noise level (6, 0, −6 dB, or a mix of the three) and tested at each noise level. Two models were used: trained only at 6 dB, and trained on all three levels. The small chosen designs with 64 mel bands, 30 calibration clips, one seed.

**Result (model trained at 6 dB; mean over fan, pump and slider; false alarms / faults found, AUC in brackets):**

| Situation | Result |
|---|---|
| Matched noise (calibrated and tested in the same noise) | 8% / 41% (0.83) |
| Calibrated quiet (6 dB), tested in more noise | **44% / 73%** (0.79): healthy clips constantly raise alerts |
| Calibrated noisy (−6 dB), tested in less noise | 11% / 40% (0.81): fewer false alarms but faults are missed |
| Calibrated on a mix of noise levels | 11% / 41% (0.82) |

The model trained on all noise levels behaved almost the same (matched 11% / 44%), so training noise barely matters.

**Per machine type, worst and best cases (false alarms / faults found):**
- Fan: calibrated at 6 dB, tested at −6 dB: 94% / 95% (useless, the device alerts on nearly everything); matched at −6 dB: 3% / 17%.
- Pump: calibrated at 6 dB, tested at −6 dB: 87% / 94%; matched at −6 dB: 3% / 20%.
- Slider: calibrated at 6 dB, tested at −6 dB: 64% / 98%; matched at −6 dB: 2% / 4%.
- Valve: nothing works at any combination (AUC about 0.5).

**What it means.**
- Calibrate in the noise the machine will really run in. Calibrating in a quiet period and then monitoring in noise floods the device with false alarms.
- Calibrating on a *mix* of noise levels avoids that flood and costs little on average (about the same faults found as matched calibration), so it is a practical mitigation, but it was only tested here on three fixed noise levels.
- In heavy noise (−6 dB, noise louder than the machine) sound alone finds few faults even when calibration matches (about 4–20% found). This is the case where the vibration sensor, which airborne noise does not affect, should help; that is still a hypothesis to be tested on real paired data.
- Not tested: real Indian factory noise, noise that varies during the day, sudden loud bursts, and our own microphone and placement. Do not claim robustness to noise.

---

## 8. Sound feature extraction on the microcontroller (PLAN)

Goal: the chip must turn raw microphone samples into exactly the same 320-number inputs that the model saw in training.

**Specification (identical to the Python code, so that results transfer):**
1. Receive samples from the microphone at 16 kHz in a rolling buffer.
2. Every 512 new samples (32 ms), take the latest 1024 samples.
3. Multiply by the Hann window (a table of 1024 numbers).
4. Fourier transform (1024-point real FFT) → 513 values.
5. Square them → energy.
6. Multiply by the 64-band mel table (a sparse table of weights, generated once from librosa and stored in the code).
7. `10·log10(energy + tiny)` → 64 numbers.
8. Keep the last 5 such rows → one 320-number input every 32 ms (the first one appears about 0.19 s after start).
9. Standardize with the calibration numbers, run the model, compute the reconstruction error.

**Rough sizes (estimates, not measured on a chip):** Hann table ≈ 4 KB, one frame buffer ≈ 4 KB (32-bit) or 2 KB (16-bit), the 5-row buffer ≈ 1.3 KB, model ≈ 200 KB (32-bit) or ≈ 50 KB (INT8), model work ≈ 50,000 multiply-adds per input. The chip has 512 KB of fast memory, 8 MB flash and 8 MB extra memory, so this should fit comfortably; it is still unproven.

**Update after model selection (section 7B):** the selected models take a single 64-band frame (slider: 32 bands), not five stacked frames. So steps 8 and the 5-row buffer above are not needed: the chip standardises one frame of 64 (or 32) numbers every 32 ms and runs a model of 2.4k–6.9k parameters on it. The rest of the specification is unchanged. If a type ends up using 32 bands, the mel table is regenerated with 32 bands.

**Plan to build and verify without hardware:** write this pipeline in portable C, feed it recorded WAV files, and compare its 320-number outputs with the Python ones on many clips (target: differences below a small tolerance). Later, moving to the ESP32 is mostly recompiling plus reading the real microphone. A script will generate the constant tables from librosa so they never drift from the Python version. If model selection changes the frame size or number of bands, the tables are regenerated.

---

## 9. Vibration and fusion

### 9.1 What exists
`dataset/clean_cwru_dataset.py` cuts CWRU vibration into 32 ms frames and keeps three numbers per frame: the RMS (average strength) of the drive-end, fan-end and base sensors.

### 9.2 PROBLEMS found
1. **A fabricated channel.** `97.mat` (the healthy file) has no base-accelerometer channel. The script silently fills it with random noise (`np.random.normal(0, 0.005, …)`), while the three fault files have a real one. So that feature separates healthy from faulty by construction.
2. **The task is too easy.** Even without that channel, the drive-end RMS alone separates healthy from faulty at AUC 0.999–1.0. A simulated low-bandwidth sensor (about 260 Hz band-pass, 1 kHz sampling, roughly what an MPU6050 gives, from memory of its datasheet and not re-checked) still gave 0.99 or more for large faults. So perfect vibration scores say nothing about our method.
3. **Leaky split.** Healthy frames overlap by 50% and were shuffled before being split into train and validation, so validation was contaminated.
4. **Thin data:** one healthy recording, one fault size, one load.

### 9.2b The fix and its results (DONE — `dataset/cwru_vibration_v2.py`)
What changed: no fabricated channel (only drive-end and fan-end are used); the healthy recording is split in time order with gaps (first 40% for the machine's own feature numbers, next 20% for the alert level, last 40% as healthy test frames); two sensor paths (full band 12 kHz as reference, and an "MPU-like" path: low-pass at 260 Hz, 1 kHz sampling); optional white sensor noise added to the MPU-like path; a no-network detector (distance from the healthy numbers, alert level = average + 3 × spread).

Results (AUC for inner-race / ball / outer-race faults, detector using both channels):

| Sensor path | Added noise (× healthy signal strength) | AUC (inner / ball / outer) | False alarms | Faults detected |
|---|---|---|---|---|
| Full band, 12 kHz | 0 | 1.00 / 1.00 / 1.00 | 3% | 100% |
| MPU-like, 1 kHz | 0 | 0.99 / 1.00 / 1.00 | 6% | 100% |
| MPU-like | 0.5 | 0.98 / 0.99 / 0.98 | 7% | 98% |
| MPU-like | 1 | 0.85 / 0.89 / 0.86 | 5% | 36% |
| MPU-like | 2 | 0.55 / 0.57 / 0.56 | 4% | 5% |
| MPU-like | 4 | 0.49 / 0.49 / 0.50 | 4% | 2% |

Health check: inside the healthy recording, AUC between its second and third block is 0.55 (full band) and 0.58 (MPU-like), close to the 0.5 of "no drift".

How to read it:
- With the fake channel gone, vibration is still easy: the three faults form clusters far from the healthy cluster (see the picture), so the near-perfect scores say little about our method.
- A low-bandwidth sensor (260 Hz, 1 kHz) loses almost nothing on these large faults.
- **The sensor-noise test is the useful new information.** Detection holds while the sensor's own noise is up to about half of the healthy machine's vibration strength, and collapses once the noise equals it (detection falls from 98% to 36% to 5%). So what matters in practice is the ratio between the real MPU6050's noise and how much the real machine vibrates. A small machine that vibrates weakly is the risk. This must be measured on the real rig; the noise levels here are relative, not MPU6050 specifications.
- Still true: one healthy recording, one fault size, one load, and healthy and faulty data come from different recordings, so rig and session differences could contribute to the separation. The sound-vibration pairing problem (9.3) is unchanged.

### 9.3 Why sound and vibration "don't go together" offline
MIMII is sound from some machines; CWRU is vibration from a different bearing rig. There is no recording where both come from the same machine, so a model trained on both would be meaningless (the earlier simulated results fell into this trap). Whether combining them improves accuracy can only be tested with data where both come from the same machine: the Jung dataset (section 4.3) or our own rig.

### 9.4 Combining the signals (design options, PLAN)
- **Early fusion (67 numbers):** 64 sound + 3 vibration in one input. Risk: three vibration numbers among 64 sound numbers may add noise and hurt, or be ignored.
- **Late fusion:** keep the sound model exactly as validated; score vibration separately; combine the two decisions with a simple rule (for example "alert if either is clearly abnormal", or a weighted vote). It cannot make the sound detector's own scores worse; the worst case is "no gain".
- The proposal says both will be compared on real paired data, and the better one adopted. The number 67 is therefore a *candidate* input size, not a fixed decision.

### 9.5 Thermal
Temperature sensing is future work. The proposal says that in this phase heat is addressed indirectly: catching a faulty machine earlier shortens the time it runs inefficiently, and the extra power it draws indicates energy wasted as heat. The Jung dataset has temperature and motor-current channels, which could give real evidence of how faults change them.

---

## 10. Quantization and the audio-in, decision-out simulation (PLAN)

**Quantization plan (about half a day of work, mostly writing and checking code):**
1. For each layer, store the weights as 8-bit integers with a scale factor.
2. Pick the scale for the numbers flowing between layers using a few healthy clips.
3. Run the model with integer arithmetic (in Python first) and compare AUC, false-alarm and detection rates with the 32-bit model.
4. Report the model size and the accuracy change. If accuracy drops too much, keep 32 bits; it still fits on the chip.
INT8 on the real chip is not possible until we have the board, so we do not claim any chip result.

**Simulation plan:** a program that takes a WAV file (or a stream of small chunks, as a microphone would deliver), computes features, runs the model, uses the first minutes to calibrate, then prints a normal/alert decision every second or so. Compare its decisions with the Python version and run it over normal and abnormal clips to measure false alarms and detection. First a Python version, then the same through the C code.

---

## 11. Accuracy and sanity checks done so far

| Check | What it showed |
|---|---|
| Dataset inventory vs Hitachi's table | All 16 type/ID counts match exactly; all files readable; format as expected |
| Hand-recomputed one spectrogram frame vs librosa | Matches to about 4 millionths |
| Features cached vs recomputed | Identical |
| Training uses healthy clips only | Fan ID 00: 604 training clips = 1011 healthy − 407 abnormal, as the split rule gives |
| Baseline vs Hitachi's published table | Correlation 0.97, mean difference +0.013 over 48 cases |
| Same case with 3 random seeds | 0.80 / 0.81 / 0.75 (shows run-to-run spread) |
| Calibration case rerun from scratch | Identical to four decimals (results are reproducible) |
| Duplicate/missing results in the baseline grid | 48 unique cases, no duplicates |
| Vibration sanity check | Revealed the fabricated channel and the "too easy" problem |

---

## 12. Timeline

**Your deck's plan (Phase-I presentation):**
- Semester one — design and documentation: Month 1 problem, survey, gaps; Month 2 architecture, signal-processing pipeline, autoencoder design; Month 3 calibration design, feasibility, bill of materials, MIMII study; Month 3.5 Phase-I report.
- Semester two — implementation: Month 4 data preprocessing and offline training; Month 5 quantization, firmware, signal-processing on chip; Month 6 calibration on device and hardware assembly; Months 6.5–7 test-rig validation, benchmarking, report and paper.

**Fixed dates:** the 50% review is in about 2 days; the KSCST proposal is due **30 November 2026**; the university requires a **patent to be filed** (filing, not grant); a go/no-go decision date on hardware and scope was proposed for **10 November**.

**Where we are:** Month 4 style work (preprocessing, baseline training) is done; the calibration experiment is done; hardware does not exist yet. By my own weighting of the whole project we were at roughly 17% before calibration; with calibration finished it is roughly 27%, about 30% after the clean vibration step, and about 39% now that model selection is also done. These percentages are my estimates.

---

## 13. Implementation plan (steps 1–5, no hardware needed)

| Step | What | Status |
|---|---|---|
| 3 | Calibration experiment (v2 with controls, 3 seeds, 3 noise levels) | DONE (section 7.4) |
| — | Model selection (12 designs, tuning machines, sealed test) | DONE (section 7B) |
| 4 | INT8 quantization and accuracy check | PLAN (on hold at your request) |
| 5 | Simulation: audio in → decision out (Python version) | PLAN |
| 2 | Sound features in portable C, verified against Python | PLAN |
| 5 | Same simulation through the C code | PLAN |
| 1 | Sound + vibration combination, tested on a real paired dataset | PLAN (needs dataset download) |
| — | Clean vibration pipeline (remove the fake channel, time-ordered split, MPU6050-like features) | DONE (section 9.2b) |
| — | Updated presentation using only real results | PLAN |
| 6 | Real hardware (board bring-up, rig, noise and energy measurements, fusion on real machine, on-chip test) | LATER, needs hardware |

Order of the next work: quantization → Python simulation → C features → C simulation → fusion on real paired data → deck. (Calibration is finished.) Open follow-ups from the calibration results: a better scoring rule for valves, and repeating the 0 dB and −6 dB runs with more seeds.

---

## 14. Energy, noise, patent (short)

**Environmental part (proposal).** The device is to record noise level and electrical power of a motor rig under healthy and fault conditions and estimate the extra noise and energy a fault causes. This is an *indicative machine-level estimate*, not a formal environmental impact assessment. Existing studies already show faults reduce motor efficiency and raise noise and temperature, so the claim is about doing it on a low-cost device, not about discovering the effect.

**Patent.** The university requires a filed patent. Software or an algorithm alone is excluded under Section 3(k) of the Indian Patents Act, so the filing should describe a hardware-tied system and method, with narrow claims. The best-looking narrow idea after a first (non-professional) web search: estimating the extra energy and noise caused by a fault, relative to that machine's own calibrated healthy baseline, on a low-cost node. File *before* publishing details anywhere. A professional prior-art search and the university IPR cell are needed; nothing here is legal advice.

---

## 15. Honesty log — mistakes we found and fixed, and open weaknesses

**Corrected:**
- The first-phase "100% accuracy" results used *simulated* sound and were deleted.
- The vibration script fabricated a channel (still to be fixed).
- A proposal sentence referred to the internal Phase-I deck; removed.
- References were checked; two entries from the old deck (an IEEE TIM 2024 paper and a Sensors 2025 paper) could not be found and were excluded.
- Citations were capped at three uses per reference.

**Open weaknesses:**
- Novelty is modest (integration and application, not a new method).
- Calibration claims are unproven until v2 finishes.
- Valves perform poorly everywhere.
- Pumps give about 40% false alarms at the device's alert level even with AUC above 0.95; the alert-level rule needs improving.
- The sealed-machine test uses one machine per type, so small differences are within noise.
- The 0.90 AUC target holds only for fans in easier conditions; qualify it by machine and noise level.
- "Thermal monitoring" is in the title but not built.
- No hardware yet; nothing is measured on the chip; INT8 on the chip is not possible yet.
- Kolok et al. (reference 5) is internally inconsistent (abstract says on-device, body says laptop; ~73% vs 93% accuracy claims), so cite it only for cost/complexity and the gap, not for accuracy.

---

## 16. How to re-run things

From `C:\dev\capstone\acoustic`:
```
python run_baseline.py --workers 3 --threads 5            # all 48 baseline cases (about 35 min)
python compare_hitachi.py                                  # table + plot vs Hitachi
python calibration_v2.py --snr 6 --seeds 0 1 2             # calibration v2 at 6 dB (hours)
python summarize_calibration_v2.py 30 6                    # readable summary (30 clips, 6 dB)
python export_csv.py --snr 6 --type fan --id id_00 --label normal --index 0
python demo.py --type fan --held id_06 --clips 16            # live demo (about 30 s; first run trains and caches the model)
python demo.py --type fan --held id_00 --clips 16            # a harder machine: shows misses honestly
python demo_gui.py                                           # the window version: pick a WAV file, see the verdict
python model_selection.py --stage tune                       # model comparison (about 30 min)
```
For the demo the SSD with `N:\dev\Sonar-acoustic-dataset` and `N:\dev\mimii_features*` must be connected.
Feature extraction (only needed once): `python extract_features.py --workers 8`. Dataset inventory: `python ..\dataset\inventory_mimii.py`.

---

## 17. Glossary

- **AUC** — ranking quality of the "strange" score; 0.5 = chance, 1.0 = perfect.
- **Autoencoder** — network that squeezes and rebuilds its input; large rebuild error = strange input.
- **Calibration** — learning a new machine's normal levels and alert level at installation.
- **Channel** — one microphone's stream inside a multi-microphone recording.
- **Edge** — processing on the device itself, no cloud.
- **False alarm (FPR)** — healthy clip wrongly flagged. **Detection (TPR)** — faulty clip correctly flagged.
- **Fusion** — combining sound and vibration (early = join features, late = join decisions).
- **ID** — one individual machine within a machine type in MIMII.
- **Log-mel** — the 64-band sound features.
- **MSME** — micro, small and medium enterprise.
- **Quantization / INT8** — storing the model with 8-bit whole numbers.
- **RMS** — root-mean-square, the average strength of a vibration signal.
- **SNR (dB)** — how loud the machine is compared with the background noise.
- **Standardizing (z-score)** — expressing numbers as distance from the average in units of spread.
- **Threshold / alert level** — the score above which the device raises the alarm.
