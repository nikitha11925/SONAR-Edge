# CODE WALKTHROUGH — every file, why it exists, how it works, what went wrong, where it stands

**Written:** 23 October 2026. **Written for:** the project team (Nikitha D, Rajika Saha, Monalisa, Sunand P Chowdary), so that any of you can explain every file to the review panel.
**Source of truth for results:** the CSV and Markdown files in `acoustic/results/` and `dataset/`. I read every code file listed here before writing this. Where a number appears, its source file is given. Where I am not sure, I say so.

**Status markers (same as PROJECT_GUIDE.md):** DONE = finished and checked. PLAN = decided, not built. PROBLEM = something wrong or weak. LEGACY = old, do not trust.

**How "line by line" is handled.** Files that carry the science (features, model, calibration, selection, evaluation) are explained function by function with the key lines quoted by number. The window-interface files (tkinter) are explained block by block, because most of their lines only place buttons on a screen.

**Relationship to PROJECT_GUIDE.md.** That guide is still correct but its status date is 21 October and it does not yet describe: `final_evaluation.py`, `ablation_calibration.py`, `evaluation_lib.py`, `dashboard_gui.py`, `own_recording_gui.py`, `own_recording_test.py`, `split_recordings.py`, `summarize_final.py`. This document covers them. The guide should be updated from this file when you are ready.

---

## 0. A correction to something I told you earlier

You asked whether the core architecture is "64-32-16-32-64 or 320-64-…". I answered that 64-32-16-32-64 is not in any file. **That was wrong.** The shapes are built from a list inside `acoustic/model_selection.py` (lines 54–69), so you cannot find the numbers by searching. The "shallow" design is `[input, 32, bottleneck, 32, input]`. With a 64-band single frame as input and a bottleneck of 16, that is exactly **64-32-16-32-64**. It is the design chosen for **valves** (5,264 parameters), and valves are the one machine type where nothing works.

So the full answer is: there is no single core architecture. There are five, shown in section 2.

---

## 1. The project in plain words

A small device (ESP32-S3 chip, one microphone, one vibration sensor) sits on a machine, learns what that machine sounds like when healthy, and raises an alert when the sound changes. It must run on the device with no cloud, and must work on a machine it has never heard before, after only a few minutes of listening ("calibration"). Later it also estimates the extra energy and noise a fault causes.

**What has actually been built so far:** all on a laptop, with the public MIMII sound recordings and one small vibration dataset (CWRU), plus a few recordings you made yourselves. Nothing runs on the chip yet. No hardware exists yet.

### 1.1 The whole idea in six steps (this is the story to tell)

1. Cut each 10-second recording into 32 ms pieces and turn each piece into 64 numbers describing how loud it is in each pitch band (the "log-mel" features).
2. Train a small neural network (an **autoencoder**) on healthy sound only. It learns to squeeze its input and rebuild it. It becomes good at rebuilding healthy sound.
3. A faulty sound is rebuilt badly. The size of the rebuilding mistake is the **score** ("how strange is this?").
4. **Calibrate** on the new machine: listen to 30 healthy clips to learn its normal level for each feature (mean and spread), then use 30 more healthy clips to set the **alert level** at *average score + 3 × spread of scores*.
5. From then on: score above the alert level means RED, otherwise GREEN.
6. Test it the hard way: the model never sees the test machine during training (leave-one-machine-out), and one machine (ID 06) is sealed away from every design choice.

---

## 2. Core architecture — how it changed over many iterations

### 2.1 The timeline of designs

| # | When | Design | Input | Parameters | What happened to it |
|---|---|---|---|---|---|
| 0 | First phase (scripts dated 10 Sep) | **67 → 32 → 16 → 32 → 67** | 64 sound numbers + 3 vibration numbers | 5,459 (my count from the layer sizes) | Trained on **simulated** sound joined to real CWRU vibration. The "100% accuracy" it reported was meaningless, because the sound half was random numbers shaped like sound and the two halves came from different machines. Results deleted. The two scripts (`train_edge_autoencoder.py`, `export_edge_model_c_header.py`) remain as LEGACY. |
| 1 | 9 Oct | **320 → 64 → 64 → 8 → 64 → 64 → 320** (Hitachi baseline) | 5 stacked frames × 64 bands | 50,760 | Re-implemented from Hitachi's paper and code. Checked against their published table: correlation 0.97 over 48 cases. This proved our data handling and metric were correct. |
| 2 | 21 Oct | Same network, but **inputs standardised per machine** ("calibration") | same | 50,760 | Turned near-chance results on new machines into AUC about 0.95 for fans, pumps and sliders (section 7). The big discovery of the project. |
| 3 | 21 Oct | **Model selection**: 12 designs compared | 1, 3 or 5 frames; 32/64/128 bands | 1,352 to 51,792 | Model size barely mattered. A single frame was enough. Picked the smallest design within 0.02 AUC of the best, per machine type. |
| 4 | 22 Oct | **Final: one small model per machine type** (table below) | one frame | 2,408 to 6,856 | Used for the final evaluation, the dashboard, and the own-recording test. |

### 2.2 The final models (from `acoustic/results/model_selection_choice_final.csv`, sizes checked by hand from the layer lists)

| Machine type | Layer sizes (input → … → output) | Parameters | Size 32-bit | Size INT8 (estimate) |
|---|---|---|---|---|
| Fan | 64 → 32 → 32 → 8 → 32 → 32 → 64 | 6,856 | 26.8 KB | 6.7 KB |
| Pump | 64 → 16 → 8 → 16 → 64 | 2,408 | 9.4 KB | 2.4 KB |
| Slider | **32** → 32 → 16 → 32 → 32 (32 mel bands) | 3,184 | 12.4 KB | 3.1 KB |
| Valve | 64 → 32 → 16 → 32 → 64 (arbitrary choice; nothing works) | 5,264 | 20.6 KB | 5.1 KB |
| *(reference)* Hitachi baseline | 320 → 64 → 64 → 8 → 64 → 64 → 320 | 50,760 | 198.3 KB | 49.6 KB |

All use ReLU between layers and a plain linear output layer. Size = parameters × 4 bytes (32-bit) or × 1 byte (INT8, **estimate only**, quantisation not done yet).

### 2.3 Why this architecture (the reasons)

- **Why an autoencoder at all?** In a factory you cannot collect every kind of fault in advance, but healthy sound is plentiful. A model trained only on healthy sound needs no fault examples. This is the standard approach for this dataset and for the DCASE machine-sound challenges.
- **Why dense (fully connected) layers, not convolutional or recurrent?** Hitachi's baseline is dense, so matching it let us check our work. Dense layers are also the simplest to run on a microcontroller (just multiply-and-add loops, no special library).
- **Why a narrow middle ("bottleneck") of 8 or 16?** The bottleneck forces the network to keep only the main patterns of healthy sound. Anything it cannot summarise (a fault) comes out badly rebuilt.
- **Why did the input shrink from 320 to 64?** Model selection showed stacking 5 frames did not help. On the chip this removes a 5-row buffer.
- **Why standardise inputs per machine?** Different machines of the same type have different loudness and tone. Without removing that, a healthy new machine looks "strange" to a model trained on other machines. Standardising each feature with the new machine's own mean and spread removes the machine-to-machine offset. This single step is what made detection work on new machines (ablation, section 6.6).
- **Why score = mean squared error averaged over a clip's frames?** It copies Hitachi. Averaging over about 300 frames makes a clip's score stable.

### 2.4 Weak points of the architecture (honest)

- Averaging over a whole 10-second clip **hides short events**. This is the likely reason valves fail: valve sound is short clicks (see PROJECT_GUIDE 7.4). Untested remedy: score the worst few frames of a clip instead of the average.
- A pure reconstruction score reacts to *any* change in the sound, including harmless ones (speed change, background noise). This is the source of the false-alarm problem on pumps and of the noise-mismatch failure (section 7).
- The model never sees time context beyond one 64 ms window.

---

## 3. Data flow (what goes in, what comes out)

```
WAV (16 kHz, 8 mics, 10 s)           own recording (phone, m4a/mp3/wav)
      |  features.load_channel0              |  own_recording_test.read_audio + cut_clips
      v                                      v
   1 channel, 160,000 samples  ---- same from here ---->  clips of exactly 10 s at 16 kHz
      |  features.logmel
      v
   log-mel table: 64 bands x 313 time steps   (stored as .npy on the N: drive)
      |  stack_frames(frames = 1 / 3 / 5)
      v
   model input vectors: (T - frames + 1) x (bands x frames)       e.g. 313 x 64 for one frame
      |  standardise:  (x - machine mean) / machine spread          <-- CALIBRATION (30 clips)
      v
   autoencoder  ->  rebuilt vector
      |  mean squared error per vector, then mean over the clip     <-- the SCORE
      v
   score > alert level ?   alert level = mean + 3 x spread of 30 more healthy clips' scores
      |
      v
   RED (alert) / GREEN (normal)
```

---

## 4. Folder map

| Folder / file | Purpose |
|---|---|
| `acoustic/` | All sound work: features, models, experiments, evaluation, interfaces |
| `acoustic/results/` | Every result file (CSV, Markdown, logs) |
| `acoustic/plots/`, `acoustic/demo_output/` | Pictures |
| `acoustic/demo_models/` | Cached trained models (`.pt`) used by the demo and dashboard |
| `dataset/` | MIMII file inventory, CWRU vibration work, raw archives (zip files are git-ignored) |
| `own_data/` | Your own recordings (raw in `healthy/`, `faulty/`; 10-second clips in `clips/`) |
| `docs/` | Project guide, review brief, literature survey, this file |
| `ppts/` | Presentations, proposal versions, MIMII paper, an old plan `implementation_plan.md` |
| `sampleandfulldataset/mimii_baseline-master/` | Hitachi's original Keras code, read but never run |
| `train_edge_autoencoder.py`, `export_edge_model_c_header.py` (root) | LEGACY first-phase scripts |

Data lives outside the repository: sound WAVs in `N:\dev\Sonar-acoustic-dataset`, extracted features in `N:\dev\mimii_features` (64 bands) and `N:\dev\mimii_features_b32`, `_b128`. **If the N: drive is not connected, nothing in `acoustic/` that loads features will run.**

---

## 5. File by file — the sound pipeline

### 5.1 `acoustic/features.py` — DONE (the foundation)

**What it is.** The single place where sound becomes numbers. Every other file imports it as `F`.
**Why it exists.** Training, testing, the demo and (later) the chip must all produce *identical* features, or results will not transfer. One file guarantees that.

| Lines | What they do |
|---|---|
| 13–14 | `DATA_ROOT`, `FEATURE_ROOT`: hard-coded paths on the N: drive. Other files change `F.FEATURE_ROOT` while running to switch between 32/64/128-band folders (a shortcut that works but is easy to break; see section 9). |
| 16–20 | Constants copied from Hitachi: 64 mel bands, window 1024 samples (64 ms), hop 512 (32 ms), power 2, 5 stacked frames. |
| 22–24 | The four machine types, the four IDs (00, 02, 04, 06), the three noise levels. |
| 27–33 | `class_dir`, `list_wavs`: build the folder path for one (noise, type, ID, normal/abnormal) and list its WAV files **sorted by name**. Sorted order matters: the Hitachi split depends on it. |
| 36–39 | `load_channel0`: read a WAV as a (160,000 × 8) table and keep column 0 (microphone 1 only). Asserts 16 kHz. *Why channel 1 only:* our device will have one microphone. |
| 42–45 | `logmel`: one librosa call (`melspectrogram`) with Hitachi's settings, then `20/power × log10(energy + epsilon)`. For power 2 this is `10 × log10(energy)`, i.e. decibel-like. `epsilon` (2.2e-16) stops log(0). Output 64 × 313 table, 32-bit. |
| 48–55 | `stack_frames`: for `frames = 5`, joins 5 neighbouring time steps side by side: output has `T − 4 = 309` rows and 320 columns. The loop copies time steps `t … t+n` into block `t`. For `frames = 1` it simply transposes (313 rows × 64 columns). |
| 58–64 | `feature_path`, `load_logmels`: where the cached `.npy` array lives and how to load it: shape (clips, bands, 313). |

**Checked:** one frame recomputed by hand (window → FFT → mel table → log) matches librosa to about 4 millionths.

**Problems / limits.** Uses librosa 1.0 while Hitachi used 0.6, so tiny numeric differences are possible. The C version for the chip must reproduce this exactly (PLAN, PROJECT_GUIDE section 8).

### 5.2 `acoustic/extract_features.py` — DONE

**What / why.** Running `logmel` on 54,057 WAV files takes about 30 minutes. This script does it once and saves the result, so every experiment afterwards loads arrays instead of re-reading audio.

| Lines | What they do |
|---|---|
| 10–21 | `job`: for one (noise, type, ID, label) group, skip if the output exists (safe to re-run), else read all files, compute log-mel, stack into one array, write to a temporary file, then rename it (`os.replace`). *Why temp-then-rename:* a crash mid-write can never leave a half-written file that looks finished. |
| 24–34 | Builds the 96 jobs (3 noise × 4 types × 4 IDs × 2 labels), runs them in a pool of 8 workers, prints progress. |

### 5.3 `acoustic/extract_bands.py` — DONE

Same as 5.2 but for **32 or 128 bands**, 6 dB only at first (`--snrs` default `['6']`), written to `N:\dev\mimii_features_b32` / `_b128`. Needed for the band-count test in model selection. `results/extract_b32_extra.log` shows it was later re-run for the other noise levels, because the slider's chosen design uses 32 bands and `final_evaluation.py` needs those features at 0 and −6 dB.

### 5.4 `acoustic/plot_spectrograms.py` and `acoustic/export_csv.py` — DONE (understanding only, no results)

- `plot_spectrograms.py`: draws normal vs abnormal spectrograms for all four machine types, the same clip at three noise levels, and the average spectrum. *Why:* to see with your own eyes what the model is up against. It is where you can see that valve faults barely change the picture.
- `export_csv.py`: writes one clip's spectrogram (64 × 313) and model inputs (309 × 320) as CSV so they can be opened in Excel. *Why:* so that a reviewer can inspect actual numbers.

### 5.5 `acoustic/baseline_ae.py` — DONE (the Hitachi baseline)

**What it is.** Our re-implementation of Hitachi's training and testing. **Why:** if our pipeline reproduces their published numbers, our data handling and metric are right. This is the project's single most valuable check.

| Lines | What they do |
|---|---|
| 1–8 | Header describing the recipe: network, Adam 1e-3, 50 epochs, batch 512, last 10% held out for monitoring only, final epoch used. |
| 18–28 | `build_model`: sizes `[320, 64, 64, 8, 64, 64, 320]`; a `Linear` layer for each step, ReLU after every layer except the last. Xavier-uniform weights and zero biases imitate Keras's defaults (Hitachi used Keras). |
| 31–33 | `stack_all`: stacks 5 frames for a list of clips: (files, 309, 320). |
| 36–40 | `split_files`: Hitachi's rule. The first N normal clips (N = number of abnormal clips) become test clips, the rest are training clips. *Why:* gives a balanced test set (N healthy + N faulty) and never trains on test clips. |
| 43–51 | `per_file_scores`: model in evaluation mode, no gradients; for each clip the score = mean over its 309 vectors of the mean squared error. |
| 54–82 | `train_model`: flatten all training vectors; hold out the last 10% for monitoring only (never used to stop early, matching Hitachi); Adam, learning rate 1e-3, MSE loss; each epoch shuffles with a seeded random generator (reproducible), runs minibatches of 512, records train and validation loss; the final epoch's model is kept. |
| 85–100 | `run_one`: load one machine's features, split, train, score test clips, compute AUC (healthy = 0, faulty = 1). Returns a record. |

**Result.** 48 cases; correlation with Hitachi's table 0.97; mean difference +0.013 (`results/baseline_vs_hitachi.md`).
**Problems.** PyTorch instead of Keras; one run per case (Hitachi averaged three); fan ID 00 at 6 dB gave 0.80 / 0.81 / 0.75 over three seeds, so gaps of a few points are random variation. Single cases such as pump ID 06 differ by up to 0.16, and the "random variation" explanation was only tested on one case.

### 5.6 `acoustic/run_baseline.py` and `acoustic/compare_hitachi.py` — DONE

- `run_baseline.py`: runs all 4 types × 4 IDs × 3 noise levels = 48 models, 3 in parallel (lines 28–55). It writes each result row the moment it finishes (`f.flush()`, line 53) and skips cases already in the CSV (lines 13–17, 44), so an interrupted run can simply be started again. The `spawn` multiprocessing context (line 47) is required on Windows.
- `compare_hitachi.py`: lines 11–16 hold Hitachi's published Table 3 by hand; lines 18–19 read our results; lines 21–45 build the comparison table and the scatter plot; line 41 prints mean difference, mean absolute difference and correlation. Note: the Hitachi numbers are typed in manually, so a typo there would silently change the comparison.

### 5.7 `acoustic/calibration_experiment.py` — DONE but FLAWED (version 1)

**What / why.** First test of "does calibration help?" on new machines. Leave-one-machine-out: hide one ID, train on the other three, treat the hidden one as a new machine.

| Lines | What they do |
|---|---|
| 32–34 | `KS = [10, 30, 100]` calibration lengths; `TRAIN_STRIDE = 4` uses every 4th vector for training, because neighbouring vectors overlap heavily and this makes training about 4 times faster. |
| 42–44 | `stats`: per-feature mean and spread (`+1e-6` avoids dividing by zero). |
| 47–54 | `file_scores`: standardise a clip with given statistics, then reconstruction error. |
| 57–58 | `tau`: the alert-level rule, mean + 3 × spread. |
| 61–64 | `metrics`: AUC plus false-alarm rate and detection rate at a given alert level. |
| 67–113 | `run_fold`: builds model P (pooled standardisation, no calibration) and model C (each training machine standardised separately), then computes **M0** no calibration, **M1** alert level only, **M2** full calibration. |

**Problems (found by checking our own work):** (1) no control to show the gain comes from the network rather than the standardising; (2) the same clips were used to standardise and to set the alert level, so the alert level came out too low (pump ID 04: 99% false alarms); (3) the "no calibration" baseline was below 0.5 AUC in 6 of 16 cases, which makes calibration look better than it is; (4) one seed and one noise level. All four are fixed in v2. Numbers in this version are reproducible (fan ID 02 re-run gave identical values), but its explanation was not sound.

### 5.8 `acoustic/calibration_v2.py` — DONE (the main calibration evidence)

**What / why.** Same experiment with the four problems fixed.

| Lines | What they do |
|---|---|
| 4–12 | Lists the fixes: calibration clips split into a statistics set and a disjoint alert-level set; two controls; several seeds. |
| 59–60 | `simple_scores`: **control S**, no neural network: the average squared z-score of a clip against the machine's own healthy numbers. |
| 81–83 | `cal_end = 2 × max(KS)` reserves 200 clips for calibration; the assertion makes sure enough healthy clips exist. `own_train` is what is left, used by the oracle. |
| 92–97 | Control S for each K: statistics from `pool[:K]`, alert level from the **next** K clips `pool[K:2K]`. |
| 99–109 | Builds the pooled-normalised set (model P), the per-machine-normalised set (model C), and the oracle's own set. |
| 111–135 | For each seed: **M0, M1** (model P), **M2** (model C, full calibration), **O** the oracle (a model trained on the new machine's own healthy clips: the best case, which a real device cannot do because it cannot train per machine). |
| 139–143, 166 | Resume logic keyed by (noise, type, ID, seed). |

**Result** (`results/calibration_v2_results.csv`; PROJECT_GUIDE 7.4 has the full table). At 6 dB, AUC averaged over four machines: fan 0.53 (none) → 0.95 (calibrated); pump 0.47 → 0.95; slider 0.80 → 0.97; valve 0.45 → 0.55. Full calibration matches or beats the per-machine best case for fans and sliders. For **pumps, the simple no-network control (0.98) beats the network (0.95)**, so for pumps the credit belongs to the standardising, not the autoencoder. Valves: no method works.

**Problem still open:** 0 dB and −6 dB were run with a single seed.

### 5.9 `acoustic/model_selection.py` — DONE (defines the models used everywhere)

**What / why.** Chooses a small model per machine type with a rule fixed *before* looking at results, so the choice cannot be tuned to the test.

| Lines | What they do |
|---|---|
| 4–11 | The protocol: IDs 00/02/04 are tuning machines, ID 06 is sealed; tune stage, final stage; rule = smallest model within 0.02 AUC of the best (0.02 is about the seed-to-seed spread seen earlier). |
| 30–33 | `K = 30`, `TRAIN_STRIDE = 4`, `TUNE_IDS`, `SEALED`. |
| 36–49 | `DESIGNS`: 12 candidates. Names: `D` deep (two hidden layers each side), `S` shallow (`in-32-bott-32-in`), `T` tiny (`in-16-bott-16-in`); `F1/F3/F5` = frames stacked; `H64/H32` = hidden width; `b8/b16` = bottleneck. `D_F5_H64_b8` is the Hitachi baseline. |
| 54–69 | `build`: **this is where all the layer sizes are made.** Deep: `[in, h, h, b, h, h, in]`; shallow: `[in, 32, b, 32, in]`; tiny: `[in, 16, b, 16, in]`. |
| 72–73 | `n_params`. |
| 76–91 | `train`: same recipe as baseline but takes the model in. **Note:** lines 79–80 cut off a 10% validation slice that is never used, so 10% of the training data is simply discarded. Harmless but wasteful (small fix possible). |
| 94–113 | `stack` (with stride), `stats`, `file_scores` (processes clips in chunks of 32 for speed; same score as before). |
| 116–122 | `tau`, `metrics` (AUC, false-alarm rate, detection rate). |
| 139–167 | `run_stage`: for each machine type and number of stacked frames, standardise each training machine with its own statistics, then for each held-out machine compute calibration statistics from `pool[:K]`, alert level from `pool[K:2K]`, train and score every design and seed. |
| 193–227 | `main`: three stages. `tune` (64 bands, all 12 designs), `bands` (winner rebuilt with 32 and 128 bands), `final` (sealed machine, baseline vs chosen design). |

**Result.** (`results/model_selection_stage1_summary.md`, `…stage2_summary.md`). Model size barely matters for fans, pumps, sliders. A 2,408-parameter pump model scores 0.954 against 0.936 for the 50,760-parameter baseline. Single frames match stacked frames. More bands help a little (pump 0.925 / 0.954 / 0.971 at 32 / 64 / 128). Valve best AUC is 0.571, so its "choice" is meaningless.
**Sealed machine (ID 06, 6 dB, 3 seeds):** chosen models match the baseline for fans and pumps and beat it for sliders (but one machine per type means differences of a few points are noise).
**Problem:** the tuning used 6 dB only. 0 dB and −6 dB were not used for choosing.

### 5.10 `acoustic/summarize_selection.py` — DONE

Implements the selection rule in code (`choose`, lines 39–42: keep designs within `TOL = 0.02` of the best mean AUC, take the one with fewest parameters). `stage1` writes `model_selection_choice.csv`; `stage2` adds the 32/128-band variants and writes `model_selection_choice_final.csv`, **the file the demo, dashboard and final evaluation all read to know which model to build.** If that file changes, every downstream result changes.

### 5.11 `acoustic/noise_mismatch.py` and `acoustic/summarize_noise_mismatch.py` — DONE

**What / why.** A real factory's noise changes during the day. If the device calibrates in a quiet moment and monitors in a loud one, what happens?

| Lines | What they do |
|---|---|
| 5–14 | Setup: model A trained at 6 dB only, model B trained at all three noise levels; calibration condition c ∈ {6, 0, −6, mix}; test condition t ∈ {6, 0, −6}. |
| 42–58 | `run_fold`: builds both models, each training machine and noise level standardised with its own statistics. |
| 62–69 | Calibration sets for each noise level; `mix` takes 10 clips from each of the three levels (lines 68–69). |
| 71–79 | For both models, every calibration condition and every test condition: alert level from the disjoint clips, then metrics. |

**Result** (`results/noise_mismatch_summary.md`; mean over fan, pump, slider, model trained at 6 dB):

| Situation | False alarms / faults found (AUC) |
|---|---|
| Calibrated and tested in the same noise | 8% / 41% (0.83) |
| Calibrated quiet (6 dB), tested in more noise | **44% / 73%** (0.79): healthy clips constantly alert |
| Calibrated noisy, tested in less noise | 11% / 40% (0.81): faults missed |
| Calibrated on a mix of noise levels | 11% / 41% (0.82) |

The model trained on all noise levels behaves the same, so training noise is not the issue. **Mitigation:** calibrate in the noise the machine runs in, or on a mix of noise levels (the mix was tested on three fixed levels only). **Not tested:** real Indian factory noise, noise that drifts during the day, loud bursts.

### 5.12 `acoustic/evaluation_lib.py` — DONE (new)

**What / why.** One place defining the metrics, used by both the batch evaluation and the dashboard, so numbers on screen are computed exactly like numbers in the results files.

| Lines | What they do |
|---|---|
| 16–30 | `full_metrics`: from healthy scores `sn`, faulty scores `sa` and the alert level → AUC, precision, recall, F1, accuracy, false-alarm rate, specificity, balanced accuracy, and the four confusion counts. Divisions are guarded against zero. |
| 33–36 | `roc_points`: the ROC curve. |
| 39–42 | `precision_at_prevalence`: what precision would be if only a fraction p of clips were faulty. *Why:* the test set is half faulty, but real factories are mostly healthy. Precision looks much worse at 1% faults. This is an honest-reporting feature. |
| 45–46 | `linear_layers`: picks the `Linear` layers out of a model. |
| 49–75 | `model_card`: parameters, multiply-adds per frame (`in × out` per layer), 32-bit and INT8 sizes, estimated RAM for activations and feature buffers, an operation count for making one feature frame (line 62 uses fixed counts: window 1024 + FFT about 25,600 + power 513 + mel + log), and **laptop** time per frame measured over 2,000 runs. |

**Problems / limits.** Everything labelled "RAM", "INT8" and "feature operations" is an **estimate**. The timing is the laptop, not the chip. The dashboard says this on screen.

### 5.13 `acoustic/final_evaluation.py` and `acoustic/summarize_final.py` — DONE (new: the headline results)

**What / why.** One run that produces every headline number: 4 types × 3 noise levels × 4 held-out machines × 3 seeds = 144 rows, with every metric. This is the table the panel should see.

| Lines | What they do |
|---|---|
| 33–39 | Read the chosen design for each type; `root_for` picks the 32/64/128-band feature folder. |
| 42–48 | `wait_for_features`: if the needed feature files do not exist yet, wait up to 90 minutes (so the run can start while features are still being extracted). |
| 51–80 | `run_case`: build the training set from the other three machines **at the same noise level**, each standardised with its own statistics; calibration on the new machine (statistics `pool[:30]`, alert level `pool[30:60]`); then for each of 3 seeds train, set the alert level, and compute all metrics. |
| 83–112 | `main`: loops, resumes, writes each case as it finishes. |
| `summarize_final.py` 19–31 | Averages over seeds and machines per type and noise level, for three subsets: all machines, sealed machine ID 06, tuning machines. Writes `results/final_summary.md`. |

**Result** (`results/final_summary.md`, all four machines, mean over seeds):

| Machine | Noise | AUC | Precision | Recall | False alarms |
|---|---|---|---|---|---|
| Fan | 6 dB | 0.92 | 0.92 | 0.66 | 5% |
| Fan | 0 dB | 0.83 | 0.87 | 0.43 | 3% |
| Fan | −6 dB | 0.71 | 0.76 | 0.21 | 3% |
| Pump | 6 dB | 0.96 | 0.77 | 0.89 | **35%** |
| Pump | 0 dB | 0.84 | 0.82 | 0.64 | 24% |
| Pump | −6 dB | 0.75 | 0.82 | 0.29 | 4% |
| Slider | 6 dB | 0.94 | 0.86 | 0.73 | 4% |
| Slider | 0 dB | 0.84 | 0.78 | 0.20 | 2% |
| Slider | −6 dB | 0.72 | 0.64 | **0.03** | 2% |
| Valve | all | 0.51–0.56 | 0.37–0.54 | 0.01–0.03 | 2–4% |

**Sealed machine (ID 06), 6 dB:** fan 1.00, pump 0.97, slider 1.00, valve 0.52 (AUC). Say this separately: it is the fairest test, but it is one machine per type.

**What the numbers say, in plain words.**
- AUC is high (0.92–0.96) for fans, pumps, sliders in moderate noise: the scores *can* separate healthy from faulty.
- But **recall is low** (fans 66%, sliders 73%, and 3% for sliders at −6 dB): the alert level is too lenient, so many real faults are not flagged even though their scores rank higher. That is an alert-level problem, not a model problem.
- **Pumps do the opposite:** 89% recall but 35% false alarms. The alert level is too strict. The same rule (mean + 3 spreads of 30 clips) is too lenient on some machines and too strict on others.
- At 1% faulty clips in real use, a 35% false-alarm rate would make almost every alert a false one (see `precision_at_prevalence`).

### 5.14 `acoustic/ablation_calibration.py` — DONE (new)

**What / why.** Answers "how much does each part of calibration add?" with the model, training and test clips held identical. Only what is learned from the new machine changes.

| Variant | Standardising uses | Alert level from |
|---|---|---|
| `none` | average of the training machines' statistics | training machines' healthy clips |
| `threshold` | the same average statistics | 30 healthy clips of the new machine |
| `full` | the new machine's own statistics (30 clips) | the next 30 clips of the new machine |

Lines 41, 56: for `none`, a few healthy clips of each training machine (`n[:K]`) are scored with their own statistics to set the alert level. Lines 57–61 define the three variants; line 63 computes the metrics for each.

**Result (I computed these averages from `results/ablation_calibration.csv`; 6 dB, 2 seeds, 4 machines each):**

| Type | AUC none → full | Recall none / threshold / full | False alarms none / threshold / full |
|---|---|---|---|
| Fan | 0.53 → 0.92 | 0.46 / 0.14 / 0.65 | 0.53 / 0.02 / 0.04 |
| Pump | 0.51 → 0.96 | 0.15 / 0.26 / 0.89 | 0.27 / 0.04 / 0.35 |
| Slider | 0.85 → 0.95 | 0.31 / 0.39 / 0.75 | 0.04 / 0.03 / 0.04 |

**Meaning.** Setting only the alert level fixes false alarms (fan 53% → 2%) but leaves detection low (14%). Only standardising with the new machine's own numbers fixes detection. **Both parts are needed.**

### 5.15 `acoustic/demo.py` — DONE (command-line live demo)

**What / why.** A demonstration that behaves like the final device: calibrates, then judges shuffled clips straight from WAV files through the real audio → features → model path.

| Lines | What they do |
|---|---|
| 39–42 | `chosen_design`: reads the chosen design for a type. |
| 49–70 | `get_model`: loads a cached model, or trains one on the other machines and saves it. The cache file name is `{type}_{held}_{snr}dB.pt` (line 51). **Problem:** the file name does not include the design. If you change the chosen design and forget `--retrain`, an old model of the wrong shape is loaded (it would crash on `load_state_dict` if the sizes differ, but would silently reuse an old model if they match). |
| 73–76 | `clip_features`: WAV → log-mel → stacked frames. |
| 79–84 | `play`: plays a clip through the speakers (Windows only, `winsound`). |
| 115–128 | Calibrates exactly as in the experiments: statistics from `pool[:30]`, alert level from `pool[30:60]`. |
| 130–137 | Picks half normal, half faulty clips at random, then shuffles. |
| 143–170 | Judges each clip, prints GREEN/RED and whether it was ok, a false alarm, caught, or missed. |
| 172–190 | Draws the picture of scores against the alert level. |

**Honest limits.** It trains and calibrates on stored features and tests on WAV files, on a laptop. It is a simulation, not the chip. A 16-clip run is a small sample (the script says so on screen).

### 5.16 `acoustic/demo_gui.py` — DONE (window version of the demo)

A tkinter window. Two parts:
- **`Engine` class (lines 33–75), no window code.** `prepare` loads/trains the model and calibrates (lines 39–57). It also remembers which clips were used for calibration so it can warn. `judge` (lines 59–75) reads any WAV (resampling if not 16 kHz), computes the score and, from the folder name, whether the clip claims to be faulty or healthy. It detects when a clip belongs to a **different machine** than the one calibrated (`other_machine`, line 71): an alert is then expected, and the window says so.
- **`App` class (lines 78–258), the screen.** Left panel: choose type and machine, "Calibrate device", "Choose WAV file", random healthy/faulty, "Play". Right panel: spectrogram, a bar of score against alert level, history table. Calibration runs in a background thread (lines 179–185) so the window does not freeze. The `parent` argument (line 79) lets the dashboard reuse this class as its first tab.

### 5.17 `acoustic/dashboard_gui.py` — DONE (new: five-tab dashboard)

| Tab | What it does | Lines |
|---|---|---|
| 1. Live clip | The demo window (re-used `demo_gui.App`) | 77 |
| 2. Evaluate | Runs the whole test set for one machine and noise level: AUC, precision, recall, F1, accuracy, false-alarm rate, confusion matrix, ROC curve, score histogram, precision at 50%/10%/1% faults, and a model card | 36–56, 83–212 |
| 3. Results table | Reads `final_evaluation_results.csv`; mean ± spread over seeds, for all / sealed / tuning machines | 215–260 |
| 4. Model card | Size and cost of each type's model, with a note on what is measured and what is only estimated | 263–291 |
| 5. My recordings | The own-recording tab (`own_recording_gui.OwnTab`) | 74–76 |

`EvalEngine.run` (lines 39–56) uses the **cached** model for the chosen machine (`D.get_model`, seed 0), so its numbers can differ slightly from the 3-seed averages in tab 3. This is expected, not a bug, but it should be said if a panel member compares them.
The text on lines 283–290 states what is measured, estimated, and not measured (power and real speed on the chip), and that the 0.90 AUC target is met for fans and sliders at 6 dB only. Keep this honesty.

### 5.18 `acoustic/own_recording_test.py`, `own_recording_gui.py`, `split_recordings.py` — DONE (new: your own sound)

**What / why.** MIMII is somebody else's microphone and somebody else's machines. The panel will want to see it work on *your* fan with *your* microphone. These three files do that.

- **`own_recording_test.py`** (the logic, also usable from the command line). `read_audio` (lines 33–49) reads WAV/MP3/FLAC through soundfile, and M4A/AAC (also WhatsApp's `.m4a.mpeg` suffix) through PyAV, first channel only. `cut_clips` (52–55) resamples to 16 kHz and cuts into exact 10-second clips, dropping the rest. `analyse` (68–96): needs `2K + 10` healthy clips (K = 30 → 70 clips, about 12 minutes) and at least 5 faulty clips, otherwise it shrinks K with a warning or refuses; it loads the MIMII model trained on machines 00, 02, 04 (line 84), calibrates on your first 30 healthy clips, sets the alert level from the next 30, and tests on a **separate healthy session** if you provide one (the fairest option) or on leftover clips (less strict, labelled as such).
- **`split_recordings.py`**: cuts a long recording into numbered 10-second WAV clips under `own_data/clips/<name>/`. *Numbered in time order matters:* calibration uses the first clips.
- **`own_recording_gui.py`**: the "My recordings" tab: folder pickers, "Calibrate and test", per-clip judging, drag-and-drop if `tkinterdnd2` is installed.

**Current outcome (limited evidence).** Your saved clips are `own_data/clips/healthy_session1`, `healthy_session2` and `faulty` (18 faulty clips). The only result I could find is a screenshot from a test run (`acoustic/demo_output/own_tab_test.png`). It shows **AUC about 0.95**, faulty clips scoring about 3 times higher than healthy ones. But the healthy scores sit very close to the alert line, and one healthy clip scored about 1.5, which is above it. I do not know the exact settings of that run, so please treat it as a first look and re-run it to get numbers you can quote.
**Limits:** one machine, one microphone, few clips; a phone recording differs from MIMII in many ways; the `other_speed` folder in `own_data` is empty, so a different-speed test has not been done.

### 5.19 `acoustic/summarize_calibration_v2.py` — DONE

Reads `calibration_v2_results.csv` and prints a plain table per machine type and method: averages over the four machines first (per seed), then mean ± spread over seeds. Usage: `python summarize_calibration_v2.py 30 6` (30 clips, 6 dB). Output `results/calibration_v2_summary_6dB_K30.md` (also K10, K100 exist).

---

## 6. File by file — vibration and the old first-phase files

### 6.1 `dataset/inventory_mimii.py` — DONE
Walks the MIMII folders, counts WAV files per (noise, type, ID, label) and opens every file header (`wave` module) to check sample rate, channels, bit depth and length. Result: 54,057 files, all 16 kHz / 8 channels / 16-bit / 10.0 s; none corrupt; counts match Hitachi's published table. *Why:* a bad or missing file silently ruins an experiment; counting first rules it out.

### 6.2 `dataset/clean_cwru_dataset.py` — LEGACY, has a PROBLEM
Cleaning chain for vibration signals: remove the mean (`remove_dc_offset`, line 56), remove spikes (`hampel_filter`, 74), band-pass 10–5000 Hz (`apply_butterworth_bandpass`, 60), cut into 32 ms frames with 50% overlap (`frame_signal`, 101), compute RMS, peak-to-peak, kurtosis, crest factor (`extract_frame_features`, 122).
**The problem (line 203):** the healthy file `97.mat` has no base-accelerometer channel, and this script fills it with random numbers (`np.random.normal(0, 0.005, ...)`), so that feature separates healthy from faulty by construction. It also shuffled overlapping frames before splitting. A warning banner is at the top. Only the cleaning functions are still used (imported by 6.3).

### 6.3 `dataset/cwru_vibration_v2.py` — DONE (clean replacement)
No fabricated channel (only drive-end and fan-end). Healthy recording split in **time order with gaps**: first 40% for statistics, next 20% for the alert level, last 40% as healthy test (`split_healthy`, lines 84–89). Two sensor paths: full 12 kHz, and an "MPU-like" path (low-pass 260 Hz, resample to 1 kHz, lines 53–58) imitating a low-bandwidth sensor. Optional white sensor noise (5 random draws averaged) to see how much margin detection has. Detector has **no neural network**: RMS per channel per 32 ms, z-score against the healthy block, score = mean squared z-score (lines 92–94).
**Result** (`dataset/cwru_vibration_v2_summary.md`): AUC about 1.0 for all three fault types with clean sensors; with sensor noise equal to 0.5× the healthy signal, detection stays at 98%; at 1× it drops to 36%; at 2× it is 5%.
**Meaning:** vibration on this dataset is easy, so near-perfect scores say little about our method. The useful finding is the noise margin: if the real MPU6050's noise is comparable to how much the real machine vibrates, detection collapses. **That ratio must be measured on the real rig.** Also: one healthy recording, one fault size, one load; healthy and faulty come from different recordings.

### 6.4 `train_edge_autoencoder.py` and `export_edge_model_c_header.py` (repository root) — LEGACY
- `train_edge_autoencoder.py`: defines `EdgeAutoencoder` 67 → 32 → 16 → 32 → 67 (lines 21–43), trains it with Adam (weight decay 1e-5), keeps the best-validation checkpoint, sets the alert level at mean + 3 × spread of validation errors (lines 119–123), and reports metrics. It reads `dataset/multimodal_fused/…npy`, **a folder that no longer exists** (the simulated data was deleted).
- `export_edge_model_c_header.py`: turns the trained weights into a C header with the inference loop written out (z-score, four layers with ReLU, mean squared error, compare with a constant `ANOMALY_THRESHOLD_TAU`).
**Why still here:** the *idea* of the C export is useful for the chip. **What must change before use:** the model shapes are different now (per-type models; the code builds `nn.Sequential`, not `encoder`/`decoder`); the standardising numbers and alert level are now learned **at install time on the device**, not fixed constants in a header; and the input is 64 or 32 numbers, not 67.

### 6.5 Reference material
`sampleandfulldataset/mimii_baseline-master/baseline.py` (lines 351–356) is Hitachi's Keras model: Dense 64, 64, 8, 64, 64, output. We read it to copy the recipe; we never ran it. `ppts/implementation_plan.md` is an old plan for the 67-feature pipeline and is out of date.

---

## 7. Problems faced and how each was handled

| # | Problem | How it was found | What we did | Status |
|---|---|---|---|---|
| 1 | First-phase "100% accuracy" came from simulated sound | Review of where the numbers came from | Deleted the results, restarted with real MIMII | DONE |
| 2 | Vibration script invented a channel | Sanity check on the healthy file | Wrote `cwru_vibration_v2.py` using only real channels; time-ordered split | DONE |
| 3 | Vibration data is too easy | Single feature separated classes at AUC 0.999 | Added the sensor-noise margin test; stopped treating CWRU scores as evidence | DONE (finding), PLAN (real rig) |
| 4 | Is our pipeline correct? | Compared to Hitachi's published table | Correlation 0.97 over 48 cases | DONE |
| 5 | New machine looks "strange" even when healthy | Leave-one-machine-out gave AUC about 0.5 | Per-machine standardising (calibration) | DONE |
| 6 | Calibration v1 had no controls, leaked calibration clips into the alert level | Self-check | v2: disjoint clips, no-network control, per-machine oracle, 3 seeds | DONE |
| 7 | Is the neural network doing the work, or only the standardising? | Control S in v2 | Pumps: standardising alone is as good. Fans/sliders: network adds value | DONE (finding) |
| 8 | Model too big for a chip | Planning | Model selection: 2.4k–6.9k parameters match 50k | DONE |
| 9 | Calibration breaks when noise changes | Noise-mismatch experiment | Calibrate in real noise or on a mix; tested only on fixed levels | PARTLY done |
| 10 | **Valves do not work** at any noise level | Every experiment | Not solved. Candidate causes: sparse clicks hidden by the clip average. Untested ideas: score the worst frames, different features | PROBLEM |
| 11 | **Pump false alarms 35%** at AUC 0.96 | Final evaluation | Known cause: alert-level rule too strict. K = 100 calibration clips cut it to about 3% but lowered detection (earlier calibration v2 run) | PROBLEM |
| 12 | **Fans/sliders miss faults** (recall 66%, 73%; slider 3% at −6 dB) | Final evaluation | Same cause, opposite direction: alert level too lenient. Not yet fixed | PROBLEM |
| 13 | Heavy noise (−6 dB) lowers AUC to about 0.7 | Final evaluation | None yet. Hypothesis: vibration sensor, which airborne noise does not affect, will help; needs paired sound + vibration data | PROBLEM |
| 14 | Random-seed variation could be mistaken for a real gap | Seeds 0/1/2 on one case | Report averages over seeds; sealed machine reported separately | DONE |
| 15 | "Sealed" machine ID 06 was seen in earlier calibration experiments | Self-check | Stated openly: no *choice* used it, but we knew its general behaviour | DONE (disclosed) |
| 16 | Our phone recordings differ from MIMII in microphone and conditions | First own-recording test | Calibration on own healthy clips; separate healthy session as fairest test; first result AUC about 0.95 but threshold looks tight | PARTLY done |
| 17 | No hardware | — | Everything is a laptop simulation and labelled so everywhere | OPEN |

---

## 8. Current outcome (what you can claim, and what you cannot)

**Can say, with the evidence:**
- Our pipeline reproduces Hitachi's published baseline (correlation 0.97).
- On a machine the model has never heard, 30 healthy clips (about 5 minutes) turn near-chance detection into AUC about 0.92–0.96 for fans, pumps and sliders at 6 dB.
- Both parts of calibration are needed: alert level alone fixes false alarms but not detection; own-machine standardising fixes detection.
- Models of 2,408 to 6,856 parameters (about 9–27 KB at 32-bit) perform like the 50,760-parameter baseline.
- Calibration must be done in the noise the machine runs in.
- A rough first test on your own fan and phone gave AUC about 0.95 (screenshot only, to be re-run).

**Cannot say:**
- That it works for valves (it does not).
- That it works in heavy noise without loss (AUC about 0.7, and recall near 0 for sliders).
- That the alert level is good (recall and false-alarm rates are poorly balanced on pumps, fans and sliders).
- That the neural network is essential for pumps.
- That it runs on the chip, or its power, speed or real memory use.
- Anything about sound + vibration fusion or temperature (no paired data yet).
- That it is novel as a method. Per-domain statistics are an existing idea (AdaFlow, ICASSP 2019, is the closest in `docs/LITERATURE_SURVEY.md`; those papers have only been read at abstract level). The contribution is the low-cost, tiny-model, unseen-machine, with-and-without comparison, with failures reported.

---

## 9. Weaknesses in the code itself (found while reading, none fixed yet)

These are small, but a reviewer reading the code could ask about them.

1. **Global path switching.** Several files do `F.FEATURE_ROOT = ...` at run time to choose the 32/64/128-band folder. It works, but any import order or parallel use could point at the wrong folder.
2. **Hard-coded N: drive paths** in `features.py`, `model_selection.py`, `demo.py`, `final_evaluation.py`, `dashboard_gui.py`. Nothing runs on another computer without edits.
3. **Cached demo models are keyed by type/machine/noise only**, not by design (`demo.py` line 51), so a changed design can load a stale model.
4. **Unused validation slice** in `model_selection.train` (lines 79–80): 10% of the training data is thrown away and never used.
5. **Hitachi's table typed by hand** in `compare_hitachi.py` (lines 11–16).
6. **Dashboard "Evaluate" uses a cached seed-0 model**, so it differs slightly from the 3-seed table.
7. **Legacy scripts** at the repository root point at data that no longer exists.
8. **`PROJECT_GUIDE.md` is behind** the code (section 0 of this file lists what is missing).
9. **Alert-level rule** (mean + 3 × spread of 30 clip scores) assumes healthy scores look like a bell curve; they are skewed, which is one likely reason it is wrong in opposite directions for pumps and for fans/sliders.

---

## 10. Future plan

### 10.1 Near term, no hardware needed (PLAN)

| Step | What and why | Notes |
|---|---|---|
| A | **Fix the alert level.** Three failure patterns (pump too strict, fan/slider too lenient, own recording tight) all point here | Options to *test*, none tried except the first: more calibration clips (K = 100 tried earlier: pump false alarms 35% → about 3%, detection lower); a percentile-based level instead of mean + 3 spreads; a tuned multiplier chosen on tuning machines only and checked on the sealed one |
| B | **Re-run the own-recording test** and save the numbers to a file; collect a second session at another speed (`own_data/other_speed` is empty) | Gives reproducible numbers the panel can rerun |
| C | **Try extra scores on top of the autoencoder** (the idea you raised: Mahalanobis distance on the 8-dimensional bottleneck, PCA reconstruction error, Isolation Forest) and combine them | Not built. My estimate of cost: Mahalanobis on the bottleneck under 1 KB; PCA with 16 components about 21 KB; a small forest about 150 KB. Keep only what raises recall at low false alarms on the **tuning** machines, then confirm on the sealed machine |
| D | **Valves:** test scoring the worst frames of a clip instead of the clip average | One-file change in the scoring function |
| E | **INT8 quantisation** and accuracy check | On hold at your request |
| F | **Audio-in, decision-out simulation** (a stream of small chunks, calibrating in the first minutes, a decision every second) | Python first, then C |

### 10.2 Chip work (PLAN; needs the board)
Features in portable C verified against `features.py` clip by clip; a table generator so the Hann window and mel table are produced from librosa, never typed; per-type model export (replacing the legacy header exporter); measure speed, memory, power. The specification is in PROJECT_GUIDE section 8: window 1024 and hop 512, Hann, FFT, power, 64-band mel, `10·log10`. With single-frame models the 5-row buffer is not needed.

### 10.3 Sound + vibration + temperature (PLAN; needs paired data)
Use the Jung et al. 2023 dataset (DOI 10.17632/ztmf3m7h5x.6) or your own rig to compare early fusion (joined features) and late fusion (separate scores, combined rule). Late fusion cannot make the sound detector worse. Whether vibration rescues the heavy-noise case is a hypothesis.

### 10.4 Environmental part and patent
Energy and noise estimates need a power sensor and a reference sound meter (hardware). Patent: filing is required; software alone is excluded in India (Section 3(k)); the narrow idea noted in PROJECT_GUIDE section 14 is estimating the extra energy and noise caused by a fault relative to the machine's own calibrated baseline, on a low-cost node. File before publishing. A professional prior-art search is still needed.

### 10.5 Dates
50% review in days; KSCST proposal due **30 November 2026**; proposed go/no-go on hardware **10 November**.

---

## 11. Questions the panel may ask, and short honest answers

| Question | Answer |
|---|---|
| What is the architecture? | One small dense autoencoder per machine type: 64-32-32-8-32-32-64 for fans, 64-16-8-16-64 for pumps, 32-32-16-32-32 for sliders, 64-32-16-32-64 for valves (which does not work). The Hitachi baseline 320-64-64-8-64-64-320 is our reference. |
| Why did it get smaller? | Model selection showed 2.4k–6.9k parameters score as well as 50k, and a single frame is enough. |
| Why does calibration matter? | Machines of one type differ in loudness and tone. Without own-machine standardising, a healthy new machine looks faulty to the model. |
| Is the test fair? | The test machine is never in training; calibration clips are never test clips; design choices used machines 00/02/04 only; ID 06 reported separately. One machine per type is a limit. |
| Why do valves fail? | Probably short, sparse clicks averaged away over 10 seconds. Not proven. |
| Why 35% false alarms on pumps with AUC 0.96? | The scores separate well, but the alert line is in the wrong place for pumps. Fixable in principle, not yet fixed. |
| Does it run on a chip? | Not yet tested. The models are small (about 2–7 KB at INT8 by estimate), but nothing has been measured on hardware. |
| Is vibration used? | Not in the results. The public vibration data is from a different machine than the sound data, so combining them would be meaningless. |
| What is new? | Not the idea of per-machine statistics. The contribution is a low-cost version (30 healthy clips, no retraining, tiny model), tested on unseen machines with a with/without comparison, failures reported. |

---

## 12. How to re-run things (from `C:\dev\capstone\acoustic`, with the N: drive connected)

```
python final_evaluation.py                 # the 144-row headline table (hours)
python summarize_final.py                  # writes results/final_summary.md
python ablation_calibration.py             # calibration ablation (fan, pump, slider, 6 dB)
python dashboard_gui.py                    # the five-tab interface
python demo.py --type fan --held id_06 --clips 16
python split_recordings.py --src "<file>" --name healthy_session1
python own_recording_test.py --healthy ..\own_data\clips\healthy_session1 --healthy-test ..\own_data\clips\healthy_session2 --faulty ..\own_data\clips\faulty --type fan
```
More commands (baseline, calibration v2, model selection, vibration) are in PROJECT_GUIDE section 16.
