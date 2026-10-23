# Review brief (read this tonight)

## 1. The project in 30 seconds
Machines make a different sound when they are faulty. We built a small program that learns what a **healthy** machine sounds like and raises an alert when a new sound does not match. A short **calibration** step (30 healthy clips, no fault examples) lets the same program work on a machine it has never heard. The program is tiny (2,400 to 6,900 numbers), so it could run on a cheap chip (ESP32). The chip version is not built yet.

## 2. What we did, in order (this is your story)
1. Used the public MIMII sound dataset: 4 machine types (fan, pump, slider, valve), 4 machines of each, healthy and faulty clips, at 3 noise levels (6, 0, -6 dB).
2. Turned each 10-second clip into a picture of sound over time and frequency (a log-mel spectrogram).
3. Trained a small autoencoder on **healthy** sound only. It learns to rebuild healthy sound well. A faulty sound rebuilds badly, so a high rebuild error means "possible fault".
4. Tested the hard way: the model trains on 3 machines and is tested on the 4th, which it never saw (leave-one-machine-out). One machine (ID 06) was kept sealed: not used to choose any settings.
5. Calibration: on the new machine, measure 30 healthy clips to set its normal level, and set the alert line at average + 3 x spread of the next 30 healthy clips.
6. Compared with and without calibration to check it really helps.
7. Built an interface (dashboard_gui.py) to pick a WAV file and see the verdict and all metrics.

## 3. Results (all measured, not estimated)
At 6 dB noise, average over the 4 machines, 3 random seeds:

| Machine | AUC | Recall (faults caught) | False alarms |
|---|---|---|---|
| Fan | 0.92 | 0.66 | 5% |
| Pump | 0.96 | 0.89 | 35% |
| Slider | 0.94 | 0.73 | 4% |
| Valve | 0.56 | 0.01 | 2% |

Noise effect (AUC): fan 0.92 -> 0.83 -> 0.71, pump 0.96 -> 0.84 -> 0.75, slider 0.94 -> 0.84 -> 0.72 (6, 0, -6 dB).

Does calibration help? Average AUC without -> with calibration (6 dB): fan 0.53 -> 0.92, pump 0.51 -> 0.96, slider 0.85 -> 0.95. Without it, fans and pumps are at guessing level.

Sealed machine ID 06 at 6 dB: fan 1.00, pump 0.97, slider 1.00, valve 0.52 (AUC). Say this one separately; it is the fairest test.

## 4. What does not work (say this yourself before they ask)
- **Valves**: AUC about 0.5 at every noise level. The valve faults are very quiet. Not solved.
- **Pumps**: ranks faults well but raises many false alarms (35%). The alert line is badly placed. Fixable in principle, not yet fixed.
- **Fans and sliders**: alert line too lenient, so some faults are missed (recall 0.66 and 0.73).
- **Heavy noise (-6 dB)**: AUC about 0.7; sliders catch almost nothing.
- **Noise change after calibration**: calibration breaks if the noise is different from when you calibrated. Calibrate in the machine's real noise.
- **No hardware yet**: power, real speed, memory on the chip, temperature sensing and sound+vibration fusion are NOT measured. The vibration work (CWRU data) is a separate, easy dataset and is not combined with sound.

## 5. Plan for the next 4 months
1. Fix the alert line (pumps, fans, sliders). 2. Build the device (ESP32 + microphone + vibration + temperature sensor). 3. Put it on a real machine, cause a fault ourselves, record sound + vibration + temperature together. 4. Test fusion and thermal on that data. 5. Measure power, speed, memory. 6. Write the paper.

## 6. Questions you will probably get
- **What is new?** Honest answer: per-machine adjustment is known in the sound-anomaly research community. Our contribution is a low-cost version (30 healthy clips, no retraining, tiny model), tested on unseen machines with a with/without comparison and failures reported. A deeper literature check is in progress (docs/LITERATURE_SURVEY.md).
- **Why does calibration matter?** Each machine has its own loudness and tone. Without calibration a normal sound from a new machine looks like a fault. With it, results go from guessing to about 0.9+.
- **Why does it fail on valves?** Valve faults barely change the sound in this dataset. Other sensors (vibration, temperature) might help; untested.
- **Will it run on a microcontroller?** The model is small enough (about 10 to 27 KB, roughly 30,000-50,000 operations per frame; estimates). We have not run it on a chip yet.
- **Is vibration used?** Not in the results. Fusion needs sound and vibration recorded from the same machine; the public data does not have it.
- **How do you know the test is fair?** Test machine never seen in training; calibration clips never used for testing; one machine sealed from all design choices.
- **What is the output?** A score per clip, compared with an alert line: above the line = alert.

## 7. Words to know (learn these 8)
- **Autoencoder**: a small network that squeezes the input and rebuilds it; trained only on healthy sound.
- **Reconstruction error**: how badly the rebuild differs from the input. High = unusual.
- **Log-mel spectrogram**: sound turned into a picture of loudness per frequency band over time, spaced like human hearing.
- **Calibration**: measuring normal for this machine (30 healthy clips) before use.
- **AUC**: chance that a random faulty clip scores higher than a random healthy one. 0.5 = guessing, 1.0 = perfect. It does not depend on the alert line.
- **Recall**: share of real faults caught. **Precision**: share of alerts that are real faults. **False-alarm rate**: share of healthy clips wrongly flagged.
- **Leave-one-machine-out**: train on three machines, test on the fourth.
- **dB (noise level)**: 6 dB = machine louder than the background; -6 dB = background louder than the machine.
