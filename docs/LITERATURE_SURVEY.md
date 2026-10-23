# Related work (verified list, 22 Oct 2026)

**How this list was made.** Web searches, then I opened the arXiv / publisher listing page of each paper and read its title, authors, year and abstract.
**No paper here has been read in full.** "Abstract" = abstract page read. "Listing" = title/venue confirmed from a search result or a review's reference list only. Methods and numbers inside the papers are NOT checked unless stated.
This file replaces an earlier version that attributed two ideas to the wrong papers (see "Corrections" at the end).

## A. The dataset and the benchmark community (DCASE = yearly sound-analysis challenge; Task 2 = machine fault detection by sound)
| # | Reference | What it is, in plain words | Checked |
|---|---|---|---|
| A1 | H. Purohit, R. Tanabe, K. Ichige, T. Endo, Y. Nikaido, K. Suefusa, Y. Kawaguchi (Hitachi). **MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection.** DCASE 2019 Workshop. arXiv:1909.09347 | Our dataset: valves, pumps, fans, slide rails; healthy and faulty sound; baseline autoencoder | Abstract |
| A2 | R. Tanabe et al. **MIMII DUE: ... with Domain Shifts due to Changes in Operational and Environmental Conditions.** 2021. arXiv:2105.02702 | Five machine types in two conditions (source, target); large performance gaps between them | Abstract |
| A3 | K. Dohi et al. **MIMII DG: ... for Domain Generalization Task.** 2022. arXiv:2205.13879 | Five machine types, three kinds of shift per type, for testing generalisation to new conditions | Abstract |
| A4 | Y. Koizumi et al. **Description and Discussion on DCASE2020 Challenge Task2: Unsupervised Anomalous Sound Detection for Machine Condition Monitoring.** arXiv:2006.05822 | First benchmark: 117 submissions from 40 teams; training uses healthy sound only | Abstract |
| A5 | Y. Kawaguchi et al. **Description and Discussion on DCASE 2021 Challenge Task 2 ... under Domain Shifted Conditions.** arXiv:2106.04492 | Adds domain shift (different conditions, products, noise); 75 submissions from 26 teams | Abstract |
| A6 | Y. Koizumi et al. **ToyADMOS: A Dataset of Miniature-Machine Operating Sounds for Anomalous Sound Detection.** WASPAA 2019. arXiv:1908.03299; N. Harada et al. **ToyADMOS2.** 2021. arXiv:2106.02369 | The other main machine-sound datasets used in DCASE | Abstract |
| A7 | K. Wilkinghoff, T. Fujimura, K. Imoto, J. Le Roux, Z.-H. Tan, T. Toda. **Handling Domain Shifts for Anomalous Sound Detection: A Review of DCASE-Related Work.** 2025. arXiv:2503.10435 | A survey of how DCASE teams deal with machines or conditions that change. **Best place to start reading.** | Opened, read the relevant parts |

## B. Closest to our calibration idea (adapting to a new machine)
| # | Reference | What it does | Checked |
|---|---|---|---|
| B1 | M. Yamaguchi, Y. Koizumi, N. Harada (NTT). **AdaFlow: Domain-Adaptive Density Estimator with Application to Anomaly Detection and Unpaired Cross-Domain Translation.** ICASSP 2019. arXiv:1812.05796 | Combines a density model with **adaptive batch normalization** so it adapts to a new data distribution in one forward pass per sample, aimed at limited compute; tested on sound anomaly detection. **Closest known idea to ours.** | Abstract (search summary) |
| B2 | K. Wilkinghoff, H. Yang, J. Ebbers, F. G. Germain, G. Wichern, J. Le Roux (MERL). **Local Density-Based Anomaly Score Normalization for Domain Generalization.** arXiv:2509.10951 (IEEE TASLP 2026). Earlier version: **Keeping the Balance: Anomaly Score Calculation for Domain Generalization.** ICASSP 2025 | Normalises anomaly scores so that one alert threshold works across different domains | Abstract (whether it needs samples from the new domain is not stated on the page) |
| B3 | P. Saengthong, T. Shinozaki. **Deep Generic Representations for Domain-Generalized Anomalous Sound Detection.** ICASSP 2025. arXiv:2409.05035 | Uses a large pre-trained audio network plus nearest-neighbour matching, no fine-tuning | Abstract (search summary) |
| B4 | N. Harada, D. Niizumi, Y. Ohishi, D. Takeuchi, M. Yasuda. **First-shot anomaly sound detection for machine condition monitoring: A domain generalization baseline.** EUSIPCO 2023 | Baseline for the first-shot task (a new machine type with few samples) | Listing only (from the review's references) |
| B5 | S. Lugani, C. Pichler, H. Wegleiter, C. Feilmayr, R. Zunino, E. Ragusa. **Few-Shot Adaptation for Anomalous Sound Detection at Low False-Positive Rate.** 2026. arXiv:2610.19893 | Adapts after the first real fault using a few faulty samples; needs fault examples, ours does not | Abstract |

## C. Small devices (microcontrollers, TinyML)
| # | Reference | What it does | Checked |
|---|---|---|---|
| C1 | C. Banbury et al. **MLPerf Tiny Benchmark.** NeurIPS 2021 (Datasets and Benchmarks). arXiv:2106.07597 | Standard benchmark for tiny devices; includes an anomaly-detection task using a small autoencoder on ToyADMOS machine sounds. Use it as the reference for how to report size, speed and accuracy on small hardware | Listing and secondary descriptions; reported numbers differ between sources, check the paper before quoting any |
| C2 | A. E. Kalor, D. Michelsanti, F. Chiariotti, Z.-H. Tan, P. Popovski. **Remote Anomaly Detection in Industry 4.0 Using Resource-Constrained Devices.** arXiv:2110.05757 | Anomaly detection (PCA and autoencoder) on vibration/acoustic data from constrained sensors over wireless links | Abstract |
| C3 | **TinyOL: TinyML with Online-Learning on Microcontrollers.** arXiv:2103.08295 | Learning on the microcontroller itself | Title and link only; authors not confirmed |

## D. Vibration, bearing data and sound + vibration
| # | Reference | What it does | Checked |
|---|---|---|---|
| D1 | W. A. Smith, R. B. Randall. **Rolling element bearing diagnostics using the Case Western Reserve University data: A benchmark study.** Mech. Syst. Signal Process. 64-65:100-131, 2015. DOI 10.1016/j.ymssp.2015.04.021 | Shows how to use the CWRU data properly; our CWRU work should follow its cautions | Listing and abstract summary |
| D2 | D. Hendriks, A. Dumond, J. Knox. **Towards better benchmarking using the CWRU bearing fault dataset.** MSSP 169:108732, 2022 | Follow-up on benchmarking with CWRU | Listing only |
| D3 | W. Jung, S.-H. Kim, S.-H. Yun, J. Bae, Y.-H. Park. **Vibration, acoustic, temperature, and motor current dataset of rotating machine under varying operating conditions for fault diagnosis.** Data in Brief 48:109049, 2023. DOI 10.1016/j.dib.2023.109049 | The one public dataset we found with sound, vibration, temperature and current from the same machine; the candidate for testing fusion | Abstract |

## E. Leads found but NOT confirmed (do not cite until opened)
- A 2024 Journal of Maritime Research paper on autoencoders for fan vibration on an Arduino (title and authors not retrieved; link from search: jmr.unican.es).
- "Hammad et al.": an LSTM autoencoder for noise anomalies on an ESP32, mentioned only inside another paper.
- A 2025 *Actuators* paper on acoustic-vibration fusion for bearing faults (multi-branch convolution with dynamic weighting): title confirmed from a listing, authors not.
- Kolok et al. (cited in the proposal): the project guide records that its abstract and body disagree; cite it only for cost and complexity, not accuracy.

## E2. Second search: IEEE Xplore and other sources (23 Oct 2026)
Checked = what I could open. IEEE Xplore pages themselves did not load for me (empty page), so IEEE items are "listing only" unless an arXiv copy was opened.

| # | Reference | What it is | Checked |
|---|---|---|---|
| E2.1 | B. Chen, L. Bondi, S. Das. **Learning to Adapt to Domain Shifts with Few-shot Samples in Anomalous Sound Detection.** 2022. arXiv:2204.01905 (also IEEE listing 9956351) | Adapts to new machine conditions with a few target samples (meta-learning, models normal samples). Number of samples and dataset not stated on the abstract page. Very close to our "adapt with few healthy clips" | Abstract |
| E2.2 | S. Abbasi, M. Famouri, M. J. Shafiee, A. Wong. **OutlierNets: Highly Compact Deep Autoencoder Network Architectures for On-Device Acoustic Anomaly Detection.** 2021. arXiv:2104.00528 | Compact autoencoders for machine-sound anomaly detection; smallest has 686 parameters and 2.7 KB; reports accuracy matching much larger published models and up to 21x lower CPU latency. Dataset and hardware not named on the abstract page. **Closest to our "tiny model" result** | Abstract |
| E2.3 | C. Banbury et al. **MicroNets: Neural Network Architectures for Deploying TinyML Applications on Commodity Microcontrollers.** 2020. arXiv:2010.11267 | Models for three tiny-device benchmark tasks including anomaly detection; run on microcontrollers. Dataset not named on the abstract page (secondary sources say MIMII slide rail for this paper and ToyCar for MLPerf Tiny; check the paper) | Abstract |
| E2.4 | A. Almaini, J. Folz, G. Ashour. **TinyML for Acoustic Anomaly Detection in IoT Sensor Networks.** 2026. arXiv:2603.26135 (IEEE listing 11277514) | MFCC plus a small classifier, 91% accuracy, on **UrbanSound8K = environmental sound, not machine sound** | Abstract |
| E2.5 | M. O. Jakobsen. **Low cost MEMS accelerometer and microphone based condition monitoring sensor, with LoRa and Bluetooth Low Energy radio.** HardwareX 18:e00525, 2024 | A low-cost sensor with a MEMS microphone and accelerometers; tested on a shaker table and on a bearing with an artificial outer-ring defect; microphone and accelerometer both showed the defect frequencies. Hardware paper; no machine-learning detector | Opened (PMC copy) |
| E2.6 | P. Kolok, M. Hodon, P. Sevcik, L. Hotz, N. Remy. **Low-Cost IoT-Based Predictive Maintenance Using Vibration.** Sensors 25(21):6610, 2025. doi:10.3390/s25216610 | **This is the "Kolok et al." in our proposal.** ESP32-C6 + MPU6050 + INMP441. Opened: the microphone is only a design concept, the system is vibration-only; Isolation Forest runs on a laptop; ROC-AUC 0.78, accuracy 0.73; the paper also quotes 93.4% and several different latencies, so it is internally inconsistent. One fault type (imbalance) | Opened (PMC copy) |

**Listing only (not opened; do not cite details until opened):**
- "Modified Overcomplete Autoencoder for Anomaly Detection Based on TinyML", IEEE Xplore 10684143. Search summary: USB-fan broken-blade faults, about 17 kB on an Arduino Nano 33 BLE Sense, 99.23% accuracy. **Very close to our fan-on-a-microcontroller work: open this first.**
- "An Adaptable and Unsupervised TinyML Anomaly Detection System for Extreme Industrial Environments" (PMC9962960): Isolation Forest on an ESP32 for submersible pumps, per search summary.
- "Edge-AI Label-Free Anomaly Detection for Home-Appliance Predictive Maintenance: Representation, Quantization, and the Limits of Cross-Appliance Transfer" (ResearchGate): title suggests it studies how well a model transfers between appliances. Directly relevant; not opened.
- DCASE 2023, 2024, 2025 Task 2 descriptions: arXiv 2305.07828, 2406.07250, 2506.10097 (the "first-shot" tasks).
- GitHub hobby projects with ESP32 + INMP441 (for example an Echo-State-Network fan-fault rig). Not papers; cite only as examples.

**Gap found (a possible strength for the proposal, not yet proven):** the search found nothing that estimates the extra energy or noise caused by a fault (the environmental-impact part). Searches returned general condition-monitoring work only. A narrower search for "fault-induced energy loss" is still needed before claiming this is new.

## F. What this means for our novelty claim (honest reading)
- **Closest prior idea:** B1 (AdaFlow). It adapts a model to a new distribution using adaptive batch normalization, aimed at limited compute, with sound anomaly detection in the abstract. Per-machine statistics are the same family of idea as ours.
- **Related:** B2 normalises scores per domain; B3 and B4 target generalisation to new machines or conditions by other means.
- **What may differ in our work (to be confirmed by reading B1-B4 properly):** (1) input-feature standardisation from only 30 healthy clips, no labels, no fault samples, no retraining; (2) very small models (2.4k-6.9k parameters) sized for a microcontroller; (3) leave-one-machine-out test on MIMII with a with/without-calibration comparison and reported failures; (4) a test on our own fan with a phone microphone; (5) a path to a real device.
- **We did not find a paper that does exactly this on MIMII.** That reflects a limited search, not proof that none exists. IEEE Xplore was not searched directly.

## G. Corrections to the earlier version of this file
- Saengthong and Shinozaki's paper is B3 (ICASSP 2025, pre-trained features plus nearest neighbours). I had described it as "domain-wise standardisation of anomaly scores" from a review's one-line note; I could not confirm that from the paper itself.
- Yamaguchi et al.'s paper is B1 (AdaFlow, ICASSP 2019). I had described it as "fine-tune only the batch-normalisation layers" from the same review; what I could confirm is that it uses adaptive batch normalisation in a density model.
- Harada et al., as cited by the review, is the first-shot baseline paper (B4), not the earlier ToyADMOS2 paper alone.

## H. To do next
- Read B1, B2, B3 and the A7 review properly (methods and results), then write the one-paragraph comparison with our method.
- Search IEEE Xplore for "TinyML acoustic anomaly detection" and "machine sound anomaly detection microcontroller".
- Open and confirm section E before citing anything from it.
