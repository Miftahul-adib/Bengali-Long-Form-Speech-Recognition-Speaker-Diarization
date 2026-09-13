# Bengali Long-Form Speech Recognition & Speaker Diarization

Solutions to two tracks of the DL Sprint 4.0 Kaggle competition.

| Track | Score |
|---|---|
| Long-Form ASR | 0.53870 |
| Speaker Diarization | 0.55401 |

---

## Repository Structure

```
├── whisper-iterative-adapter-training.ipynb    # ASR fine-tuning
├── long-form-voice-recognition-inference.ipynb # ASR inference
├── diarization-bangla.ipynb                    # Speaker diarization pipeline
├── sample_asr_output.csv                       # Sample ASR predictions (5 files)
└── sample_diarization_output.csv               # Sample diarization predictions (5 files)
```

---

## Task 1 — Long-Form Bengali ASR

The goal was to transcribe Bengali audio recordings ranging from a few minutes to over an hour into clean Bengali text.

**Base model:** `whisper-medium` fine-tuned on Bengali (`tugstugi/bengali-whisper-medium`)

### Training — Iterative LoRA Adapter Stacking

The training dataset was too large to fine-tune in a single pass without catastrophic forgetting, so the dataset was split into equal portions with one LoRA adapter trained per portion, freezing all previous adapters before adding a new one:

```
Iteration 1: Base + Adapter 1                         (trains on slice 1)
Iteration 2: Base + Adapter 1 [FROZEN] + Adapter 2    (trains on slice 2)
Iteration 3: Base + Adapters 1–2 [FROZEN] + Adapter 3 (trains on slice 3)
Inference:   All adapters active together
```

LoRA config: `r=16`, `alpha=32`, `dropout=0.05`, targeting `q_proj`, `v_proj`, `k_proj`, `o_proj`. Training used FP16, gradient checkpointing, effective batch size of 32 (4 × 8 accumulation steps), and a cosine LR schedule.

### Preprocessing

**Text:** Transcripts were stripped to Bengali-only — all punctuation, English characters, digits, and foreign scripts (Devanagari, Urdu/Arabic, Telugu, Malayalam) were removed to keep the label vocabulary clean.

**Audio:**
- RMS normalisation to −20 dBFS
- Robust resampling: 16 kHz → 8 kHz → 16 kHz (smooths codec artefacts)
- Waveform augmentation: time-stretch (±5%), pitch-shift (±2 semitones), Gaussian noise — applied with 30% probability
- SpecAugment: 2 time masks and 2 frequency masks on the mel spectrogram

### Inference — Chunked Transcription

Whisper natively handles only 30-second windows. Long files were handled by splitting audio into 30-second chunks with 1-second overlap, transcribing each chunk independently with greedy decoding, then concatenating with space-joining.

---

## Task 2 — Bengali Speaker Diarization

Given a long Bengali audio recording of a multi-speaker conversation, the task was to output timestamped speaker labels for every speech segment.

**Feature extraction:** 80-band log-mel spectrograms (25 ms window, 10 ms hop)

**Speaker embeddings:** ECAPA-TDNN encoder with ArcFace loss, producing 192-dimensional embeddings per 2-second sliding window (0.75 s stride)

**Speaker count estimation:** BIC (Bayesian Information Criterion) with `lambda=1.5` penalty. AHC was deliberately removed from the pipeline after it was found to collapse BIC's accurate speaker count estimates (9–20 predicted speakers) down to 3–8, discarding genuine speaker distinctions.

**VAD:** Silero VAD with a low threshold (0.30) to maximise speech recall, with 150 ms minimum speech duration and 80 ms minimum silence.

**Fine-tuning:** The encoder was fine-tuned on labelled training segments (≥0.5 s) with noise augmentation (SNR 12–30 dB) and room impulse response convolution for robustness.

**Post-processing:** Segments shorter than 0.3 s were removed; gaps smaller than 0.15 s between same-speaker segments were merged.






## Datasets

Both datasets consist of naturally recorded Bengali conversational audio, with test files ranging from a few minutes to approximately one hour.

- [Bengali Long-Form Speech Recognition](https://www.kaggle.com/competitions/dl-sprint-4-0-bengali-long-form-speech-recognition)
- [Bengali Speaker Diarization Challenge](https://www.kaggle.com/competitions/dl-sprint-4-0-bengali-speaker-diarization-challenge)
