---
marp: true
theme: default
paginate: true
size: 16:9
header: 'CarSpec AI · Module 2 Pitch'
footer: 'hanfuzhao781-carspec-ai.hf.space'
style: |
    section {
        background: #faf7f2;
        color: #0a0a0a;
        font-family: 'Space Grotesk', sans-serif;
        padding: 60px 80px;
    }
    h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 800;
        font-size: 64px;
        letter-spacing: -0.04em;
        line-height: 0.95;
        color: #0a0a0a;
    }
    h2 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 800;
        font-size: 48px;
        letter-spacing: -0.03em;
        color: #0a0a0a;
        border-bottom: 3px solid #dc2626;
        padding-bottom: 12px;
        margin-bottom: 32px;
    }
    h3 {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 14px;
        color: #dc2626;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-bottom: 12px;
    }
    table {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 16px;
        width: 100%;
        border-collapse: collapse;
    }
    th {
        background: #0a0a0a;
        color: #faf7f2;
        padding: 12px 16px;
        text-align: left;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-size: 12px;
    }
    td {
        padding: 12px 16px;
        border-bottom: 1px solid #d6d3d1;
    }
    code {
        background: #0a0a0a;
        color: #dc2626;
        padding: 2px 8px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 14px;
    }
    strong { color: #dc2626; }
    section.title {
        background: #0a0a0a;
        color: #faf7f2;
    }
    section.title h1 { color: #faf7f2; }
    section.title h2 { color: #dc2626; border-bottom-color: #dc2626; }
    section.title h3 { color: #faf7f2; opacity: 0.6; }
    blockquote {
        border-left: 4px solid #dc2626;
        padding-left: 24px;
        font-style: italic;
        color: #44403c;
    }
---

<!-- _class: title -->

# CarSpec AI
## Read a car like a spec sheet.
### Module 2 Pitch · Computer Vision · 5 min

---

## 01 / Problem

> Traditional vehicle recognition predicts **one** attribute and behaves like a black box.

**Two pains:**

1. **Single-task** - one attribute at a time
2. **Black box** - users get a label, no clue how it was reached

**CarSpec AI** predicts three correlated attributes in one pass and shows the visual features behind each call.

---

## 02 / Hypothesis

Vehicle attributes are **correlated**:

| Type | Doors | Seats |
|---|---|---|
| coupe | 2 | 2 |
| sedan | 4 | 5 |
| MPV | 5 | 7 |

A **multi-task** model with shared backbone should pick up these correlations and push accuracy up - and the handcrafted features leave a trail a human can audit.

---

## 03 / Method - Three Models

| | Naive | Classical | Deep |
|---|---|---|---|
| **Algorithm** | Majority class | Random Forest | MobileNetV2 |
| **Input** | - | 50-D handcrafted | 224×224 RGB |
| **Heads** | 1 | 3 (independent) | 3 (shared) |
| **Why** | Sanity floor | Interpretable | Multi-task joint |

**Backbone**: MobileNetV2 (ImageNet pretrained) → 256-D FC → 3 heads (5/3/3 classes).

---

## 04 / Pipeline

```
[ CompCars 136k images ]
        ↓
   224×224 + normalize
        ↓
   ┌────────────┬──────────────┐
   │  50-D      │   MobileNetV2 │
   │ handcrafted│   backbone    │
   │  features  │      ↓        │
   │   ↓        │   3 heads     │
   │ RF (×3)    │   multi-task  │
   └────────────┴──────────────┘
        ↓
 predictions + confidence + explanations
```

**50-D features**: HSV histogram (24) · aspect ratio · edge density · body proportions · symmetry · HOG stats (3) · LBP (16).

---

## 05 / Live Demo

### https://hanfuzhao781-carspec-ai.hf.space

- Upload vehicle photo → 3 attributes in one pass
- Top-5 candidates + confidence-gated feedback
- Classical vs Deep side-by-side comparison
- 50-D feature breakdown

**Health check**: `GET /health` → `{"status":"ok","models_loaded":7}`

---

## 06 / Results

| Model | car_type | door_count | seat_count |
|---|---|---|---|
| Naive (majority) | 0.229 | 0.577 | 0.576 |
| Classical (RF) | 0.407 | 0.611 | 0.609 |
| **Deep (MobileNetV2)** | **0.771** *(top5=1.0)* | **0.819** *(top5=1.0)* | **0.860** *(top5=1.0)* |

Evaluated on 969 held-out real car photos (80/20 split of 4,869 images crawled via Bing). Deep model correctly predicts **5/5** demo samples with confidence 0.97–0.98. Scaling from 100 → 4,869 images lifted car_type accuracy from 0.55 → 0.77 (+22 pts).

---

## 07 / Insights

1. **Multi-task lifts door/seat** - 0.82/0.86 vs 0.77 car_type. Shared backbone picks up car_type↔door↔seat correlations.
2. **Deep beats classical by 36 pts** on car_type (0.77 vs 0.41) - learned features beat handcrafted on real photos.
3. **Classical doubles naive** (0.41 vs 0.23) - 50-D handcrafted features carry real discriminative signal.
4. **Robustness** - mean accuracy 0.543 under gaussian noise (3 severity levels); classical barely moves, good fallback.
5. **Handcrafted features read out as sentences**: "dominant color is red", "aspect ratio 1.2 → leans SUV".

---

## 08 / Engineering

- **Code**: 12 modular scripts (crawl → clean → train → eval → deploy)
- **Models**: 6 `.pkl` + 1 `.pt` hosted on HF Hub (auto-downloaded at runtime)
- **Deployment**: HF Space · gunicorn · Python 3.11 · Dockerfile
- **CI**: keep-alive cron every 6h
- **Git**: 19 PRs, branch-based workflow, PR template
- **Reports**: `TECHNICAL_REPORT.md` (15 sections) · `GRADING.md` (rubric map) · `README.md`

---

<!-- _class: title -->

## 09 / Future

> If I had another semester...

1. Train on full **136k CompCars** images with GPU
2. **Grad-CAM** heatmaps for CNN attention
3. **SE-Block / CBAM** attention modules
4. **Multi-view fusion** using CompCars view annotations
5. **TensorRT / ONNX** for real-time inference

---

<!-- _class: title -->

# Thank you.
## Questions?

### https://hanfuzhao781-carspec-ai.hf.space
### github.com/hanfuzhao/carspec-ai
