# GRADING.md - Rubric Mapping

Each rubric item points to the file/section/URL where the evidence lives.

## 1. Project Topic & Originality
- **Topic**: Computer vision - vehicle multi-attribute recognition (car type / door count / seat count). See `TECHNICAL_REPORT.md` §1 Problem Statement.
- **Originality**: Multi-task learning + interpretable handcrafted features + experiments on robustness / gating / head-tail (robustness / confidence gating / head-tail). See `TECHNICAL_REPORT.md` §3 Related Work, last paragraph; `README.md` → Originality Statement.
- **Evidence**: `TECHNICAL_REPORT.md`, `README.md`.

## 2. Modeling Requirements (three models)
| Requirement | Code | Trained Model | Report Section |
|------|------|-----------|------|
| Naive baseline (majority + random) | `scripts/model.py::NaiveBaseline` | `models/naive_*.pkl` | `TECHNICAL_REPORT.md` §5.1, §8.1 |
| Classical ML (non-deep) | `scripts/model.py::ClassicalModel` (Random Forest + 50-dim features) | `models/classical_*.pkl` | `TECHNICAL_REPORT.md` §5.2, §8.1 |
| Deep Learning | `scripts/model.py::DeepMultiTaskModel` (MobileNetV2 multi-task) | `models/deep_multitask.pt` | `TECHNICAL_REPORT.md` §5.3, §8.1 |

Training pipeline: `setup.py` (9 steps). Reproduce with `python setup.py`.

## 3. Experiments & Analysis
| Requirement | Code | Output | Report Section |
|------|------|------|------|
| Robustness (4 corruptions × 3 severities) | `scripts/experiment.py::robustness_experiment` | `data/outputs/robustness.png`; `metrics.json.robustness` | `TECHNICAL_REPORT.md` §10.4 |
| Confidence gating / selective prediction | `scripts/experiment.py::confidence_gating_experiment` | `data/outputs/confidence_curve.png` + `confidence_analysis.json`; `metrics.json.confidence_gating` | `TECHNICAL_REPORT.md` §10.5 |
| Head/tail analysis | `scripts/experiment.py::head_tail_analysis` | `metrics.json.head_tail` (with `gap`) | `TECHNICAL_REPORT.md` §10.6 |
| Data size sensitivity | `scripts/experiment.py::data_size_sensitivity` | `metrics.json.data_size_sensitivity` | `TECHNICAL_REPORT.md` §10.2/§10.3 |
| Error analysis (≥5 mispredictions with confidence + test_index) | `scripts/experiment.py::error_analysis` | `metrics.json.error_cases` (15 cases) | `TECHNICAL_REPORT.md` §9 |
| Confusion matrices | `scripts/experiment.py::plot_confusion_matrix` | `data/outputs/confusion_matrix.png` + `.npy`; `data/outputs/plots/cm_*.png` | `TECHNICAL_REPORT.md` §8.3 |
| Quantitative model comparison | `scripts/experiment.py::plot_model_comparison` | `data/outputs/model_comparison.png`; `metrics.json` | `TECHNICAL_REPORT.md` §8.1 |

## 4. Interactive Application
| Requirement | Location |
|------|------|
| Flask app (inference-only) | `main.py` |
| `/health` endpoint | `main.py` → `/health` |
| `/predict` endpoint returns `prediction` / `confidence` / `top_k` / `feedback` | `main.py::predict` |
| 400 error for non-image / unsupported format | `main.py::validate_upload` + `@app.errorhandler(400)` |
| 413 error for files > 16MB | `main.py::MAX_CONTENT_LENGTH` + `@app.errorhandler(413)` |
| `/samples` endpoint (lists sample images) | `main.py::samples` |
| Drag-and-drop upload + preview | `templates/index.html`, `static/js/app.js` |
| Confidence-based feedback box (color-coded) | `static/js/app.js::renderFeedback`; `static/css/style.css::.feedback-*` |
| Top-5 candidate list with confidence bars | `static/js/app.js::renderTopK` |
| Sample image gallery | `static/js/app.js::loadSamples`; `static/samples/sample_*.jpg` |
| Mobile-responsive UI | `static/css/style.css` `@media (max-width: 768px)` |
| Deployed demo URL | https://hanfuzhao781-carspec-ai.hf.space |
| Docker build | `Dockerfile` (uses `requirements-deploy.txt` for lightweight image) |
| Keep-alive workflow | `.github/workflows/keep-alive.yml` (pings every 6 hours, real URL set) |

## 5. Written Report
| Section | Location |
|------|------|
| Problem Statement | `TECHNICAL_REPORT.md` §1 |
| Data Sources | `TECHNICAL_REPORT.md` §2 |
| Related Work | `TECHNICAL_REPORT.md` §3 |
| Evaluation Strategy & Metrics (with rationale) | `TECHNICAL_REPORT.md` §4 |
| Modeling Approach | `TECHNICAL_REPORT.md` §5 |
| Data Processing Pipeline (with per-step rationale) | `TECHNICAL_REPORT.md` §6 |
| Hyperparameter Tuning Strategy | `TECHNICAL_REPORT.md` §7 |
| Models Evaluated (quantitative comparison, three models) | `TECHNICAL_REPORT.md` §8.1 |
| Results Analysis | `TECHNICAL_REPORT.md` §8.2 |
| Confusion Matrix | `TECHNICAL_REPORT.md` §8.3 |
| Error Analysis (5+ cases with confidence + test_index + root cause + mitigation) | `TECHNICAL_REPORT.md` §9 |
| Experiment Write-Up: plan / results / interpretation / recommendation | `TECHNICAL_REPORT.md` §10.1–§10.6 |
| Robustness Experiment | `TECHNICAL_REPORT.md` §10.4 |
| Confidence Gating / Selective Prediction | `TECHNICAL_REPORT.md` §10.5 |
| Head/Tail Analysis | `TECHNICAL_REPORT.md` §10.6 |
| Recommendations | `TECHNICAL_REPORT.md` §11 |
| Conclusions | `TECHNICAL_REPORT.md` §12 |
| Future Work | `TECHNICAL_REPORT.md` §13 |
| Commercial Viability | `TECHNICAL_REPORT.md` §14 |
| Ethics Statement | `TECHNICAL_REPORT.md` §15 |
| Accuracy numbers consistent with `metrics.json` | `TECHNICAL_REPORT.md` §8.1 ↔ `data/outputs/metrics.json` |

## 6. Pitch
- **File**: `PITCH.md` (local deliverable, not in repo) - 5-minute script with problem/motivation, method overview, live demo link, results/insights, future work. Slide deck: `PITCH_SLIDES.pptx` (local).

## 7. Code & Repository / Git Best Practices
| Requirement | Evidence |
|------|------|
| Branch-based workflow + reviewed PRs | 24 PRs merged into `main` (commit history visible via `git log`) |
| PR template | `.github/pull_request_template.md` |
| Keep-alive workflow (real URL) | `.github/workflows/keep-alive.yml` |
| `.gitignore` (covers `__pycache__`, `*.pyc`, `data/processed/*.npz`, large models) | `.gitignore` |

## 8. Code Quality & Practices
| Requirement | Evidence |
|------|------|
| Modular code (functions/classes only) | `scripts/*.py`, `main.py`, `setup.py` |
| No top-level executable code outside `if __name__ == "__main__"` | All `.py` files |
| Notebooks only in `notebooks/` | `notebooks/.gitkeep` (no notebooks submitted) |
| Docstrings on modules/functions | All `.py` files |
| EDA module present | `scripts/eda.py` |

## 9. Reproducibility & Documentation
| Requirement | Evidence |
|------|------|
| `requirements.txt` lists torch, torchvision, scikit-learn, scikit-image, matplotlib, Flask, gunicorn, huggingface_hub, numpy, Pillow, pandas | `requirements.txt` |
| Lightweight `requirements-deploy.txt` for Docker | `requirements-deploy.txt` |
| `python setup.py` runs end-to-end | `setup.py` (9-step pipeline; tested end-to-end, 351s on CPU) |
| `python main.py` runs from committed/HF-downloaded models | `main.py::load_models` |
| Makefile targets | `Makefile` (setup/train/run/deploy/clean) |
| README quick-start (pip install / setup.py / main.py / docker) | `README.md` → Quick Start |

## 10. README
| Requirement | Evidence |
|------|------|
| Project title & domain | `README.md` |
| Quick-start commands (pip install / setup.py / main.py / docker) | `README.md` → Quick Start |
| Model locations (with class paths) | `README.md` → Three Model Locations table |
| Deployment URL | `README.md` → Online Demo |
| Repository layout | `README.md` → Project Structure |
| Originality statement | `README.md` → Originality Statement |
| Data/citations | `README.md` → Dataset |
| Evaluation outputs documented | `README.md` → Evaluation Outputs |

## 11. Deployed Demo Verification
- **URL**: https://hanfuzhao781-carspec-ai.hf.space
- **Health check**: `GET /health` → `{"status":"ok","models_loaded":7,"loaded_models":[...]}`
- **Prediction**: `POST /predict` with a vehicle image returns JSON with `success`, `prediction`, `confidence`, `top_k`, `feedback`, `classical`, `deep`, `explanations`, `features`.
- **Error handling**: 400 for non-image / unsupported format, 413 for files > 16MB, server does not crash.
- **Samples**: `GET /samples` returns list of 5 sample images in `static/samples/`.
- **Keep-alive**: `.github/workflows/keep-alive.yml` pings every 6 hours (cron `0 */6 * * *`).

## 12. Evaluation Outputs (data/outputs/)
| File | Origin |
|------|------|
| `metrics.json` | `setup.py` → `run_full_evaluation()` - contains `naive_majority`, `naive_random`, `classical`, `deep`, `robustness`, `head_tail`, `error_cases`, `confidence_gating`, `data_size_sensitivity`, `meta` |
| `model_comparison.png` | `setup.py` step 9 → `plot_model_comparison()` |
| `confusion_matrix.png` + `.npy` | `setup.py` step 9 |
| `robustness.png` | `scripts/experiment.py::robustness_experiment` |
| `confidence_curve.png` | `scripts/experiment.py::confidence_gating_experiment` |
| `confidence_analysis.json` | `scripts/experiment.py::confidence_gating_experiment` |
| `sample_images.png` | `scripts/eda.py::plot_sample_images` |
| `eda_stats.json` | `scripts/eda.py::run_eda` |
| `plots/cm_classical_*.png`, `cm_deep_*.png` | `scripts/experiment.py::evaluate_model` |
| `plots/eda_dist_*.png` | `scripts/eda.py::plot_class_distribution` |
