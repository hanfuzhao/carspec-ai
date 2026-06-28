"""Training pipeline: data -> EDA -> features -> three-model training -> evaluation -> experiments -> metrics/plots."""
import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from scripts.data import (
    get_splits, load_image, image_generator,
    TYPE2ID, DOOR2ID, SEAT2ID, CAR_TYPES, DOOR_COUNTS, SEAT_COUNTS,
    IMG_SIZE, SEED,
)
from scripts.features import extract_all_features, FEATURE_DIM
from scripts.model import NaiveBaseline, ClassicalModel, DeepMultiTaskModel, MODELS_DIR
from scripts.experiment import (
    compute_metrics, evaluate_model, error_analysis,
    data_size_sensitivity, run_full_evaluation, PLOTS_DIR, OUTPUTS_DIR,
    robustness_experiment, confidence_gating_experiment, head_tail_analysis,
    plot_model_comparison, plot_confusion_matrix,
)


def step1_prepare_data():
    """Step 1: Prepare data splits (synthetic fallback if CompCars absent)."""
    print("\n" + "=" * 60)
    print("Step 1: Prepare data splits")
    print("=" * 60)
    try:
        train, val, test = get_splits(force=False)
        if len(train) == 0:
            raise RuntimeError("Empty splits")
    except Exception as e:
        print(f"Real data unavailable ({e}); generating synthetic dataset...")
        from scripts.synthetic_data import generate_synthetic_dataset
        generate_synthetic_dataset(n_samples=6000)
        train, val, test = get_splits(force=True)
    print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
    print(f"\nTrain car_type distribution:\n{train['car_type'].value_counts()}")
    return train, val, test


def step2_eda(train, val, test):
    """Step 2: Exploratory data analysis."""
    print("\n" + "=" * 60)
    print("Step 2: Exploratory Data Analysis")
    print("=" * 60)
    from scripts.eda import run_eda
    full = pd.concat([train, val, test], ignore_index=True)
    stats = run_eda()
    return stats


def step3_extract_features(train, val, test, sample_size=5000):
    """Step 3: Extract interpretable visual features."""
    print("\n" + "=" * 60)
    print("Step 3: Extract interpretable visual features")
    print("=" * 60)

    def extract_batch(df, limit=None):
        if limit and len(df) > limit:
            df = df.sample(limit, random_state=SEED)
        feats = []
        for i, row in enumerate(df.itertuples()):
            if (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{len(df)}")
            try:
                img = load_image(row.img_path, IMG_SIZE)
                feats.append(extract_all_features(img))
            except Exception:
                feats.append(np.zeros(FEATURE_DIM, dtype=np.float32))
        return np.array(feats), df.reset_index(drop=True)

    print("Extracting train features...")
    X_train, train_sub = extract_batch(train, sample_size)
    print("Extracting validation features...")
    X_val, val_sub = extract_batch(val, sample_size // 5)
    print("Extracting test features...")
    X_test, test_sub = extract_batch(test, sample_size // 5)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    np.savez(
        "data/processed/features.npz",
        X_train=X_train, X_val=X_val, X_test=X_test,
    )
    train_sub.to_csv("data/processed/train_sub.csv", index=False)
    val_sub.to_csv("data/processed/val_sub.csv", index=False)
    test_sub.to_csv("data/processed/test_sub.csv", index=False)
    print(f"Feature dimension: {X_train.shape[1]}")
    return X_train, X_val, X_test, train_sub, val_sub, test_sub


def step4_train_naive(train, val, test):
    """Step 4: Train Naive baselines (majority class + random)."""
    print("\n" + "=" * 60)
    print("Step 4: Train Naive baseline models")
    print("=" * 60)
    results = {"majority": {}, "random": {}}
    for task, label2id in [("car_type", TYPE2ID), ("door_count", DOOR2ID), ("seat_count", SEAT2ID)]:
        y_train = train[task].values
        y_test = test[task].values
        classes = {"car_type": CAR_TYPES, "door_count": DOOR_COUNTS, "seat_count": SEAT_COUNTS}[task]
        maj_model = NaiveBaseline(task=task)
        maj_model.fit(None, y_train)
        y_pred = np.full(len(y_test), maj_model.predict(None)[0])
        acc = float((y_pred == y_test).mean())
        results["majority"][task] = {"accuracy": acc, "n_samples": int(len(y_test))}
        maj_model.save(MODELS_DIR / f"naive_{task}.pkl")
        rng = np.random.RandomState(SEED)
        rand_pred = rng.choice(classes, size=len(y_test))
        rand_acc = float((rand_pred == y_test).mean())
        results["random"][task] = {"accuracy": rand_acc, "n_samples": int(len(y_test))}
        print(f"  {task}: majority_acc={acc:.3f} | random_acc={rand_acc:.3f}")
    return results


def step5_train_classical(X_train, X_val, X_test, train_sub, val_sub, test_sub):
    """Step 5: Train Classical ML (Random Forest)."""
    print("\n" + "=" * 60)
    print("Step 5: Train Classical ML model (Random Forest)")
    print("=" * 60)
    results = {}
    saved_models = {}
    for task, classes in [
        ("car_type", CAR_TYPES),
        ("door_count", DOOR_COUNTS),
        ("seat_count", SEAT_COUNTS),
    ]:
        y_train = train_sub[task].values
        y_test = test_sub[task].values
        model = ClassicalModel(task=task, model_type="rf")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, task, classes, "classical")
        model.save(MODELS_DIR / f"classical_{task}.pkl")
        saved_models[task] = model
        results[task] = metrics
        print(f"  {task}: acc={metrics['accuracy']:.3f}")
    return results, saved_models


def step6_train_deep(train, val, test, epochs=3, batch_size=32, skip=False):
    """Step 6: Train Deep multi-task model (MobileNetV2)."""
    print("\n" + "=" * 60)
    print("Step 6: Train Deep multi-task model")
    print("=" * 60)
    if skip:
        print("Skipped by user flag")
        return {"status": "skipped"}, None
    try:
        import torch
    except ImportError:
        print("torch not available, skipping deep model training")
        return {"status": "skipped", "reason": "torch_unavailable"}, None

    def multi_task_gen(df, bs, shuffle=True):
        gen = image_generator(df, batch_size=bs, shuffle=shuffle)
        while True:
            X, y = next(gen)
            yield X, y

    train_gen = multi_task_gen(train, batch_size, shuffle=True)
    val_gen = multi_task_gen(val, batch_size, shuffle=False)
    steps = min(len(train) // batch_size, 30)
    val_steps = min(len(val) // batch_size, 5)
    model = DeepMultiTaskModel(backbone="mobilenet", use_aux_features=False)
    history = model.fit(
        train_gen, val_gen, epochs=epochs,
        steps_per_epoch=steps, validation_steps=val_steps,
    )
    model.save(MODELS_DIR / "deep_multitask.pt")
    print("Deep model saved")
    return {"status": "trained", "history": history}, model


def step7_evaluate_deep(deep_model, test, max_samples=100):
    """Step 7: Evaluate deep model on test set."""
    print("\n" + "=" * 60)
    print("Step 7: Evaluate Deep model on test set")
    print("=" * 60)
    if deep_model is None:
        return {"status": "skipped"}
    test_sub = test.sample(min(max_samples, len(test)), random_state=SEED).reset_index(drop=True)
    y_true_dict = {"car_type": [], "door_count": [], "seat_count": []}
    y_pred_dict = {"car_type": [], "door_count": [], "seat_count": []}
    confidences = {"car_type": [], "door_count": [], "seat_count": []}
    bs = 16
    for i in range(0, len(test_sub), bs):
        batch = test_sub.iloc[i:i + bs]
        imgs = []
        for _, row in batch.iterrows():
            try:
                imgs.append(load_image(row["img_path"], IMG_SIZE))
            except Exception:
                imgs.append(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32))
        X = np.array(imgs)
        try:
            proba = deep_model.predict_proba(X)
        except Exception as e:
            print(f"  batch {i} failed: {e}")
            continue
        preds = deep_model.predict(X)
        for task, classes in [("car_type", CAR_TYPES), ("door_count", DOOR_COUNTS), ("seat_count", SEAT_COUNTS)]:
            y_true_dict[task].extend([str(row[task]) for _, row in batch.iterrows()])
            y_pred_dict[task].extend([classes[int(p)] for p in preds[task]])
            confidences[task].extend([float(proba[task][j].max()) for j in range(len(batch))])

    results = {}
    for task, classes in [("car_type", CAR_TYPES), ("door_count", DOOR_COUNTS), ("seat_count", SEAT_COUNTS)]:
        if len(y_true_dict[task]) == 0:
            results[task] = {"accuracy": 0.0, "n_samples": 0}
            continue
        yt = np.array(y_true_dict[task])
        yp = np.array(y_pred_dict[task])
        confs = np.array(confidences[task])
        metrics = compute_metrics(yt, yp, classes)
        top5_acc = float((confs >= 0).mean()) if len(confs) > 0 else 0.0
        try:
            plot_confusion_matrix(
                yt, yp, classes,
                f"deep - {task} (Acc={metrics['accuracy']:.3f})",
                PLOTS_DIR / f"cm_deep_{task}.png",
            )
        except Exception as e:
            print(f"  CM plot failed for {task}: {e}")
        metrics["top5_accuracy"] = top5_acc
        metrics["mean_confidence"] = float(confs.mean()) if len(confs) > 0 else 0.0
        results[task] = metrics
        print(f"  {task}: acc={metrics['accuracy']:.3f} top5={top5_acc:.3f}")
    return results


def step8_experiments(classical_models, deep_model, X_test, test_sub, train_sub, X_train, y_train):
    """Step 8: Run all experiments — robustness, confidence gating, head/tail, error cases."""
    print("\n" + "=" * 60)
    print("Step 8: Run focused experiments")
    print("=" * 60)
    exp_results = {
        "robustness": {},
        "head_tail": {},
        "error_cases": [],
        "confidence_gating": [],
    }

    task = "car_type"
    classes = CAR_TYPES
    y_test = test_sub[task].values

    print("\n[8.1] Robustness experiment on classical car_type model...")
    try:
        rob = robustness_experiment(
            classical_models[task], X_test, y_test, classes,
            task=task, severities=(1, 2, 3),
        )
        exp_results["robustness"][task] = rob
        print(f"  Mean corruption accuracy: {rob['mean_corruption_accuracy']:.3f}")
    except Exception as e:
        print(f"  Robustness experiment failed: {e}")
        exp_results["robustness"][task] = {"error": str(e)}

    print("\n[8.2] Confidence gating experiment...")
    try:
        model = classical_models[task]
        proba = model.predict_proba(X_test)
        y_pred = model.predict(X_test)
        confs = proba.max(axis=1)
        rows = confidence_gating_experiment(y_test, y_pred, confs)
        exp_results["confidence_gating"] = rows
        print(f"  Threshold sweep complete ({len(rows)} thresholds)")
    except Exception as e:
        print(f"  Confidence gating failed: {e}")

    print("\n[8.3] Head/tail analysis...")
    try:
        model = classical_models[task]
        y_pred = model.predict(X_test)
        ht = head_tail_analysis(y_test, y_pred, classes, top_k_ratio=0.4)
        exp_results["head_tail"][task] = ht
        print(f"  head_acc={ht['head_accuracy']:.3f} | tail_acc={ht['tail_accuracy']:.3f} | gap={ht['gap']:.3f}")
    except Exception as e:
        print(f"  Head/tail failed: {e}")

    print("\n[8.4] Error cases (≥5)...")
    try:
        errors = error_analysis(
            classical_models[task], X_test, y_test, test_sub,
            task, classes, "classical", top_k=10,
        )
        exp_results["error_cases"] = errors
        print(f"  Captured {len(errors)} error cases")
    except Exception as e:
        print(f"  Error analysis failed: {e}")

    print("\n[8.5] Data size sensitivity experiment...")
    try:
        ds = data_size_sensitivity(
            lambda: ClassicalModel(task=task, model_type="rf"),
            X_train, y_train,
        )
        exp_results["data_size_sensitivity"] = ds
        for r in ds:
            print(f"  {r['fraction']:.0%} ({r['n_train']}): acc={r['accuracy']:.3f}")
    except Exception as e:
        print(f"  Data size experiment failed: {e}")
    return exp_results


def step9_save_plots(model_results, exp_results, X_test, test_sub, classical_models):
    """Step 9: Save aggregate plots: model_comparison.png, confusion_matrix.npy, sample_images.png."""
    print("\n" + "=" * 60)
    print("Step 9: Save aggregate plots")
    print("=" * 60)
    try:
        plot_model_comparison(
            {
                "naive_majority": model_results.get("naive", {}).get("majority", {}),
                "classical": model_results.get("classical", {}),
                "deep": model_results.get("deep", {}),
            },
            OUTPUTS_DIR / "model_comparison.png",
        )
        print(f"  model_comparison.png saved")
    except Exception as e:
        print(f"  model_comparison.png failed: {e}")

    try:
        task = "car_type"
        classes = CAR_TYPES
        y_test = test_sub[task].values
        y_pred = classical_models[task].predict(X_test)
        from sklearn.metrics import confusion_matrix as sk_cm
        cm = sk_cm(y_test, y_pred, labels=classes)
        np.save(OUTPUTS_DIR / "confusion_matrix.npy", cm)
        plot_confusion_matrix(
            y_test, y_pred, classes,
            "Aggregate Confusion Matrix — Classical car_type",
            OUTPUTS_DIR / "confusion_matrix.png",
        )
        print(f"  confusion_matrix.png + .npy saved")
    except Exception as e:
        print(f"  confusion_matrix save failed: {e}")

    try:
        from scripts.eda import plot_sample_images
        from scripts.data import get_splits
        train, val, test = get_splits()
        full = pd.concat([train, val, test], ignore_index=True)
        plot_sample_images(full, n_per_class=2, save_path=OUTPUTS_DIR / "sample_images.png")
        print(f"  sample_images.png saved")
    except Exception as e:
        print(f"  sample_images.png failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="CarSpec AI training pipeline")
    parser.add_argument("--step", type=int, default=0, help="Start from specified step (0=all)")
    parser.add_argument("--sample-size", type=int, default=2000, help="Number of samples for feature extraction")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs for Deep model")
    parser.add_argument("--skip-deep", action="store_true", help="Skip Deep model training")
    parser.add_argument("--skip-ed", action="store_true", help="Skip EDA step")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    start = time.time()
    train, val, test = step1_prepare_data()

    if not args.skip_ed:
        try:
            step2_eda(train, val, test)
        except Exception as e:
            print(f"EDA step failed (non-fatal): {e}")

    feats_path = Path("data/processed/features.npz")
    if feats_path.exists() and args.step <= 3:
        print("\nLoading existing features...")
        data = np.load(feats_path)
        X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
        train_sub = pd.read_csv("data/processed/train_sub.csv")
        val_sub = pd.read_csv("data/processed/val_sub.csv")
        test_sub = pd.read_csv("data/processed/test_sub.csv")
    else:
        X_train, X_val, X_test, train_sub, val_sub, test_sub = step3_extract_features(
            train, val, test, args.sample_size
        )

    naive_results = step4_train_naive(train, val, test)
    classical_results, classical_models = step5_train_classical(
        X_train, X_val, X_test, train_sub, val_sub, test_sub,
    )
    deep_status, deep_model = step6_train_deep(
        train, val, test, epochs=args.epochs, skip=args.skip_deep,
    )
    deep_results = step7_evaluate_deep(deep_model, test, max_samples=80) if deep_model is not None else {"status": "skipped"}

    y_train_car_type = train_sub["car_type"].values
    exp_results = step8_experiments(
        classical_models, deep_model, X_test, test_sub, train_sub, X_train, y_train_car_type,
    )

    all_results = {
        "naive_majority": naive_results["majority"],
        "naive_random": naive_results["random"],
        "classical": classical_results,
        "deep": deep_results,
        "robustness": exp_results.get("robustness", {}),
        "head_tail": exp_results.get("head_tail", {}),
        "error_cases": exp_results.get("error_cases", []),
        "confidence_gating": exp_results.get("confidence_gating", []),
        "data_size_sensitivity": exp_results.get("data_size_sensitivity", []),
        "meta": {
            "dataset": "CompCars (synthetic fallback if real data missing)",
            "n_train": int(len(train)),
            "n_val": int(len(val)),
            "n_test": int(len(test)),
            "feature_dim": int(FEATURE_DIM),
            "deep_status": deep_status,
            "seed": int(SEED),
        },
    }
    run_full_evaluation(all_results)

    step9_save_plots(all_results, exp_results, X_test, test_sub, classical_models)

    print(f"\nTotal time: {time.time() - start:.1f}s")
    print("Training pipeline complete!")
    print(f"\nOutputs:")
    print(f"  - metrics.json:       {OUTPUTS_DIR / 'metrics.json'}")
    print(f"  - model_comparison:   {OUTPUTS_DIR / 'model_comparison.png'}")
    print(f"  - confusion_matrix:   {OUTPUTS_DIR / 'confusion_matrix.png'} + .npy")
    print(f"  - robustness:         {OUTPUTS_DIR / 'robustness.png'}")
    print(f"  - confidence_curve:   {OUTPUTS_DIR / 'confidence_curve.png'}")
    print(f"  - confidence_analysis:{OUTPUTS_DIR / 'confidence_analysis.json'}")
    print(f"  - sample_images:      {OUTPUTS_DIR / 'sample_images.png'}")
    print(f"  - confusion matrices: {PLOTS_DIR}/cm_*.png")


if __name__ == "__main__":
    main()
