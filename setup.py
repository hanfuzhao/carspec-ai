"""Training pipeline: data preparation -> feature extraction -> three-model training -> evaluation -> experiment."""
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
    data_size_sensitivity, run_full_evaluation, PLOTS_DIR,
)


def step1_prepare_data():
    """Step 1: Prepare data splits."""
    print("\n" + "=" * 60)
    print("Step 1: Prepare data splits")
    print("=" * 60)
    train, val, test = get_splits(force=False)
    print(f"Train set: {len(train):,} | Validation set: {len(val):,} | Test set: {len(test):,}")
    print(f"\nTrain set car type distribution:\n{train['car_type'].value_counts()}")
    return train, val, test


def step2_extract_features(train, val, test, sample_size=5000):
    """Step 2: Extract interpretable visual features."""
    print("\n" + "=" * 60)
    print("Step 2: Extract interpretable visual features")
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
            except Exception as e:
                feats.append(np.zeros(FEATURE_DIM, dtype=np.float32))
        return np.array(feats), df.reset_index(drop=True)
    print("Extracting train set features...")
    X_train, train_sub = extract_batch(train, sample_size)
    print("Extracting validation set features...")
    X_val, val_sub = extract_batch(val, sample_size // 5)
    print("Extracting test set features...")
    X_test, test_sub = extract_batch(test, sample_size // 5)
    # Save
    np.savez(
        "data/processed/features.npz",
        X_train=X_train, X_val=X_val, X_test=X_test,
    )
    train_sub.to_csv("data/processed/train_sub.csv", index=False)
    val_sub.to_csv("data/processed/val_sub.csv", index=False)
    test_sub.to_csv("data/processed/test_sub.csv", index=False)
    print(f"Feature dimension: {X_train.shape[1]}")
    return X_train, X_val, X_test, train_sub, val_sub, test_sub


def step3_train_naive(train, val, test):
    """Step 3: Train Naive baseline model."""
    print("\n" + "=" * 60)
    print("Step 3: Train Naive baseline model")
    print("=" * 60)
    results = {}
    for task, label2id in [("car_type", TYPE2ID), ("door_count", DOOR2ID), ("seat_count", SEAT2ID)]:
        y_train = train[task].values
        y_test = test[task].values
        model = NaiveBaseline(task=task)
        model.fit(None, y_train)
        y_pred = model.predict(None)
        y_pred = np.full(len(y_test), y_pred[0])
        metrics = compute_metrics(y_test, y_pred)
        results[task] = metrics
        model.save(MODELS_DIR / f"naive_{task}.pkl")
        print(f"  {task}: acc={metrics['accuracy']:.3f}")
    return results


def step4_train_classical(X_train, X_val, X_test, train_sub, val_sub, test_sub):
    """Step 4: Train Classical ML model."""
    print("\n" + "=" * 60)
    print("Step 4: Train Classical ML model (Random Forest)")
    print("=" * 60)
    results = {}
    for task, label2id, classes in [
        ("car_type", TYPE2ID, CAR_TYPES),
        ("door_count", DOOR2ID, DOOR_COUNTS),
        ("seat_count", SEAT2ID, SEAT_COUNTS),
    ]:
        y_train = train_sub[task].values
        y_test = test_sub[task].values
        model = ClassicalModel(task=task, model_type="rf")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, task, classes, "classical")
        model.save(MODELS_DIR / f"classical_{task}.pkl")
        results[task] = metrics
        print(f"  {task}: acc={metrics['accuracy']:.3f}")
    return results


def step5_train_deep(train, val, test, epochs=20, batch_size=32, use_aux=False):
    """Step 5: Train Deep multi-task model."""
    print("\n" + "=" * 60)
    print("Step 5: Train Deep multi-task model (MobileNetV2)")
    print("=" * 60)
    # Build generator
    def multi_task_gen(df, batch_size, shuffle=True):
        gen = image_generator(df, batch_size=batch_size, shuffle=shuffle)
        while True:
            X, y = next(gen)
            yield X, y
    train_gen = multi_task_gen(train, batch_size, shuffle=True)
    val_gen = multi_task_gen(val, batch_size, shuffle=False)
    steps = min(len(train) // batch_size, 50)
    val_steps = min(len(val) // batch_size, 10)
    model = DeepMultiTaskModel(backbone="mobilenet", use_aux_features=use_aux)
    history = model.fit(
        train_gen, val_gen, epochs=epochs,
        steps_per_epoch=steps, validation_steps=val_steps,
    )
    model.save(MODELS_DIR / "deep_multitask.pt")
    print("Deep model saved")
    return model, history


def step6_experiment(X_train, y_train, model_factory):
    """Step 6: Data size sensitivity experiment."""
    print("\n" + "=" * 60)
    print("Step 6: Data size sensitivity experiment")
    print("=" * 60)
    results = data_size_sensitivity(model_factory, X_train, y_train)
    for r in results:
        print(f"  {r['fraction']:.0%} ({r['n_train']} samples): acc={r['accuracy']:.3f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="CarSpec AI training pipeline")
    parser.add_argument("--step", type=int, default=0, help="Start from specified step (0=all)")
    parser.add_argument("--sample-size", type=int, default=5000, help="Number of samples for feature extraction")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs for Deep model")
    parser.add_argument("--skip-deep", action="store_true", help="Skip Deep model training")
    args = parser.parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()
    # Step 1
    train, val, test = step1_prepare_data()
    # Step 2
    feats_path = Path("data/processed/features.npz")
    if feats_path.exists() and args.step <= 2:
        print("Loading existing features...")
        data = np.load(feats_path)
        X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
        train_sub = pd.read_csv("data/processed/train_sub.csv")
        val_sub = pd.read_csv("data/processed/val_sub.csv")
        test_sub = pd.read_csv("data/processed/test_sub.csv")
    else:
        X_train, X_val, X_test, train_sub, val_sub, test_sub = step2_extract_features(
            train, val, test, args.sample_size
        )
    # Step 3
    naive_results = step3_train_naive(train, val, test)
    # Step 4
    classical_results = step4_train_classical(X_train, X_val, X_test, train_sub, val_sub, test_sub)
    # Step 5
    deep_results = {}
    if not args.skip_deep:
        try:
            deep_model, history = step5_train_deep(train, val, test, args.epochs)
            deep_results = {"status": "trained"}
        except Exception as e:
            print(f"Deep model training failed: {e}")
            deep_results = {"status": "failed", "error": str(e)}
    # Step 6
    y_train_car_type = train_sub["car_type"].values
    exp_results = step6_experiment(
        X_train, y_train_car_type,
        lambda: ClassicalModel(task="car_type", model_type="rf"),
    )
    # Summary
    all_results = {
        "naive": naive_results,
        "classical": classical_results,
        "deep": deep_results,
        "experiment_data_size": exp_results,
    }
    run_full_evaluation(all_results)
    print(f"\nTotal time: {time.time() - start:.1f}s")
    print("Training pipeline complete!")


if __name__ == "__main__":
    main()
