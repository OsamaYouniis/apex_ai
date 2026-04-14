"""
training/train_dl.py
─────────────────────
Deep Learning training scripts:

  A) Posture Classifier (TensorFlow/Keras)
     Input  : 51 pose keypoint features (17 joints × x,y,confidence)
     Output : correct (1) / incorrect (0) posture
     Saved  : ai_models/dl_models/pose_classifier.keras

  B) Chatbot fine-tuning helper (HuggingFace DistilGPT-2)
     Fine-tunes a tiny GPT-2 variant on fitness Q&A pairs.
     Saved  : ai_models/dl_models/chatbot_model/

Run:
    cd apex-ai-project
    python training/train_dl.py [--task pose|chatbot|all]
"""

import argparse
import sys
from pathlib import Path

ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "datasets"
MODEL_DIR = ROOT / "ai_models" / "dl_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# A  ·  POSTURE CLASSIFIER (TensorFlow)
# ═══════════════════════════════════════════════════════════════════════════════
def train_pose_classifier():
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import joblib

    print("\n" + "═" * 60)
    print("  A · Posture Classifier (TensorFlow/Keras)")
    print("═" * 60)

    # ── Load dataset ──────────────────────────────────────────────────────────
    path = DATA_DIR / "pose_keypoints.csv"
    if not path.exists():
        print("⚠️  pose_keypoints.csv missing — generating …")
        sys.path.insert(0, str(DATA_DIR))
        from generate_datasets import generate_pose_keypoints
        generate_pose_keypoints()

    df = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c != "label"]
    X = df[feat_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.float32)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                               random_state=42, stratify=y)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    # Save scaler (needed at inference time)
    scaler_path = MODEL_DIR / "pose_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"   Scaler saved → {scaler_path}")

    # ── Model architecture ────────────────────────────────────────────────────
    model = keras.Sequential([
        keras.layers.Input(shape=(X_tr.shape[1],)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),       # binary output
    ], name="posture_classifier")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # ── Training ──────────────────────────────────────────────────────────────
    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True,
                                       monitor="val_accuracy"),
        keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=1),
    ]

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_te, y_te),
        epochs=100,
        batch_size=64,
        callbacks=callbacks,
        verbose=1,
    )

    _, test_acc = model.evaluate(X_te, y_te, verbose=0)
    print(f"\n   Test accuracy : {test_acc:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    save_path = MODEL_DIR / "pose_classifier.keras"
    model.save(save_path)
    print(f"   ✅  Model saved → {save_path}")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# B  ·  CHATBOT FINE-TUNING (HuggingFace DistilGPT-2)
# ═══════════════════════════════════════════════════════════════════════════════
FITNESS_QA = [
    ("What is BMI?",
     "BMI (Body Mass Index) is weight in kg divided by height in metres squared. A healthy range is 18.5–24.9."),
    ("How many calories should I eat to lose weight?",
     "Create a 300–500 kcal daily deficit from your TDEE. Avoid going below 1200 kcal (women) or 1500 kcal (men)."),
    ("What is TDEE?",
     "Total Daily Energy Expenditure — the total calories you burn including exercise. Calculated as BMR × activity multiplier."),
    ("How often should I train?",
     "Beginners: 3 days/week. Intermediate: 4 days. Advanced: 5–6 days. Allow 48 h recovery per muscle group."),
    ("What should I eat before a workout?",
     "A meal with complex carbs and protein 1–2 h before training. Example: oats + protein shake or chicken + rice."),
    ("How much protein do I need?",
     "Aim for 1.6–2.2 g per kg of bodyweight daily to support muscle growth and repair."),
    ("What is progressive overload?",
     "Gradually increasing workout difficulty — more weight, reps, or reduced rest — to continually challenge muscles."),
    ("How do I fix bad posture?",
     "Strengthen your core and posterior chain. Stretch hip flexors and chest. Practice shoulder retraction exercises daily."),
    ("Is cardio necessary for fat loss?",
     "Not strictly — a calorie deficit drives fat loss. Cardio helps create that deficit and improves cardiovascular health."),
    ("How long until I see results?",
     "With consistent training and nutrition, most people notice visible changes in 8–12 weeks. Strength gains start in 2–3 weeks."),
]


def train_chatbot_model():
    """
    Fine-tunes DistilGPT-2 on fitness Q&A.
    This is a lightweight demo — for production use a larger model or Claude API.
    """
    try:
        from transformers import (
            AutoTokenizer, AutoModelForCausalLM,
            TrainingArguments, Trainer, DataCollatorForLanguageModeling
        )
        import torch
        from torch.utils.data import Dataset
    except ImportError:
        print("⚠️  transformers / torch not installed.  "
              "Skipping chatbot fine-tuning.  "
              "The backend will use the Anthropic Claude API as primary chatbot.")
        return

    print("\n" + "═" * 60)
    print("  B · Chatbot Fine-tuning (DistilGPT-2)")
    print("═" * 60)

    MODEL_NAME = "distilgpt2"
    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # Build training texts as "<Q> ... <A> ..." pairs
    texts = [
        f"<Q> {q} <A> {a} {tokenizer.eos_token}"
        for q, a in FITNESS_QA * 20          # repeat for more training steps
    ]

    class FitnessDataset(Dataset):
        def __init__(self, texts, tokenizer, max_length=128):
            self.encodings = tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
        def __len__(self):
            return len(self.encodings["input_ids"])
        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.encodings.items()}

    dataset   = FitnessDataset(texts, tokenizer)
    collator  = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    save_path = str(MODEL_DIR / "chatbot_model")
    args = TrainingArguments(
        output_dir=save_path,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=50,
        logging_steps=20,
        overwrite_output_dir=True,
        no_cuda=not torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=base_model,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
    )
    trainer.train()

    base_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"   ✅  Chatbot model saved → {save_path}")
    return base_model


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="APEX AI — DL Training")
    parser.add_argument("--task", default="all",
                        choices=["pose", "chatbot", "all"],
                        help="Which model to train")
    args = parser.parse_args()

    if args.task in ("pose", "all"):
        train_pose_classifier()

    if args.task in ("chatbot", "all"):
        train_chatbot_model()

    print("\n✅  DL training complete.")
