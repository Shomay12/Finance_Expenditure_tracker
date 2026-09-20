"""
Transaction Categorization Supervised Machine Learning Pipeline.
Implements:
1. Feature Extraction (TF-IDF N-grams for cleaned text + One-Hot Encoded categorical + Scaled amount)
2. Model Training: TF-IDF Baseline (Logistic Regression) vs Random Forest vs XGBoost Classifier
3. Comprehensive Evaluation: Accuracy, Precision, Recall, Macro F1, Weighted F1, Confusion Matrix
4. Model Serialization with metadata.json to /models/transaction_classifier/
5. Clean, structured inference function
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# Relative imports / helper functions
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.preprocessing import clean_merchant_text, clean_description_text

SEED = 42

class TransactionClassifierPipeline:
    def __init__(self, model_type="xgboost"):
        self.model_type = model_type
        self.label_encoder = LabelEncoder()
        self.pipeline = None
        self.metadata = {}

    def build_pipeline(self, model_type="xgboost"):
        """
        Builds a feature extraction + model pipeline.
        Features:
        - text_features (TF-IDF char/word n-grams)
        - payment_method, transaction_type (OneHotEncoder)
        - amount, log_amount (StandardScaler)
        """
        text_transformer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=1500,
            sublinear_tf=True
        )
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")
        numerical_transformer = StandardScaler()

        preprocessor = ColumnTransformer(
            transformers=[
                ("text", text_transformer, "text_features"),
                ("cat", categorical_transformer, ["payment_method", "transaction_type"]),
                ("num", numerical_transformer, ["amount", "log_amount"])
            ]
        )

        if model_type == "logistic_regression":
            classifier = LogisticRegression(
                max_iter=1000,
                C=2.5,
                solver="saga",
                random_state=SEED,
                n_jobs=-1
            )
        elif model_type == "random_forest":
            classifier = RandomForestClassifier(
                n_estimators=150,
                max_depth=20,
                random_state=SEED,
                n_jobs=-1
            )
        elif model_type == "xgboost":
            classifier = XGBClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.15,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=SEED,
                tree_method="hist",
                eval_metric="mlogloss",
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        return Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ])

    def train_and_compare(self, train_path: str, val_path: str, test_path: str) -> dict:
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)

        # Encode multi-class labels (Category::Subcategory)
        y_train = self.label_encoder.fit_transform(train_df["full_target"])
        y_val = self.label_encoder.transform(val_df["full_target"])
        y_test = self.label_encoder.transform(test_df["full_target"])

        models_to_test = ["logistic_regression", "random_forest", "xgboost"]
        comparison_results = {}

        print("\n" + "=" * 70)
        print("MODEL BENCHMARKING: TRANSACTION CATEGORIZATION")
        print("=" * 70)

        best_score = -1.0
        best_model_name = None
        best_pipeline = None

        for m_name in models_to_test:
            print(f"\nTraining [{m_name.upper()}]...")
            pipe = self.build_pipeline(m_name)
            pipe.fit(train_df, y_train)

            # Evaluate on Validation set
            val_preds = pipe.predict(val_df)
            val_acc = accuracy_score(y_val, val_preds)
            val_p, val_r, val_f1_macro, _ = precision_recall_fscore_support(y_val, val_preds, average="macro", zero_division=0)
            _, _, val_f1_weighted, _ = precision_recall_fscore_support(y_val, val_preds, average="weighted", zero_division=0)

            print(f"Validation Metrics [{m_name}]:")
            print(f"  Accuracy:    {val_acc:.4f}")
            print(f"  Macro F1:    {val_f1_macro:.4f}")
            print(f"  Weighted F1: {val_f1_weighted:.4f}")

            comparison_results[m_name] = {
                "val_accuracy": float(val_acc),
                "val_macro_f1": float(val_f1_macro),
                "val_weighted_f1": float(val_f1_weighted)
            }

            if val_f1_macro > best_score:
                best_score = val_f1_macro
                best_model_name = m_name
                best_pipeline = pipe

        print(f"\n--> Best Validated Model: [{best_model_name.upper()}] (Macro F1: {best_score:.4f})")
        self.pipeline = best_pipeline
        self.model_type = best_model_name

        # Final evaluation on held-out TEST set
        test_preds = self.pipeline.predict(test_df)
        test_probs = self.pipeline.predict_proba(test_df)

        test_acc = accuracy_score(y_test, test_preds)
        test_p_macro, test_r_macro, test_f1_macro, _ = precision_recall_fscore_support(y_test, test_preds, average="macro", zero_division=0)
        test_p_wt, test_r_wt, test_f1_wt, _ = precision_recall_fscore_support(y_test, test_preds, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, test_preds)

        print("\n" + "=" * 70)
        print(f"FINAL HELD-OUT TEST EVALUATION [{best_model_name.upper()}]")
        print("=" * 70)
        print(f"Test Accuracy:    {test_acc:.4f}")
        print(f"Test Precision:   {test_p_macro:.4f} (Macro), {test_p_wt:.4f} (Weighted)")
        print(f"Test Recall:      {test_r_macro:.4f} (Macro), {test_r_wt:.4f} (Weighted)")
        print(f"Test Macro F1:    {test_f1_macro:.4f}")
        print(f"Test Weighted F1: {test_f1_wt:.4f}")
        print("\nClassification Report (Top Classes Sample):")
        print(classification_report(y_test, test_preds, target_names=self.label_encoder.classes_, digits=4, zero_division=0))

        # Prepare Metadata
        self.metadata = {
            "model_name": f"TransactionClassifier_{best_model_name}",
            "version": "1.0.0",
            "training_dataset": "raw_transactions_classification.csv",
            "taxonomy_version": "standardized_v1",
            "category_mapping_version": "v1.0",
            "preprocessing_version": "v1.0_tfidf_normalized",
            "training_date": datetime.now().isoformat(),
            "n_train_samples": int(len(train_df)),
            "n_classes": int(len(self.label_encoder.classes_)),
            "classes": self.label_encoder.classes_.tolist(),
            "feature_list": ["text_features (merchant_clean + desc_clean)", "payment_method", "transaction_type", "amount", "log_amount"],
            "model_benchmarking": comparison_results,
            "test_metrics": {
                "accuracy": float(test_acc),
                "precision_macro": float(test_p_macro),
                "recall_macro": float(test_r_macro),
                "macro_f1": float(test_f1_macro),
                "weighted_f1": float(test_f1_wt),
                "confusion_matrix_shape": list(cm.shape)
            }
        }
        return self.metadata

    def save(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.pipeline, os.path.join(model_dir, "model.pkl"))
        joblib.dump(self.label_encoder, os.path.join(model_dir, "label_encoder.pkl"))
        with open(os.path.join(model_dir, "metadata.json"), "w") as f:
            json.dump(self.metadata, f, indent=4)
        print(f"[Model Saved] Successfully stored classifier artifacts at {model_dir}")

    @classmethod
    def load(cls, model_dir: str):
        instance = cls()
        instance.pipeline = joblib.load(os.path.join(model_dir, "model.pkl"))
        instance.label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
        with open(os.path.join(model_dir, "metadata.json"), "r") as f:
            instance.metadata = json.load(f)
        return instance

    def predict_transaction(
        self,
        merchant_raw: str,
        description: str,
        amount: float,
        transaction_type: str = "debit",
        payment_method: str = "Credit Card",
        date: str = None
    ) -> dict:
        """
        Performs inference on a single incoming transaction and returns structured JSON output.
        """
        m_clean = clean_merchant_text(merchant_raw)
        d_clean = clean_description_text(description)
        text_feat = f"{m_clean} {d_clean}".strip()
        log_amt = np.log1p(max(0.0, float(amount)))

        input_df = pd.DataFrame([{
            "text_features": text_feat,
            "payment_method": payment_method,
            "transaction_type": transaction_type,
            "amount": float(amount),
            "log_amount": log_amt
        }])

        proba = self.pipeline.predict_proba(input_df)[0]
        pred_idx = np.argmax(proba)
        conf = float(proba[pred_idx])
        full_label = self.label_encoder.classes_[pred_idx]

        cat, subcat = full_label.split("::")

        return {
            "category": cat,
            "subcategory": subcat,
            "category_confidence": round(conf, 4)
        }


def run_training_pipeline(data_dir: str, model_dir: str):
    train_path = os.path.join(data_dir, "train", "classification_train.csv")
    val_path = os.path.join(data_dir, "validation", "classification_val.csv")
    test_path = os.path.join(data_dir, "test", "classification_test.csv")

    clf_module = TransactionClassifierPipeline()
    clf_module.train_and_compare(train_path, val_path, test_path)
    clf_module.save(model_dir)

    # Demo inference in Indian INR context
    sample_tests = [
        ("SWIGGY*BANGALORE FOOD", "Dinner order paneer butter masala naan", 650.0, "debit", "UPI"),
        ("TATA CONSULTANCY SERVICES SALARY", "Monthly salary credit NEFT direct deposit", 85000.0, "credit", "Direct Deposit"),
        ("NETFLIX INDIA MONTHLY PLAN", "Monthly 4K streaming plan subscription", 649.0, "debit", "UPI AutoPay"),
        ("GROWW MUTUAL FUND SIP AUTO", "Monthly index fund SIP investment deduction", 15000.0, "transfer", "Auto Debit"),
        ("BHARAT PETROLEUM BPCL BUNKER", "Automobile petrol fuel fillup", 3200.0, "debit", "Credit Card")
    ]

    print("\n" + "=" * 70)
    print("INFERENCE DEMO - STRUCTURED OUTPUTS")
    print("=" * 70)
    for m, d, a, tt, pm in sample_tests:
        res = clf_module.predict_transaction(m, d, a, tt, pm)
        print(f"Input: {m} | Rs.{a} --> {json.dumps(res, indent=2)}")

if __name__ == "__main__":
    base_data = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
    base_model = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/transaction_classifier"))
    run_training_pipeline(base_data, base_model)
