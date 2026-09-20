"""
Contextual Spending Anomaly Detection Engine.
Detects financial transactions that deviate significantly from a user's baseline spending habits.
Implements:
1. Historical contextual feature engineering (user-category baselines, z-scores, percentiles, frequency)
2. Isolation Forest (primary) benchmarked against Local Outlier Factor (LOF) and One-Class SVM
3. Interpretable Anomaly Reason Generation
4. Artifact serialization to /models/anomaly_detector/
5. Structured JSON inference API
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

SEED = 42

class ContextualAnomalyDetector:
    def __init__(self, contamination: float = 0.015):
        self.contamination = contamination
        self.model = None
        self.scaler = StandardScaler()
        self.user_category_baselines = {}
        self.user_merchant_baselines = {}
        self.user_overall_baselines = {}
        self.feature_names = [
            "log_amount",
            "amount_to_user_mean_ratio",
            "amount_to_category_mean_ratio",
            "category_amount_zscore",
            "user_amount_percentile",
            "merchant_seen_frequency",
            "category_spending_share",
            "is_weekend",
            "hour"
        ]
        self.metadata = {}

    def compute_user_baselines(self, df_history: pd.DataFrame):
        """
        Computes historical user baseline statistics up to historical training point.
        No future leakage is allowed.
        """
        # User overall baselines
        for uid, u_df in df_history.groupby("user_id"):
            amt = u_df[u_df["category"] == "EXPENSE"]["amount"]
            if len(amt) == 0:
                amt = u_df["amount"]
            self.user_overall_baselines[uid] = {
                "mean": float(amt.mean()) if len(amt) > 0 else 500.0,
                "std": float(amt.std()) if len(amt) > 1 and amt.std() > 0 else 100.0,
                "median": float(amt.median()) if len(amt) > 0 else 500.0,
                "total_txns": len(u_df),
                "amounts_list": amt.tolist()
            }

        # User category baselines
        for (uid, subcat), g_df in df_history.groupby(["user_id", "subcategory"]):
            amt = g_df["amount"]
            self.user_category_baselines[(uid, subcat)] = {
                "mean": float(amt.mean()),
                "std": float(amt.std()) if len(amt) > 1 and amt.std() > 0 else 1.0,
                "median": float(amt.median()),
                "max": float(amt.max()),
                "count": len(amt),
                "amounts_list": amt.tolist()
            }

        # User merchant baselines
        for (uid, m_name), g_df in df_history.groupby(["user_id", "merchant"]):
            self.user_merchant_baselines[(uid, m_name)] = len(g_df)

    def register_user_history(self, user_id: str, df_user: pd.DataFrame):
        """
        Dynamically registers or updates historical baselines for a specific user.
        Ensures strict user data isolation without affecting any other user's baseline.
        """
        uid = str(user_id)
        if df_user.empty:
            return

        amt = df_user[df_user["category"] == "EXPENSE"]["amount"] if "category" in df_user.columns else df_user["amount"]
        if len(amt) == 0:
            amt = df_user["amount"]

        self.user_overall_baselines[uid] = {
            "mean": float(amt.mean()) if len(amt) > 0 else 500.0,
            "std": float(amt.std()) if len(amt) > 1 and amt.std() > 0 else 100.0,
            "median": float(amt.median()) if len(amt) > 0 else 500.0,
            "total_txns": len(df_user),
            "amounts_list": amt.tolist()
        }

        # User category baselines
        subcat_col = "subcategory" if "subcategory" in df_user.columns else ("category" if "category" in df_user.columns else None)
        if subcat_col:
            for subcat, g_df in df_user.groupby(subcat_col):
                c_amt = g_df["amount"]
                self.user_category_baselines[(uid, subcat)] = {
                    "mean": float(c_amt.mean()),
                    "std": float(c_amt.std()) if len(c_amt) > 1 and c_amt.std() > 0 else 1.0,
                    "median": float(c_amt.median()),
                    "max": float(c_amt.max()),
                    "count": len(c_amt),
                    "amounts_list": c_amt.tolist()
                }

        # User merchant baselines
        if "merchant" in df_user.columns:
            for m_name, g_df in df_user.groupby("merchant"):
                self.user_merchant_baselines[(uid, m_name)] = len(g_df)

    def extract_features_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts contextual behavioral features for each transaction.
        """
        feats = []
        for _, row in df.iterrows():
            uid = row["user_id"]
            subcat = row["subcategory"]
            m_name = row["merchant"]
            amt = float(row["amount"])
            is_weekend = int(row.get("is_weekend", 0))
            hour = int(row.get("hour", 12))

            u_base = self.user_overall_baselines.get(uid, {"mean": 500.0, "std": 200.0, "total_txns": 10, "amounts_list": [amt]})
            c_base = self.user_category_baselines.get((uid, subcat), {"mean": amt, "std": 1.0, "count": 1, "amounts_list": [amt]})
            m_freq = self.user_merchant_baselines.get((uid, m_name), 0)

            # Ratios and contextual z-scores
            u_mean = u_base["mean"] if u_base["mean"] > 0 else 1.0
            c_mean = c_base["mean"] if c_base["mean"] > 0 else 1.0
            c_std = c_base["std"] if c_base["std"] > 0 else 1.0

            ratio_user = amt / u_mean
            ratio_cat = amt / c_mean
            z_cat = (amt - c_mean) / c_std

            # Empirical percentile in user's history
            hist_amts = u_base.get("amounts_list", [amt])
            pctile = (np.sum(np.array(hist_amts) <= amt) / max(1, len(hist_amts)))

            # Relative merchant frequency
            m_freq_norm = min(1.0, m_freq / max(1, u_base["total_txns"]))

            # Category spending share
            cat_share = c_base["count"] / max(1, u_base["total_txns"])

            feats.append({
                "log_amount": np.log1p(amt),
                "amount_to_user_mean_ratio": ratio_user,
                "amount_to_category_mean_ratio": ratio_cat,
                "category_amount_zscore": z_cat,
                "user_amount_percentile": pctile,
                "merchant_seen_frequency": m_freq_norm,
                "category_spending_share": cat_share,
                "is_weekend": is_weekend,
                "hour": hour
            })

        return pd.DataFrame(feats)[self.feature_names]

    def fit_and_benchmark(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
        print("\n" + "=" * 70)
        print("CONTEXTUAL ANOMALY DETECTION: TRAINING & BENCHMARKING")
        print("=" * 70)

        # 1. Compute historical baselines
        self.compute_user_baselines(train_df)

        # 2. Extract feature matrices
        X_train = self.extract_features_df(train_df)
        X_val = self.extract_features_df(val_df)

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        models = {
            "IsolationForest": IsolationForest(
                n_estimators=150,
                contamination=self.contamination,
                random_state=SEED,
                n_jobs=-1
            ),
            "OneClassSVM": OneClassSVM(
                kernel="rbf",
                gamma="scale",
                nu=self.contamination
            ),
            "LocalOutlierFactor": LocalOutlierFactor(
                n_neighbors=20,
                contamination=self.contamination,
                novelty=True,
                n_jobs=-1
            )
        }

        bench_results = {}
        for name, mdl in models.items():
            mdl.fit(X_train_scaled)
            val_preds = mdl.predict(X_val_scaled)
            # sklearn: -1 is anomaly, 1 is inlier
            n_anomalies = int(np.sum(val_preds == -1))
            pct_anomalies = float(n_anomalies / len(val_preds) * 100)
            
            print(f"Model [{name}]: Detected {n_anomalies} anomalies ({pct_anomalies:.2f}%) on validation set.")
            bench_results[name] = {
                "val_anomalies_detected": n_anomalies,
                "val_anomaly_percentage": pct_anomalies
            }

        # Primary production model: Isolation Forest
        self.model = models["IsolationForest"]

        self.metadata = {
            "model_name": "ContextualAnomalyDetector_IsolationForest",
            "version": "1.0.0",
            "training_dataset": "historical_train.csv",
            "category_mapping_version": "v1.0",
            "preprocessing_version": "v1.0_historical_contextual",
            "contamination_rate": float(self.contamination),
            "training_date": datetime.now().isoformat(),
            "n_train_samples": int(len(train_df)),
            "feature_list": self.feature_names,
            "evaluation_metrics": bench_results,
            "user_baselines_count": len(self.user_overall_baselines)
        }
        return self.metadata

    def explain_anomaly(
        self,
        user_id: str,
        subcategory: str,
        merchant: str,
        amount: float,
        score: float
    ) -> str:
        """
        Synthesizes human-interpretable contextual justification for anomaly detection.
        """
        c_base = self.user_category_baselines.get((user_id, subcategory))
        u_base = self.user_overall_baselines.get(user_id)

        if not c_base:
            return f"First-time transaction in category '{subcategory}' with high amount (₹{amount:,.2f})."

        c_mean = c_base["mean"]
        ratio = amount / c_mean if c_mean > 0 else 1.0

        if ratio >= 4.0:
            return f"Amount (₹{amount:,.2f}) is {ratio:.1f}x higher than user's normal '{subcategory}' spending average of ₹{c_mean:,.2f}."
        elif amount > c_base["max"]:
            return f"Amount (₹{amount:,.2f}) exceeds previous maximum '{subcategory}' spend of ₹{c_base['max']:,.2f}."
        elif u_base and amount > (u_base["mean"] + 3 * u_base["std"]):
            return f"Amount deviates more than 3 standard deviations from user's overall spending distribution."
        elif self.user_merchant_baselines.get((user_id, merchant), 0) == 0 and ratio > 2.0:
            return f"Unfamiliar merchant '{merchant}' with elevated category amount (₹{amount:,.2f} vs avg ₹{c_mean:,.2f})."
        else:
            return f"Contextual pattern deviation detected (Isolation score: {score:.2f})."

    def predict_transaction(
        self,
        user_id: str,
        merchant: str,
        subcategory: str,
        amount: float,
        is_weekend: int = 0,
        hour: int = 14
    ) -> dict:
        """
        Inference API returning structured anomaly assessment.
        """
        df_single = pd.DataFrame([{
            "user_id": user_id,
            "merchant": merchant,
            "subcategory": subcategory,
            "amount": float(amount),
            "is_weekend": is_weekend,
            "hour": hour
        }])

        X = self.extract_features_df(df_single)
        X_scaled = self.scaler.transform(X)

        # Isolation forest score: lower score = more anomalous
        raw_score = self.model.decision_function(X_scaled)[0]
        pred = self.model.predict(X_scaled)[0]

        # Contextual check
        c_base = self.user_category_baselines.get((user_id, subcategory))
        c_mean = c_base["mean"] if c_base and c_base["mean"] > 0 else 0
        ratio = amount / c_mean if c_mean > 0 else 1.0

        # Normalize score to 0.0 - 1.0 (where 1.0 is highest anomaly risk)
        normalized_anomaly_score = float(np.clip(1.0 / (1.0 + np.exp(raw_score * 8)), 0.01, 0.99))
        
        # Severe contextual departure trigger
        is_context_outlier = (c_base is not None and ratio >= 3.5) or (c_base is None and amount > 5000)
        is_anom = bool(pred == -1 or normalized_anomaly_score > 0.65 or is_context_outlier)
        if is_anom:
            normalized_anomaly_score = max(normalized_anomaly_score, 0.75 if is_context_outlier else 0.70)

        reason = self.explain_anomaly(user_id, subcategory, merchant, amount, normalized_anomaly_score) if is_anom else "Transaction aligns with historical user baseline."

        return {
            "is_anomaly": is_anom,
            "anomaly_score": round(normalized_anomaly_score, 4),
            "reason": reason
        }

    def save(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.model, os.path.join(model_dir, "model.pkl"))
        joblib.dump(self.scaler, os.path.join(model_dir, "scaler.pkl"))
        joblib.dump({
            "user_overall_baselines": self.user_overall_baselines,
            "user_category_baselines": self.user_category_baselines,
            "user_merchant_baselines": self.user_merchant_baselines
        }, os.path.join(model_dir, "baselines.pkl"))
        with open(os.path.join(model_dir, "metadata.json"), "w") as f:
            json.dump(self.metadata, f, indent=4)
        print(f"[Model Saved] Successfully stored anomaly detector artifacts at {model_dir}")

    @classmethod
    def load(cls, model_dir: str):
        instance = cls()
        instance.model = joblib.load(os.path.join(model_dir, "model.pkl"))
        instance.scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
        base_dict = joblib.load(os.path.join(model_dir, "baselines.pkl"))
        instance.user_overall_baselines = base_dict["user_overall_baselines"]
        instance.user_category_baselines = base_dict["user_category_baselines"]
        instance.user_merchant_baselines = base_dict["user_merchant_baselines"]
        with open(os.path.join(model_dir, "metadata.json"), "r") as f:
            instance.metadata = json.load(f)
        return instance


def run_training_pipeline(data_dir: str, model_dir: str):
    train_path = os.path.join(data_dir, "train", "historical_train.csv")
    val_path = os.path.join(data_dir, "validation", "historical_val.csv")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    detector = ContextualAnomalyDetector(contamination=0.015)
    detector.fit_and_benchmark(train_df, val_df)
    detector.save(model_dir)

    print("\n" + "=" * 70)
    print("CONTEXTUAL ANOMALY DETECTION DEMO")
    print("=" * 70)

    # Test Cases (Indian Context):
    # 1. Normal food spend for USR_0001 (approx Rs 250 - 500) -> NOT anomaly
    # 2. Contextual anomaly: Rs 11,500 luxury banquet for USR_0001 -> ANOMALY
    # 3. High absolute amount: Rs 28,000 rent payment for USR_0001 -> NOT anomaly (expected rent)
    # 4. Sudden emergency hospital bill Rs 55,000 -> ANOMALY

    test_cases = [
        ("USR_0001", "Chai Point", "Food", 320.0, 0, 13),
        ("USR_0001", "Taj Palace 5-Star Luxury Banquet", "Food", 11500.0, 1, 23),
        ("USR_0001", "NoBroker RentPay / Landlord Transfer", "Rent & Housing", 28000.0, 0, 10),
        ("USR_0001", "Apollo Hospitals Emergency Care", "Healthcare", 55000.0, 0, 15)
    ]

    for uid, merch, subcat, amt, is_wk, hr in test_cases:
        res = detector.predict_transaction(uid, merch, subcat, amt, is_wk, hr)
        print(f"\nUser: {uid} | Merchant: {merch} | Category: {subcat} | Amount: ₹{amt:,.2f}")
        print(f"--> Result: {json.dumps(res, indent=2)}")


if __name__ == "__main__":
    base_data = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
    base_model = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/anomaly_detector"))
    run_training_pipeline(base_data, base_model)
