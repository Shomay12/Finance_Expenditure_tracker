"""
User Segmentation & Behavioral Profiling Engine.
Aggregates longitudinal historical transactions into rich 15-dimensional user feature vectors.
Implements:
1. User-level feature engineering & Category Entropy calculation
2. Missing-value handling & StandardScaler
3. PCA Decomposition & Multi-k K-Means Benchmarking (k=2..6)
4. Evaluation: Silhouette Score & Davies-Bouldin Index
5. Descriptive Behavioral Persona Assignment (without psychological/status claims)
6. Model Serialization to /models/segmentation/
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

SEED = 42

class UserSegmenter:
    def __init__(self, n_clusters: int = 4):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=3, random_state=SEED)
        self.kmeans = None
        self.cluster_personas = {}
        self.feature_names = [
            "monthly_income",
            "monthly_expense",
            "savings_ratio",
            "transaction_count",
            "average_transaction",
            "food_ratio",
            "shopping_ratio",
            "transport_ratio",
            "subscription_ratio",
            "recurring_expense_ratio",
            "spending_variability",
            "weekend_spending_ratio",
            "merchant_count",
            "category_count",
            "category_entropy"
        ]
        self.metadata = {}

    def extract_user_vectors(self, df_history: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates transaction history into ONE rich behavioral feature vector per user.
        """
        df = df_history.copy()
        df["date_dt"] = pd.to_datetime(df["date"])
        df["is_weekend"] = (df["date_dt"].dt.dayofweek >= 5).astype(int)

        # Number of distinct active months in dataset
        n_months_total = max(1, df["date_dt"].dt.to_period("M").nunique())

        user_rows = []
        for uid, u_df in df.groupby("user_id"):
            income_txns = u_df[u_df["category"] == "INCOME"]
            expense_txns = u_df[u_df["category"] == "EXPENSE"]
            transfer_txns = u_df[u_df["category"] == "TRANSFER"]

            tot_income = float(income_txns["amount"].sum())
            tot_expense = float(expense_txns["amount"].sum())
            tot_transfers = float(transfer_txns["amount"].sum())

            monthly_income = tot_income / n_months_total
            monthly_expense = tot_expense / n_months_total

            # Savings ratio: (income - expense) / income or transfer-to-investment/savings
            savings_ratio = float(np.clip((monthly_income - monthly_expense) / max(1.0, monthly_income), 0.0, 0.95))

            txn_count = len(u_df) / n_months_total
            avg_txn = float(expense_txns["amount"].mean()) if len(expense_txns) > 0 else 500.0

            # Subcategory spending proportions
            sub_sums = expense_txns.groupby("subcategory")["amount"].sum()
            food_spend = float(sub_sums.get("Food", 0.0))
            shopping_spend = float(sub_sums.get("Shopping", 0.0))
            transport_spend = float(sub_sums.get("Transport", 0.0) + sub_sums.get("Fuel", 0.0))
            subscription_spend = float(sub_sums.get("Subscriptions", 0.0))
            recurring_spend = float(
                sub_sums.get("Subscriptions", 0.0) + 
                sub_sums.get("Bills & Utilities", 0.0) + 
                sub_sums.get("Rent & Housing", 0.0)
            )

            tot_exp_safe = max(1.0, tot_expense)
            food_ratio = food_spend / tot_exp_safe
            shopping_ratio = shopping_spend / tot_exp_safe
            transport_ratio = transport_spend / tot_exp_safe
            sub_ratio = subscription_spend / tot_exp_safe
            rec_ratio = recurring_spend / tot_exp_safe

            # Spending variability (std / mean of daily expense spend)
            daily_exp = expense_txns.groupby(expense_txns["date_dt"].dt.date)["amount"].sum()
            spending_variability = float(daily_exp.std() / (daily_exp.mean() + 1e-3)) if len(daily_exp) > 1 else 0.5

            # Weekend spending ratio
            weekend_exp = float(expense_txns[expense_txns["is_weekend"] == 1]["amount"].sum())
            weekend_ratio = float(weekend_exp / tot_exp_safe)

            merchant_count = int(u_df["merchant"].nunique())
            category_count = int(u_df["subcategory"].nunique())

            # Category Entropy (Shannon entropy of category distribution)
            cat_probs = (sub_sums / tot_exp_safe).values
            cat_probs = cat_probs[cat_probs > 0]
            cat_entropy = float(entropy(cat_probs, base=2)) if len(cat_probs) > 0 else 0.0

            user_rows.append({
                "user_id": uid,
                "monthly_income": monthly_income,
                "monthly_expense": monthly_expense,
                "savings_ratio": savings_ratio,
                "transaction_count": txn_count,
                "average_transaction": avg_txn,
                "food_ratio": food_ratio,
                "shopping_ratio": shopping_ratio,
                "transport_ratio": transport_ratio,
                "subscription_ratio": sub_ratio,
                "recurring_expense_ratio": rec_ratio,
                "spending_variability": spending_variability,
                "weekend_spending_ratio": weekend_ratio,
                "merchant_count": merchant_count,
                "category_count": category_count,
                "category_entropy": cat_entropy
            })

        user_df = pd.DataFrame(user_rows)
        return user_df

    def train_and_evaluate(self, df_history: pd.DataFrame, model_dir: str) -> dict:
        print("\n" + "=" * 70)
        print("USER SEGMENTATION: FEATURE AGGREGATION & K-MEANS CLUSTERING")
        print("=" * 70)

        user_df = self.extract_user_vectors(df_history)
        print(f"Extracted feature vectors for {len(user_df)} users with {len(self.feature_names)} behavioral features.")

        X = user_df[self.feature_names].values
        X_scaled = self.scaler.fit_transform(X)
        X_pca = self.pca.fit_transform(X_scaled)

        # Multi-k hyperparameter sweep
        k_range = range(2, min(7, len(user_df)))
        sweep_results = {}
        best_sil = -1.0
        best_k = 4

        print("\nClustering Hyperparameter Search (k=2..6):")
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=SEED, n_init=15)
            labels = km.fit_predict(X_scaled)
            sil = silhouette_score(X_scaled, labels)
            db = davies_bouldin_score(X_scaled, labels)
            sweep_results[f"k_{k}"] = {
                "k": k,
                "silhouette_score": float(sil),
                "davies_bouldin_index": float(db)
            }
            print(f"  k={k} | Silhouette Score: {sil:.4f} | Davies-Bouldin Index: {db:.4f}")

            if sil > best_sil:
                best_sil = sil
                best_k = k

        print(f"\n--> Selected Optimal Clusters: k={best_k} (Silhouette: {best_sil:.4f})")
        self.n_clusters = best_k
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=SEED, n_init=20)
        cluster_labels = self.kmeans.fit_predict(X_scaled)
        user_df["cluster"] = cluster_labels

        # Assign descriptive behavioral personas based on cluster centroid profiles
        cluster_profiles = user_df.groupby("cluster")[self.feature_names].mean()
        self.cluster_personas = self._assign_behavioral_personas(cluster_profiles)

        print("\nBehavioral Segment Personas Identified:")
        for c_id, p_info in self.cluster_personas.items():
            print(f"\n[Cluster {c_id}] Persona: \"{p_info['persona_name']}\" (Users: {np.sum(cluster_labels == c_id)})")
            print(f"  Profile: {p_info['description']}")
            print(f"  Key Traits: Weekend Ratio={p_info['weekend_ratio']:.2f}, Recurring Ratio={p_info['recurring_ratio']:.2f}, Savings Ratio={p_info['savings_ratio']:.2f}")

        self.metadata = {
            "model_name": "UserSegmenter_KMeans",
            "version": "1.0.0",
            "training_dataset": "raw_historical_transactions.csv",
            "category_mapping_version": "v1.0",
            "preprocessing_version": "v1.0_user_vector_aggregation",
            "optimal_k": int(self.n_clusters),
            "training_date": datetime.now().isoformat(),
            "n_users_clustered": int(len(user_df)),
            "feature_list": self.feature_names,
            "pca_explained_variance_ratio": [float(v) for v in self.pca.explained_variance_ratio_],
            "evaluation_metrics": {
                "silhouette_score": float(best_sil),
                "davies_bouldin_index": float(davies_bouldin_score(X_scaled, cluster_labels))
            },
            "hyperparameter_sweep": sweep_results,
            "cluster_personas": self.cluster_personas
        }

        self.save(model_dir)
        return self.metadata

    def _assign_behavioral_personas(self, centroids: pd.DataFrame) -> dict:
        """
        Creates descriptive behavioral personas reflecting empirical spending patterns.
        Strictly avoids psychological, moral, or status claims.
        """
        personas = {}
        for c_id, row in centroids.iterrows():
            wk_ratio = float(row["weekend_spending_ratio"])
            rec_ratio = float(row["recurring_expense_ratio"])
            sav_ratio = float(row["savings_ratio"])
            food_r = float(row["food_ratio"])
            sub_r = float(row["subscription_ratio"])
            ent = float(row["category_entropy"])

            # Rank-based persona differentiation
            if sav_ratio >= centroids["savings_ratio"].median() and rec_ratio >= centroids["recurring_expense_ratio"].median():
                name = "High-Savings Structured Planner"
                desc = "Characterized by above-average savings allocations and high structured recurring commitments."
            elif wk_ratio >= centroids["weekend_spending_ratio"].median():
                name = "Discretionary Weekend & Lifestyle Spender"
                desc = "Characterized by concentrated weekend spending and higher discretionary dining/retail expense ratios."
            elif sub_r >= centroids["subscription_ratio"].median():
                name = "Subscription & Digital Services Consumer"
                desc = "Characterized by active recurring digital subscriptions, utilities, and telecom commitments."
            else:
                name = "Balanced Diversified Spender"
                desc = "Characterized by evenly distributed category allocations across weekday utilities, groceries, and travel."

            personas[int(c_id)] = {
                "persona_name": name,
                "description": desc,
                "weekend_ratio": round(wk_ratio, 3),
                "recurring_ratio": round(rec_ratio, 3),
                "savings_ratio": round(sav_ratio, 3),
                "subscription_ratio": round(sub_r, 3),
                "category_entropy": round(ent, 3),
                "centroid_values": {col: round(float(row[col]), 3) for col in centroids.columns}
            }
        return personas

    def predict_user_segment(self, user_features_dict: dict) -> dict:
        """
        Inference function returning structured user segmentation profile.
        """
        df_single = pd.DataFrame([user_features_dict])[self.feature_names]
        X_scaled = self.scaler.transform(df_single.values)
        cluster_id = int(self.kmeans.predict(X_scaled)[0])
        persona = self.cluster_personas.get(cluster_id, {
            "persona_name": f"Segment_{cluster_id}",
            "description": "General behavioral segment"
        })

        return {
            "cluster_id": cluster_id,
            "persona_name": persona["persona_name"],
            "persona_description": persona["description"],
            "weekend_spending_ratio": round(float(user_features_dict.get("weekend_spending_ratio", 0.0)), 3),
            "subscription_ratio": round(float(user_features_dict.get("subscription_ratio", 0.0)), 3),
            "savings_ratio": round(float(user_features_dict.get("savings_ratio", 0.0)), 3)
        }

    def save(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.kmeans, os.path.join(model_dir, "model.pkl"))
        joblib.dump(self.scaler, os.path.join(model_dir, "scaler.pkl"))
        joblib.dump(self.pca, os.path.join(model_dir, "pca.pkl"))
        with open(os.path.join(model_dir, "metadata.json"), "w") as f:
            json.dump(self.metadata, f, indent=4)
        print(f"[Model Saved] Successfully stored segmentation artifacts at {model_dir}")

    @classmethod
    def load(cls, model_dir: str):
        instance = cls()
        instance.kmeans = joblib.load(os.path.join(model_dir, "model.pkl"))
        instance.scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
        instance.pca = joblib.load(os.path.join(model_dir, "pca.pkl"))
        with open(os.path.join(model_dir, "metadata.json"), "r") as f:
            instance.metadata = json.load(f)
        instance.cluster_personas = {int(k): v for k, v in instance.metadata.get("cluster_personas", {}).items()}
        return instance


def run_segmentation_pipeline(data_path: str, model_dir: str):
    df_hist = pd.read_csv(data_path)
    segmenter = UserSegmenter()
    segmenter.train_and_evaluate(df_hist, model_dir)


if __name__ == "__main__":
    hist_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw/raw_historical_transactions.csv"))
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/segmentation"))
    run_segmentation_pipeline(hist_path, model_dir)
