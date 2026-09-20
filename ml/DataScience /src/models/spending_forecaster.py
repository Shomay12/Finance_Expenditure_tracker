"""
Temporal Spending Forecasting Engine.
Predicts remaining monthly spend and projected month-end total spend per user.
Implements:
1. Daily aggregated time-series feature engineering (Rolling 7d/30d, MTD, velocity, days remaining)
2. Strict Chronological Splitting (Train: Months 1-5, Val: Month 6, Test: Months 7-8)
3. Model Progression: Historical Average, Moving Average Baseline vs Ridge vs Random Forest vs XGBoost Regressor
4. Evaluation: MAE, RMSE, MAPE
5. Serialization to /models/forecasting/
6. Structured JSON Forecast Output
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

SEED = 42

def calculate_mape(y_true, y_pred):
    mask = y_true > 100.0
    if np.sum(mask) == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

class SpendingForecaster:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.model_name = "XGBoostRegressor"
        self.feature_cols = [
            "month_to_date_spending",
            "days_elapsed",
            "days_remaining",
            "average_daily_spending_mtd",
            "rolling_7_day_spending",
            "rolling_30_day_spending",
            "spending_velocity",
            "previous_month_spending",
            "monthly_recurring_committed"
        ]
        self.metadata = {}

    def build_daily_features(self, df_history: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms raw transaction stream into user-daily time series with non-leaking rolling metrics.
        """
        df = df_history.copy()
        df["date_dt"] = pd.to_datetime(df["date"])
        df["day"] = df["date_dt"].dt.date
        df["year_month"] = df["date_dt"].dt.to_period("M")

        # Focus on expenses only for spending forecast
        expenses = df[df["category"] == "EXPENSE"].copy()

        # Identify fixed recurring commitments per user (Indian context)
        rec_merchants = {"netflix", "spotify", "broadband", "rent", "lease", "gym", "internet", "airtel", "fibernet", "cult", "sip", "nobroker"}
        expenses["is_recurring_commit"] = expenses["merchant"].apply(
            lambda x: any(rm in str(x).lower() for rm in rec_merchants)
        )

        daily_user = expenses.groupby(["user_id", "year_month", "day"]).agg(
            daily_total=("amount", "sum"),
            daily_recurring=("amount", lambda s: s[expenses.loc[s.index, "is_recurring_commit"]].sum())
        ).reset_index()

        # Build complete user date grid to prevent missing day gaps
        all_user_records = []
        for (uid, ym), g in daily_user.groupby(["user_id", "year_month"]):
            days_in_m = ym.days_in_month
            month_start = ym.to_timestamp().date()
            dates = [month_start + pd.Timedelta(days=i) for i in range(days_in_m)]
            grid_df = pd.DataFrame({"day": dates})
            grid_df["user_id"] = uid
            grid_df["year_month"] = ym
            merged = pd.merge(grid_df, g, on=["user_id", "year_month", "day"], how="left").fillna(0.0)
            merged = merged.sort_values("day").reset_index(drop=True)

            merged["day_num"] = [(d - month_start).days + 1 for d in merged["day"]]
            merged["days_elapsed"] = merged["day_num"]
            merged["days_remaining"] = days_in_m - merged["days_elapsed"]
            merged["month_to_date_spending"] = merged["daily_total"].cumsum()

            # Average daily MTD
            merged["average_daily_spending_mtd"] = merged["month_to_date_spending"] / merged["days_elapsed"]

            # Total month target spend
            actual_month_total = float(merged["daily_total"].sum())
            merged["actual_month_total"] = actual_month_total
            merged["target_remaining_spending"] = actual_month_total - merged["month_to_date_spending"]

            all_user_records.append(merged)

        df_daily_all = pd.concat(all_user_records, ignore_index=True)
        df_daily_all = df_daily_all.sort_values(by=["user_id", "day"]).reset_index(drop=True)

        # Compute rolling window metrics (7d, 30d) strictly backwards
        df_daily_all["rolling_7_day_spending"] = df_daily_all.groupby("user_id")["daily_total"].transform(
            lambda s: s.shift(1).rolling(7, min_periods=1).sum()
        ).fillna(0.0)

        df_daily_all["rolling_30_day_spending"] = df_daily_all.groupby("user_id")["daily_total"].transform(
            lambda s: s.shift(1).rolling(30, min_periods=1).sum()
        ).fillna(0.0)

        # Spending velocity: recent 7d daily pace vs 30d daily pace
        recent_daily_rate = df_daily_all["rolling_7_day_spending"] / 7.0
        monthly_daily_rate = df_daily_all["rolling_30_day_spending"] / 30.0 + 1e-3
        df_daily_all["spending_velocity"] = (recent_daily_rate / monthly_daily_rate).clip(0.1, 5.0)

        # Previous month spending
        user_monthly_totals = df_daily_all.groupby(["user_id", "year_month"])["daily_total"].sum().reset_index()
        user_monthly_totals["prev_month_spending"] = user_monthly_totals.groupby("user_id")["daily_total"].shift(1)
        
        df_daily_all = pd.merge(
            df_daily_all,
            user_monthly_totals[["user_id", "year_month", "prev_month_spending"]],
            on=["user_id", "year_month"],
            how="left"
        )
        df_daily_all["previous_month_spending"] = df_daily_all["prev_month_spending"].fillna(df_daily_all["month_to_date_spending"] * 2.0)

        # Monthly recurring committed
        user_recurring_totals = df_daily_all.groupby(["user_id", "year_month"])["daily_recurring"].sum().reset_index()
        df_daily_all = pd.merge(
            df_daily_all,
            user_recurring_totals.rename(columns={"daily_recurring": "monthly_recurring_committed"}),
            on=["user_id", "year_month"],
            how="left"
        ).fillna(0.0)

        # Filter out day 0/31 extremes where days_remaining is 0 or day is 1
        df_model = df_daily_all[(df_daily_all["days_remaining"] > 0) & (df_daily_all["days_elapsed"] >= 3)].copy()
        return df_model

    def train_and_evaluate(self, raw_hist_path: str, model_dir: str) -> dict:
        print("\n" + "=" * 70)
        print("SPENDING FORECASTING: TEMPORAL TRAINING & BENCHMARKING")
        print("=" * 70)

        df_raw = pd.read_csv(raw_hist_path)
        df_features = self.build_daily_features(df_raw)

        unique_months = sorted(df_features["year_month"].unique())
        print(f"Total temporal span: {len(unique_months)} months: {unique_months}")

        # Chronological Split: Train = Months 1-5, Val = Month 6, Test = Months 7-8
        train_months = unique_months[:5]
        val_months = [unique_months[5]] if len(unique_months) > 5 else [unique_months[-1]]
        test_months = unique_months[6:] if len(unique_months) > 6 else [unique_months[-1]]

        print(f"Train Months: {train_months} | Val Months: {val_months} | Test Months: {test_months}")

        train_data = df_features[df_features["year_month"].isin(train_months)]
        val_data = df_features[df_features["year_month"].isin(val_months)]
        test_data = df_features[df_features["year_month"].isin(test_months)]

        X_train = train_data[self.feature_cols]
        y_train = train_data["target_remaining_spending"]

        X_val = val_data[self.feature_cols]
        y_val = val_data["target_remaining_spending"]

        X_test = test_data[self.feature_cols]
        y_test = test_data["target_remaining_spending"]

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        # Baseline 1: Historical / Linear Daily Rate Extrapolation
        base_val_preds = val_data["average_daily_spending_mtd"] * val_data["days_remaining"]
        base_mae = mean_absolute_error(y_val, base_val_preds)
        base_rmse = np.sqrt(mean_squared_error(y_val, base_val_preds))
        base_mape = calculate_mape(y_val.values, base_val_preds.values)

        print(f"\n[BASELINE 1 - MTD Daily Rate Extrapolation]:")
        print(f"  Val MAE: ₹{base_mae:,.2f} | RMSE: ₹{base_rmse:,.2f} | MAPE: {base_mape:.2f}%")

        # Baseline 2: 30-Day Moving Average Extrapolation
        base2_val_preds = (val_data["rolling_30_day_spending"] / 30.0) * val_data["days_remaining"]
        base2_mae = mean_absolute_error(y_val, base2_val_preds)
        base2_rmse = np.sqrt(mean_squared_error(y_val, base2_val_preds))
        base2_mape = calculate_mape(y_val.values, base2_val_preds.values)

        print(f"\n[BASELINE 2 - Rolling 30d Moving Avg Extrapolation]:")
        print(f"  Val MAE: ₹{base2_mae:,.2f} | RMSE: ₹{base2_rmse:,.2f} | MAPE: {base2_mape:.2f}%")

        # ML Models
        ml_models = {
            "RidgeRegression": Ridge(alpha=1.0, random_state=SEED),
            "RandomForestRegressor": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=SEED, n_jobs=-1),
            "XGBoostRegressor": XGBRegressor(n_estimators=120, max_depth=5, learning_rate=0.08, subsample=0.85, random_state=SEED, n_jobs=-1)
        }

        benchmarks = {
            "Baseline_MTD_Extrapolation": {"mae": float(base_mae), "rmse": float(base_rmse), "mape": float(base_mape)},
            "Baseline_30d_MovingAvg": {"mae": float(base2_mae), "rmse": float(base2_rmse), "mape": float(base2_mape)}
        }

        best_val_mae = float("inf")
        best_model = None
        best_name = None

        for name, mdl in ml_models.items():
            mdl.fit(X_train_scaled, y_train)
            preds = mdl.predict(X_val_scaled)
            mae = mean_absolute_error(y_val, preds)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            mape = calculate_mape(y_val.values, preds)

            print(f"\n[ML Model - {name}]:")
            print(f"  Val MAE: ₹{mae:,.2f} | RMSE: ₹{rmse:,.2f} | MAPE: {mape:.2f}%")

            benchmarks[name] = {"mae": float(mae), "rmse": float(rmse), "mape": float(mape)}

            if mae < best_val_mae:
                best_val_mae = mae
                best_model = mdl
                best_name = name

        print(f"\n--> Best Validated Forecaster: [{best_name}] with Val MAE: ₹{best_val_mae:,.2f}")
        self.model = best_model
        self.model_name = best_name

        # Final Test Evaluation on held-out months (Months 7-8)
        test_preds = self.model.predict(X_test_scaled)
        test_mae = mean_absolute_error(y_test, test_preds)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        test_mape = calculate_mape(y_test.values, test_preds)

        print("\n" + "=" * 70)
        print(f"FINAL HELD-OUT CHRONOLOGICAL TEST RESULTS [{best_name}]")
        print("=" * 70)
        print(f"Test MAE:   ₹{test_mae:,.2f}")
        print(f"Test RMSE:  ₹{test_rmse:,.2f}")
        print(f"Test MAPE:  {test_mape:.2f}%")

        self.metadata = {
            "model_name": f"SpendingForecaster_{best_name}",
            "version": "1.0.0",
            "training_dataset": "raw_historical_transactions.csv",
            "category_mapping_version": "v1.0",
            "preprocessing_version": "v1.0_daily_rolling_temporal",
            "chronological_split": {
                "train_months": [str(m) for m in train_months],
                "val_months": [str(m) for m in val_months],
                "test_months": [str(m) for m in test_months]
            },
            "training_date": datetime.now().isoformat(),
            "n_train_samples": int(len(train_data)),
            "feature_list": self.feature_cols,
            "benchmarks": benchmarks,
            "evaluation_metrics": {
                "test_mae": float(test_mae),
                "test_rmse": float(test_rmse),
                "test_mape": float(test_mape)
            }
        }

        self.save(model_dir)
        return self.metadata

    def predict_spending(
        self,
        current_mtd_spending: float,
        days_elapsed: int,
        days_remaining: int,
        rolling_7_day: float,
        rolling_30_day: float,
        previous_month_spending: float,
        monthly_recurring_committed: float
    ) -> dict:
        """
        Inference API returning structured forecasting projection.
        """
        avg_daily = current_mtd_spending / max(1, days_elapsed)
        r7_daily = rolling_7_day / 7.0
        r30_daily = rolling_30_day / 30.0 + 1e-3
        velocity = float(np.clip(r7_daily / r30_daily, 0.1, 5.0))

        feat_df = pd.DataFrame([{
            "month_to_date_spending": float(current_mtd_spending),
            "days_elapsed": int(days_elapsed),
            "days_remaining": int(days_remaining),
            "average_daily_spending_mtd": float(avg_daily),
            "rolling_7_day_spending": float(rolling_7_day),
            "rolling_30_day_spending": float(rolling_30_day),
            "spending_velocity": velocity,
            "previous_month_spending": float(previous_month_spending),
            "monthly_recurring_committed": float(monthly_recurring_committed)
        }])[self.feature_cols]

        X_scaled = self.scaler.transform(feat_df)
        pred_remaining = max(0.0, float(self.model.predict(X_scaled)[0]))
        pred_month_end = float(current_mtd_spending + pred_remaining)

        # Expected error bound based on test MAE
        expected_error = self.metadata.get("test_metrics", {}).get("mae", 1500.0)
        confidence = float(np.clip(1.0 - (expected_error / max(1000.0, pred_month_end)), 0.65, 0.95))

        return {
            "current_spending": round(current_mtd_spending, 2),
            "predicted_remaining": round(pred_remaining, 2),
            "predicted_month_end": round(pred_month_end, 2),
            "forecast_error_margin": round(expected_error, 2),
            "forecast_confidence": round(confidence, 3)
        }

    def save(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.model, os.path.join(model_dir, "model.pkl"))
        joblib.dump(self.scaler, os.path.join(model_dir, "scaler.pkl"))
        with open(os.path.join(model_dir, "metadata.json"), "w") as f:
            json.dump(self.metadata, f, indent=4)
        print(f"[Model Saved] Successfully stored forecaster artifacts at {model_dir}")

    @classmethod
    def load(cls, model_dir: str):
        instance = cls()
        instance.model = joblib.load(os.path.join(model_dir, "model.pkl"))
        instance.scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
        with open(os.path.join(model_dir, "metadata.json"), "r") as f:
            instance.metadata = json.load(f)
        return instance


def run_forecasting_pipeline(raw_hist_path: str, model_dir: str):
    forecaster = SpendingForecaster()
    forecaster.train_and_evaluate(raw_hist_path, model_dir)

    print("\n" + "=" * 70)
    print("LIVE SPENDING FORECAST DEMO")
    print("=" * 70)

    # Demo scenarios:
    # 1. Mid-month user with Rs 31,500 spend on Day 15 (15 days remaining)
    # 2. End-of-month user with Rs 58,000 spend on Day 25 (5 days remaining)
    demos = [
        ("Mid-Month Spending Projection (Day 15)", 31500.0, 15, 15, 7800.0, 32000.0, 62000.0, 35000.0),
        ("Late-Month Spending Projection (Day 25)", 58000.0, 25, 5, 6200.0, 56000.0, 65000.0, 35000.0)
    ]

    for title, mtd, d_el, d_rem, r7, r30, prev, rec in demos:
        out = forecaster.predict_spending(mtd, d_el, d_rem, r7, r30, prev, rec)
        print(f"\nScenario: {title}")
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    hist_raw = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw/raw_historical_transactions.csv"))
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/forecasting"))
    run_forecasting_pipeline(hist_raw, model_path)
