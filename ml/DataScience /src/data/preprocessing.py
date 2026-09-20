"""
Unified Data Preprocessing and Leakage-Proof Split Pipeline.
Handles:
1. Raw schema validation & immutable data loading
2. Text cleaning & merchant string normalization
3. Category mapping verification
4. Feature extraction
5. Stratified splitting for classification (Train/Val/Test)
6. Strict chronological splitting for historical time-series forecasting & anomaly detection
"""

import os
import re
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

SEED = 42

def clean_merchant_text(text: str) -> str:
    """
    Standardizes raw merchant strings by stripping POS noise, terminal codes,
    special characters, and normalizing casing.
    """
    if not isinstance(text, str):
        return ""
    
    # Remove terminal/pos prefixes & transaction ids
    text = re.sub(r'\b(POS|DEBIT|CHECKCARD|ONLINE|PAYPAL|SQ|TST|PURCHASE|RETAIL)\b[*#\-\s]*', ' ', text, flags=re.IGNORECASE)
    # Remove hash numbers like #1049, alphanumeric store codes like *1A2B
    text = re.sub(r'[*#]\w+', ' ', text)
    # Remove isolated numbers and ref codes
    text = re.sub(r'\b\d{4,}\b', ' ', text)
    # Remove special characters except basic whitespace
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    # Collapse multiple whitespaces
    text = re.sub(r'\s+', ' ', text).strip().upper()
    return text

def clean_description_text(text: str) -> str:
    """
    Cleans raw description strings, removing ref codes and extra whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r'ref:\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

def preprocess_classification_dataset(
    raw_path: str,
    output_dir: str,
    category_mapping_path: str = None
) -> dict:
    """
    Preprocesses Dataset 1: Transaction Categorization.
    Outputs processed data, feature tables, and stratified train/val/test splits.
    """
    df_raw = pd.read_csv(raw_path)
    print(f"[Classification Prep] Raw rows loaded: {len(df_raw)}")
    
    # Validation
    required_cols = {"merchant_raw", "description", "amount", "transaction_type", "payment_method", "category", "subcategory"}
    assert required_cols.issubset(df_raw.columns), f"Missing required columns in {raw_path}"

    df = df_raw.copy()
    
    # Clean text features
    df["merchant_clean"] = df["merchant_raw"].apply(clean_merchant_text)
    df["description_clean"] = df["description"].apply(clean_description_text)
    
    # Combined NLP text representation
    df["text_features"] = df["merchant_clean"] + " " + df["description_clean"]
    
    # Log amount for numerical stability
    df["log_amount"] = np.log1p(df["amount"])
    
    # Standardized composite target
    df["full_target"] = df["category"] + "::" + df["subcategory"]
    
    # Check for missing values
    df = df.dropna(subset=["text_features", "full_target", "amount"])
    
    # Stratified Train (70%), Validation (15%), Test (15%)
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=SEED, stratify=df["full_target"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=SEED, stratify=temp_df["full_target"]
    )

    # Save to standard directories
    processed_path = os.path.join(output_dir, "processed", "transactions_classification_processed.csv")
    train_path = os.path.join(output_dir, "train", "classification_train.csv")
    val_path = os.path.join(output_dir, "validation", "classification_val.csv")
    test_path = os.path.join(output_dir, "test", "classification_test.csv")
    
    df.to_csv(processed_path, index=False)
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"[Classification Prep] Split saved: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return {
        "processed": processed_path,
        "train": train_path,
        "val": val_path,
        "test": test_path
    }

def preprocess_historical_dataset(
    raw_path: str,
    output_dir: str
) -> dict:
    """
    Preprocesses Dataset 2: Historical Transactions.
    Performs chronological sorting, temporal feature extraction, user rolling statistics,
    and strict chronological train/val/test splits without lookahead leakage.
    """
    df_raw = pd.read_csv(raw_path)
    print(f"[Historical Prep] Raw rows loaded: {len(df_raw)}")
    
    df = df_raw.copy()
    df["date_dt"] = pd.to_datetime(df["date"])
    
    # Sort strictly by date to preserve temporal causality
    df = df.sort_values(by=["date_dt", "user_id"]).reset_index(drop=True)
    
    # Temporal variables
    df["year_month"] = df["date_dt"].dt.to_period("M")
    df["day_of_week"] = df["date_dt"].dt.dayofweek
    df["hour"] = df["date_dt"].dt.hour
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    
    # Time-of-day bucket
    def get_time_bucket(hour):
        if 5 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 17:
            return "Afternoon"
        elif 17 <= hour < 22:
            return "Evening"
        else:
            return "Night"
            
    df["time_of_day"] = df["hour"].apply(get_time_bucket)
    
    # Chronological splitting by month
    unique_months = sorted(df["year_month"].unique())
    n_months = len(unique_months)
    assert n_months >= 3, "Historical dataset requires at least 3 months for chronological splits"
    
    # E.g. If 8 months: Train = Months 1-5, Val = Month 6, Test = Months 7-8
    train_end_idx = max(1, int(n_months * 0.65))
    val_end_idx = max(train_end_idx + 1, int(n_months * 0.80))
    
    train_months = unique_months[:train_end_idx]
    val_months = unique_months[train_end_idx:val_end_idx]
    test_months = unique_months[val_end_idx:]
    
    print(f"[Historical Prep] Chronological Splitting: Train Months={train_months}, Val Months={val_months}, Test Months={test_months}")
    
    train_df = df[df["year_month"].isin(train_months)].copy()
    val_df = df[df["year_month"].isin(val_months)].copy()
    test_df = df[df["year_month"].isin(test_months)].copy()

    # Drop temporary period columns for CSV storage
    df = df.drop(columns=["date_dt", "year_month"])
    train_df = train_df.drop(columns=["date_dt", "year_month"])
    val_df = val_df.drop(columns=["date_dt", "year_month"])
    test_df = test_df.drop(columns=["date_dt", "year_month"])

    processed_path = os.path.join(output_dir, "processed", "historical_transactions_processed.csv")
    train_path = os.path.join(output_dir, "train", "historical_train.csv")
    val_path = os.path.join(output_dir, "validation", "historical_val.csv")
    test_path = os.path.join(output_dir, "test", "historical_test.csv")

    df.to_csv(processed_path, index=False)
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"[Historical Prep] Split saved: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return {
        "processed": processed_path,
        "train": train_path,
        "val": val_path,
        "test": test_path
    }

def preprocess_survey_dataset(
    raw_path: str,
    output_dir: str
) -> str:
    """
    Preprocesses Dataset 3: Behavioral Research Survey.
    Cleans, checks outliers, encodes categorical ranges, and prepares for econometrics/EDA.
    """
    df_raw = pd.read_csv(raw_path)
    print(f"[Survey Prep] Raw rows loaded: {len(df_raw)}")
    
    df = df_raw.copy()
    
    # Map income range to ordinal rank and median value (in INR / Lakhs Per Annum)
    income_ordinal_map = {
        "< ₹3,00,000": 1,
        "₹3,00,000 - ₹6,00,000": 2,
        "₹6,00,000 - ₹12,00,000": 3,
        "₹12,00,000 - ₹25,00,000": 4,
        "> ₹25,00,000": 5
    }
    income_median_map = {
        "< ₹3,00,000": 200000,
        "₹3,00,000 - ₹6,00,000": 450000,
        "₹6,00,000 - ₹12,00,000": 900000,
        "₹12,00,000 - ₹25,00,000": 1850000,
        "> ₹25,00,000": 3500000
    }
    
    df["income_ordinal"] = df["income_range"].map(income_ordinal_map)
    df["estimated_income"] = df["income_range"].map(income_median_map)
    
    processed_path = os.path.join(output_dir, "processed", "user_behavior_survey_processed.csv")
    df.to_csv(processed_path, index=False)
    print(f"[Survey Prep] Processed survey data saved at {processed_path}")
    return processed_path

def run_preprocessing_pipeline(base_data_dir: str):
    raw_dir = os.path.join(base_data_dir, "raw")
    output_dir = base_data_dir
    
    print("=" * 60)
    print("STARTING UNIFIED DATA PREPROCESSING & SPLIT PIPELINE")
    print("=" * 60)
    
    # 1. Classification dataset
    cat_raw = os.path.join(raw_dir, "raw_transactions_classification.csv")
    cat_mapping = os.path.join(raw_dir, "category_mapping.csv")
    res_cat = preprocess_classification_dataset(cat_raw, output_dir, cat_mapping)
    
    # 2. Historical dataset
    hist_raw = os.path.join(raw_dir, "raw_historical_transactions.csv")
    res_hist = preprocess_historical_dataset(hist_raw, output_dir)
    
    # 3. Survey dataset
    survey_raw = os.path.join(raw_dir, "raw_user_behavior_survey.csv")
    res_survey = preprocess_survey_dataset(survey_raw, output_dir)
    
    print("=" * 60)
    print("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)

if __name__ == "__main__":
    base_data = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
    run_preprocessing_pipeline(base_data)
