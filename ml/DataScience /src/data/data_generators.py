"""
Data Generators for AI-Based Personal Finance Tracking & Intelligence System.
INDIAN CONTEXT (INR ₹):
Generates reproducible, statistically grounded raw datasets tailored for Indian financial behavior:
1. Transaction Categorization (Dataset 1) - UPI, NetBanking, Indian Merchants, INR
2. Historical Transaction Data (Dataset 2) - Longitudinal multi-user Indian financial patterns
3. User Behavior / Research Survey Dataset (Dataset 3) - Indian payment ecosystem (UPI fragmentation, SIPs, tracking friction)
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

def generate_transaction_classification_dataset(
    output_path: str,
    n_samples: int = 6000
) -> pd.DataFrame:
    """
    Generates Dataset 1: Raw transactions for supervised category & subcategory classification in INR.
    Features Indian merchants, UPI payment handles, NEFT/IMPS remarks, credit cards, debit cards.
    """
    merchants_by_subcategory = {
        # INCOME
        ("INCOME", "Salary"): [
            ("TATA CONSULTANCY SERVICES SALARY", "Monthly salary credit NEFT/IMPS", (35000, 180000), "credit", ["Direct Deposit", "Net Banking"]),
            ("INFOSYS LTD PAYROLL CR", "Corporate payroll net salary payout", (40000, 220000), "credit", ["Direct Deposit", "Net Banking"]),
            ("WIPRO TECHNOLOGIES SALARY CR", "Monthly employee salary disbursement", (30000, 160000), "credit", ["Direct Deposit"]),
            ("ACCENTURE INDIA PAYROLL DIRECT", "Monthly net salary credit", (45000, 250000), "credit", ["Direct Deposit"])
        ],
        ("INCOME", "Freelance"): [
            ("UPWORK FOREX INWARD REMITTANCE", "Freelance UI/UX design project fee", (8000, 65000), "credit", ["Wire Transfer", "Bank Transfer"]),
            ("FIVERR GIG PAYOUT RAZORPAY", "Web development consulting payment", (4000, 45000), "credit", ["UPI", "Bank Transfer"]),
            ("RAZORPAY MERCHANT SETTLEMENT", "Freelance software contract invoice", (12000, 95000), "credit", ["UPI", "Net Banking"])
        ],
        ("INCOME", "Business Income"): [
            ("BHARATPE QR SETTLEMENT", "Retail store daily UPI merchant collection", (15000, 120000), "credit", ["UPI", "Bank Transfer"]),
            ("PAYTM BUSINESS SETTLEMENT CR", "Shop sales disbursement", (10000, 150000), "credit", ["UPI", "Bank Transfer"])
        ],
        ("INCOME", "Interest"): [
            ("HDFC BANK FD INTEREST CR", "Quarterly fixed deposit interest payout", (500, 15000), "credit", ["Bank Credit"]),
            ("ICICI BANK SB INTEREST CREDIT", "Savings account quarterly interest credit", (150, 4500), "credit", ["Bank Credit"]),
            ("SBI TERM DEPOSIT INTEREST", "Cumulative term deposit interest", (800, 25000), "credit", ["Bank Credit"])
        ],
        ("INCOME", "Other Income"): [
            ("INCOME TAX REFUND CPC BANGALORE", "Direct tax refund FY2024-25 credit", (2500, 45000), "credit", ["Direct Deposit"]),
            ("CRED CASHBACK REWARD TO BANK", "Credit card bill payment cashback reward", (50, 1500), "credit", ["UPI", "Statement Credit"]),
            ("GOOGLE PAY REWARD CASHBACK", "Scratch card UPI transaction cashback", (20, 500), "credit", ["UPI"])
        ],

        # EXPENSE
        ("EXPENSE", "Food"): [
            ("SWIGGY*BANGALORE RESTAURANT", "Food delivery dinner order biryani", (180, 1400), "debit", ["UPI", "Credit Card", "Debit Card"]),
            ("ZOMATO FOOD ORDER #49102", "Weekend lunch food delivery burger pizza", (220, 1800), "debit", ["UPI", "Credit Card", "Paytm Wallet"]),
            ("STARBUCKS COFFEE CONNAUGHT PL", "Cappuccino and classic cold coffee", (250, 950), "debit", ["UPI", "Credit Card"]),
            ("CHAI POINT CYBER CITY", "Ginger tea samosa and banana cake", (80, 350), "debit", ["UPI", "Paytm"]),
            ("HALDIRAMS RESTAURANT NOIDA", "North Indian thali and sweets", (300, 1500), "debit", ["UPI", "Debit Card"]),
            ("DOMINOS PIZZA ONLINE ORDER", "Farmhouse medium pizza garlic bread", (350, 1200), "debit", ["UPI", "Credit Card"])
        ],
        ("EXPENSE", "Groceries"): [
            ("ZEPTO QUICK COMMERCE", "10-min grocery delivery milk bread curd", (150, 1800), "debit", ["UPI", "Credit Card", "Paytm"]),
            ("BLINKIT COMMERCE GURGAON", "Weekly fresh fruits vegetables eggs", (250, 3200), "debit", ["UPI", "Credit Card"]),
            ("SWIGGY INSTAMART GROCERY", "Pantry staples oil wheat dal spices", (300, 4500), "debit", ["UPI", "Credit Card"]),
            ("BIGBASKET SUPERMARKET ORDER", "Monthly household monthly grocery order", (1500, 9500), "debit", ["UPI", "Credit Card", "Net Banking"]),
            ("DMART AVENUE SUPERMARTS", "In-store discounted FMCG household groceries", (1200, 8500), "debit", ["Debit Card", "UPI", "Credit Card"])
        ],
        ("EXPENSE", "Shopping"): [
            ("AMAZON SELLER SERVICES IN", "Online electronics apparel kitchenware", (499, 25000), "debit", ["Credit Card", "UPI", "Amazon Pay", "Debit Card"]),
            ("FLIPKART INTERNET PVT LTD", "Smartphone accessories audio headphones", (599, 35000), "debit", ["UPI", "Credit Card", "Debit Card"]),
            ("MYNTRA DESIGNS BANGALORE", "Branded shirts jeans casual sneakers", (899, 8500), "debit", ["UPI", "Credit Card"]),
            ("NYKAA E-RETAIL BEAUTY", "Skincare grooming perfumes and cosmetics", (450, 5500), "debit", ["UPI", "Credit Card"]),
            ("CROMA ELECTRONICS STORE", "Home appliance laptop monitor accessories", (1200, 45000), "debit", ["Credit Card", "Debit Card", "UPI"])
        ],
        ("EXPENSE", "Transport"): [
            ("UBER INDIA RIDES BANGALORE", "Cab ride home from office commute", (90, 1200), "debit", ["UPI", "Credit Card", "Paytm"]),
            ("OLA CABS TRIP #94812", "City taxi rideshare transport", (110, 950), "debit", ["UPI", "Ola Money"]),
            ("RAPIDO BIKE TAXI RIDE", "Quick metro station last-mile commute", (35, 220), "debit", ["UPI", "Paytm"]),
            ("DELHI METRO DMRC CARD RECHARGE", "Subway smart card automated reload", (200, 1000), "debit", ["UPI", "Debit Card"]),
            ("IHMCL FASTAG TOLL PLAZA", "National highway automated toll deduction", (65, 850), "debit", ["Auto Debit", "FASTag Wallet"])
        ],
        ("EXPENSE", "Fuel"): [
            ("INDIAN OIL CORP PETROL PUMP", "Two-wheeler petrol tank fuel refill", (300, 2000), "debit", ["UPI", "Debit Card"]),
            ("BHARAT PETROLEUM BPCL BUNKER", "Automobile unleaded petrol fillup", (1500, 6000), "debit", ["Credit Card", "UPI", "Debit Card"]),
            ("HPCL PETROL OUTLET PUNE", "Diesel fuel refill for SUV", (2000, 6500), "debit", ["Credit Card", "Debit Card", "UPI"])
        ],
        ("EXPENSE", "Bills & Utilities"): [
            ("BESCOM ELECTRICITY BILL", "Monthly home electricity utility power bill", (800, 5500), "debit", ["UPI", "Net Banking", "Auto Debit"]),
            ("TATA POWER DDL BILL PAY", "Monthly residential power consumption", (1200, 6500), "debit", ["UPI", "Net Banking"]),
            ("ACT FIBERNET BROADBAND BILL", "Monthly high-speed fiber internet plan", (799, 1499), "debit", ["UPI", "Credit Card", "Auto Debit"]),
            ("AIRTEL PREPAID/POSTPAID BILL", "Mobile recharge and 5G family plan", (299, 1199), "debit", ["UPI", "Auto Debit", "Credit Card"]),
            ("JIO FIBER BROADBAND RECHARGE", "Prepaid unlimited home internet recharge", (399, 999), "debit", ["UPI", "JioMoney"]),
            ("INDRAPRASTHA GAS IGL BILL", "Piped cooking natural gas monthly bill", (400, 1800), "debit", ["UPI", "Net Banking"])
        ],
        ("EXPENSE", "Rent & Housing"): [
            ("NOBROKER RENTPAY TO LANDLORD", "Monthly 2BHK residential flat house rent", (12000, 55000), "debit", ["UPI", "Credit Card", "Bank Transfer"]),
            ("CRED RENTPAY DIRECT WIRE", "Monthly apartment lease payment to owner", (15000, 65000), "debit", ["Credit Card", "UPI"]),
            ("SOCIETY MAINTENANCE CHARGES", "Monthly gated community maintenance water", (2000, 8500), "debit", ["UPI", "Net Banking"])
        ],
        ("EXPENSE", "Healthcare"): [
            ("APOLLO PHARMACY MEDICAL", "Prescription diabetes and blood pressure drugs", (200, 4500), "debit", ["UPI", "Credit Card", "Debit Card"]),
            ("TATA 1MG HEALTHCARE STORE", "Online diagnostic medicines and vitamins", (350, 3800), "debit", ["UPI", "Credit Card"]),
            ("DR LAL PATHLABS DIAGNOSTICS", "Full body health checkup blood testing", (800, 6500), "debit", ["UPI", "Credit Card"]),
            ("MAX HEALTHCARE CLINIC DOCTOR", "Outpatient specialist doctor consultation", (700, 2500), "debit", ["UPI", "Credit Card"])
        ],
        ("EXPENSE", "Education"): [
            ("PHYSICSWALLAH COURSE FEE", "Engineering exam prep test series", (999, 6500), "debit", ["UPI", "Net Banking"]),
            ("UNACADEMY SUBSCRIPTION", "UPSC / Tech live learning plus batch", (2500, 28000), "debit", ["Credit Card", "UPI", "Net Banking"]),
            ("DPS SCHOOL QUARTERLY TUITION", "School term tuition fees and bus charges", (15000, 65000), "debit", ["Net Banking", "UPI"])
        ],
        ("EXPENSE", "Entertainment"): [
            ("BOOKMYSHOW PVR INOX MOVIES", "Weekend cinema IMAX movie tickets snacks", (400, 2200), "debit", ["UPI", "Credit Card"]),
            ("STEAM GAMES INDIA VALVE", "Digital video game download purchase", (399, 3999), "debit", ["Credit Card", "UPI"]),
            ("SMAASH ENTERTAINMENT BOWLING", "Weekend arcade gaming and bowling", (600, 3500), "debit", ["UPI", "Credit Card"])
        ],
        ("EXPENSE", "Travel"): [
            ("IRCTC RAILWAY TICKET BOOKING", "Train journey 3rd AC express tickets", (450, 4200), "debit", ["UPI", "Net Banking", "Credit Card"]),
            ("MAKEMYTRIP INDIA FLIGHTS", "Roundtrip domestic airline tickets flight", (3500, 24000), "debit", ["Credit Card", "UPI", "Net Banking"]),
            ("INDIGO AIRLINES TICKETS", "Direct domestic airfare booking", (3000, 18000), "debit", ["Credit Card", "UPI"])
        ],
        ("EXPENSE", "Subscriptions"): [
            ("NETFLIX INDIA MONTHLY PLAN", "Monthly 4K Ultra HD video streaming", (199, 649), "debit", ["Credit Card", "Auto Debit", "UPI"]),
            ("SPOTIFY INDIA PREMIUM MUSIC", "Individual music streaming monthly subscription", (119, 179), "debit", ["UPI AutoPay", "Credit Card", "Paytm"]),
            ("HOTSTAR DISNEY PLUS ANNUAL", "Cricket IPL and movie streaming sub", (299, 1499), "debit", ["UPI", "Credit Card"]),
            ("AMAZON PRIME INDIA MEMBERSHIP", "Prime video and free fast shipping sub", (299, 1499), "debit", ["Credit Card", "UPI"]),
            ("YOUTUBE PREMIUM INDIA", "Ad-free video streaming family plan", (129, 189), "debit", ["UPI AutoPay", "Credit Card"]),
            ("OPENAI CHATGPT PLUS SUB", "Monthly AI assistant subscription in INR", (1650, 2100), "debit", ["Credit Card"])
        ],
        ("EXPENSE", "Insurance"): [
            ("HDFC ERGO HEALTH INSURANCE", "Annual family floater health policy premium", (8000, 32000), "debit", ["Net Banking", "Credit Card"]),
            ("LIC OF INDIA LIFE POLICY PREM", "Quarterly life insurance policy premium", (2500, 15000), "debit", ["Net Banking", "UPI"]),
            ("ACKO GENERAL VEHICLE INSUR", "Annual two-wheeler car insurance renewal", (1200, 8500), "debit", ["UPI", "Credit Card"])
        ],
        ("EXPENSE", "Personal Care"): [
            ("URBAN COMPANY SALON AT HOME", "Men grooming haircut and massage services", (350, 2200), "debit", ["UPI", "Credit Card"]),
            ("ENRICH SALON & SPA SERVICES", "Hair styling facial skincare treatments", (600, 4500), "debit", ["UPI", "Credit Card"]),
            ("CULT.FIT GYM MEMBERSHIP", "Monthly elite fitness center membership", (1200, 3500), "debit", ["UPI AutoPay", "Credit Card"])
        ],
        ("EXPENSE", "Gifts & Donations"): [
            ("PM CARES FUND DONATION", "National relief charitable contribution", (500, 10000), "debit", ["UPI", "Net Banking"]),
            ("CRY CHILD RIGHTS CHARITY", "Monthly child welfare education donation", (300, 3000), "debit", ["Auto Debit", "UPI"])
        ],
        ("EXPENSE", "Fees & Charges"): [
            ("HDFC BANK ATM OUT OF NETWORK FEE", "Non-home bank ATM transaction surcharge", (21, 50), "debit", ["Bank Debit"]),
            ("ICICI CREDIT CARD ANNUAL FEE", "Annual credit card membership charges", (500, 3500), "debit", ["Statement Debit"]),
            ("UPI TRANSACTION CHARGE/PENALTY", "Bank ECS bounce penalty charges", (250, 590), "debit", ["Bank Debit"])
        ],
        ("EXPENSE", "Other Expense"): [
            ("LOCAL DRY CLEANERS AND LAUNDRY", "Steam press dry cleaning clothes", (150, 1200), "debit", ["UPI", "Cash"]),
            ("URBAN COMPANY PLUMBER CARPENTER", "Home tap repair carpentry maintenance", (299, 1800), "debit", ["UPI", "Cash"])
        ],

        # TRANSFER
        ("TRANSFER", "Bank Transfer"): [
            ("PHONEPE P2P UPI TRANSFER", "Split dinner bill with friend UPI payment", (100, 5000), "transfer", ["UPI"]),
            ("GPAY UPI TO RAHUL SHARMA", "Instant peer to peer money transfer", (200, 15000), "transfer", ["UPI"]),
            ("IMPS/NEFT TO SISTER ACCOUNT", "Inter-bank savings account funds transfer", (1000, 45000), "transfer", ["Net Banking"])
        ],
        ("TRANSFER", "Credit Card Payment"): [
            ("CRED HDFC CARD BILL PAY", "Monthly credit card outstanding bill payment", (3500, 45000), "transfer", ["UPI", "Net Banking"]),
            ("ICICI CREDIT CARD AUTO DEBIT", "Monthly statement balance autopay deduction", (4000, 60000), "transfer", ["Auto Debit", "Net Banking"])
        ],
        ("TRANSFER", "Investment"): [
            ("ZERODHA BROKING FUND ADD", "UPI funds add to Demat trading account", (1500, 40000), "transfer", ["UPI", "Net Banking"]),
            ("GROWW MUTUAL FUND SIP AUTO", "Monthly index fund SIP investment deduction", (1000, 25000), "transfer", ["Auto Debit", "UPI AutoPay"]),
            ("KITE ZERODHA EQUITY BUY", "Stocks portfolio investment purchase", (2000, 50000), "transfer", ["UPI", "Net Banking"])
        ],
        ("TRANSFER", "Savings"): [
            ("SBI RECURRING DEPOSIT RD CR", "Monthly recurring deposit savings installment", (1000, 20000), "transfer", ["Auto Debit", "Net Banking"]),
            ("HDFC FIXED DEPOSIT CREATE", "Term deposit lock-in placement for 1 year", (10000, 100000), "transfer", ["Net Banking"])
        ],
        ("TRANSFER", "Cash Withdrawal"): [
            ("SBI ATM CASH WITHDRAWAL", "ATM machine currency cash withdrawal", (500, 10000), "transfer", ["Debit Card"]),
            ("HDFC BANK ATM WITHDRAWAL", "Self cash withdrawal from ATM outlet", (1000, 20000), "transfer", ["Debit Card"])
        ],
        ("TRANSFER", "Other Transfer"): [
            ("PAYTM WALLET TOPUP LOAD", "Money transfer from bank account to wallet", (500, 5000), "transfer", ["UPI", "Debit Card"])
        ]
    }

    records = []
    base_date = datetime(2025, 1, 1)
    all_keys = list(merchants_by_subcategory.keys())
    
    weights = []
    for cat, subcat in all_keys:
        if cat == "EXPENSE":
            if subcat in ["Food", "Groceries", "Shopping", "Transport", "Subscriptions"]:
                weights.append(10.0)
            elif subcat in ["Bills & Utilities", "Rent & Housing", "Fuel"]:
                weights.append(5.0)
            else:
                weights.append(2.5)
        elif cat == "INCOME":
            weights.append(3.0)
        else: # TRANSFER
            weights.append(4.0)
    
    weights = np.array(weights) / sum(weights)

    for i in range(n_samples):
        idx = np.random.choice(len(all_keys), p=weights)
        cat, subcat = all_keys[idx]
        template = random.choice(merchants_by_subcategory[(cat, subcat)])
        merchant_template, desc_template, (min_amt, max_amt), t_type, pay_methods = template
        
        # Add realistic noise/variations to merchant text in Indian payment gateway formats
        noise_prefixes = ["", "UPI-", "POS ", "PAYTM*", "RAZORPAY*", "BHARATPE-", "CRED-", "BILLDESK-"]
        noise_suffixes = ["", " BANGALORE", " MUMBAI", " DELHI", " NOIDA", " PUNE", " HYD", " STORE"]
        
        clean_prefix = random.choice(noise_prefixes) if random.random() < 0.4 else ""
        clean_suffix = random.choice(noise_suffixes) if random.random() < 0.4 else ""
        
        merchant_raw = f"{clean_prefix}{merchant_template}{clean_suffix}".strip()
        desc = f"{desc_template} ref:{random.randint(10000, 99999)}" if random.random() < 0.3 else desc_template
        amount = round(random.uniform(min_amt, max_amt), 2)
        
        txn_date = base_date + timedelta(days=random.randint(0, 364), hours=random.randint(0, 23), minutes=random.randint(0, 59))
        payment_method = random.choice(pay_methods)
        
        records.append({
            "transaction_id": f"TXN_C_{i+1:06d}",
            "date": txn_date.strftime("%Y-%m-%d %H:%M:%S"),
            "merchant_raw": merchant_raw,
            "description": desc,
            "amount": amount,
            "transaction_type": t_type,
            "payment_method": payment_method,
            "category": cat,
            "subcategory": subcat
        })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[Dataset 1] Generated {len(df)} categorization records in INR at {output_path}")
    return df


def generate_historical_transactions_dataset(
    output_path: str,
    n_users: int = 40,
    n_months: int = 8
) -> pd.DataFrame:
    """
    Generates Dataset 2: Multi-user longitudinal historical transactions in Indian INR context.
    Maintains temporal sequence. Contains Indian salary schedules, monthly house rent,
    UPI autopay subscriptions, daily Swiggy/Zepto/Uber spends, weekend dining, and contextual anomalies.
    """
    start_date = datetime(2025, 1, 1)
    end_date = start_date + timedelta(days=n_months * 30)
    
    all_transactions = []
    global_txn_counter = 1

    for u_idx in range(1, n_users + 1):
        user_id = f"USR_{u_idx:04d}"
        
        # Indian salary and lifestyle parameters in INR
        salary_day = random.choice([1, 5, 28, 30, 31])
        monthly_salary = round(random.uniform(40000, 180000), 2)
        rent_amount = round(random.uniform(0.22, 0.38) * monthly_salary, 2)
        rent_day = min(salary_day + 2, 28)
        
        # Subscriptions
        has_netflix = random.random() < 0.8
        netflix_day = random.randint(1, 28)
        netflix_amt = 649.0 if random.random() < 0.6 else 199.0
        
        has_spotify = random.random() < 0.75
        spotify_day = random.randint(1, 28)
        spotify_amt = 119.0
        
        has_cult_fit = random.random() < 0.45
        cult_day = random.randint(1, 28)
        cult_amt = round(random.uniform(1499, 2999), 2)
        
        has_broadband = True
        broadband_day = random.randint(5, 20)
        broadband_amt = round(random.choice([799.0, 999.0, 1199.0, 1499.0]), 2)

        # Baseline spending metrics for context (INR)
        user_food_mean = random.uniform(220, 550)
        user_grocery_mean = random.uniform(800, 2500)
        user_shopping_mean = random.uniform(1200, 4500)

        # Generate day by day
        cur_date = start_date
        while cur_date < end_date:
            day_of_month = cur_date.day
            is_weekend = cur_date.weekday() >= 5
            
            # 1. Monthly Salary (NEFT/Direct Deposit)
            if day_of_month == salary_day:
                all_transactions.append({
                    "user_id": user_id,
                    "transaction_id": f"TXN_H_{global_txn_counter:07d}",
                    "date": cur_date.strftime("%Y-%m-%d 09:30:00"),
                    "merchant": "TCS / Infosys Monthly Payroll Salary",
                    "amount": monthly_salary,
                    "category": "INCOME",
                    "subcategory": "Salary",
                    "transaction_type": "credit",
                    "payment_method": "Direct Deposit"
                })
                global_txn_counter += 1
            
            # 2. Monthly House Rent (NoBroker / Bank Wire)
            if day_of_month == rent_day:
                all_transactions.append({
                    "user_id": user_id,
                    "transaction_id": f"TXN_H_{global_txn_counter:07d}",
                    "date": cur_date.strftime("%Y-%m-%d 10:15:00"),
                    "merchant": "NoBroker RentPay / Landlord Transfer",
                    "amount": rent_amount,
                    "category": "EXPENSE",
                    "subcategory": "Rent & Housing",
                    "transaction_type": "debit",
                    "payment_method": "UPI"
                })
                global_txn_counter += 1
            
            # 3. Fixed Subscriptions & Recurring Bills
            if has_netflix and day_of_month == netflix_day:
                all_transactions.append({
                    "user_id": user_id,
                    "transaction_id": f"TXN_H_{global_txn_counter:07d}",
                    "date": cur_date.strftime("%Y-%m-%d 04:00:00"),
                    "merchant": "Netflix India Monthly Plan",
                    "amount": netflix_amt,
                    "category": "EXPENSE",
                    "subcategory": "Subscriptions",
                    "transaction_type": "debit",
                    "payment_method": "UPI AutoPay"
                })
                global_txn_counter += 1
                
            if has_spotify and day_of_month == spotify_day:
                all_transactions.append({
                    "user_id": user_id,
                    "transaction_id": f"TXN_H_{global_txn_counter:07d}",
                    "date": cur_date.strftime("%Y-%m-%d 04:05:00"),
                    "merchant": "Spotify India Premium Music",
                    "amount": spotify_amt,
                    "category": "EXPENSE",
                    "subcategory": "Subscriptions",
                    "transaction_type": "debit",
                    "payment_method": "UPI AutoPay"
                })
                global_txn_counter += 1
                
            if has_cult_fit and day_of_month == cult_day:
                all_transactions.append({
                    "user_id": user_id,
                    "transaction_id": f"TXN_H_{global_txn_counter:07d}",
                    "date": cur_date.strftime("%Y-%m-%d 08:00:00"),
                    "merchant": "Cult.fit Elite Gym Membership",
                    "amount": cult_amt,
                    "category": "EXPENSE",
                    "subcategory": "Personal Care",
                    "transaction_type": "debit",
                    "payment_method": "Credit Card"
                })
                global_txn_counter += 1
                
            if has_broadband and day_of_month == broadband_day:
                all_transactions.append({
                    "user_id": user_id,
                    "transaction_id": f"TXN_H_{global_txn_counter:07d}",
                    "date": cur_date.strftime("%Y-%m-%d 11:30:00"),
                    "merchant": "Airtel / ACT Fibernet Broadband",
                    "amount": broadband_amt,
                    "category": "EXPENSE",
                    "subcategory": "Bills & Utilities",
                    "transaction_type": "debit",
                    "payment_method": "UPI"
                })
                global_txn_counter += 1

            # 4. Daily Food Delivery / Coffee / Dining (Swiggy, Zomato, Chai Point)
            if random.random() < (0.85 if is_weekend else 0.65):
                n_food = random.choice([1, 2, 3] if is_weekend else [1, 2])
                for _ in range(n_food):
                    amt = round(np.random.normal(user_food_mean * (1.3 if is_weekend else 1.0), user_food_mean * 0.25), 2)
                    amt = max(60.0, amt)
                    all_transactions.append({
                        "user_id": user_id,
                        "transaction_id": f"TXN_H_{global_txn_counter:07d}",
                        "date": (cur_date + timedelta(hours=random.randint(8, 22), minutes=random.randint(0, 59))).strftime("%Y-%m-%d %H:%M:%S"),
                        "merchant": random.choice(["Swiggy Food Order", "Zomato Restaurant", "Chai Point", "Starbucks India", "Haldiram's", "Domino's Pizza"]),
                        "amount": amt,
                        "category": "EXPENSE",
                        "subcategory": "Food",
                        "transaction_type": "debit",
                        "payment_method": random.choice(["UPI", "Credit Card", "Debit Card", "Paytm"])
                    })
                    global_txn_counter += 1

            # Groceries (Blinkit, Zepto, DMart, BigBasket)
            if (is_weekend and random.random() < 0.50) or (not is_weekend and random.random() < 0.15):
                amt = round(np.random.normal(user_grocery_mean, user_grocery_mean * 0.2), 2)
                amt = max(150.0, amt)
                all_transactions.append({
                    "user_id": user_id,
                    "transaction_id": f"TXN_H_{global_txn_counter:07d}",
                    "date": (cur_date + timedelta(hours=random.randint(10, 20), minutes=random.randint(0, 59))).strftime("%Y-%m-%d %H:%M:%S"),
                    "merchant": random.choice(["Blinkit Quick Commerce", "Zepto Groceries", "BigBasket Supermarket", "DMart Supercenter"]),
                    "amount": amt,
                    "category": "EXPENSE",
                    "subcategory": "Groceries",
                    "transaction_type": "debit",
                    "payment_method": random.choice(["UPI", "Credit Card", "Debit Card"])
                })
                global_txn_counter += 1

            # Transport / Commute (Uber, Ola, Rapido, Metro)
            if not is_weekend and random.random() < 0.65:
                amt = round(random.choice([45.0, 95.0, 180.0, 320.0, 480.0]) + random.uniform(-10, 25), 2)
                all_transactions.append({
                    "user_id": user_id,
                    "transaction_id": f"TXN_H_{global_txn_counter:07d}",
                    "date": (cur_date + timedelta(hours=random.randint(7, 19), minutes=random.randint(0, 59))).strftime("%Y-%m-%d %H:%M:%S"),
                    "merchant": random.choice(["Uber India Rides", "Ola Cabs Commute", "Rapido Bike Taxi", "Namma Metro / Delhi Metro DMRC"]),
                    "amount": amt,
                    "category": "EXPENSE",
                    "subcategory": "Transport",
                    "transaction_type": "debit",
                    "payment_method": random.choice(["UPI", "Paytm", "Credit Card"])
                })
                global_txn_counter += 1

            # Online Shopping (Amazon India, Flipkart, Myntra)
            if random.random() < (0.28 if is_weekend else 0.08):
                amt = round(np.random.normal(user_shopping_mean, user_shopping_mean * 0.35), 2)
                amt = max(299.0, amt)
                all_transactions.append({
                    "user_id": user_id,
                    "transaction_id": f"TXN_H_{global_txn_counter:07d}",
                    "date": (cur_date + timedelta(hours=random.randint(11, 21), minutes=random.randint(0, 59))).strftime("%Y-%m-%d %H:%M:%S"),
                    "merchant": random.choice(["Amazon India Marketplace", "Flipkart Internet", "Myntra Fashion Store", "Nykaa E-Retail", "Croma Electronics"]),
                    "amount": amt,
                    "category": "EXPENSE",
                    "subcategory": "Shopping",
                    "transaction_type": "debit",
                    "payment_method": random.choice(["UPI", "Credit Card", "Debit Card"])
                })
                global_txn_counter += 1

            # Monthly Mutual Fund SIP / Investment (Zerodha, Groww)
            if day_of_month == min(salary_day + 1, 28):
                save_amt = round(monthly_salary * random.uniform(0.12, 0.30), 2)
                all_transactions.append({
                    "user_id": user_id,
                    "transaction_id": f"TXN_H_{global_txn_counter:07d}",
                    "date": cur_date.strftime("%Y-%m-%d 14:00:00"),
                    "merchant": "Groww / Zerodha Mutual Fund SIP",
                    "amount": save_amt,
                    "category": "TRANSFER",
                    "subcategory": "Investment",
                    "transaction_type": "transfer",
                    "payment_method": "UPI AutoPay"
                })
                global_txn_counter += 1

            # 5. Contextual Outliers / Anomalies (INR)
            # Anomaly 1: Massive luxury nightclub / 5-star hotel banquet (e.g. ₹9,000 - ₹18,000 vs ₹350 food mean)
            if random.random() < 0.004:
                anom_amt = round(user_food_mean * random.uniform(15.0, 30.0), 2)
                all_transactions.append({
                    "user_id": user_id,
                    "transaction_id": f"TXN_H_{global_txn_counter:07d}",
                    "date": (cur_date + timedelta(hours=23, minutes=random.randint(0, 59))).strftime("%Y-%m-%d %H:%M:%S"),
                    "merchant": "Taj Palace 5-Star Luxury Banquet",
                    "amount": anom_amt,
                    "category": "EXPENSE",
                    "subcategory": "Food",
                    "transaction_type": "debit",
                    "payment_method": "Credit Card"
                })
                global_txn_counter += 1

            # Anomaly 2: Emergency Hospital Care
            if random.random() < 0.002:
                all_transactions.append({
                    "user_id": user_id,
                    "transaction_id": f"TXN_H_{global_txn_counter:07d}",
                    "date": (cur_date + timedelta(hours=random.randint(10, 17), minutes=random.randint(0, 59))).strftime("%Y-%m-%d %H:%M:%S"),
                    "merchant": "Apollo Hospitals Emergency Care",
                    "amount": round(random.uniform(25000, 85000), 2),
                    "category": "EXPENSE",
                    "subcategory": "Healthcare",
                    "transaction_type": "debit",
                    "payment_method": "Credit Card"
                })
                global_txn_counter += 1

            cur_date += timedelta(days=1)

    df_hist = pd.DataFrame(all_transactions)
    df_hist["date_dt"] = pd.to_datetime(df_hist["date"])
    df_hist = df_hist.sort_values(by=["date_dt", "user_id"]).drop(columns=["date_dt"]).reset_index(drop=True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_hist.to_csv(output_path, index=False)
    print(f"[Dataset 2] Generated {len(df_hist)} historical records in INR across {n_users} users at {output_path}")
    return df_hist


def generate_user_behavior_survey_dataset(
    output_path: str,
    n_respondents: int = 1500
) -> pd.DataFrame:
    """
    Generates Dataset 3: Financial tracking behavior research dataset for the Indian market.
    Addresses: "Why do people struggle to track where their money goes each month?"
    Statistically models UPI fragmentation (PhonePe, GPay, Paytm, CRED), micro-transactions,
    and manual tracking difficulty across Indian income brackets (LPA - Lakhs Per Annum).
    """
    records = []
    
    # Indian Income Brackets in INR (Lakhs per annum)
    income_brackets = [
        "< ₹3,00,000",
        "₹3,00,000 - ₹6,00,000",
        "₹6,00,000 - ₹12,00,000",
        "₹12,00,000 - ₹25,00,000",
        "> ₹25,00,000"
    ]
    income_weights = [0.20, 0.35, 0.25, 0.14, 0.06]

    for i in range(1, n_respondents + 1):
        user_id = f"RESP_IN_{i:05d}"
        income_range = np.random.choice(income_brackets, p=income_weights)
        
        # Payment fragmentation: number of UPI apps, cards & wallets (GPay, PhonePe, Paytm, CRED, Amazon Pay, 2 Bank Debit/Credit Cards)
        payment_method_count = np.random.choice([1, 2, 3, 4, 5, 6, 7], p=[0.04, 0.12, 0.28, 0.28, 0.16, 0.08, 0.04])
        
        # Monthly digital transaction count (High velocity in India due to QR/UPI ₹10-₹500 micro-transactions)
        base_txn = 35 + (payment_method_count * 15) + np.random.normal(20, 12)
        transaction_count = max(15, int(base_txn))
        
        # Active subscriptions (OTT, music, gym, broadband, software)
        sub_base = 2 + (payment_method_count * 0.9) + np.random.normal(2, 1.5)
        subscription_count = max(0, int(sub_base))
        
        # Expense tracking frequency: 1=Never, 5=Daily
        expense_tracking_frequency = np.random.choice([1, 2, 3, 4, 5], p=[0.22, 0.32, 0.23, 0.16, 0.07])
        
        # Reliance on manual tracking (Excel, WhatsApp notes, mental tally): 1=None, 5=Heavy
        manual_tracking_frequency = np.random.choice([1, 2, 3, 4, 5], p=[0.14, 0.20, 0.26, 0.25, 0.15])
        
        # Budgeting adherence frequency: 1=Never, 5=Strict
        budgeting_frequency = np.random.choice([1, 2, 3, 4, 5], p=[0.24, 0.28, 0.22, 0.18, 0.08])
        
        # Frequency of missed / forgotten transactions (Driven by high volume UPI micro-payments)
        missed_txn_latent = 0.38 * payment_method_count + 0.022 * transaction_count - 0.42 * expense_tracking_frequency + np.random.normal(1.7, 0.75)
        missed_transaction_frequency = int(np.clip(round(missed_txn_latent), 1, 5))
        
        # Impulse spending tendency: 1=Low, 5=High
        impulse_spending_tendency = np.random.choice([1, 2, 3, 4, 5], p=[0.12, 0.24, 0.34, 0.20, 0.10])
        
        # UPI / Digital wallet usage percentage (Typically very high in urban India, 70%-100%)
        digital_wallet_usage_pct = round(np.clip(np.random.normal(82 + payment_method_count * 2.5, 10), 30, 100), 1)

        # Monthly savings rate as % of income
        savings_latent = 28 - (impulse_spending_tendency * 3.5) + (budgeting_frequency * 3.0) + np.random.normal(0, 4)
        savings_behavior_pct = round(np.clip(savings_latent, 0, 65), 1)

        # Perceived Financial Tracking Difficulty (Score 1 to 10)
        difficulty_latent = (
            1.2
            + 0.55 * payment_method_count
            + 0.025 * transaction_count
            + 0.28 * subscription_count
            + 0.65 * missed_transaction_frequency
            + 0.35 * (manual_tracking_frequency if expense_tracking_frequency > 1 else 0.8)
            - 0.50 * budgeting_frequency
            + 0.30 * impulse_spending_tendency
            + np.random.normal(0, 0.65)
        )
        financial_tracking_difficulty = int(np.clip(round(difficulty_latent), 1, 10))

        records.append({
            "user_id": user_id,
            "income_range": income_range,
            "transaction_count": transaction_count,
            "payment_method_count": payment_method_count,
            "subscription_count": subscription_count,
            "expense_tracking_frequency": expense_tracking_frequency,
            "manual_tracking_frequency": manual_tracking_frequency,
            "missed_transaction_frequency": missed_transaction_frequency,
            "budgeting_frequency": budgeting_frequency,
            "impulse_spending_tendency": impulse_spending_tendency,
            "digital_wallet_usage_pct": digital_wallet_usage_pct,
            "savings_behavior_pct": savings_behavior_pct,
            "financial_tracking_difficulty": financial_tracking_difficulty
        })

    df_survey = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_survey.to_csv(output_path, index=False)
    print(f"[Dataset 3] Generated {len(df_survey)} Indian financial behavior survey records at {output_path}")
    return df_survey

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw"))
    generate_transaction_classification_dataset(os.path.join(base_dir, "raw_transactions_classification.csv"), n_samples=6000)
    generate_historical_transactions_dataset(os.path.join(base_dir, "raw_historical_transactions.csv"), n_users=40, n_months=8)
    generate_user_behavior_survey_dataset(os.path.join(base_dir, "raw_user_behavior_survey.csv"), n_respondents=1500)
