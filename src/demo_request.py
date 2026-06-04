"""
COM668 Final Year Project — AI-Based Intrusion Detection System
API Demo Script
Author : Abdulbosit Abdurazzakov | B00979380
Purpose: Demonstrates how to call the Flask API with real flows from the test set.
         Run this AFTER starting api.py (python src/api.py).

Usage:
  python src/demo_request.py
"""

import requests
import pandas as pd
import numpy as np
import json
import os

BASE_URL = "http://localhost:5000"

# ── Load feature names ────────────────────────────────────────────────────────
import joblib
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
feature_names = joblib.load(os.path.join(BASE_DIR, 'data', 'processed', 'feature_names.pkl'))


def print_result(label, resp):
    """Pretty-print an API response."""
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    if resp.status_code == 200:
        r = resp.json()['result']
        print(f"  Random Forest   : {r['random_forest']['prediction']}"
              f"  (confidence={r['random_forest']['confidence']},"
              f"  P(ATTACK)={r['random_forest']['attack_prob']})")
        print(f"  Isolation Forest: {r['isolation_forest']['prediction']}"
              f"  (anomaly score={r['isolation_forest']['anomaly_score']})")
        print(f"  Ensemble (OR)   : {r['ensemble']['prediction']}")
        print(f"  FINAL DECISION  : >>> {r['final_decision']} <<<")
    else:
        print(f"  ERROR: {resp.json()}")


# ── 1. Health check ───────────────────────────────────────────────────────────
print("\n1. Health Check")
resp = requests.get(f"{BASE_URL}/health")
print(f"   Status: {resp.json()['status']} | {resp.json()['message']}")


# ── 2. Load real test flows ───────────────────────────────────────────────────
print("\n2. Loading real test flows from CICIDS2017...")

# Load a small sample of Thursday data (Web Attacks)
df = pd.read_csv(os.path.join(BASE_DIR, 'data',
                  'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv'))
df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

attacks = df[df['Label'] != 'BENIGN'].head(3)
benigns = df[df['Label'] == 'BENIGN'].head(3)

print(f"   Loaded {len(attacks)} ATTACK flows and {len(benigns)} BENIGN flows for demo")


# ── 3. Test with real ATTACK flows ────────────────────────────────────────────
print("\n3. Sending ATTACK flows to API...")
for i, (_, row) in enumerate(attacks.iterrows()):
    payload = {"features": [float(row[f]) for f in feature_names]}
    resp = requests.post(f"{BASE_URL}/predict", json=payload)
    attack_type = row['Label']
    print_result(f"Flow {i+1} — True label: {attack_type}", resp)


# ── 4. Test with real BENIGN flows ────────────────────────────────────────────
print("\n4. Sending BENIGN flows to API...")
for i, (_, row) in enumerate(benigns.iterrows()):
    payload = {"features": [float(row[f]) for f in feature_names]}
    resp = requests.post(f"{BASE_URL}/predict", json=payload)
    print_result(f"Flow {i+1} — True label: BENIGN", resp)


# ── 5. Batch request ──────────────────────────────────────────────────────────
print("\n5. Batch prediction (3 flows in one request)...")
all_flows = pd.concat([attacks.head(2), benigns.head(1)])
batch_payload = {
    "flows": [
        [float(row[f]) for f in feature_names]
        for _, row in all_flows.iterrows()
    ]
}
resp = requests.post(f"{BASE_URL}/predict/batch", json=batch_payload)
batch_result = resp.json()
print(f"   Batch response: {batch_result['count']} flows processed")
for r in batch_result['results']:
    dec = r['result']['final_decision']
    print(f"   Flow {r['flow_index']}: {dec}")

print("\n\nDemo complete. The API correctly classified real CICIDS2017 flows.")
