import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

DATA = "outputs/tables/beat_features.csv"
MODEL = "outputs/models/LogisticRegression.joblib"
OUT = "outputs/figures"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(DATA)
X = df.drop(columns=["label","record_id","r_peak_sample"], errors="ignore")
y = df["label"].values
features = X.columns

bundle = joblib.load(MODEL)
clf = bundle["model"].named_steps["clf"]

# ---------- Figure 3: Feature Importance ----------
coef = np.abs(clf.coef_).mean(axis=0)
idx = np.argsort(coef)[-10:]

plt.figure(figsize=(6,4))
plt.barh(features[idx], coef[idx])
plt.xlabel("Absolute Coefficient")
plt.title("Top-10 Important Features (Logistic Regression)")
plt.tight_layout()
plt.savefig(f"{OUT}/fig_feature_importance.png", dpi=300)
plt.close()

# ---------- Figure 4: Spectral Patterns ----------
plt.figure(figsize=(7,4))
for lab in sorted(df["label"].unique()):
    row = df[df["label"]==lab].iloc[0]
    bands = [row[f"fft_band_{i}"] for i in range(5)]
    plt.plot(bands, marker="o", label=lab)

plt.xlabel("Frequency Band Index")
plt.ylabel("Energy")
plt.title("Representative ECG Beat Spectral Patterns")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/fig_ecg_examples.png", dpi=300)
plt.close()

print("Saved Figure 3 & 4")