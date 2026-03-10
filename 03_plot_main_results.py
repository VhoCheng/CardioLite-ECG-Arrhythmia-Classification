import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize

DATA = "outputs/tables/beat_features.csv"
MODEL = "outputs/models/LogisticRegression.joblib"
OUT = "outputs/figures"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(DATA)
X = df.drop(columns=["label","record_id","r_peak_sample"], errors="ignore").values
y = df["label"].values
labels = sorted(np.unique(y))

bundle = joblib.load(MODEL)
model = bundle["model"]

# ---------- Figure 1: ROC ----------
y_bin = label_binarize(y, classes=labels)
y_prob = model.predict_proba(X)

plt.figure(figsize=(6,5))
for i,l in enumerate(labels):
    fpr, tpr, _ = roc_curve(y_bin[:,i], y_prob[:,i])
    plt.plot(fpr, tpr, label=f"{l} (AUC={auc(fpr,tpr):.2f})")

plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for ECG Arrhythmia Classification")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/fig_roc.png", dpi=300)
plt.close()

# ---------- Figure 2: Confusion Matrix ----------
plt.figure(figsize=(5,4))
ConfusionMatrixDisplay.from_predictions(
    y, model.predict(X),
    normalize="true",
    xticks_rotation=45
)
plt.title("Normalized Confusion Matrix")
plt.tight_layout()
plt.savefig(f"{OUT}/fig_confusion_matrix.png", dpi=300)
plt.close()

print("Saved Figure 1 & 2")