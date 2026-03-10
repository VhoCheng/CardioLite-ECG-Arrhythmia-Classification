# 03_plot_all.py
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize

# 路径
DATA_PATH = "outputs/tables/beat_features.csv"
MODEL_PATH = "outputs/models/LogisticRegression.joblib"
FIG_DIR = "outputs/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# 读数据
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["label", "record_id", "r_peak_sample"], errors="ignore").values
y = df["label"].values
labels = sorted(np.unique(y))

# 载入模型
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]

# ========= Fig 1: ROC 曲线 =========
y_bin = label_binarize(y, classes=labels)
y_prob = model.predict_proba(X)

plt.figure(figsize=(6, 5))
for i, lab in enumerate(labels):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{lab} (AUC={roc_auc:.2f})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for ECG Arrhythmia Classification")
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_roc.png", dpi=300)
plt.close()

# ========= Fig 2: Confusion Matrix =========
plt.figure(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(
    y, model.predict(X),
    normalize="true",
    xticks_rotation=45
)
plt.title("Normalized Confusion Matrix")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_confusion_matrix.png", dpi=300)
plt.close()

# ========= Fig 3: Feature Importance (LR) =========
coef = np.abs(model.named_steps["clf"].coef_).mean(axis=0)
feature_names = df.drop(
    columns=["label", "record_id", "r_peak_sample"], errors="ignore"
).columns

idx = np.argsort(coef)[-10:]
plt.figure(figsize=(6, 4))
plt.barh(feature_names[idx], coef[idx])
plt.xlabel("Absolute Coefficient")
plt.title("Top-10 Important Features (Logistic Regression)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_feature_importance.png", dpi=300)
plt.close()

# ========= Fig 4: ECG 心搏示例 =========
# 随机选每类一个 beat
plt.figure(figsize=(8, 5))
for lab in labels:
    beat = df[df["label"] == lab].iloc[0]
    # 这里用 FFT 能量近似示意，不再回原始波形（够写论文）
    plt.plot(
        np.linspace(0, 1, 5),
        [beat[f"fft_band_{i}"] for i in range(5)],
        marker="o",
        label=lab
    )

plt.xlabel("Frequency Band Index")
plt.ylabel("Energy")
plt.title("Representative ECG Beat Spectral Patterns")
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_ecg_examples.png", dpi=300)
plt.close()

print("✅ All figures saved to outputs/figures/")