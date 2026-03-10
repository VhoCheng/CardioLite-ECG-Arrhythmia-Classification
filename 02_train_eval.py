# 02_train_eval.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib

# 路径
DATA_PATH = "outputs/tables/beat_features.csv"
OUT_DIR = "outputs"
os.makedirs(f"{OUT_DIR}/models", exist_ok=True)
os.makedirs(f"{OUT_DIR}/tables", exist_ok=True)

# 1️⃣ 读数据
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["label", "record_id", "r_peak_sample"], errors="ignore")
y = df["label"].values
groups = df["record_id"].values
feature_names = X.columns.tolist()
X = X.values

# 2️⃣ 按 record 分组划分（防止数据泄漏）
gss = GroupShuffleSplit(test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# 3️⃣ 定义对比模型（全部 CPU）
models = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=3000,
            class_weight="balanced"
        ))
    ]),
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1
    ),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=42
        ))
    ])
}

# 4️⃣ 训练 + 评估
results = []

for name, model in models.items():
    print(f"\n=== Training {name} ===")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    auc = roc_auc_score(
        pd.get_dummies(y_test),
        y_prob,
        multi_class="ovr"
    )

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Macro_F1": macro_f1,
        "ROC_AUC_OVR": auc
    })

    joblib.dump(
        {"model": model, "features": feature_names},
        f"{OUT_DIR}/models/{name}.joblib"
    )

# 5️⃣ 保存论文表格
results_df = pd.DataFrame(results)
results_df.to_csv(f"{OUT_DIR}/tables/results.csv", index=False)

print("\n✅ Saved: outputs/tables/results.csv")
print(results_df)