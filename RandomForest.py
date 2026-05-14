import os

os.chdir("DATA")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error

# ====== 1. Пути к файлам ======
BASE_FILE = "frccsc_processed.csv"

CLUSTER_FILES = {
    "Normalize":        "path_processed/frccsc_processed_normalize.csv",
    "PREP":             "path_processed/frccsc_processed_prep_01.csv",
    "BERT":             "path_processed/frccsc_processed_bert.csv",
    "TF-IDF":           "path_processed/frccsc_processed_tfidf.csv",
    "Word2Vec":         "path_processed/frccsc_processed_w2v.csv",
    "Label":            "path_processed/frccsc_processed_struct.csv",
}

TARGET_COL = "ElapsedRaw"

# ====== 2. Загрузка базового файла ======
df_base = pd.read_csv(BASE_FILE)
print("Базовый файл:", df_base.shape)

# Целевая переменная
y = df_base[TARGET_COL].values

# Столбцы, которые НЕ используем как признаки
drop_cols_common = ["State", "Start", "End", TARGET_COL]

# Признаки для baseline-модели: WorkDir оставляем
feature_cols_base = [c for c in df_base.columns if c not in drop_cols_common]

# Признаки для моделей с кластерами: WorkDir убираем
feature_cols_clust_base = [c for c in feature_cols_base if c != "WorkDir"]

print("Признаков в baseline-модели:", len(feature_cols_base))
print("Список baseline-признаков:", feature_cols_base)

# ====== 3. Общее разбиение train/test (одинаковое для всех моделей) ======
indices = np.arange(len(df_base))
train_idx, test_idx = train_test_split(
    indices, test_size=0.2, random_state=42, shuffle=True
)

def make_splits(df, feature_cols):
    X = df[feature_cols]
    X_train = X.loc[train_idx].reset_index(drop=True)
    X_test  = X.loc[test_idx].reset_index(drop=True)
    y_train = y[train_idx]
    y_test  = y[test_idx]
    return X_train, X_test, y_train, y_test

# ====== 4. Функция для обучения и оценки модели ======
def build_and_evaluate(X_train, X_test, y_train, y_test, desc=""):
    # Отделяем числовые и категориальные признаки
    num_cols = X_train.select_dtypes(exclude=["object", "category"]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    print(f"\n[{desc}]")
    print("  Числовые признаки :", num_cols)
    print("  Категориальные    :", cat_cols)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
        ],
        remainder="drop",
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", model),
    ])

    # Обучение
    print(X_train.columns[X_train.columns.duplicated()])
    pipe.fit(X_train, y_train)

    # Предсказание
    y_pred = pipe.predict(X_test)

    # Метрики

    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print(f"  MAE  = {mae:.2f} сек")
    print(f"  RMSE = {rmse:.2f} сек")
    print(f"  MAPE = {mape:.2f} %")

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "model": pipe,
        "y_pred": y_pred
    }

# ====== 5. Baseline-модель (без PathClusterID) ======
df_base["UID"] = df_base["UID"].astype("category")
df_base["GID"] = df_base["GID"].astype("category")

Xb_train, Xb_test, y_train, y_test = make_splits(df_base, feature_cols_base)
res_base = build_and_evaluate(Xb_train, Xb_test, y_train, y_test,
                              desc="Baseline (без PathClusterID)")

# ====== 6. Модели с PathClusterID ======
results = []
predictions = {}

results.append(("Baseline", res_base["MAE"], res_base["RMSE"], res_base["MAPE"]))
predictions["Baseline"] = res_base["y_pred"]

for name, filepath in CLUSTER_FILES.items():
    print(f"\n=== Модель с кластерами {name} ===")
    df_clust = pd.read_csv(filepath)
    print("  Файл с кластерами:", filepath, df_clust.shape)

    assert len(df_clust) == len(df_base), f"Размер df_clust != df_base для {name}"

    if "JobIDRaw" in df_base.columns and "JobIDRaw" in df_clust.columns:
        same_ids = (df_base["JobIDRaw"].values == df_clust["JobIDRaw"].values).all()
        if not same_ids:
            print("  [WARNING] JobIDRaw не совпадает по порядку между base и", name)

    if "PathClusterID" not in df_clust.columns:
        raise ValueError(f"В файле {filepath} нет столбца PathClusterID")

    df_clust["PathClusterID"] = df_clust["PathClusterID"].astype("category")

    feature_cols_clust = feature_cols_clust_base + ["PathClusterID"]

    Xc_train, Xc_test, _, _ = make_splits(df_clust, feature_cols_clust)

    res = build_and_evaluate(
        Xc_train, Xc_test, y_train, y_test,
        desc=f"With PathClusterID ({name})"
    )

    results.append((name, res["MAE"], res["RMSE"], res["MAPE"]))
    predictions[name] = res["y_pred"]

# ====== 8. Сводное сравнение всех моделей ======
summary = pd.DataFrame(results, columns=["Модель", "MAE", "RMSE", "MAPE"])

print("\n=== Сводное сравнение моделей ===")
print(summary)

import matplotlib.pyplot as plt
import numpy as np

methods_plot = summary["Модель"].tolist()
mae_plot = summary["MAE"].tolist()
rmse_plot = summary["RMSE"].tolist()
mape_plot = summary["MAPE"].tolist()

x = np.arange(len(methods_plot))
colors = ["steelblue"] * 3 + ["green"] + ["steelblue"] * 7


# ---------------- MAE ----------------
plt.figure(figsize=(8,4))
bars = plt.bar(x, mae_plot, color=colors, alpha=0.9, width=0.4)
plt.xticks(x, methods_plot)
plt.ylabel("MAE (сек)")
plt.title("MAE")
plt.grid(axis="y", alpha=0.3)



plt.tight_layout()
plt.show()


# ---------------- RMSE ----------------
plt.figure(figsize=(8,4))
bars = plt.bar(x, rmse_plot, color=colors, alpha=0.9, width=0.4)
plt.xticks(x, methods_plot)
plt.ylabel("RMSE (сек)")
plt.title("RMSE")
plt.grid(axis="y", alpha=0.3)



plt.tight_layout()
plt.show()


# ---------------- MAPE ----------------
plt.figure(figsize=(8,4))
bars = plt.bar(x, mape_plot, color=colors, alpha=0.9, width=0.4)
plt.xticks(x, methods_plot)
plt.ylabel("MAPE (%)")
plt.title("MAPE")
plt.grid(axis="y", alpha=0.3)

for bar, val in zip(bars, mape_plot):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
             f"{val:.2f}%", ha="center", va="bottom")

plt.tight_layout()
plt.show()

# NOISE ROBUSTNESS: CONFIG
NOISE_CLUSTER_DIR = "noise_clusters"
N_NOISE_RUNS = 5

NOISY_CLUSTER_FILES = {
    "Normalize":  [os.path.join(NOISE_CLUSTER_DIR, f"noise_{i}_normalize.csv") for i in range(1, N_NOISE_RUNS + 1)],
    "PREP": [os.path.join(NOISE_CLUSTER_DIR, f"noise_{i}_prep_01.csv") for i in range(1, N_NOISE_RUNS + 1)],
    "BERT":       [os.path.join(NOISE_CLUSTER_DIR, f"noise_{i}_bert.csv") for i in range(1, N_NOISE_RUNS + 1)],
    "TF-IDF":     [os.path.join(NOISE_CLUSTER_DIR, f"noise_{i}_tfidf.csv") for i in range(1, N_NOISE_RUNS + 1)],
    "Word2Vec":   [os.path.join(NOISE_CLUSTER_DIR, f"noise_{i}_w2v.csv") for i in range(1, N_NOISE_RUNS + 1)],
    "Label":   [os.path.join(NOISE_CLUSTER_DIR, f"noise_{i}_struct.csv") for i in range(1, N_NOISE_RUNS + 1)],
}

for model_name, files in NOISY_CLUSTER_FILES.items():
    print(model_name, "->", len(files), "files")

def mean_std_ci95(values):
    vals = np.asarray(values, dtype=float)
    n = len(vals)

    mean_val = np.mean(vals)
    std_val = np.std(vals, ddof=1) if n > 1 else 0.0

    t_crit = 2.776 if n == 5 else 1.96
    half_ci = t_crit * std_val / np.sqrt(n) if n > 1 else 0.0

    return mean_val, std_val, mean_val - half_ci, mean_val + half_ci

# EVALUATION ON 5 NOISY FILES
noise_results_rows = []
noise_predictions_by_run = {}

for run_id in range(1, N_NOISE_RUNS + 1):
    print(f"\n========================================")
    print(f"NOISE RUN {run_id}")
    print(f"========================================")

    preds_this_run = {}

    for model_name, file_list in NOISY_CLUSTER_FILES.items():
        filepath = file_list[run_id - 1]
        print(f"\n--- {model_name} | file: {filepath}")

        df_clust = pd.read_csv(filepath)

        assert len(df_clust) == len(df_base), f"Size mismatch for {model_name}, run {run_id}"
        if "PathClusterID" not in df_clust.columns:
            raise ValueError(f"{filepath}: no PathClusterID column")

        df_clust["PathClusterID"] = df_clust["PathClusterID"].astype("category")

        feature_cols_clust = feature_cols_base + ["PathClusterID"]
        Xc_train, Xc_test, y_train, y_test = make_splits(df_clust, feature_cols_clust)

        res = build_and_evaluate(
            Xc_train, Xc_test, y_train, y_test,
            desc=f"Noise run {run_id} | {model_name}"
        )

        noise_results_rows.append({
            "run_id": run_id,
            "model": model_name,
            "MAE": res["MAE"],
            "RMSE": res["RMSE"],
            "MAPE": res["MAPE"],
        })

        preds_this_run[model_name] = res["y_pred"]

    noise_predictions_by_run[run_id] = preds_this_run

noise_results_df = pd.DataFrame(noise_results_rows)
print(noise_results_df.sort_values(["model", "run_id"]))

# SUMMARY: MEAN + 95% CI
agg_rows = []

for model_name, g in noise_results_df.groupby("model"):
    mae_mean, mae_std, mae_lo, mae_hi = mean_std_ci95(g["MAE"].values)
    rmse_mean, rmse_std, rmse_lo, rmse_hi = mean_std_ci95(g["RMSE"].values)
    mape_mean, mape_std, mape_lo, mape_hi = mean_std_ci95(g["MAPE"].values)

    agg_rows.append({
        "Модель": model_name,

        "MAE_mean": mae_mean,
        "MAE_std": mae_std,
        "MAE_CI95_low": mae_lo,
        "MAE_CI95_high": mae_hi,

        "RMSE_mean": rmse_mean,
        "RMSE_std": rmse_std,
        "RMSE_CI95_low": rmse_lo,
        "RMSE_CI95_high": rmse_hi,

        "MAPE_mean": mape_mean,
        "MAPE_std": mape_std,
        "MAPE_CI95_low": mape_lo,
        "MAPE_CI95_high": mape_hi,
    })

noise_summary = pd.DataFrame(agg_rows)
print(noise_summary)

import matplotlib.pyplot as plt
import numpy as np

def plot_metric_with_ci(df_summary, mean_col, low_col, high_col, title, ylabel, fmt="{:.2f}"):
    models_plot = df_summary["Модель"].tolist()
    x = np.arange(len(models_plot))

    means = df_summary[mean_col].values
    lows = df_summary[low_col].values
    highs = df_summary[high_col].values
    errs = np.vstack([means - lows, highs - means])

    colors = ["steelblue"] * len(models_plot)

    plt.figure(figsize=(8, 4))

    for i in range(len(models_plot)):
        plt.errorbar(
            x[i],
            means[i],
            yerr=[[errs[0, i]], [errs[1, i]]],
            fmt="o",
            color=colors[i],
            capsize=5,
            elinewidth=2
        )

    plt.xticks(x, models_plot)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.3)
    plt.ylim(bottom=0)

    plt.tight_layout()
    plt.show()

# Определение порядка
desired_order_noise_plots = ['Normalize', 'PREP', 'BERT', 'TF-IDF', 'Word2Vec', 'Label']

noise_summary_ordered = noise_summary.set_index('Модель').loc[desired_order_noise_plots].reset_index()

plot_metric_with_ci(noise_summary_ordered, "MAE_mean", "MAE_CI95_low", "MAE_CI95_high", "Noise robustness: MAE", "MAE")
plot_metric_with_ci(noise_summary_ordered, "RMSE_mean", "RMSE_CI95_low", "RMSE_CI95_high", "Noise robustness: RMSE", "RMSE")

