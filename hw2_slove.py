import pandas as pd
import numpy as np

# ---------- Load data ----------
df = pd.read_csv("/Users/fahmimshahriar/Documents/Project/ML_Zoomcamp/car_fuel_efficiency_2026.csv")

print("=" * 60)
print("ML ZOOMCAMP 2026 - HOMEWORK 2 (Regression)")
print("=" * 60)

# Features used in the homework (the 4 candidates listed in Q1)
features = ["engine_displacement", "horsepower", "vehicle_weight", "model_year"]

# ---------- Q1: Column with missing values ----------
print("\n--- Q1: Column with missing values ---")
missing_counts = df[features].isnull().sum()
print(missing_counts)
cols_with_missing = missing_counts[missing_counts > 0].index.tolist()
print(f"Q1 Answer: {cols_with_missing}")
# Answer: horsepower

# ---------- Q2: Median for horsepower ----------
print("\n--- Q2: Median for horsepower ---")
hp_median = df['horsepower'].median()
print(f"Q2 Answer (median horsepower) = {hp_median}")
# Answer: 149 (closest match for the 2026 dataset)

# ---------- Helper functions ----------
def train_linear_regression(X, y, r=0.0):
    """Train ridge regression with regularization r."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    XTX = X.T @ X
    reg = r * np.eye(XTX.shape[0])
    reg[0, 0] = 0  # don't regularize bias term
    XTX = XTX + reg
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv @ X.T @ y
    return w[0], w[1:]

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def prepare_X(df_in, base_df, feature_names):
    """Fill NAs with values from base_df (training mean or 0)."""
    return df_in[feature_names].fillna(0).values

def split_data(df_in, seed=42):
    """60/20/20 split with seed."""
    np.random.seed(seed)
    n = len(df_in)
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_val = int(0.2 * n)
    n_test = int(0.2 * n)
    n_train = n - n_val - n_test
    df_train = df_in.iloc[idx[:n_train]].reset_index(drop=True)
    df_val = df_in.iloc[idx[n_train:n_train+n_val]].reset_index(drop=True)
    df_test = df_in.iloc[idx[n_train+n_val:]].reset_index(drop=True)
    return df_train, df_val, df_test

# ---------- Q3: Filling NAs with 0 vs mean ----------
print("\n--- Q3: Filling NAs (0 vs mean) - RMSE comparison ---")
df_train, df_val, df_test = split_data(df, seed=42)

# Homework uses log-transformed target
y_train_log = np.log1p(df_train['fuel_efficiency_mpg'].values)
y_val_log = np.log1p(df_val['fuel_efficiency_mpg'].values)

# Option A: fill NAs with 0
X_train_0 = df_train[features].fillna(0).values
X_val_0 = df_val[features].fillna(0).values

# Option B: fill NAs with training mean
means = df_train[features].mean()
X_train_mean = df_train[features].fillna(means).values
X_val_mean = df_val[features].fillna(means).values

# Train both models
w0_b, w_0 = train_linear_regression(X_train_0, y_train_log, r=0.0)
w0_m, w_m = train_linear_regression(X_train_mean, y_train_log, r=0.0)

pred_val_0 = w0_b + X_val_0 @ w_0
pred_val_mean = w0_m + X_val_mean @ w_m

rmse_0 = rmse(y_val_log, pred_val_0)
rmse_mean = rmse(y_val_log, pred_val_mean)
print(f"RMSE with 0    = {round(rmse_0, 4)}")
print(f"RMSE with mean = {round(rmse_mean, 4)}")
if abs(rmse_0 - rmse_mean) < 1e-6:
    print("Q3 Answer: Both are equally good")
else:
    print(f"Q3 Answer: {'With 0' if rmse_0 < rmse_mean else 'With mean'}")
# Answer: Both are equally good

# ---------- Q4: Best regularization r ----------
print("\n--- Q4: Best regularization parameter r ---")
r_values = [0, 0.01, 0.1, 1, 5, 10, 100]
results = {}
for r in r_values:
    w0, w = train_linear_regression(X_train_0, y_train_log, r=r)
    pred_val = w0 + X_val_0 @ w
    r_val = rmse(y_val_log, pred_val)
    results[r] = round(r_val, 2)  # round to 2 decimals as per homework
    print(f"  r={r:<6} -> RMSE={round(r_val, 4)} (rounded: {round(r_val, 2)})")

best_r = min(results, key=results.get)
print(f"Q4 Answer: best r = {best_r}")
# Answer: 0

# ---------- Q5: RMSE standard deviation across seeds ----------
print("\n--- Q5: RMSE standard deviation across different random seeds ---")
rmse_scores = []
for seed in range(0, 10):
    df_tr, df_vl, df_te = split_data(df, seed=seed)
    y_tr = np.log1p(df_tr['fuel_efficiency_mpg'].values)
    y_vl = np.log1p(df_vl['fuel_efficiency_mpg'].values)
    X_tr = df_tr[features].fillna(0).values
    X_vl = df_vl[features].fillna(0).values
    w0, w = train_linear_regression(X_tr, y_tr, r=best_r)
    pred = w0 + X_vl @ w
    rmse_scores.append(rmse(y_vl, pred))

rmse_scores = np.array(rmse_scores)
print(f"RMSE scores across 10 seeds: {np.round(rmse_scores, 4)}")
print(f"Mean RMSE = {round(rmse_scores.mean(), 4)}")
print(f"Std  RMSE = {round(rmse_scores.std(), 4)}")
print(f"Q5 Answer: std RMSE = {round(rmse_scores.std(), 3)}")
# Answer: 0.006

# ---------- Q6: Evaluation on test set (train+val combined) ----------
print("\n--- Q6: RMSE on test set using train+val combined with best r ---")
# Combine train + val
df_full_train = pd.concat([df_train, df_val], ignore_index=True)
y_full_train_log = np.log1p(df_full_train['fuel_efficiency_mpg'].values)
y_test_log = np.log1p(df_test['fuel_efficiency_mpg'].values)
X_full_train = df_full_train[features].fillna(0).values
X_test = df_test[features].fillna(0).values

w0_full, w_full = train_linear_regression(X_full_train, y_full_train_log, r=best_r)
pred_test = w0_full + X_test @ w_full
rmse_test = rmse(y_test_log, pred_test)
print(f"Q6 Answer: Test RMSE = {round(rmse_test, 3)}")
# Answer: 0.236

print("\n" + "=" * 60)
print("SUMMARY OF ANSWERS")
print("=" * 60)
print(f"Q1. Column with missing values : {cols_with_missing}")
print(f"Q2. Median horsepower          : {hp_median}")
print(f"Q3. Filling NAs                : {'With 0' if rmse_0 < rmse_mean else ('With mean' if rmse_mean < rmse_0 else 'Both equally good')}")
print(f"Q4. Best regularization r      : {best_r}")
print(f"Q5. RMSE std deviation         : {round(rmse_scores.std(), 3)}")
print(f"Q6. Test RMSE (train+val)      : {round(rmse_test, 3)}")