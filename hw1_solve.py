import pandas as pd
import numpy as np

# Q1: Pandas version
print(f"Q1: Pandas version = {pd.__version__}")

# Load data
df = pd.read_csv("/Users/fahmimshahriar/Documents/Project/ML_Zoomcamp/car_fuel_efficiency_2026.csv")

# Q2: Records count
print(f"Q2: Records count = {len(df)}")

# Q3: Fuel types
print(f"Q3: Fuel types = {df['fuel_type'].nunique()}")

# Q4: Missing values per column
missing_counts = df.isnull().sum()
cols_with_missing = (missing_counts > 0).sum()
print(f"Q4: Columns with missing values = {cols_with_missing}")
print("   Per-column missing:")
print(missing_counts[missing_counts > 0].to_string())

# Q5: Max fuel efficiency of Asia cars
asia_max = df[df['origin'] == 'Asia']['fuel_efficiency_mpg'].max()
print(f"Q5: Max fuel efficiency (Asia) = {asia_max}")

# Q6: Median horsepower change
median_before = df['horsepower'].median()
mode_val = df['horsepower'].mode()[0]
print(f"   Median horsepower before = {median_before}")
print(f"   Mode horsepower = {mode_val}")
df_filled = df.copy()
df_filled['horsepower'] = df_filled['horsepower'].fillna(mode_val)
median_after = df_filled['horsepower'].median()
print(f"   Median horsepower after  = {median_after}")
if median_after > median_before:
    print("Q6: Yes, it increased")
elif median_after < median_before:
    print("Q6: Yes, it decreased")
else:
    print("Q6: No")

# Q7: Linear regression on first 7 Asia cars
asia = df[df['origin'] == 'Asia'][['vehicle_weight', 'model_year']].head(7)
X = asia.to_numpy()
XTX = X.T @ X
XTX_inv = np.linalg.inv(XTX)
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
w = XTX_inv @ X.T @ y
print(f"Q7: Sum of w = {w.sum()}")
print(f"   w = {w}")