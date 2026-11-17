import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

# fetch dataset 
default_of_credit_card_clients = fetch_ucirepo(id=350) 
  
# data (as pandas dataframes) 
X = default_of_credit_card_clients.data.features 
y = default_of_credit_card_clients.data.targets 
  
column_names = {
    "X1": "LIMIT_BAL",
    "X2": "SEX",
    "X3": "EDUCATION",
    "X4": "MARRIAGE",
    "X5": "AGE",
    "X6": "PAY_0",
    "X7": "PAY_2",
    "X8": "PAY_3",
    "X9": "PAY_4",
    "X10": "PAY_5",
    "X11": "PAY_6",
    "X12": "BILL_AMT1",
    "X13": "BILL_AMT2",
    "X14": "BILL_AMT3",
    "X15": "BILL_AMT4",
    "X16": "BILL_AMT5",
    "X17": "BILL_AMT6",
    "X18": "PAY_AMT1",
    "X19": "PAY_AMT2",
    "X20": "PAY_AMT3",
    "X21": "PAY_AMT4",
    "X22": "PAY_AMT5",
    "X23": "PAY_AMT6"
}

df = X.copy().rename(columns=column_names)

# Combine features with target
df["default_next_month"] = y.iloc[:, 0].to_numpy()

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst five records:")
print(df.head())

# Missing values
print("Missing values by column:")
print(df.isna().sum())

# Duplicate records
print("\nDuplicate rows:", df.duplicated().sum())

# Categorical values
for col in ["SEX", "EDUCATION", "MARRIAGE", "PAY_0"]:
    print(f"\n{col} value counts:")
    print(df[col].value_counts().sort_index())

# Financial variable statistics
financial_cols = [
    "LIMIT_BAL",
    "BILL_AMT1",
    "BILL_AMT2",
    "PAY_AMT1",
    "PAY_AMT2"
]

print("\nFinancial variable statistics:")
print(
    df[financial_cols]
    .describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99])
    .round(2)
)

df_clean = df.copy()

# Combine unknown/undocumented education categories
df_clean["EDUCATION"] = df_clean["EDUCATION"].replace({
    5: 0,
    6: 0
})

# Verify target values
assert set(df_clean["default_next_month"].unique()) == {0, 1}

# Verify credit limits and age
assert (df_clean["LIMIT_BAL"] > 0).all()
assert (df_clean["AGE"] > 0).all()

print("Cleaned dataset shape:", df_clean.shape)

print("\nEducation categories after cleaning:")
print(df_clean["EDUCATION"].value_counts().sort_index())

print("\nRemaining missing values:")
print(df_clean.isna().sum().sum())

TARGET = "default_next_month"

# First split: 70% training, 30% temporary
train_df, temp_df = train_test_split(
    df_clean,
    test_size=0.30,
    random_state=42,
    stratify=df_clean[TARGET]
)

# Second split: divide remaining 30% equally
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df[TARGET]
)

# Reset indices
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Verify split sizes
assert len(train_df) == 21000
assert len(val_df) == 4500
assert len(test_df) == 4500

for name, data in [
    ("Training", train_df),
    ("Validation", val_df),
    ("Testing", test_df)
]:
    print(
        f"{name}: {len(data):,} records | "
        f"Default rate: {data[TARGET].mean():.2%}"
    )

default_by_status = (
    train_df.groupby("PAY_0")[TARGET]
    .agg(["mean", "count"])
    .reset_index()
)

default_by_status.columns = [
    "Repayment Status",
    "Default Rate",
    "Customers"
]

print(default_by_status)

plt.figure(figsize=(10, 5))

plt.bar(
    default_by_status["Repayment Status"].astype(str),
    default_by_status["Default Rate"] * 100
)

plt.title("Default Rate by Most Recent Repayment Status")
plt.xlabel("September Repayment Status")
plt.ylabel("Default Rate (%)")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()

analysis_df = train_df.copy()

analysis_df["LIMIT_GROUP"] = pd.qcut(
    analysis_df["LIMIT_BAL"],
    q=5,
    duplicates="drop"
)

default_by_limit = (
    analysis_df.groupby(
        "LIMIT_GROUP",
        observed=True
    )[TARGET]
    .agg(["mean", "count"])
)

print(default_by_limit)

plt.figure(figsize=(10, 5))

plt.bar(
    range(len(default_by_limit)),
    default_by_limit["mean"] * 100
)

plt.xticks(
    range(len(default_by_limit)),
    [str(x) for x in default_by_limit.index],
    rotation=30,
    ha="right"
)

plt.title("Default Rate by Credit Limit Group")
plt.xlabel("Credit Limit Range (NTD)")
plt.ylabel("Default Rate (%)")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()

output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

train_df.to_csv(output_dir / "train.csv", index=False)
val_df.to_csv(output_dir / "validation.csv", index=False)
test_df.to_csv(output_dir / "test.csv", index=False)

print("All datasets saved successfully.")