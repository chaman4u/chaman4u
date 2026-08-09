import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("week_3_mls/data/tourism.csv")
df.drop(columns=["CustomerID"], inplace=True)

# The 'ProductPitched' column (similar to 'Type' in other datasets)
# is intentionally left as raw strings. The training pipeline one-hot-encodes it,
# and the serving application also expects raw values. Encoding it here
# would make training and serving use different representations, leading to issues.

X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# stratify=y keeps the (imbalanced) target ratio consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print("ProductPitched values kept as:", sorted(X["ProductPitched"].unique()))
