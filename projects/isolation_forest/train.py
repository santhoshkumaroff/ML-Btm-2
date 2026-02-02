
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("creditcard.csv")

features = [c for c in df.columns if c != "Class"]
X = df[features]
y = df["Class"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

iso = IsolationForest(n_estimators=200, contamination=0.002, random_state=42)
df["anomaly"] = iso.fit_predict(X_scaled)
df["anomaly"] = df["anomaly"].map({1: 0, -1: 1})

X_final = df[features + ["anomaly"]]

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.25, stratify=y, random_state=42
)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

with open("fraud_pipeline.pkl", "wb") as f:
    pickle.dump({
        "scaler": scaler,
        "iso": iso,
        "clf": clf,
        "features": features
    }, f)

print("Fraud model trained & saved")
