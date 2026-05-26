import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder
)

from sklearn.neighbors import KNeighborsClassifier

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------
df = pd.read_csv("data/insurance.csv")

# ---------------------------------------------------
# CREATE CLASSIFICATION TARGET
# ---------------------------------------------------
median_value = df["charges"].median()

df["insurance_category"] = np.where(
    df["charges"] > median_value,
    1,
    0
)

# ---------------------------------------------------
# DROP ORIGINAL TARGET
# ---------------------------------------------------
df.drop("charges", axis=1, inplace=True)

# ---------------------------------------------------
# ENCODE CATEGORICAL COLUMNS
# ---------------------------------------------------
le = LabelEncoder()

categorical_columns = df.select_dtypes(
    include=['object']
).columns

for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# ---------------------------------------------------
# FEATURES AND TARGET
# ---------------------------------------------------
X = df.drop("insurance_category", axis=1)

y = df["insurance_category"]

# ---------------------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------------------------
# FEATURE SCALING
# ---------------------------------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# ---------------------------------------------------
# CREATE MODEL
# ---------------------------------------------------
model = KNeighborsClassifier(
    n_neighbors=5
)

# ---------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------
model.fit(X_train, y_train)

# ---------------------------------------------------
# CREATE MODELS FOLDER
# ---------------------------------------------------
os.makedirs("models", exist_ok=True)

# ---------------------------------------------------
# SAVE MODEL
# ---------------------------------------------------
pickle.dump(
    model,
    open("models/knn_classifier.pkl", "wb")
)

pickle.dump(
    scaler,
    open("models/scaler.pkl", "wb")
)

print("Model Saved Successfully")