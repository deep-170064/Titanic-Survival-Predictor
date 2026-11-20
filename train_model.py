"""Train the Project 04 Titanic Decision Tree and save it for the Streamlit app."""
from pathlib import Path
import joblib
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

FEATURES = ["pclass", "sex", "age", "fare"]
TARGET = "survived"
BASE_DIR = Path(__file__).parent

# Same selected columns and preprocessing used in Project 04.
df = sns.load_dataset("titanic")[["survived", *FEATURES]].dropna().copy()
df["sex"] = (df["sex"] == "female").astype(int)

X = df[FEATURES]
y = df[TARGET]

# Reproduce the Project 04 split and classifier configuration.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

model_path = BASE_DIR / "model.joblib"
joblib.dump(model, model_path)

print(f"Saved trained model to: {model_path}")
print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")
print(f"Training accuracy: {model.score(X_train, y_train):.3f}")
print(f"Test accuracy: {model.score(X_test, y_test):.3f}")
print(f"Feature order: {FEATURES}")
