import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "model.joblib"
FEATURES = ["pclass", "sex", "age", "fare"]

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

st.title("🚢 Titanic Survival Predictor")
st.write(
    "Explore how passenger class, sex, age and fare relate to the model's "
    "historical Titanic survival predictions."
)

@st.cache_resource
def load_model():
    """Load the trained Scikit-Learn model once per app session."""
    return joblib.load(MODEL_PATH)

st.subheader("Passenger details")
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Ticket class", options=[1, 2, 3], index=2,
                          help="1 = first class, 2 = second class, 3 = third class")
    sex_label = st.radio("Sex", options=["Female", "Male"], horizontal=True)

with col2:
    age = st.slider("Age", min_value=1, max_value=80, value=30, step=1)
    fare = st.slider("Fare paid (£)", min_value=0.0, max_value=512.0,
                     value=30.0, step=1.0)

st.caption("The model expects features in this exact order: pclass, sex, age, fare.")

if st.button("Predict survival probability", type="primary", use_container_width=True):
    try:
        model = load_model()
        sex = 1 if sex_label == "Female" else 0
        row = pd.DataFrame([[pclass, sex, age, fare]], columns=FEATURES)

        if row.isna().any().any():
            st.error("Please check the inputs. All passenger details are required.")
        else:
            probabilities = model.predict_proba(row)[0]
            classes = list(model.classes_)
            survived_probability = float(probabilities[classes.index(1)])
            predicted_class = int(model.predict(row)[0])

            st.metric("Estimated chance of survival", f"{survived_probability:.1%}")
            if predicted_class == 1:
                st.success("Model prediction: Survived")
            else:
                st.warning("Model prediction: Did not survive")

            st.progress(survived_probability)
            st.caption(
                "This probability is the output of an educational Decision Tree "
                "trained on the Seaborn Titanic dataset. It is not historical certainty, "
                "professional advice, or a real-world risk assessment."
            )
    except FileNotFoundError:
        st.error(
            "The model file (model.joblib) is missing. Run `python train_model.py` "
            "in this project folder to train and save it, then restart the app."
        )
    except Exception as exc:
        st.error(f"Prediction could not be completed. Please check the model and inputs. Details: {exc}")

st.divider()
st.caption(
    "InternCircle Project 05 | Streamlit + Scikit-Learn | Educational demonstration only. "
    "No personal data is collected or stored."
)
