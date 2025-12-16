import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay

st.set_page_config(layout="wide")
st.title("🧠 Interactive Model Interpretability Playground")

# =========================
# 1. DATA UPLOAD
# =========================
st.sidebar.header("1️⃣ Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV file to begin (e.g., Iris, Titanic, or your own dataset).")
    st.stop()

df = pd.read_csv(uploaded_file)
st.write("### Dataset Preview")
st.dataframe(df.head())

# =========================
# 2. TARGET SELECTION
# =========================
st.sidebar.header("2️⃣ Select Target")
target = st.sidebar.selectbox("Target column", df.columns)

X = df.drop(columns=[target])
y = df[target]

# Encode categorical target
if y.dtype == "object":
    y = LabelEncoder().fit_transform(y)
    problem_type = "classification"
else:
    problem_type = "regression"

# Encode categorical features
X_encoded = pd.get_dummies(X, drop_first=True)

# =========================
# 3. MODEL TRAINING
# =========================
st.sidebar.header("3️⃣ Train Model")

test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=test_size, random_state=42
)

if problem_type == "classification":
    model = RandomForestClassifier(n_estimators=100, random_state=42)
else:
    model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

st.success("Model trained successfully!")

# =========================
# 4. GLOBAL FEATURE IMPORTANCE
# =========================
st.header("📊 Global Feature Importance")

importances = model.feature_importances_
feat_df = pd.DataFrame({
    "feature": X_encoded.columns,
    "importance": importances
}).sort_values("importance", ascending=False)

st.bar_chart(feat_df.set_index("feature"))

# =========================
# 5. SHAP EXPLANATIONS
# =========================
st.header("🔍 SHAP Explanations")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

st.subheader("SHAP Summary Plot")
fig, ax = plt.subplots()
if problem_type == "classification":
    shap.summary_plot(shap_values[0], X_test, show=False)
else:
    shap.summary_plot(shap_values, X_test, show=False)
st.pyplot(fig)

# =========================
# 6. LOCAL EXPLANATION
# =========================
st.subheader("🧩 Local Explanation (Single Instance)")

idx = st.slider("Select test instance", 0, len(X_test)-1, 0)
instance = X_test.iloc[[idx]]

fig2, ax2 = plt.subplots()
if problem_type == "classification":
    shap.force_plot(
        explainer.expected_value[0],
        shap_values[0][idx],
        instance,
        matplotlib=True,
        show=False
    )
else:
    shap.force_plot(
        explainer.expected_value,
        shap_values[idx],
        instance,
        matplotlib=True,
        show=False
    )
st.pyplot(fig2)

# =========================
# 7. PDP & ICE
# =========================
st.header("📈 PDP & ICE Plots")

feature = st.selectbox("Select feature", X_encoded.columns)

fig3, ax3 = plt.subplots()
PartialDependenceDisplay.from_estimator(
    model,
    X_train,
    [feature],
    kind="both",  # PDP + ICE
    ax=ax3
)
st.pyplot(fig3)

st.success("Interpretability analysis complete!")
