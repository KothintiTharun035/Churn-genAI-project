import json
import plotly.express as px
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# 1. Project paths and configuration
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = BASE_DIR / "artifacts" / "random_forest.joblib"
PREPROCESSOR_PATH = BASE_DIR / "artifacts" / "preprocessor.joblib"

st.set_page_config(
    page_title="ChurnIQ | Customer Analytics",
    page_icon="📊",
    layout="wide",
)

# --------------------------------------------------
# 2. Load dataset and trained artifacts
# --------------------------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"], errors="coerce"
    )
    return data

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor

try:
    df = load_data()
    model, preprocessor = load_artifacts()
except Exception as exc:
    st.error(f"Unable to load project files: {exc}")
    st.stop()

# --------------------------------------------------
# 3. Dashboard header
# --------------------------------------------------
st.title("📊 ChurnIQ")
st.subheader("Customer Churn Prediction & Explainable GenAI")
st.caption(
    "Machine learning • Explainable AI • Customer retention analytics"
)

# --------------------------------------------------
# 4. Customer selector
# --------------------------------------------------
st.sidebar.header("Customer Selection")

customer_ids = df["customerID"].astype(str).tolist()
selected_id = st.sidebar.selectbox(
    "Select a customer",
    customer_ids,
)

customer_row = df.loc[
    df["customerID"].astype(str) == selected_id
].iloc[[0]]

# Keep the original customer features for prediction.
X_customer = customer_row.drop(
    columns=["customerID", "Churn"],
    errors="ignore",
)


# --------------------------------------------------
# 5. Predict churn probability
# --------------------------------------------------
try:
    X_transformed = preprocessor.transform(X_customer)
    churn_probability = float(
        model.predict_proba(X_transformed)[0, 1]
    )
except Exception as exc:
    st.error(f"Prediction failed: {exc}")
    st.stop()

# --------------------------------------------------
# 5A. Apply classification threshold
# --------------------------------------------------
CHURN_THRESHOLD = 0.40

predicted_churn = int(
    churn_probability >= CHURN_THRESHOLD
)

if predicted_churn:
    st.warning("Predicted churn: Yes")
else:
    st.success("Predicted churn: No")

st.caption(
    f"Decision threshold: {CHURN_THRESHOLD:.0%}. "
    "Churn probability and the final classification are different."
)



   # --------------------------------------------------
# 6. Customer overview
# --------------------------------------------------
st.markdown("### Customer Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Customer ID", selected_id)
col2.metric("Tenure", f"{int(customer_row['tenure'].iloc[0])} months")
col3.metric(
    "Monthly Charges",
    f"${float(customer_row['MonthlyCharges'].iloc[0]):,.2f}",
)
col4.metric(
    "Churn Probability",
    f"{churn_probability:.2%}",
)

if churn_probability >= 0.60:
    st.warning("Higher predicted churn risk")
elif churn_probability >= 0.30:
    st.info("Moderate predicted churn risk")
else:
    st.success("Lower predicted churn risk")

# --------------------------------------------------
# 7. Customer details
# --------------------------------------------------
st.markdown("### Customer Profile")

profile_columns = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "Contract",
    "InternetService",
    "OnlineSecurity",
    "TechSupport",
    "PaymentMethod",
    "TotalCharges",
]

available_columns = [
    column for column in profile_columns
    if column in customer_row.columns
]

st.dataframe(
    customer_row[available_columns].reset_index(drop=True),
    use_container_width=True,
    hide_index=True,
)

# --------------------------------------------------
# 8. Model information
# --------------------------------------------------
st.markdown("### Model Information")
st.write(
    "Prediction generated using the trained Random Forest classifier "
    "and the saved preprocessing pipeline."
)
st.caption(
    "Risk bands are illustrative dashboard thresholds, not calibrated "
    "business decisions. Validate them before operational use."
)




# --------------------------------------------------
# 9. Explainable AI + Gemini retention advice
# --------------------------------------------------
import os
import numpy as np
import shap
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

st.markdown("---")
st.header("🤖 Explainable AI & Retention Advisor")
st.write(
    "Understand the prediction, explore its main drivers, "
    "and generate a policy-grounded retention recommendation."
)

# Load API key from the project's .env file
load_dotenv(BASE_DIR / ".env")
api_key = os.getenv("GEMINI_API_KEY")

# Load policy files
@st.cache_data
def load_policies():
    policy_dir = BASE_DIR / "policies"
    policies = []

    if policy_dir.exists():
        for path in sorted(policy_dir.glob("*.txt")):
            text = path.read_text(encoding="utf-8").strip()
            if text:
                policies.append({
                    "source": path.name,
                    "text": text
                })

    return policies


# Load embedding model once and reuse it
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# Explain this selected customer's prediction
if st.button("🔍 Explain Prediction & Generate AI Advice"):
    try:
        with st.spinner("Calculating SHAP explanations..."):
            # Dense input is convenient for TreeExplainer
            dense_customer = (
                X_transformed.toarray()
                if hasattr(X_transformed, "toarray")
                else np.asarray(X_transformed)
            )

            explainer = shap.TreeExplainer(model)
            explanation = explainer(dense_customer)

            if len(explanation.values.shape) == 3:
                explanation = explanation[:, :, 1]

            shap_values = np.asarray(explanation.values[0]).reshape(-1)
            feature_names = preprocessor.get_feature_names_out()
            feature_values = dense_customer[0]

            # Create a readable list of the strongest drivers.
            # Skip inactive one-hot categories in the business summary.
            drivers = []

            for i, name in enumerate(feature_names):
                name = str(name)
                value = float(feature_values[i])
                contribution = float(shap_values[i])

                if name.startswith("cat__") and value == 0:
                    continue

                drivers.append({
                    "feature": name,
                    "value": value,
                    "contribution": contribution,
                    "direction": (
                        "decreases model output"
                        if contribution < 0
                        else "increases model output"
                    )
                })

            drivers.sort(
                key=lambda item: abs(item["contribution"]),
                reverse=True
            )
            drivers = drivers[:8]

        st.subheader("SHAP Explanation")
        st.caption(
            "SHAP values explain model behavior, not causation. "
            "Contributions are not necessarily percentage-point changes."
        )

        
        if drivers:
            driver_df = pd.DataFrame(drivers)

            # Convert encoded feature names into readable labels
            def readable_feature(name):
                name = name.replace("num__", "").replace("cat__", "")

                replacements = {
                    "Contract_Two year": "Two-year contract",
                    "Contract_One year": "One-year contract",
                    "Contract_Month-to-month": "Month-to-month contract",
                    "InternetService_Fiber optic": "Fiber-optic internet",
                    "InternetService_DSL": "DSL internet",
                    "InternetService_No": "No internet service",
                    "OnlineSecurity_Yes": "Online security",
                    "OnlineSecurity_No": "No online security",
                    "TechSupport_Yes": "Tech support",
                    "TechSupport_No": "No tech support",
                    "OnlineBackup_Yes": "Online backup",
                    "OnlineBackup_No": "No online backup",
                    "PaymentMethod_Electronic check": "Electronic check",
                    "PaymentMethod_Credit card (automatic)": "Automatic credit card",
                    "PaymentMethod_Bank transfer (automatic)": "Automatic bank transfer",
                    "PaymentMethod_Mailed check": "Mailed check",
                    "MonthlyCharges": "Monthly charges",
                    "TotalCharges": "Total charges",
                    "tenure": "Customer tenure",
                }

                return replacements.get(name, name.replace("_", " "))

            driver_df["Feature"] = driver_df["feature"].apply(readable_feature)
            driver_df["Impact"] = driver_df["contribution"].apply(
                lambda value: (
                    "Decreases model output"
                    if value < 0
                    else "Increases model output"
                )
            )

            driver_df = driver_df.sort_values(
                "contribution", key=lambda s: s.abs(), ascending=True
            )

            st.subheader("SHAP Feature Contributions")

            fig = px.bar(
                driver_df,
                x="contribution",
                y="Feature",
                orientation="h",
                color="Impact",
                color_discrete_map={
                    "Increases model output": "#E76F51",
                    "Decreases model output": "#2A9D8F",
                },
                custom_data=["value", "Impact"],
                labels={
                    "contribution": "SHAP contribution",
                    "Feature": "Customer feature",
                    "Impact": "Direction",
                },
                title="What influenced this churn prediction?",
            )

            fig.add_vline(
                x=0,
                line_width=1,
                line_color="gray",
            )

            fig.update_layout(
                height=max(350, 48 * len(driver_df)),
                xaxis_title="SHAP contribution (model output scale)",
                yaxis_title="",
                legend_title="Contribution direction",
                margin=dict(l=20, r=20, t=70, b=40),
            )

            fig.update_traces(
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "SHAP contribution: %{x:.4f}<br>"
                    "Encoded feature value: %{customdata[0]:.4f}<br>"
                    "Direction: %{customdata[1]}"
                    "<extra></extra>"
                )
            )

            st.plotly_chart(fig, use_container_width=True)

            # Readable feature table below the chart
            display_df = driver_df[
                ["Feature", "value", "contribution", "direction"]
            ].rename(
                columns={
                    "value": "Encoded value",
                    "contribution": "SHAP contribution",
                    "direction": "Direction",
                }
            )

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
            )


        # Retrieve relevant policy documents
        policies = load_policies()
        retrieved_policies = []

        if policies:
            with st.spinner("Retrieving relevant policies..."):
                embedding_model = load_embedding_model()

                policy_embeddings = embedding_model.encode(
                    [p["text"] for p in policies],
                    normalize_embeddings=True,
                    convert_to_numpy=True
                )

                query = (
                    "Customer retention recommendations for a customer "
                    f"with predicted churn probability {churn_probability:.2%}. "
                    "Relevant plans, contracts, service quality and approved offers. "
                    + " ".join(d["feature"] for d in drivers)
                )

                query_embedding = embedding_model.encode(
                    [query],
                    normalize_embeddings=True,
                    convert_to_numpy=True
                )[0]

                similarities = policy_embeddings @ query_embedding
                best_indices = np.argsort(similarities)[::-1][:3]

                retrieved_policies = [
                    {
                        **policies[i],
                        "similarity": float(similarities[i])
                    }
                    for i in best_indices
                ]

            with st.expander("📚 Retrieved policy evidence"):
                for policy in retrieved_policies:
                    st.markdown(f"**Source: {policy['source']}**")
                    st.write(policy["text"])

        else:
            st.warning("No policy text files were found in the policies folder.")

        # Generate Gemini recommendation
        if not api_key:
            st.error(
                "GEMINI_API_KEY was not found. Check that your project "
                ".env file contains GEMINI_API_KEY=your_key."
            )

        else:
            with st.spinner("Gemini is preparing the retention report..."):
                client = genai.Client(api_key=api_key)

                customer_details = {
                    "customer_id": selected_id,
                    "tenure_months": int(customer_row["tenure"].iloc[0]),
                    "contract": str(customer_row["Contract"].iloc[0]),
                    "monthly_charges": float(
                        customer_row["MonthlyCharges"].iloc[0]
                    ),
                    "total_charges": float(
                        customer_row["TotalCharges"].iloc[0]
                    ) if pd.notna(customer_row["TotalCharges"].iloc[0]) else None,
                    "internet_service": str(
                        customer_row["InternetService"].iloc[0]
                    ),
                    "online_security": str(
                        customer_row["OnlineSecurity"].iloc[0]
                    ),
                    "tech_support": str(
                        customer_row["TechSupport"].iloc[0]
                    ),
                    "payment_method": str(
                        customer_row["PaymentMethod"].iloc[0]
                    )
                }

                prompt = f"""
You are a customer retention analyst.

Write a concise report using ONLY the verified information below.

CUSTOMER DETAILS:
{json.dumps(customer_details, indent=2)}

ACTUAL RANDOM FOREST CHURN PROBABILITY:
{churn_probability:.6f} (as a percentage: {churn_probability:.2%})

VERIFIED SHAP DRIVERS:
{json.dumps(drivers, indent=2)}

RETRIEVED COMPANY POLICIES:
{json.dumps(retrieved_policies, indent=2)}

Rules:
- Never invent customer details, probabilities, or SHAP contributions.
- Positive contributions increase the explained model output;
  negative contributions decrease it.
- SHAP contributions are not causal effects.
- Do not describe inactive categories as customer characteristics.
- Do not promise offers or discounts absent from the policy evidence.
- Do not assume the customer has a problem merely because a service
  is absent.
- Clearly distinguish model findings from recommendations.

Use these headings:
1. Risk Assessment
2. Main Model Drivers
3. Recommended Actions
4. Supporting Policies
5. Limitations
"""

                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )

                if response.text:
                    st.subheader("✨ AI-Generated Retention Report")
                    st.markdown(response.text)
                else:
                    st.warning("Gemini returned an empty response.")

    except Exception as exc:
        st.error(f"Unable to generate the explanation or AI advice: {exc}")
        st.info(
            "Check the error message, your installed packages, "
            "policy files, and Gemini API availability."
        )
