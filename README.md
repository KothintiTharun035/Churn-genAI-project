# Customer Churn Prediction + Explainable GenAI

An end-to-end machine learning application that predicts customer churn, explains predictions using Explainable AI (SHAP), and generates AI-powered customer retention recommendations using Retrieval-Augmented Generation (RAG) and Google Gemini.

**Live Demo:** https://churn-genai-project-d32267hgmx7xd3vsvrad9i.streamlit.app/

## Project Overview

Customer churn is a major challenge for subscription-based businesses. Identifying customers who are likely to leave helps businesses take proactive steps to improve customer retention.

This project combines machine learning, explainable AI, semantic policy retrieval, and generative AI to provide understandable churn predictions and actionable retention suggestions through an interactive Streamlit dashboard.

## Key Features

- **Churn Prediction:** Predicts the probability that a customer will leave using trained machine learning models.
- **Model Comparison:** Evaluates Logistic Regression, Random Forest, and Gradient Boosting.
- **Explainable AI:** Uses SHAP to identify factors that contribute positively or negatively to an individual prediction.
- **Interactive Dashboard:** Displays customer information, churn probability, prediction status, and explanation charts.
- **Policy Retrieval (RAG):** Retrieves relevant information from retention offers, service quality guidelines, and contract guidelines.
- **GenAI Recommendations:** Uses Google Gemini to generate customer-specific explanations and retention recommendations grounded in retrieved policy information.
- **Threshold-Based Classification:** Supports a configurable churn classification threshold.
- **Deployed Application:** Accessible through Streamlit Community Cloud.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Analysis | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Explainable AI | SHAP |
| Generative AI | Google Gemini API |
| Semantic Retrieval | Sentence Transformers |
| Visualization | Plotly |
| Web Application | Streamlit |
| Model Persistence | Joblib |
| Environment Configuration | python-dotenv |
| Deployment | Streamlit Community Cloud |

## Machine Learning Models

The project compares three classification algorithms:

1. **Logistic Regression** — a baseline classification model.
2. **Random Forest** — an ensemble model used for the interactive prediction dashboard.
3. **Gradient Boosting** — an ensemble model evaluated for predictive performance.

Models are evaluated using accuracy, precision, recall, F1-score, and ROC-AUC.

The Random Forest model is used in the deployed dashboard. A configurable classification threshold determines the predicted churn class, while the probability score provides a separate estimate of churn likelihood.

## Explainable AI with SHAP

SHAP (SHapley Additive exPlanations) helps explain why a model produces a prediction.

For an individual customer, the application:
- Calculates feature contributions to the model prediction.
- Identifies factors that increase or decrease predicted churn probability.
- Displays a visualization of the most relevant contributing features.
- Helps users interpret model outputs rather than relying on a probability score alone.

SHAP explanations describe model behavior; they do not establish that a factor causes customer churn.

## Retrieval-Augmented Generation (RAG)

The application retrieves relevant text from sample policy documents before generating retention recommendations.

### Retrieval workflow

1. Load the policy documents.
2. Encode policy text and the customer-related query using a Sentence Transformer model.
3. Rank policy text by semantic similarity.
4. Select the most relevant policy passages.
5. Provide the retrieved evidence, customer information, churn probability, and SHAP drivers to Google Gemini.
6. Generate a customer churn explanation and potential retention actions.

The policy documents are illustrative examples for demonstrating the RAG workflow, not official policies from a real company.

## Dataset

The project uses the Telco Customer Churn dataset, commonly distributed as `WA_Fn-UseC_-Telco-Customer-Churn.csv`.

- **Records:** 7,043 customers
- **Original features:** 21 columns, including the churn target
- **Target variable:** `Churn`
- **Prediction task:** Binary classification (`Yes` or `No`)

The dataset includes customer demographics, account information, subscribed services, contract types, tenure, and billing details.

## Project Structure

```text
churn-genai-project/
│
├── app.py
├── customer_churn_explainable_genai.ipynb
├── requirements.txt
├── .gitignore
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── artifacts/
│   ├── random_forest.joblib
│   └── preprocessor.joblib
│
└── policies/
    ├── retention_offers.txt
    ├── service_quality.txt
    └── contract_guidelines.txt
```

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/churn-genai-project.git
cd churn-genai-project
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure the Gemini API key

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
```

Obtain an API key from [Google AI Studio](https://aistudio.google.com/).

Never commit `.env` or expose your API key in source code.

### 5. Launch the application

```bash
python -m streamlit run app.py
```

Open the local URL shown in your terminal, usually `http://localhost:8501`.

## Deployment

The application is deployed using Streamlit Community Cloud.

**Live application:** https://churn-genai-project-d32267hgmx7xd3vsvrad9i.streamlit.app/

For deployment, configure `GEMINI_API_KEY` in the app's Streamlit secrets settings instead of committing a local `.env` file.

## Security and Responsible Use

- Never publish API keys, passwords, or other credentials.
- Use appropriate access controls when working with real customer data.
- Treat churn probabilities and SHAP explanations as decision-support information, not guaranteed outcomes.
- Review AI-generated recommendations for accuracy, relevance, and policy compliance.
- Validate the model on appropriate data before using it in a production business environment.

## Future Improvements

- Add cross-validation and hyperparameter tuning.
- Compare precision-recall trade-offs and calibrate predicted probabilities.
- Evaluate performance across customer segments.
- Add monitoring for model drift and prediction quality.
- Expand the policy knowledge base and improve retrieval evaluation.
- Add automated tests and more robust error handling.
- Integrate a secure business database for real-time customer analytics.

## Author

**Tharun Kothinti**

Computer Science and Engineering Graduate | Machine Learning | Explainable AI | Generative AI | Python

---

If you find this project useful, consider giving the repository a ⭐ on GitHub.