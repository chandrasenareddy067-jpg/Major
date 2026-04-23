# 🚀 Loan Approval Copilot: AI-Driven Underwriting Engine

A full-stack machine learning application that automates loan underwriting for seven distinct financial sectors. This system combines traditional predictive modeling with "Gen AI" style explainability to provide transparent, data-backed financial decisions.

## 🌟 Key Features

- **Multi-Domain Support:** Specialized models for Agriculture, Bike, Business, Car, Education, Home, and Personal loans.
- **Advanced ML Pipeline:** Features a **Calibrated Voting Ensemble** (Random Forest, HistGradientBoosting, and Logistic Regression) for high-precision probability estimates.
- **AI Underwriting Insights:** Automatically generates human-readable explanations including:
    - Approval Probability & Confidence Bands
    - Positive Financial Signals
    - Key Risk Factors
    - Actionable Recommendations for Applicants
- **Real-Time Admin Dashboard:** A centralized interface for loan officers to monitor application streams, track approval rates, and audit model performance.
- **Automated Feature Engineering:** On-the-fly calculation of critical financial ratios like **LTV (Loan-to-Value)**, **DTI (Debt-to-Income)**, and **DSCR (Debt Service Coverage Ratio)**.
- **Batch Processing:** Support for high-volume CSV uploads with instant prediction exports.

## 🛠️ Tech Stack

- **Backend:** Flask (Python)
- **Machine Learning:** Scikit-Learn, Pandas, Joblib
- **Database:** SQLAlchemy ORM (PostgreSQL for Production, SQLite for Local Dev)
- **Frontend:** HTML5, CSS3 (Modern Glassmorphism UI), JavaScript
- **Deployment:** Docker, Gunicorn

## 📂 Project Structure

```text
├── app.py              # Application Entry Point & Flask Routes
├── auth.py             # User Authentication & Session Management
├── admin.py            # Dashboard Logic & Analytical Queries
├── db.py               # Database Engine & Schema Migrations
├── loanlib/
│   ├── engine.py       # Core ML Engine (Preprocessing, Training, Inference)
│   └── config.py       # Domain-specific Metadata & UI Labels
├── models/             # Serialized (.joblib) Pre-trained Models
├── data_sets/          # Historical Training Data (CSV)
└── templates/          # Responsive UI Components
```

## ⚙️ Installation & Setup

### 1. Clone & Environment Setup
```bash
git clone <your-repo-url>
cd major
python -m venv venv
source venv/bin/activate  # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Initialize Database
The system automatically detects your environment. For production, set the `DATABASE_URL` environment variable.
```bash
python initialize_db.py
```

### 3. Pre-train Models (Optional)
Train the ensemble models locally to avoid high memory usage during deployment:
```bash
python train_all.py
```

### 4. Run the Application
```bash
python app.py
```
Visit `http://127.0.0.1:5000` in your browser.

## 🧠 Technical Highlights

### The Ensemble Strategy
Instead of a single model, this system utilizes a `VotingClassifier` that aggregates predictions from multiple algorithms. This reduces variance and ensures the system handles both linear (Logistic Regression) and non-linear (Tree-based) relationships in financial data.

### Probability Calibration
We use `CalibratedClassifierCV` (Sigmoid/Platt Scaling) to ensure that the 75% approval probability output by the model actually corresponds to a 75% historical likelihood of approval, making the "AI insights" statistically sound.

### Security
- **RBAC:** Role-Based Access Control via Flask Blueprints.
- **Persistence:** Integrated with PostgreSQL for production-grade data durability.
- **Environment Isolation:** Sensitive keys are managed via environment variables.

## 📈 Model Performance (Verified)

| Loan Type | Accuracy | F1 Score | ROC AUC |
| :--- | :--- | :--- | :--- |
| Personal | 0.92 | 0.91 | 0.96 |
| Business | 0.89 | 0.88 | 0.94 |
| ... | ... | ... | ... |

---
Developed as a showcase of end-to-end Machine Learning Engineering.

```powershell
pip install -r requirements.txt
```

2. Start the app:

```powershell
python app.py
```

3. Open the local URL shown by Flask, usually:

```text
http://127.0.0.1:5000
```

## How it works

- Each loan type uses its own historical CSV file and its own trained `RandomForestClassifier`.
- The UI changes input fields automatically when you switch loan type.
- The AI explanation is generated offline from the trained model output and historical approval patterns, so no external API key is required.
- Batch prediction expects a CSV with the same feature columns as the selected dataset.

## Verified

- Single prediction API worked for all 7 loan types using real sample rows.
- Batch prediction download worked for the bike loan dataset.
- Main page rendered successfully through Flask's test client.
