# FicZon Inc — Sales Effectiveness & Lead Quality Prediction

> **IABAC Data Science Project** | Using machine learning to pre-categorize lead quality and improve sales conversion rates for a digital-first IT solutions provider.

---

## Table of Contents

- [Business Context](#business-context)
- [Project Objective](#project-objective)
- [Project Structure](#project-structure)
- [Data Source](#data-source)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Models & Results](#models--results)
- [Key Findings](#key-findings)
- [Installation & Usage](#installation--usage)
- [Dependencies](#dependencies)

---

## Business Context

**FicZon Inc** is an IT solutions provider offering a range of products from on-premises software to SaaS-based platforms. Its primary lead generation channels are digital — website traffic and inbound inquiries.

FicZon's business growth is heavily dependent on **sales force effectiveness**. As the market matures and competition increases, the company has been experiencing a dip in sales. A critical bottleneck identified is **lead quality** — the manual, staff-dependent process of categorizing whether a lead is worth pursuing.

While a quality review process exists, it is largely useful for post-analysis rather than real-time conversion. Sales agents spend time on low-quality or irrelevant leads, reducing their effectiveness on high-potential opportunities.

**FicZon wants to explore Machine Learning** to automatically pre-categorize lead quality — enabling the sales team to prioritize the right leads, reduce manual effort, and significantly improve conversion rates and overall sales effectiveness.

---

## Project Objective

Build a machine learning classification model that can **predict the status/category of a sales lead** based on available features (source, agent, location, delivery mode, timing), so that:

- High-potential leads are identified and prioritized early.
- Low-quality or junk leads are filtered out automatically.
- Sales agents can focus their effort on leads most likely to convert.

---

## Project Structure

```
Sales_Effectiveness-main/
│
├── Sales_effective.ipynb      # Main Jupyter Notebook (full pipeline)
└── sales_effect.pkl           # Saved Random Forest model (best generalized performer)
```

---

## Data Source

The dataset is loaded directly from a **MySQL database** hosted on a remote server:

```python
connection = mysql.connector.connect(
    host='18.136.157.135',
    user='dm_team2',
    database='project_sales'
)
data = pd.read_sql_query("select * from data", connection)
```

This live database connection reflects real operational CRM data from FicZon's lead management system.

---

## Dataset

The dataset contains sales lead records with the following features. All features are categorical in nature.

### Feature Reference

| Feature | Description |
|---|---|
| `Created` | Timestamp of when the lead/inquiry was created (date + time) |
| `Product_ID` | Unique identifier for the product or inquiry type |
| `Source` | Lead origin channel (e.g., Call, Web, Email, Walk-in) |
| `Mobile` | Mobile number associated with the lead |
| `EMAIL` | Email address of the lead |
| `Sales_Agent` | The sales agent assigned to the lead |
| `Location` | Geographic location of the customer/inquiry |
| `Delivery_Mode` | Mode of product/service delivery (Mode-1 through Mode-5) |
| `Status` | **Target variable** — Current state of the lead (see below) |

### Target Variable: `Status`

| Status | Description |
|---|---|
| `Open` | Lead received; no action taken yet |
| `Potential` | Shows interest but no commitment yet |
| `In Progress Positive` | Being actively worked on with a positive trajectory |
| `Not Responding` | Customer not responding to follow-up |
| `Just Enquiry` | Customer asked for information only; no purchase intent |
| `Junk Lead` | Irrelevant, fake, or unqualified lead |
| `Converted` | Successfully converted to a sale |
| `In Progress Negative` | Being worked on but trending unfavorably |
| `LOST` | Opportunity lost; customer no longer engaged |
| `Long Term` | Not immediately actionable; flagged for future follow-up |

### Key Descriptive Statistics

- **Product_ID 18** is the most frequently occurring product in leads.
- **Call** is the most common lead source channel.
- **Sales-Agent-4** handles the highest volume of leads.
- **Mode-5** is the most used delivery mode, followed by Mode-3 and Mode-2.
- **"Potential"** is the most common lead status, followed by "In Progress Positive" and "Open."
- **Bangalore** is the highest-volume location.
- **Sales-Agent-11** handles the highest number of leads; Sales-Agent-6 handles the lowest.

---

## Methodology

### 1. Data Cleaning

- **Duplicate removal** — duplicate rows identified and dropped.
- **Hidden missing values** — blank strings (`""`, `" "`) in columns were replaced with `NaN` and handled:
  - `Product_ID` → imputed with mode value (`18`)
  - `Source` → imputed with most frequent value (`"Call"`)
  - `Mobile` → imputed with `"Unknown"`
  - `Sales_Agent` → imputed with most active agent (`"Sales-Agent-4"`)
  - `Location` → imputed with `"Other Locations"`
- **Dropped columns** — `Product_ID`, `EMAIL`, and `Mobile` were removed as they are identifiers/contact fields that do not contribute to lead quality prediction.

### 2. Feature Engineering

The `Created` timestamp was decomposed into meaningful numerical features:

| New Feature | Description |
|---|---|
| `year` | Year of inquiry |
| `month` | Month of inquiry |
| `day` | Day of the month |
| `hour` | Hour of the day |
| `minute` | Minute |
| `weekday` | Day of the week (0=Monday, 6=Sunday) |

The original `Created` column was then dropped.

### 3. Encoding

- **One-Hot Encoding** applied to `Delivery_Mode` — creating binary indicator columns (`Delivery_Mode-1` through `Delivery_Mode-5`).
- **Label Encoding** applied to `Source`, `Sales_Agent`, `Location`, and `Status`.

### 4. Outlier Detection

Boxplots were generated for continuous features (`Source`, `day`, `minute`). No significant outliers were detected.

### 5. Feature Scaling

**StandardScaler** applied to all features used for modeling:
`Source`, `Sales_Agent`, `Location`, `Status`, `year`, `month`, `day`, `hour`, `minute`, `weekday`

### 6. Dimensionality Reduction — PCA

Principal Component Analysis (PCA) was applied to reduce feature dimensionality:

- Full PCA was fit to determine cumulative explained variance.
- **13 principal components** were found to explain over **90% of the variance**.
- Final model training used a PCA-reduced feature matrix (`x1`) with 10 principal components for the standardized data.

### 7. Train-Test Split

- **Split ratio:** 75% training / 25% testing
- **Random state:** 1

---

## Models & Results

Seven classification algorithms were trained and evaluated on PCA-transformed features:

| Model | Test Accuracy | F1 Score (Test) | Notes |
|---|---|---|---|
| **Random Forest** | **99.51%** | **99.31%** | High accuracy + strong generalization |
| **XGBoost** | **100.00%** | **100.00%** | Perfect score on test set |
| **Gradient Boosting** | **100.00%** | **100.00%** | Perfect score on test set |
| ANN (MLP) | 99.29% | 99.10% | High potential; strong performance |
| Decision Tree | 100.00% | 100.00% | Caution: prone to overfitting |
| Logistic Regression | 90.35% | 89.64% | Lower performance; struggles with class separation |
| SVM | — | — | Trained but underperformed vs. ensemble methods |

### Model Recommendations

**High Potential Models:**
- **Random Forest** — Best balance of accuracy and generalization. Chosen as the saved production model.
- **XGBoost / Gradient Boosting** — Near-perfect performance; suitable for further hyperparameter tuning and validation against unseen data.
- **ANN** — Strong performance; viable alternative with more data.

**Lower Potential Models:**
- **Logistic Regression** — Linear model unable to capture the complexity of 11 lead status categories.
- **Decision Tree** — Perfect training accuracy but likely overfitting; poor generalization expected on new data without pruning.

### Saved Model

The **Random Forest** classifier was selected and saved as the production model:

```python
import pickle
with open("sales_effect.pkl", "wb") as f:
    pickle.dump(random_forest_model, f)
```

It was chosen over XGBoost/GBM due to better expected generalization (100% test accuracy from boosting models raises overfitting concerns on this dataset).

---

## Key Findings

### EDA Insights

- All features in the dataset are **categorical** — no traditional bivariate or multivariate numerical analysis was applicable.
- The **"Potential"** status dominates the dataset, followed by "In Progress Positive" and "Open" — indicating that most leads in the pipeline are mid-funnel and not yet converted.
- **Mode-5** is the most utilized delivery mode, significantly ahead of others, suggesting it may be the standard offering.
- **Bangalore** generates the highest number of leads, pointing to geographic concentration of FicZon's customer base.
- Lead **source "Call"** dominates all other channels, meaning direct outreach is the primary acquisition method.

### Temporal Patterns

By engineering time-based features from the `Created` timestamp, the model can capture patterns such as:
- Which days of the week leads are more likely to convert.
- Whether time of day correlates with lead quality or response.
- Seasonal or monthly patterns in lead volume and conversion.

### Correlation Analysis

A heatmap of Pearson correlations was used to identify relationships between encoded features and the target variable, informing the PCA step and confirming which features carry the most predictive signal.

---

## Installation & Usage

### Prerequisites

- Python 3.8+
- Jupyter Notebook
- MySQL access (or a local copy of the dataset)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/Sales_Effectiveness.git
cd Sales_Effectiveness

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook Sales_effective.ipynb
```

### Database Connection

The notebook connects to a remote MySQL database. Update the connection credentials if running against a different instance:

```python
connection = mysql.connector.connect(
    host='YOUR_HOST',
    user='YOUR_USER',
    password='YOUR_PASSWORD',
    database='project_sales'
)
```

### Using the Saved Model

```python
import pickle

# Load the trained Random Forest model
with open("sales_effect.pkl", "rb") as f:
    model = pickle.load(f)

# Predict lead status on new data (after preprocessing + PCA transformation)
predictions = model.predict(X_new_pca)
```

> **Important:** New data must go through the same preprocessing pipeline (missing value handling, encoding, scaling, and PCA transformation with the same fitted scaler and PCA object) before being passed to the model.

---

## Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
mysql-connector-python
pickle
jupyter
```

Install all at once:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost mysql-connector-python jupyter
```

---

## License

This project was developed as part of an **IABAC (International Association of Business Analytics Certifications)** certification program. All data is sourced from a controlled training environment for educational purposes.
