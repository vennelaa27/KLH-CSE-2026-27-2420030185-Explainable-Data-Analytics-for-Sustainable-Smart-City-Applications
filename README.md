# Explainable Data Analytics for Sustainable Smart City Applications

An integrated machine learning framework for analyzing **Air Quality, Weather, and Energy Consumption** using predictive modeling, feature engineering, explainable AI, and cross-domain analytics to support sustainable smart-city decision-making.

## 📌 Project Overview

Smart cities generate large amounts of heterogeneous data related to environmental conditions, weather patterns, and energy usage. Analyzing these data sources independently can limit our understanding of their relationships and their implications for urban sustainability.

This project develops an explainable data analytics framework that integrates three smart-city domains to explore patterns, predict domain-specific outcomes, interpret model behavior, and derive data-driven insights.

The framework combines data preprocessing, exploratory data analysis (EDA), feature engineering, feature selection, machine learning, ensemble learning, SHAP-based explainability, and an interactive Streamlit dashboard.

**The primary objective is to make machine learning predictions more interpretable and useful for sustainable smart-city applications.**

## 🎯 Objectives

- Analyze heterogeneous datasets representing air quality, weather, and energy consumption.
- Perform data cleaning, preprocessing, and exploratory data analysis.
- Engineer and select relevant features for predictive modeling.
- Develop and evaluate multiple machine learning models.
- Implement a stacking ensemble to combine complementary predictive models.
- Apply SHAP (SHapley Additive exPlanations) to interpret model predictions.
- Explore cross-domain patterns connecting environmental, weather, and energy-related information.
- Develop an interactive dashboard for presenting predictions, visualizations, model results, and explainability insights.
- Support evidence-based decision-making for sustainable urban development.

## 🌍 Domains and Datasets

The project investigates three complementary smart-city domains.

| Domain | Prediction Target | Purpose |
|---|---|---|
| Air Quality | `Pollution_Score` | Analyze pollution-related patterns and identify influential features. |
| Weather | `Temperature (C)` | Model temperature using relevant weather attributes while avoiding target leakage. |
| Energy Consumption | `EnergyConsumption` | Analyze energy usage and evaluate predictive performance. |

Each dataset follows a domain-specific preprocessing and feature-engineering workflow. Features that directly reveal the target or introduce target leakage are excluded where applicable.

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Gradient Boosting:** CatBoost, Gradient Boosting Regressor
- **Explainable AI:** SHAP
- **Interactive Dashboard:** Streamlit
- **Development Environment:** Jupyter Notebook, PyCharm
- **Version Control:** Git and GitHub

## ⚙️ Methodology

### 1. Data Collection and Preprocessing

- Load and inspect the datasets for all three domains.
- Identify missing values, duplicate records, and inconsistencies.
- Apply appropriate data-cleaning and preprocessing techniques.
- Prepare features and target variables for model training.
- Exclude leakage-prone features to improve the validity of model evaluation.

### 2. Exploratory Data Analysis (EDA)

Conduct exploratory analysis to understand data distributions, relationships, and domain-specific patterns.

Key activities include:

- Statistical summaries and data-quality analysis.
- Univariate and multivariate visualizations.
- Correlation analysis.
- Feature-versus-target relationship analysis.
- Identification of potentially influential variables.

### 3. Feature Engineering and Feature Selection

Transform the original datasets into model-ready representations through domain-specific feature engineering and encoding.

Feature selection and leakage audits help identify useful predictors while preventing features that reveal the target from entering the models.

### 4. Machine Learning Model Development

Train and compare multiple regression models, including:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Other domain-appropriate regression and ensemble models

Evaluate model performance using multiple metrics to understand predictive accuracy and generalization.

### 5. Proposed Stacking Ensemble — LCG-Stack

The project implements an **OOF Linear–CatBoost–Gradient Boosting Stacking Ensemble (LCG-Stack)**.

The architecture combines three base learners:

- Linear Regression
- CatBoost Regressor
- Gradient Boosting Regressor

A Linear Regression meta-learner combines their predictions to generate the final output. Out-of-fold (OOF) predictions are used to train the meta-learner, helping reduce the risk of meta-model training leakage.

The ensemble is evaluated against individual models rather than assumed to be superior in every domain.

### 6. Model Evaluation

Model performance is assessed using a comprehensive set of regression metrics:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)
- Median Absolute Error
- Explained Variance
- Maximum Error
- Normalized RMSE
- Adjusted R²
- Mean Absolute Percentage Error (MAPE), where appropriate

These metrics provide complementary perspectives on prediction errors and model fit. MAPE is treated cautiously when actual target values are zero, near zero, or otherwise unsuitable for percentage-based evaluation.

### 7. Explainable AI Using SHAP

SHAP is used to interpret model predictions and investigate the contribution of individual features.

The explainability workflow includes:

- Global feature-importance analysis.
- Feature contribution analysis.
- Local prediction explanations.
- Visualization of positive and negative feature contributions.
- Domain-specific interpretation of model behavior.

This component helps make model outputs more transparent and supports a better understanding of the factors associated with predicted air quality, weather, and energy outcomes.

### 8. Cross-Domain Analytics

The framework explores relationships across the three smart-city domains by combining selected features and contextual information where appropriate.

The cross-domain analysis investigates how environmental, weather-related, and energy-related information can be examined together to support a broader understanding of smart-city systems.

The combined analysis is treated as a separate analytical component, with its own preprocessing and evaluation considerations.

### 9. Interactive Streamlit Dashboard

The completed dashboard brings the project's analytical workflow together in an interactive interface.

The dashboard presents the project's major components, including:

- Project overview and domain information.
- Data preprocessing and exploratory analysis.
- Feature engineering and feature selection.
- Model comparison and evaluation results.
- Interactive model predictions.
- SHAP explainability visualizations.
- Cross-domain analytical insights.
- Smart-city sustainability and decision-support interpretations.

The dashboard provides an accessible way to explore the analytical results without requiring users to execute every notebook cell manually.

## 📊 Key Features

- **Multi-domain analytics:** Covers Air Quality, Weather, and Energy Consumption.
- **Leakage-aware modeling:** Applies domain-specific feature exclusions and evaluation safeguards.
- **Comparative model evaluation:** Assesses multiple regression models using several performance metrics.
- **Stacking ensemble:** Combines Linear Regression, CatBoost, and Gradient Boosting with a Linear Regression meta-learner.
- **Explainable predictions:** Uses SHAP to investigate feature contributions and model behavior.
- **Cross-domain exploration:** Examines selected relationships across heterogeneous smart-city data.
- **Interactive dashboard:** Provides a unified interface for predictions, visualizations, and analytical findings.
- **Sustainability-oriented insights:** Connects predictive analytics with potential smart-city decision-support applications.

## 📁 Project Structure

The main implementation is organized as follows:

```text
Project_Implementation/
│
├── DAV_Analysis.ipynb
├── dashboard.py
│
├── cleaned_data/
├── datasets/
├── Dashboard_Resources/
│
├── EDA_Graphs/
│   ├── Air Quality/
│   ├── Weather/
│   └── Energy/
│
├── Feature_engineered_datasets/
│   ├── Cross_Domain_Fusion/
│   ├── Explainability/
│   ├── air_quality_feature_engineered/
│   ├── energy_consumption_feature_engineered/
│   └── weather_feature_engineered/
│
├── Feature_selected_datasets/
│
├── Hybrid_Model/
│
└── SHAP_Results/
    ├── Air_Quality/
    │   ├── Images/
    │   └── Tables/
    ├── Weather/
    │   ├── Images/
    │   └── Tables/
    └── Energy/
        ├── Images/
        └── Tables/
```

*Note: This represents the main implementation structure. Additional files, result tables, notebooks, or supporting resources may be present in the complete repository.*

## 🚀 Getting Started

### Prerequisites

Install Python and ensure that the required project datasets and supporting files are available locally.

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

### 2. Navigate to the Project Directory

```bash
cd Project_Implementation
```

Adjust the path if `Project_Implementation/` is nested inside another repository directory.

### 3. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

Install the libraries required by the notebook and dashboard:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn catboost shap streamlit
```

If the repository includes a `requirements.txt` file, use that file as the source of truth instead:

```bash
pip install -r requirements.txt
```

### 5. Launch the Dashboard

From the directory containing `dashboard.py`, run:

```bash
streamlit run dashboard.py
```

Streamlit will display a local URL that you can open in your browser.

**Important:** Ensure that the datasets, saved model artifacts, SHAP outputs, and dashboard resources are available at the paths expected by the application.

### 6. Explore the Notebook

Open `DAV_Analysis.ipynb` in Jupyter Notebook or JupyterLab to inspect the implementation, preprocessing, feature engineering, model development, evaluation, and explainability workflows.

## 📈 Results and Evaluation

The project evaluates individual regression models and the proposed LCG-Stack ensemble across the three domains.

Results are documented through model comparison tables, evaluation metrics, prediction outputs, and SHAP visualizations.

The evaluation emphasizes:

- Comparative predictive performance.
- Error characteristics across different domains.
- Model interpretability.
- The importance of domain-specific preprocessing.
- The value and limitations of combining heterogeneous smart-city data.

Performance is assessed separately for each domain because their targets, feature distributions, and prediction difficulties differ. No single model is assumed to be the best for every task.

Refer to the repository's saved evaluation results and dashboard for the actual reported metrics.

## 🌱 Applications in Sustainable Smart Cities

The framework can support exploratory analysis and decision-support research in areas such as:

- Environmental monitoring and pollution analysis.
- Weather-pattern analysis.
- Energy-consumption monitoring.
- Identification of influential environmental and operational factors.
- Interpretation of machine learning predictions.
- Data-driven sustainability planning.

The outputs are analytical estimates and insights, not substitutes for validated environmental monitoring systems, operational forecasts, or expert policy decisions.

## 🔬 Research Contribution and Novelty

The main contribution is an **integrated, explainable, multi-domain data analytics framework** designed for sustainable smart-city applications.

Its distinguishing aspects include:

1. Joint consideration of Air Quality, Weather, and Energy Consumption.
2. A consistent analytical workflow covering preprocessing, feature engineering, model evaluation, and explainability.
3. An OOF-based stacking ensemble for combining complementary regression learners.
4. SHAP-based interpretation of predictions across multiple domains.
5. Cross-domain exploratory analysis connecting heterogeneous data sources.
6. An interactive dashboard that makes the analytical workflow and findings accessible.

The contribution lies in the integration, application, evaluation, and interpretation of existing machine learning techniques rather than proposing an entirely new learning algorithm.

## 🔮 Future Enhancements

Potential future extensions include:

- Integration of real-time IoT and smart-city sensor data.
- Time-series forecasting for environmental and energy variables.
- Broader cross-domain feature integration.
- Deployment through a cloud-hosted dashboard.
- Model monitoring and periodic retraining.
- Additional explainability and sustainability-oriented decision-support features.

These are potential extensions and are not claimed as completed features of the current implementation.

## ⚠️ Limitations

- Model performance depends on dataset quality, coverage, and representativeness.
- Relationships observed in the available data do not necessarily establish causation.
- Cross-domain findings depend on the availability and compatibility of relevant features.
- SHAP explains model behavior but does not establish causal relationships.
- Predictions should be interpreted within the limitations of the datasets and evaluation procedures.

## 📄 License

Add a suitable open-source license if you intend to distribute the project publicly. If no license is added, the repository should not be assumed to grant permission for unrestricted reuse.

## 🙌 Acknowledgements

This project uses machine learning, data analysis, and explainable AI techniques to investigate sustainable smart-city applications. Acknowledgements and dataset citations should be added for the original dataset providers and any external resources used in the implementation.
