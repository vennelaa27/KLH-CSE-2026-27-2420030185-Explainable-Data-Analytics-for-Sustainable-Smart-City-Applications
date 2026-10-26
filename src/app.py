import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from pathlib import Path

# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "Dashboard_Artifacts"

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="DAV Smart City Analytics",
    page_icon="🏙️",
    layout="wide"
)


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title("🏙️ DAV Smart City Analytics")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate to",
    [
        "🏠 Project Overview",
        "📂 Data & Preprocessing",
        "⚙️ Feature Engineering",
        "🎯 Feature Selection",
        "🤖 Model Development",
        "📊 Model Evaluation",
        "🔮 Interactive Prediction",
        "🔍 Explainable AI",
        "🔗 Cross-Domain Analytics",
        "🏙️ Smart-City Decision Support",
        "ℹ️ About Project"
    ]
)


# =========================================================
# PROJECT OVERVIEW
# =========================================================

if page == "🏠 Project Overview":

    st.title(
        "🏙️ Explainable Data Analytics for "
        "Sustainable Smart City Applications"
    )

    st.markdown(
        """
        ### 🎯 Project Objective

        This project develops an **explainable and adaptable data analytics
        framework** for sustainable smart-city applications.

        The framework integrates data analytics and machine learning across
        three important smart-city domains:

        - 🌫️ **Air Quality**
        - 🌦️ **Weather**
        - ⚡ **Energy**
        """
    )

    st.markdown("---")

    # -----------------------------------------------------
    # PROJECT DOMAINS
    # -----------------------------------------------------

    st.subheader("🌐 Smart-City Domains")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🌫️ Air Quality")
        st.write(
            "Analysis and prediction of Pollution Score "
            "using environmental and air-quality features."
        )

    with col2:
        st.markdown("### 🌦️ Weather")
        st.write(
            "Analysis and prediction of Temperature using "
            "weather and temporal features."
        )

    with col3:
        st.markdown("### ⚡ Energy")
        st.write(
            "Analysis and prediction of Energy Consumption "
            "using building and operational features."
        )

    st.markdown("---")

    # -----------------------------------------------------
    # WORKFLOW
    # -----------------------------------------------------

    st.subheader("🔄 End-to-End Project Workflow")

    st.info(
        """
        **Data**
        ->
        **Data Cleaning**
        ->
        **EDA**
        ->
        **Feature Engineering**
        ->
        **Feature Selection**
        ->
        **Baseline Models**
        ->
        **Proposed Hybrid Model**
        ->
        **Model Evaluation**
        ->
        **SHAP Explainability**
        ->
        **Cross-Domain Analytics**
        ->
        **Smart-City Decision Support**
        """
    )

    st.markdown("---")

    # -----------------------------------------------------
    # PROPOSED HYBRID MODEL
    # -----------------------------------------------------

    st.subheader("⭐ Proposed Hybrid Model")

    st.markdown(
        """
        The proposed model combines three different base learners
        and a meta-learner using **5-fold Out-of-Fold (OOF) stacking**.
        """
    )

    st.code(
        """
                    INPUT FEATURES
                          │
            ┌─────────────┼─────────────┐
            ↓             ↓             ↓
       Linear Reg.    CatBoost    Gradient Boosting
            └─────────────┼─────────────┘
                          ↓
                 5-Fold OOF Predictions
                          ↓
                Linear Regression
                  Meta-Learner
                          ↓
                   ⭐ FINAL PREDICTION
        """,
        language="text"
    )

    st.success(
        """
        Proposed Hybrid Architecture:
        Linear Regression + CatBoost + Gradient Boosting
        → Linear Regression Meta-Learner
        """
    )

    st.markdown("---")

    # -----------------------------------------------------
    # EXPLAINABILITY
    # -----------------------------------------------------

    st.subheader("🔍 Explainable AI")

    st.write(
        """
        SHAP-based explainability is used to understand how individual
        input features contribute to model predictions.

        The dashboard provides both:

        - **Global explanations** — overall feature importance
        - **Prediction explanations** — feature contributions for an
          individual prediction
        """
    )

    st.markdown("---")

    # -----------------------------------------------------
    # PROJECT HIGHLIGHT
    # -----------------------------------------------------

    st.subheader("💡 Project Highlight")

    st.success(
        "The final framework connects prediction, explainability, "
        "cross-domain evidence, and smart-city decision support "
        "in a single interactive dashboard."
    )


# =========================================================
# DATA & PREPROCESSING
# =========================================================

elif page == "📂 Data & Preprocessing":

    st.title("📂 Data & Preprocessing")

    st.markdown(
        """
        This section presents the datasets used in the project,
        their characteristics, preprocessing activities, and
        exploratory data analysis.
        """
    )

    # -----------------------------------------------------
    # SUB-NAVIGATION
    # -----------------------------------------------------

    subsection = st.radio(
        "Select a section",
        [
            "📊 Dataset Overview",
            "🧹 Data Cleaning",
            "📈 Exploratory Data Analysis"
        ],
        horizontal=True
    )

    st.markdown("---")

    # =====================================================
    # DATASET OVERVIEW
    # =====================================================

    if subsection == "📊 Dataset Overview":

        st.header("📊 Dataset Overview")

        summary_file = (
            ARTIFACTS_DIR
            / "data_overview"
            / "dataset_summary.xlsx"
        )

        if summary_file.exists():

            summary_df = pd.read_excel(summary_file)

            st.dataframe(
                summary_df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.warning(
                "Dataset summary file was not found."
            )

        st.markdown("---")

        st.subheader("🎯 Prediction Targets")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "🌫️ Air Quality",
                "Pollution Score"
            )

        with col2:
            st.metric(
                "🌦️ Weather",
                "Temperature (C)"
            )

        with col3:
            st.metric(
                "⚡ Energy",
                "Energy Consumption"
            )

        st.markdown("---")

        feature_file = (
            ARTIFACTS_DIR
            / "data_overview"
            / "feature_information.xlsx"
        )

        if feature_file.exists():

            st.subheader("🔎 Feature Information")

            feature_df = pd.read_excel(feature_file)

            st.dataframe(
                feature_df,
                use_container_width=True,
                hide_index=True
            )

    # =====================================================
    # DATA CLEANING
    # =====================================================

    elif subsection == "🧹 Data Cleaning":

        st.header("🧹 Data Cleaning")

        st.markdown(
            """
            The datasets were examined and prepared before applying
            exploratory analysis and machine-learning models.

            The preprocessing stage included:

            - Missing-value inspection
            - Duplicate-row inspection
            - Data-type verification
            - Target and feature identification
            - Removal or handling of unsuitable records where required
            - Preparation of datasets for subsequent feature engineering
              and model development

            The cleaned datasets were then used for EDA, feature
            engineering, feature selection, and model development.
            """
        )

        st.info(
            "The dashboard presents the cleaning methodology here rather "
            "than duplicating a separate data-cleaning artifact."
        )

        st.subheader("📋 Cleaning Approach by Domain")

        cleaning_data = pd.DataFrame(
            {
                "Domain": [
                    "Air Quality",
                    "Weather",
                    "Energy"
                ],
                "Target": [
                    "Pollution Score",
                    "Temperature (C)",
                    "Energy Consumption"
                ],
                "Main Checks": [
                    "Missing values, duplicates, data types, target leakage",
                    "Missing values, duplicates, temporal features, data types",
                    "Missing values, duplicates, data types, derived variables"
                ]
            }
        )

        st.dataframe(
            cleaning_data,
            use_container_width=True,
            hide_index=True
        )

    # =====================================================
    # EDA
    # =====================================================

    elif subsection == "📈 Exploratory Data Analysis":

        st.header("📈 Exploratory Data Analysis")

        st.markdown(
            """
            Exploratory Data Analysis was performed to understand
            feature distributions, relationships, correlations,
            and patterns within each smart-city domain.
            """
        )

        eda_base = ARTIFACTS_DIR / "eda"

        domain = st.selectbox(
            "Select Domain",
            [
                "Air Quality",
                "Weather",
                "Energy"
            ]
        )

        domain_folder = domain.replace(" ", "_")

        st.info(
            f"Select an EDA image from the **{domain}** analysis."
        )

        if eda_base.exists():

            image_files = list(
                eda_base.rglob("*.png")
            ) + list(
                eda_base.rglob("*.jpg")
            ) + list(
                eda_base.rglob("*.jpeg")
            )

            domain_images = [
                img for img in image_files
                if domain.lower().replace(" ", "_")
                in str(img).lower()
            ]

            if domain_images:

                image_names = [
                    img.name for img in domain_images
                ]

                selected_image = st.selectbox(
                    "Select EDA Visualization",
                    image_names
                )

                selected_path = next(
                    img for img in domain_images
                    if img.name == selected_image
                )

                st.image(
                    str(selected_path),
                    caption=selected_image,
                    use_container_width=True
                )

            else:

                st.warning(
                    f"No EDA images found for {domain}."
                )

        else:

            st.warning(
                "EDA artifact folder was not found."
            )



# =========================================================
# FEATURE ENGINEERING
# =========================================================

elif page == "⚙️ Feature Engineering":

    st.title("⚙️ Feature Engineering")

    st.markdown(
        """
        Feature engineering was performed to transform the original
        datasets into more informative representations for machine
        learning.

        The engineered datasets were prepared during the project
        implementation and saved for subsequent feature selection
        and model development.
        """
    )

    st.markdown("---")

    # -----------------------------------------------------
    # LOAD FEATURE ENGINEERING SUMMARY
    # -----------------------------------------------------

    feature_engineering_file = (
        ARTIFACTS_DIR
        / "feature_engineering"
        / "feature_engineering_summary.xlsx"
    )

    if feature_engineering_file.exists():

        feature_engineering_df = pd.read_excel(
            feature_engineering_file
        )

        st.subheader("📋 Engineered Feature Summary")

        st.dataframe(
            feature_engineering_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "Feature engineering summary file was not found."
        )

    st.markdown("---")

    # -----------------------------------------------------
    # DOMAIN-WISE ENGINEERED FEATURES
    # -----------------------------------------------------

    st.subheader("🧩 Domain-wise Engineered Features")

    domain = st.selectbox(
        "Select Domain",
        [
            "Air Quality",
            "Weather",
            "Energy"
        ]
    )

    if domain == "Air Quality":

        st.markdown("### 🌫️ Air Quality")

        st.write(
            "Ratio-based features were created to capture relationships "
            "between important air-quality variables."
        )

        air_features = pd.DataFrame(
            {
                "Engineered Feature": [
                    "PM_Ratio",
                    "NOx_NO2_Ratio",
                    "CO_CO2_Ratio"
                ],
                "Purpose": [
                    "Represents the relationship between PM10 and PM2.5.",
                    "Represents the relationship between NOx and NO2.",
                    "Represents the relationship between CO and CO2."
                ]
            }
        )

        st.dataframe(
            air_features,
            use_container_width=True,
            hide_index=True
        )

    elif domain == "Weather":

        st.markdown("### 🌦️ Weather")

        st.write(
            "Temporal and weather-condition features were derived "
            "to capture time-dependent and environmental patterns."
        )

        weather_features = pd.DataFrame(
            {
                "Engineered Feature": [
                    "Year",
                    "Month",
                    "Hour",
                    "DayOfWeek",
                    "Season",
                    "Temperature_Difference",
                    "Wind_Chill_Indicator"
                ],
                "Purpose": [
                    "Extracts the year from the date-time information.",
                    "Extracts the month from the date-time information.",
                    "Extracts the hour from the date-time information.",
                    "Identifies the day of the week.",
                    "Represents the seasonal period.",
                    "Captures the difference between temperature measures.",
                    "Indicates the presence of wind-chill conditions."
                ]
            }
        )

        st.dataframe(
            weather_features,
            use_container_width=True,
            hide_index=True
        )

    elif domain == "Energy":

        st.markdown("### ⚡ Energy")

        st.write(
            "Temporal, operational-status, and normalized consumption "
            "features were created for energy analysis."
        )

        energy_features = pd.DataFrame(
            {
                "Engineered Feature": [
                    "Hour",
                    "Month",
                    "Weekend",
                    "HVAC_Status",
                    "Lighting_Status",
                    "Energy_Per_Occupant",
                    "Renewable_Energy_Ratio",
                    "Energy_Per_SquareFoot"
                ],
                "Purpose": [
                    "Captures the hour of energy usage.",
                    "Captures the month of energy usage.",
                    "Identifies whether the observation falls on a weekend.",
                    "Represents HVAC operational status.",
                    "Represents lighting operational status.",
                    "Normalizes energy consumption by occupancy.",
                    "Represents the contribution of renewable energy.",
                    "Normalizes energy consumption by building area."
                ]
            }
        )

        st.dataframe(
            energy_features,
            use_container_width=True,
            hide_index=True
        )

    st.markdown("---")

    # -----------------------------------------------------
    # PURPOSE
    # -----------------------------------------------------

    st.subheader("🎯 Why Feature Engineering?")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📈 Better Representation")
        st.write(
            "Transforms raw variables into features that better "
            "represent domain-specific patterns."
        )

    with col2:
        st.markdown("### 🔎 Domain Information")
        st.write(
            "Introduces temporal, ratio-based, operational, and "
            "normalized information."
        )

    with col3:
        st.markdown("### 🤖 Model Preparation")
        st.write(
            "Provides informative inputs for feature selection "
            "and subsequent machine-learning models."
        )


# =========================================================
# FEATURE SELECTION
# =========================================================

elif page == "🎯 Feature Selection":

    st.title("🎯 Feature Selection")

    st.markdown(
        """
        Feature selection was performed to identify the most relevant
        input variables for each smart-city domain.

        The selected features were then used during model development
        to improve the representation of the prediction problem while
        avoiding unnecessary variables.
        """
    )

    st.markdown("---")

    # -----------------------------------------------------
    # LOAD FEATURE SELECTION SUMMARY
    # -----------------------------------------------------

    feature_selection_file = (
        ARTIFACTS_DIR
        / "feature_selection"
        / "feature_selection_summary.xlsx"
    )

    if feature_selection_file.exists():

        feature_selection_df = pd.read_excel(
            feature_selection_file
        )

        st.subheader("📋 Selected Feature Summary")

        st.dataframe(
            feature_selection_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "Feature selection summary file was not found."
        )

    st.markdown("---")

    # -----------------------------------------------------
    # DOMAIN-WISE FEATURE SELECTION
    # -----------------------------------------------------

    st.subheader("🔎 Domain-wise Selected Features")

    domain = st.selectbox(
        "Select Domain",
        [
            "Air Quality",
            "Weather",
            "Energy"
        ]
    )

    domain_df = feature_selection_df[
        feature_selection_df["Dataset"] == domain
    ].copy() if feature_selection_file.exists() else pd.DataFrame()

    if not domain_df.empty:

        st.markdown(f"### {domain}")

        st.metric(
            "Selected Input Features",
            domain_df["Total_Selected_Input_Features"].iloc[0]
        )

        st.dataframe(
            domain_df[
                [
                    "Selected_Feature",
                    "Data_Type",
                    "Unique_Values",
                    "Missing_Values",
                    "Target"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            f"No feature-selection information found for {domain}."
        )

    st.markdown("---")

    # -----------------------------------------------------
    # TARGET VARIABLES
    # -----------------------------------------------------

    st.subheader("🎯 Prediction Targets")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🌫️ Air Quality")
        st.write("Target: **Pollution_Score**")

    with col2:
        st.markdown("### 🌦️ Weather")
        st.write("Target: **Temperature (C)**")

    with col3:
        st.markdown("### ⚡ Energy")
        st.write("Target: **EnergyConsumption**")

    st.markdown("---")

    # -----------------------------------------------------
    # PURPOSE OF FEATURE SELECTION
    # -----------------------------------------------------

    st.subheader("💡 Purpose of Feature Selection")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🎯 Relevance")
        st.write(
            "Focuses the model on informative variables "
            "for the prediction task."
        )

    with col2:
        st.markdown("### 🧹 Reduce Redundancy")
        st.write(
            "Helps avoid unnecessary or less useful input "
            "variables."
        )

    with col3:
        st.markdown("### 🤖 Model Efficiency")
        st.write(
            "Provides a more focused feature set for "
            "machine-learning models."
        )


# =========================================================
# MODEL DEVELOPMENT
# =========================================================

elif page == "🤖 Model Development":

    st.title("🤖 Model Development")

    st.markdown(
        """
        This section presents the machine-learning models developed
        for the three smart-city domains.

        The project compares conventional baseline models with the
        proposed hybrid stacking model.
        """
    )

    st.markdown("---")

    # -----------------------------------------------------
    # SUB-NAVIGATION
    # -----------------------------------------------------

    subsection = st.radio(
        "Select a section",
        [
            "📌 Baseline Models",
            "⭐ Proposed Hybrid Model",
            "🏗️ Model Architecture",
            "💾 Saved Model Artifacts"
        ],
        horizontal=True
    )

    st.markdown("---")

    # =====================================================
    # BASELINE MODELS
    # =====================================================

    if subsection == "📌 Baseline Models":

        st.header("📌 Baseline Models")

        st.markdown(
            """
            Three conventional regression algorithms were used as
            baseline models for comparison with the proposed hybrid
            approach.
            """
        )

        baseline_models = pd.DataFrame(
            {
                "Model": [
                    "Linear Regression",
                    "Decision Tree",
                    "Random Forest"
                ],
                "Type": [
                    "Linear regression",
                    "Tree-based regression",
                    "Ensemble tree-based regression"
                ],
                "Purpose": [
                    "Provides a simple linear reference",
                    "Captures non-linear relationships",
                    "Provides a strong conventional ensemble baseline"
                ]
            }
        )

        st.dataframe(
            baseline_models,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

        st.subheader("🌐 Baselines Across Domains")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 🌫️ Air Quality")
            st.write("Linear Regression")
            st.write("Decision Tree")
            st.write("Random Forest")

        with col2:
            st.markdown("### 🌦️ Weather")
            st.write("Linear Regression")
            st.write("Decision Tree")
            st.write("Random Forest")

        with col3:
            st.markdown("### ⚡ Energy")
            st.write("Linear Regression")
            st.write("Decision Tree")
            st.write("Random Forest")

    # =====================================================
    # PROPOSED HYBRID MODEL
    # =====================================================

    elif subsection == "⭐ Proposed Hybrid Model":

        st.header("⭐ Proposed Hybrid Model")

        st.markdown(
            """
            The proposed approach uses **stacking ensemble learning**.

            Three different base learners generate predictions using
            5-fold Out-of-Fold (OOF) training. These predictions are then
            passed to a Linear Regression meta-learner to produce the
            final prediction.
            """
        )

        st.markdown("---")

        st.subheader("🧩 Hybrid Components")

        hybrid_components = pd.DataFrame(
            {
                "Stage": [
                    "Base Learner 1",
                    "Base Learner 2",
                    "Base Learner 3",
                    "Stacking Stage"
                ],
                "Model": [
                    "Linear Regression",
                    "CatBoost",
                    "Gradient Boosting",
                    "Linear Regression Meta-Learner"
                ],
                "Role": [
                    "Linear modeling perspective",
                    "Non-linear gradient boosting with categorical-data capability",
                    "Non-linear boosting perspective",
                    "Combines base-model predictions"
                ]
            }
        )

        st.dataframe(
            hybrid_components,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

        st.subheader("🎯 Why a Hybrid Model?")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 📐 Linear Perspective")
            st.write(
                "Linear Regression captures approximately linear "
                "relationships between inputs and the target."
            )

        with col2:
            st.markdown("### 🌳 Non-Linear Perspective")
            st.write(
                "CatBoost and Gradient Boosting can capture complex "
                "non-linear patterns."
            )

        with col3:
            st.markdown("### 🔗 Prediction Combination")
            st.write(
                "The meta-learner combines predictions from the "
                "different base learners."
            )

        st.markdown("---")

        st.success(
            """
            ⭐ Final Proposed Architecture:

            Linear Regression + CatBoost + Gradient Boosting
            → 5-Fold OOF Stacking
            → Linear Regression Meta-Learner
            """
        )

    # =====================================================
    # MODEL ARCHITECTURE
    # =====================================================

    elif subsection == "🏗️ Model Architecture":

        st.header("🏗️ Proposed Hybrid Model Architecture")

        st.markdown(
            """
            The architecture below illustrates how the input features
            flow through the proposed stacking framework.
            """
        )

        st.code(
            """
                         INPUT FEATURES
                               │
                ┌──────────────┼──────────────┐
                ↓              ↓              ↓
          Linear Regression  CatBoost   Gradient Boosting
                │              │              │
                └──────────────┼──────────────┘
                               ↓
                      5-Fold OOF Predictions
                               ↓
                    Linear Regression
                      Meta-Learner
                               ↓
                       ⭐ FINAL PREDICTION
            """,
            language="text"
        )

        st.markdown("---")

        st.subheader("🔄 Stacking Process")

        steps = [
            (
                "1️⃣",
                "Prepare Input Features",
                "The selected features are passed through the saved domain-specific preprocessing pipeline."
            ),
            (
                "2️⃣",
                "Train Base Learners",
                "Linear Regression, CatBoost, and Gradient Boosting learn complementary patterns."
            ),
            (
                "3️⃣",
                "Generate OOF Predictions",
                "Five-fold Out-of-Fold predictions are generated without using the validation fold for training."
            ),
            (
                "4️⃣",
                "Train Meta-Learner",
                "The OOF predictions become inputs to the Linear Regression meta-learner."
            ),
            (
                "5️⃣",
                "Final Prediction",
                "The trained stacking framework produces the final prediction."
            )
        ]

        for number, title, description in steps:

            st.markdown(f"### {number} {title}")
            st.write(description)

    # =====================================================
    # SAVED MODEL ARTIFACTS
    # =====================================================

    elif subsection == "💾 Saved Model Artifacts":

        st.header("💾 Saved Model Artifacts")

        st.markdown(
            """
            The dashboard uses the saved model and preprocessing
            artifacts generated during project implementation.
            
            **No model retraining is performed inside Streamlit.**
            """
        )

        st.markdown("---")

        model_base = ARTIFACTS_DIR / "models"
        preprocessor_base = ARTIFACTS_DIR / "preprocessors"

        domains = {
            "🌫️ Air Quality": "air",
            "🌦️ Weather": "weather",
            "⚡ Energy": "energy"
        }

        for domain_name, folder_name in domains.items():

            st.subheader(domain_name)

            model_folder = model_base / folder_name
            preprocessor_folder = (
                preprocessor_base / folder_name
            )

            if model_folder.exists():

                model_files = sorted(
                    model_folder.glob("*.joblib")
                )

                if model_files:

                    for model_file in model_files:
                        st.write(
                            f"✅ {model_file.name}"
                        )

                else:

                    st.warning(
                        "No model artifacts found."
                    )

            if preprocessor_folder.exists():

                preprocessor_files = sorted(
                    preprocessor_folder.glob("*.joblib")
                )

                if preprocessor_files:

                    for preprocessor_file in preprocessor_files:
                        st.write(
                            f"⚙️ {preprocessor_file.name}"
                        )

            st.markdown("---")


# =========================================================
# MODEL EVALUATION
# =========================================================

elif page == "📊 Model Evaluation":

    st.title("📊 Model Evaluation")

    st.markdown(
        """
        This section presents the final evaluation and comparative
        analysis of the baseline and proposed hybrid models across
        the three smart-city domains.
        """
    )

    # -----------------------------------------------------
    # LOAD EVALUATION FILE
    # -----------------------------------------------------

    evaluation_file = (
        ARTIFACTS_DIR
        / "evaluation"
        / "final_model_evaluation.xlsx"
    )

    if not evaluation_file.exists():

        st.error(
            "Final model evaluation file was not found."
        )

        st.stop()

    evaluation_df = pd.read_excel(
        evaluation_file
    )

    # -----------------------------------------------------
    # SUB-NAVIGATION
    # -----------------------------------------------------

    subsection = st.radio(
        "Select a section",
        [
            "📐 Evaluation Metrics",
            "🌫️ Air Quality",
            "🌦️ Weather",
            "⚡ Energy",
            "📈 Overall Comparison",
            "🔍 Comparative Findings"
        ],
        horizontal=True
    )

    st.markdown("---")

    # =====================================================
    # EVALUATION METRICS
    # =====================================================

    if subsection == "📐 Evaluation Metrics":

        st.header("📐 Evaluation Metrics")

        st.markdown(
            """
            Multiple regression metrics are used to evaluate prediction
            accuracy, error magnitude, and goodness of fit.
            """
        )

        metric_info = pd.DataFrame(
            {
                "Metric": [
                    "MAE",
                    "MSE",
                    "RMSE",
                    "R²",
                    "Median AE",
                    "Explained Variance",
                    "Maximum Error",
                    "NRMSE (%)",
                    "Adjusted R²",
                    "MAPE (%)"
                ],
                "Meaning": [
                    "Average absolute prediction error",
                    "Average squared prediction error",
                    "Root mean squared prediction error",
                    "Proportion of target variance explained",
                    "Median absolute prediction error",
                    "Proportion of variance explained by predictions",
                    "Largest absolute prediction error",
                    "Normalized RMSE expressed as a percentage",
                    "R² adjusted for number of predictors",
                    "Mean absolute percentage error"
                ],
                "Better": [
                    "Lower ↓",
                    "Lower ↓",
                    "Lower ↓",
                    "Higher ↑",
                    "Lower ↓",
                    "Higher ↑",
                    "Lower ↓",
                    "Lower ↓",
                    "Higher ↑",
                    "Lower ↓"
                ]
            }
        )

        st.dataframe(
            metric_info,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

        st.subheader("📋 Final Evaluation Results")

        st.dataframe(
            evaluation_df,
            use_container_width=True,
            hide_index=True
        )

    # =====================================================
    # AIR QUALITY
    # =====================================================

    elif subsection == "🌫️ Air Quality":

        st.header("🌫️ Air Quality Model Evaluation")

        air_df = evaluation_df[
            evaluation_df["Dataset"] == "Air Quality"
        ].copy()

        st.markdown(
            """
            Comparison of Linear Regression, Decision Tree,
            Random Forest, and the Proposed Hybrid model for
            Pollution Score prediction.
            """
        )

        st.dataframe(
            air_df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

        st.subheader("📊 Metric-wise Comparison")

        air_metric = st.selectbox(
            "Select Metric",
            [
                "MAE",
                "MSE",
                "RMSE",
                "R2",
                "Median_AE",
                "Explained_Variance",
                "Max_Error",
                "NRMSE_percent",
                "Adjusted_R2",
                "MAPE_percent"
            ],
            key="air_metric"
        )

        air_chart = air_df[
            ["Model", air_metric]
        ].dropna()

        if not air_chart.empty:

            st.bar_chart(
                air_chart.set_index("Model")[air_metric]
            )

        st.markdown("---")

        st.info(
            """
            The Air Quality results show perfect performance for
            Linear Regression and the Proposed Hybrid model in the
            final evaluation artifact. These unusually perfect
            results should be interpreted carefully in relation to
            the dataset and target construction.
            """
        )

    # =====================================================
    # WEATHER
    # =====================================================

    elif subsection == "🌦️ Weather":

        st.header("🌦️ Weather Model Evaluation")

        weather_df = evaluation_df[
            evaluation_df["Dataset"] == "Weather"
        ].copy()

        st.markdown(
            """
            Comparison of the four evaluated models for
            Temperature prediction.
            """
        )

        st.dataframe(
            weather_df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

        st.subheader("📊 Metric-wise Comparison")

        weather_metric = st.selectbox(
            "Select Metric",
            [
                "MAE",
                "MSE",
                "RMSE",
                "R2",
                "Median_AE",
                "Explained_Variance",
                "Max_Error",
                "NRMSE_percent",
                "Adjusted_R2",
                "MAPE_percent"
            ],
            key="weather_metric"
        )

        weather_chart = weather_df[
            ["Model", weather_metric]
        ].dropna()

        if not weather_chart.empty:

            st.bar_chart(
                weather_chart.set_index("Model")[weather_metric]
            )

        else:

            st.info(
                f"No available values for {weather_metric}."
            )

        st.markdown("---")

        st.info(
            """
            Random Forest provides the strongest final Weather
            performance, while the Proposed Hybrid model also
            performs strongly but does not surpass Random Forest.
            """
        )

    # =====================================================
    # ENERGY
    # =====================================================

    elif subsection == "⚡ Energy":

        st.header("⚡ Energy Model Evaluation")

        energy_df = evaluation_df[
            evaluation_df["Dataset"] == "Energy"
        ].copy()

        st.markdown(
            """
            Comparison of the four evaluated models for
            Energy Consumption prediction.
            """
        )

        st.dataframe(
            energy_df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

        st.subheader("📊 Metric-wise Comparison")

        energy_metric = st.selectbox(
            "Select Metric",
            [
                "MAE",
                "MSE",
                "RMSE",
                "R2",
                "Median_AE",
                "Explained_Variance",
                "Max_Error",
                "NRMSE_percent",
                "Adjusted_R2",
                "MAPE_percent"
            ],
            key="energy_metric"
        )

        energy_chart = energy_df[
            ["Model", energy_metric]
        ].dropna()

        if not energy_chart.empty:

            st.bar_chart(
                energy_chart.set_index("Model")[energy_metric]
            )

        else:

            st.info(
                f"No available values for {energy_metric}."
            )

        st.markdown("---")

        st.info(
            """
            Linear Regression provides the strongest final Energy
            performance, while the Proposed Hybrid model produces
            a very similar result and outperforms the Decision Tree
            and Random Forest on several metrics.
            """
        )

    # =====================================================
    # OVERALL COMPARISON
    # =====================================================

    elif subsection == "📈 Overall Comparison":

        st.header("📈 Overall Comparative Model Performance")

        st.markdown(
            """
            The following visualization was generated during the
            project implementation and provides the overall comparison
            of the evaluated models using five key performance metrics.
            """
        )

        # -------------------------------------------------
        # EXISTING COMPARATIVE GRAPH
        # -------------------------------------------------

        comparative_graph = (
            BASE_DIR
            / "Hybrid_Model"
            / "Results"
            / "DAV_Comparative_Model_Performance.png"
        )

        if comparative_graph.exists():

            st.image(
                str(comparative_graph),
                use_container_width=True
            )

            st.caption(
                "DAV_Comparative_Model_Performance — "
                "R², NRMSE, MAE, Explained Variance, and Maximum Error"
            )

        else:

            st.warning(
                "DAV_Comparative_Model_Performance.png was not found."
            )

        st.markdown("---")

        # -------------------------------------------------
        # ADDITIONAL FIVE METRICS
        # -------------------------------------------------

        st.subheader(
            "📊 Additional Comparative Metrics"
        )

        st.markdown(
            """
            The existing project graph contains five evaluation
            metrics. The complementary visualization below presents
            the remaining five metrics from the final evaluation:
            MSE, RMSE, Median Absolute Error, Adjusted R², and MAPE.
            """
        )

        additional_metrics = [
            "MSE",
            "RMSE",
            "Median_AE",
            "Adjusted_R2",
            "MAPE_percent"
        ]

        selected_additional_metric = st.selectbox(
            "Select Additional Metric",
            additional_metrics,
            key="additional_metric"
        )

        comparison_df = evaluation_df[
            ["Dataset", "Model", selected_additional_metric]
        ].copy()

        comparison_df = comparison_df.dropna(
            subset=[selected_additional_metric]
        )

        if not comparison_df.empty:

            # Create a readable model-domain label
            comparison_df["Comparison"] = (
                comparison_df["Dataset"]
                + " - "
                + comparison_df["Model"]
            )

            plot_df = comparison_df[
                ["Comparison", selected_additional_metric]
            ].set_index("Comparison")

            st.bar_chart(
                plot_df[
                    selected_additional_metric
                ]
            )

        else:

            st.info(
                f"No available values for "
                f"{selected_additional_metric}."
            )

        st.markdown("---")

        # -------------------------------------------------
        # COMPLETE OVERALL TABLE
        # -------------------------------------------------

        st.subheader(
            "📋 Complete Comparative Evaluation"
        )

        st.dataframe(
            evaluation_df,
            use_container_width=True,
            hide_index=True
        )

    # =====================================================
    # COMPARATIVE FINDINGS
    # =====================================================

    elif subsection == "🔍 Comparative Findings":

        st.header("🔍 Comparative Findings")

        st.markdown(
            """
            The final comparison highlights how model performance
            varies across heterogeneous smart-city domains.
            """
        )

        # -------------------------------------------------
        # AIR QUALITY
        # -------------------------------------------------

        st.subheader("🌫️ Air Quality")

        st.markdown(
            """
            - Linear Regression and the Proposed Hybrid model achieve
              perfect results in the final evaluation.
            - Random Forest also performs strongly.
            - Decision Tree has comparatively higher prediction error.
            - The perfect Air Quality results should be interpreted
              carefully rather than generalized to unseen datasets.
            """
        )

        # -------------------------------------------------
        # WEATHER
        # -------------------------------------------------

        st.subheader("🌦️ Weather")

        st.markdown(
            """
            - Random Forest achieves the strongest Weather performance.
            - The Proposed Hybrid model performs strongly but does not
              surpass Random Forest.
            - This demonstrates that the hybrid framework is not assumed
              to outperform every individual baseline.
            """
        )

        # -------------------------------------------------
        # ENERGY
        # -------------------------------------------------

        st.subheader("⚡ Energy")

        st.markdown(
            """
            - Linear Regression provides the strongest final Energy
              performance.
            - The Proposed Hybrid model produces a very similar result.
            - The Proposed Hybrid model performs better than Decision
              Tree and Random Forest on several evaluation measures.
            """
        )

        # -------------------------------------------------
        # OVERALL
        # -------------------------------------------------

        st.subheader("🌐 Overall Research Finding")

        st.success(
            """
            The comparative analysis demonstrates that model
            effectiveness depends on the characteristics of each
            smart-city domain.

            Therefore, the proposed hybrid framework is presented as
            a robust and adaptable modelling approach rather than as
            a model that is guaranteed to outperform every baseline
            in every domain.
            """
        )

        st.markdown("---")

        st.subheader("⚠️ Missing Metric Values")

        st.write(
            """
            The final evaluation artifact contains unavailable values
            for some metrics. These are preserved as missing values
            rather than estimated or artificially filled.

            • Weather: MAPE is unavailable.

            • Energy: Adjusted R² is unavailable for the baseline
              models.

            These missing values do not affect the other reported
            evaluation metrics.
            """
        )


# =========================================================
# INTERACTIVE PREDICTION
# =========================================================

elif page == "🔮 Interactive Prediction":

    st.title("🔮 Interactive Prediction")

    st.markdown(
        """
        This section allows users to enter domain-specific input
        values and generate predictions using the saved proposed
        hybrid models.

        The dashboard does not retrain models. It loads the
        pre-trained preprocessing pipelines, base learners, and
        meta-learners generated during project implementation.
        """
    )

    st.markdown("---")

    # -----------------------------------------------------
    # SUB-NAVIGATION
    # -----------------------------------------------------

    prediction_domain = st.radio(
        "Select Prediction Domain",
        [
            "🌫️ Air Quality",
            "🌦️ Weather",
            "⚡ Energy"
        ],
        horizontal=True
    )

    st.markdown("---")

    # =====================================================
    # AIR QUALITY
    # =====================================================

    if prediction_domain == "🌫️ Air Quality":

        st.header("🌫️ Air Quality Prediction")

        st.markdown(
            """
            Enter the environmental and air-quality-related features
            below to predict the Pollution Score using the proposed
            hybrid model.
            """
        )

        # -------------------------------------------------
        # LOAD SAVED ARTIFACTS
        # -------------------------------------------------

        air_model_dir = (
            ARTIFACTS_DIR
            / "models"
            / "air"
        )

        air_preprocessor_path = (
            ARTIFACTS_DIR
            / "preprocessors"
            / "air"
            / "preprocessor_air_hybrid.joblib"
        )

        air_lr_path = (
            air_model_dir
            / "lr_final_air.joblib"
        )

        air_cb_path = (
            air_model_dir
            / "cb_final_air.joblib"
        )

        air_gb_path = (
            air_model_dir
            / "gb_final_air.joblib"
        )

        air_meta_path = (
            air_model_dir
            / "meta_lr_air.joblib"
        )

        required_files = [
            air_preprocessor_path,
            air_lr_path,
            air_cb_path,
            air_gb_path,
            air_meta_path
        ]

        missing_files = [
            file.name
            for file in required_files
            if not file.exists()
        ]

        if missing_files:

            st.error(
                "Missing Air Quality model artifacts: "
                + ", ".join(missing_files)
            )

            st.stop()

        # -------------------------------------------------
        # LOAD ARTIFACTS
        # -------------------------------------------------

        import joblib

        air_preprocessor = joblib.load(
            air_preprocessor_path
        )

        air_lr_model = joblib.load(
            air_lr_path
        )

        air_cb_model = joblib.load(
            air_cb_path
        )

        air_gb_model = joblib.load(
            air_gb_path
        )

        air_meta_model = joblib.load(
            air_meta_path
        )

        # -------------------------------------------------
        # INPUT FEATURES
        # -------------------------------------------------

        st.subheader("📝 Input Features")

        col1, col2 = st.columns(2)

        with col1:

            pm25 = st.number_input(
                "PM2.5",
                min_value=0.0,
                value=50.0
            )

            pm10 = st.number_input(
                "PM10",
                min_value=0.0,
                value=80.0
            )

            nox = st.number_input(
                "NOx",
                min_value=0.0,
                value=30.0
            )

            no2 = st.number_input(
                "NO2",
                min_value=0.0,
                value=20.0
            )

            so2 = st.number_input(
                "SO2",
                min_value=0.0,
                value=10.0
            )

            vocs = st.number_input(
                "VOCs",
                min_value=0.0,
                value=20.0
            )

            co = st.number_input(
                "CO",
                min_value=0.0,
                value=1.0
            )

        with col2:

            co2 = st.number_input(
                "CO2",
                min_value=0.0,
                value=400.0
            )

            ch4 = st.number_input(
                "CH4",
                min_value=0.0,
                value=2.0
            )

            temperature = st.number_input(
                "Temperature",
                value=25.0
            )

            humidity = st.number_input(
                "Humidity",
                min_value=0.0,
                max_value=100.0,
                value=60.0
            )

            wind_direction = st.number_input(
                "Wind Direction",
                min_value=0.0,
                max_value=360.0,
                value=180.0
            )

            location_type = st.text_input(
                "Location Type",
                value="Urban"
            )

            source_label = st.text_input(
                "Source Label",
                value="Station"
            )

        st.markdown("---")

        # -------------------------------------------------
        # PREDICT BUTTON
        # -------------------------------------------------

        if st.button(
            "🔮 Predict Pollution Score",
            use_container_width=True
        ):

            try:

                # -------------------------------------------------
                # CREATE INPUT DATAFRAME
                # -------------------------------------------------

                air_input = pd.DataFrame(
                    {
                        "PM2.5": [pm25],
                        "PM10": [pm10],
                        "NOx": [nox],
                        "NO2": [no2],
                        "SO2": [so2],
                        "VOCs": [vocs],
                        "CO": [co],
                        "CO2": [co2],
                        "CH4": [ch4],
                        "Temperature": [temperature],
                        "Humidity": [humidity],
                        "Wind_Direction": [wind_direction],
                        "Location_Type": [location_type],
                        "Source_Label": [source_label]
                    }
                )

                # -------------------------------------------------
                # FEATURE ENGINEERING
                # -------------------------------------------------
                # These three features were used during training.
                # They must be created before preprocessing.

                air_input["PM_Ratio"] = (
                    air_input["PM2.5"]
                    / air_input["PM10"].replace(0, np.nan)
                )

                air_input["NOx_NO2_Ratio"] = (
                    air_input["NOx"]
                    / air_input["NO2"].replace(0, np.nan)
                )

                air_input["CO_CO2_Ratio"] = (
                    air_input["CO"]
                    / air_input["CO2"].replace(0, np.nan)
                )

                # Handle invalid values caused by division by zero
                air_input = air_input.replace(
                    [np.inf, -np.inf],
                    np.nan
                )

                air_input = air_input.fillna(0)

                # -------------------------------------------------
                # APPLY SAVED PREPROCESSING
                # -------------------------------------------------

                air_encoded = (
                    air_preprocessor.transform(
                        air_input
                    )
                )

                # Convert sparse matrix if required
                if hasattr(
                    air_encoded,
                    "toarray"
                ):

                    air_encoded = (
                        air_encoded.toarray()
                    )

                # -------------------------------------------------
                # BASE MODEL PREDICTIONS
                # -------------------------------------------------

                pred_lr = air_lr_model.predict(
                    air_encoded
                )[0]

                pred_cb = air_cb_model.predict(
                    air_encoded
                )[0]

                pred_gb = air_gb_model.predict(
                    air_encoded
                )[0]

                # -------------------------------------------------
                # META-LEARNER INPUT
                # -------------------------------------------------

                meta_input = np.array(
                    [
                        [
                            pred_lr,
                            pred_cb,
                            pred_gb
                        ]
                    ]
                )

                # -------------------------------------------------
                # FINAL HYBRID PREDICTION
                # -------------------------------------------------

                final_prediction = (
                    air_meta_model.predict(
                        meta_input
                    )[0]
                )

                # -------------------------------------------------
                # DISPLAY RESULT
                # -------------------------------------------------

                st.markdown("---")

                st.subheader(
                    "⭐ Predicted Pollution Score"
                )

                st.metric(
                    "Pollution Score",
                    f"{final_prediction:.4f}"
                )

                st.markdown("---")

                st.subheader(
                    "🧩 Base Model Predictions"
                )

                prediction_table = pd.DataFrame(
                    {
                        "Model": [
                            "Linear Regression",
                            "CatBoost",
                            "Gradient Boosting",
                            "Proposed Hybrid"
                        ],
                        "Prediction": [
                            pred_lr,
                            pred_cb,
                            pred_gb,
                            final_prediction
                        ]
                    }
                )

                st.dataframe(
                    prediction_table,
                    use_container_width=True,
                    hide_index=True
                )

                st.success(
                    """
                    Prediction generated successfully using the
                    saved Air Quality hybrid model.
                    """
                )

            except Exception as e:

                st.error(
                    f"Prediction failed: {e}"
                )

        # =====================================================
    # WEATHER
    # =====================================================

    elif prediction_domain == "🌦️ Weather":

        st.header("🌦️ Weather Prediction")

        st.markdown(
            """
            Enter the weather-related features below to predict
            Temperature using the proposed hybrid model.
            """
        )

        # -------------------------------------------------
        # LOAD SAVED ARTIFACTS
        # -------------------------------------------------

        weather_model_dir = (
            ARTIFACTS_DIR
            / "models"
            / "weather"
        )

        weather_preprocessor_path = (
            ARTIFACTS_DIR
            / "preprocessors"
            / "weather"
            / "preprocessor_weather_hybrid.joblib"
        )

        weather_lr_path = (
            weather_model_dir
            / "lr_final_weather.joblib"
        )

        weather_cb_path = (
            weather_model_dir
            / "cb_final_weather.joblib"
        )

        weather_gb_path = (
            weather_model_dir
            / "gb_final_weather.joblib"
        )

        weather_meta_path = (
            weather_model_dir
            / "meta_lr_weather.joblib"
        )

        required_files = [
            weather_preprocessor_path,
            weather_lr_path,
            weather_cb_path,
            weather_gb_path,
            weather_meta_path
        ]

        missing_files = [
            file.name
            for file in required_files
            if not file.exists()
        ]

        if missing_files:

            st.error(
                "Missing Weather model artifacts: "
                + ", ".join(missing_files)
            )

            st.stop()

        # -------------------------------------------------
        # LOAD ARTIFACTS
        # -------------------------------------------------

        import joblib

        weather_preprocessor = joblib.load(
            weather_preprocessor_path
        )

        weather_lr_model = joblib.load(
            weather_lr_path
        )

        weather_cb_model = joblib.load(
            weather_cb_path
        )

        weather_gb_model = joblib.load(
            weather_gb_path
        )

        weather_meta_model = joblib.load(
            weather_meta_path
        )

        # -------------------------------------------------
        # INPUT FEATURES
        # -------------------------------------------------

        st.subheader("📝 Input Features")

        col1, col2 = st.columns(2)

        with col1:

            summary = st.text_input(
                "Summary",
                value="Partly Cloudy"
            )

            precip_type = st.selectbox(
                "Precip Type",
                [
                    "rain",
                    "snow"
                ]
            )

            apparent_temperature = st.number_input(
                "Apparent Temperature (C)",
                value=20.0
            )

            humidity = st.number_input(
                "Humidity",
                min_value=0.0,
                max_value=1.0,
                value=0.70
            )

            wind_speed = st.number_input(
                "Wind Speed (km/h)",
                min_value=0.0,
                value=10.0
            )

        with col2:

            wind_bearing = st.number_input(
                "Wind Bearing (degrees)",
                min_value=0.0,
                max_value=360.0,
                value=180.0
            )

            visibility = st.number_input(
                "Visibility (km)",
                min_value=0.0,
                value=10.0
            )

            pressure = st.number_input(
                "Pressure (millibars)",
                min_value=0.0,
                value=1013.0
            )

            year = st.number_input(
                "Year",
                min_value=2000,
                max_value=2100,
                value=2015,
                step=1
            )

            month = st.number_input(
                "Month",
                min_value=1,
                max_value=12,
                value=6,
                step=1
            )

        col3, col4 = st.columns(2)

        with col3:

            hour = st.number_input(
                "Hour",
                min_value=0,
                max_value=23,
                value=12,
                step=1
            )

            day_of_week = st.selectbox(
                "DayOfWeek",
                [
                    "Monday",
                    "Tuesday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                    "Saturday",
                    "Sunday"
                ]
            )

        with col4:

            season = st.selectbox(
                "Season",
                [
                    "Winter",
                    "Spring",
                    "Summer",
                    "Autumn"
                ]
            )

            wind_chill_indicator = st.selectbox(
                "Wind_Chill_Indicator",
                [
                    "No",
                    "Yes"
                ]
            )

        st.markdown("---")

        # -------------------------------------------------
        # PREDICT BUTTON
        # -------------------------------------------------

        if st.button(
            "🔮 Predict Temperature",
            use_container_width=True
        ):

            try:

                # -------------------------------------------------
                # CREATE INPUT DATAFRAME
                # -------------------------------------------------

                weather_input = pd.DataFrame(
                    {
                        "Summary": [summary],
                        "Precip Type": [precip_type],
                        "Apparent Temperature (C)": [
                            apparent_temperature
                        ],
                        "Humidity": [humidity],
                        "Wind Speed (km/h)": [
                            wind_speed
                        ],
                        "Wind Bearing (degrees)": [
                            wind_bearing
                        ],
                        "Visibility (km)": [
                            visibility
                        ],
                        "Pressure (millibars)": [
                            pressure
                        ],
                        "Year": [year],
                        "Month": [month],
                        "Hour": [hour],
                        "DayOfWeek": [day_of_week],
                        "Season": [season],
                        "Wind_Chill_Indicator": [
                            wind_chill_indicator
                        ]
                    }
                )

                # -------------------------------------------------
                # APPLY SAVED PREPROCESSING
                # -------------------------------------------------

                weather_encoded = (
                    weather_preprocessor.transform(
                        weather_input
                    )
                )

                # Convert sparse matrix if required
                if hasattr(
                    weather_encoded,
                    "toarray"
                ):

                    weather_encoded = (
                        weather_encoded.toarray()
                    )

                # -------------------------------------------------
                # BASE MODEL PREDICTIONS
                # -------------------------------------------------

                pred_lr = weather_lr_model.predict(
                    weather_encoded
                )[0]

                pred_cb = weather_cb_model.predict(
                    weather_encoded
                )[0]

                pred_gb = weather_gb_model.predict(
                    weather_encoded
                )[0]

                # -------------------------------------------------
                # META-LEARNER INPUT
                # -------------------------------------------------

                meta_input = np.array(
                    [
                        [
                            pred_lr,
                            pred_cb,
                            pred_gb
                        ]
                    ]
                )

                # -------------------------------------------------
                # FINAL HYBRID PREDICTION
                # -------------------------------------------------

                final_prediction = (
                    weather_meta_model.predict(
                        meta_input
                    )[0]
                )

                # -------------------------------------------------
                # DISPLAY RESULT
                # -------------------------------------------------

                st.markdown("---")

                st.subheader(
                    "⭐ Predicted Temperature"
                )

                st.metric(
                    "Temperature (°C)",
                    f"{final_prediction:.2f} °C"
                )

                st.markdown("---")

                st.subheader(
                    "🧩 Base Model Predictions"
                )

                prediction_table = pd.DataFrame(
                    {
                        "Model": [
                            "Linear Regression",
                            "CatBoost",
                            "Gradient Boosting",
                            "Proposed Hybrid"
                        ],
                        "Prediction (°C)": [
                            pred_lr,
                            pred_cb,
                            pred_gb,
                            final_prediction
                        ]
                    }
                )

                st.dataframe(
                    prediction_table,
                    use_container_width=True,
                    hide_index=True
                )

                st.success(
                    """
                    Prediction generated successfully using the
                    saved Weather hybrid model.
                    """
                )

            except Exception as e:

                st.error(
                    f"Prediction failed: {e}"
                )

        # =====================================================
    # ENERGY
    # =====================================================

    elif prediction_domain == "⚡ Energy":

        st.header("⚡ Energy Consumption Prediction")

        st.markdown(
            """
            Enter the building, environmental, and operational
            features below to predict Energy Consumption using
            the proposed hybrid model.
            """
        )

        # -------------------------------------------------
        # LOAD SAVED ARTIFACTS
        # -------------------------------------------------

        energy_model_dir = (
            ARTIFACTS_DIR
            / "models"
            / "energy"
        )

        energy_preprocessor_path = (
            ARTIFACTS_DIR
            / "preprocessors"
            / "energy"
            / "energy_preprocessor_c14.joblib"
        )

        energy_lr_path = (
            energy_model_dir
            / "lr_final_lrcbgb_c14.joblib"
        )

        energy_cb_path = (
            energy_model_dir
            / "cb_final_lrcbgb_c14.joblib"
        )

        energy_gb_path = (
            energy_model_dir
            / "gb_final_lrcbgb_c14.joblib"
        )

        energy_meta_path = (
            energy_model_dir
            / "meta_lr_lrcbgb_c14.joblib"
        )

        required_files = [
            energy_preprocessor_path,
            energy_lr_path,
            energy_cb_path,
            energy_gb_path,
            energy_meta_path
        ]

        missing_files = [
            file.name
            for file in required_files
            if not file.exists()
        ]

        if missing_files:

            st.error(
                "Missing Energy model artifacts: "
                + ", ".join(missing_files)
            )

            st.stop()

        # -------------------------------------------------
        # LOAD ARTIFACTS
        # -------------------------------------------------

        import joblib

        energy_preprocessor = joblib.load(
            energy_preprocessor_path
        )

        energy_lr_model = joblib.load(
            energy_lr_path
        )

        energy_cb_model = joblib.load(
            energy_cb_path
        )

        energy_gb_model = joblib.load(
            energy_gb_path
        )

        energy_meta_model = joblib.load(
            energy_meta_path
        )

        # -------------------------------------------------
        # INPUT FEATURES
        # -------------------------------------------------

        st.subheader("📝 Input Features")

        col1, col2 = st.columns(2)

        with col1:

            temperature = st.number_input(
                "Temperature",
                value=25.0
            )

            humidity = st.number_input(
                "Humidity",
                min_value=0.0,
                max_value=100.0,
                value=60.0
            )

            square_footage = st.number_input(
                "SquareFootage",
                min_value=0.0,
                value=2000.0
            )

            occupancy = st.number_input(
                "Occupancy",
                min_value=0.0,
                value=20.0
            )

            hvac_usage = st.selectbox(
                "HVACUsage",
                [
                    "Low",
                    "Medium",
                    "High"
                ]
            )

            lighting_usage = st.selectbox(
                "LightingUsage",
                [
                    "Low",
                    "Medium",
                    "High"
                ]
            )

            renewable_energy = st.number_input(
                "RenewableEnergy",
                min_value=0.0,
                value=10.0
            )

        with col2:

            day_of_week = st.selectbox(
                "DayOfWeek",
                [
                    "Monday",
                    "Tuesday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                    "Saturday",
                    "Sunday"
                ]
            )

            holiday = st.selectbox(
                "Holiday",
                [
                    "No",
                    "Yes"
                ]
            )

            hour = st.number_input(
                "Hour",
                min_value=0,
                max_value=23,
                value=12,
                step=1
            )

            month = st.number_input(
                "Month",
                min_value=1,
                max_value=12,
                value=6,
                step=1
            )

            weekend = st.selectbox(
                "Weekend",
                [
                    "No",
                    "Yes"
                ]
            )

            hvac_status = st.number_input(
                "HVAC_Status",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                step=1.0
            )

            lighting_status = st.number_input(
                "Lighting_Status",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                step=1.0
            )

        st.markdown("---")

        # -------------------------------------------------
        # PREDICT BUTTON
        # -------------------------------------------------

        if st.button(
            "🔮 Predict Energy Consumption",
            use_container_width=True
        ):

            try:

                # -------------------------------------------------
                # CREATE INPUT DATAFRAME
                # -------------------------------------------------

                energy_input = pd.DataFrame(
                    {
                        "Temperature": [temperature],
                        "Humidity": [humidity],
                        "SquareFootage": [square_footage],
                        "Occupancy": [occupancy],
                        "HVACUsage": [hvac_usage],
                        "LightingUsage": [lighting_usage],
                        "RenewableEnergy": [renewable_energy],
                        "DayOfWeek": [day_of_week],
                        "Holiday": [holiday],
                        "Hour": [hour],
                        "Month": [month],
                        "Weekend": [weekend],
                        "HVAC_Status": [hvac_status],
                        "Lighting_Status": [lighting_status]
                    }
                )

                # -------------------------------------------------
                # ENSURE EXACT TRAINING DATA TYPES
                # -------------------------------------------------

                numeric_columns = [
                    "Temperature",
                    "Humidity",
                    "SquareFootage",
                    "Occupancy",
                    "RenewableEnergy",
                    "Hour",
                    "Month",
                    "HVAC_Status",
                    "Lighting_Status"
                ]

                categorical_columns = [
                    "HVACUsage",
                    "LightingUsage",
                    "DayOfWeek",
                    "Holiday",
                    "Weekend"
                ]

                for column in numeric_columns:

                    energy_input[column] = pd.to_numeric(
                        energy_input[column],
                        errors="coerce"
                    )

                for column in categorical_columns:

                    energy_input[column] = (
                        energy_input[column]
                        .astype(str)
                    )

                # -------------------------------------------------
                # CHECK NUMERIC VALUES
                # -------------------------------------------------

                if energy_input[
                    numeric_columns
                ].isnull().any().any():

                    st.error(
                        "Please enter valid numeric values "
                        "for all numerical Energy features."
                    )

                    st.stop()

                # -------------------------------------------------
                # APPLY SAVED PREPROCESSOR
                # -------------------------------------------------

                energy_encoded = (
                    energy_preprocessor.transform(
                        energy_input
                    )
                )

                # Convert sparse matrix if required
                if hasattr(
                    energy_encoded,
                    "toarray"
                ):

                    energy_encoded = (
                        energy_encoded.toarray()
                    )

                # -------------------------------------------------
                # BASE MODEL PREDICTIONS
                # -------------------------------------------------

                pred_lr = energy_lr_model.predict(
                    energy_encoded
                )[0]

                pred_cb = energy_cb_model.predict(
                    energy_encoded
                )[0]

                pred_gb = energy_gb_model.predict(
                    energy_encoded
                )[0]

                # -------------------------------------------------
                # META-LEARNER INPUT
                # -------------------------------------------------

                meta_input = np.array(
                    [
                        [
                            pred_lr,
                            pred_cb,
                            pred_gb
                        ]
                    ],
                    dtype=float
                )

                # -------------------------------------------------
                # FINAL HYBRID PREDICTION
                # -------------------------------------------------

                final_prediction = (
                    energy_meta_model.predict(
                        meta_input
                    )[0]
                )

                # -------------------------------------------------
                # DISPLAY RESULT
                # -------------------------------------------------

                st.markdown("---")

                st.subheader(
                    "⭐ Predicted Energy Consumption"
                )

                st.metric(
                    "Energy Consumption",
                    f"{final_prediction:.2f}"
                )

                st.markdown("---")

                st.subheader(
                    "🧩 Base Model Predictions"
                )

                prediction_table = pd.DataFrame(
                    {
                        "Model": [
                            "Linear Regression",
                            "CatBoost",
                            "Gradient Boosting",
                            "Proposed Hybrid"
                        ],
                        "Prediction": [
                            pred_lr,
                            pred_cb,
                            pred_gb,
                            final_prediction
                        ]
                    }
                )

                st.dataframe(
                    prediction_table,
                    use_container_width=True,
                    hide_index=True
                )

                st.success(
                    """
                    Prediction generated successfully using the
                    saved Energy hybrid model.
                    """
                )

            except Exception as e:

                st.error(
                    f"Prediction failed: {e}"
                )


# ============================================================
# EXPLAINABLE AI
# ============================================================

elif page == "🔍 Explainable AI":

    st.title("🔍 Explainable AI")

    shap_section = st.radio(
        "Select Explainability Section",
        [
            "🌍 Global SHAP",
            "🔎 Prediction Explanation"
        ],
        horizontal=True
    )

    # ========================================================
    # GLOBAL SHAP
    # ========================================================

    if shap_section == "🌍 Global SHAP":

        st.subheader("🌍 Global SHAP Explainability")

        st.write(
            """
            SHAP (SHapley Additive exPlanations) is used to understand
            how input features contribute to the predictions of the
            proposed hybrid models.
            """
        )

        shap_domain = st.selectbox(
            "Select Domain",
            [
                "🌫️ Air Quality",
                "🌦️ Weather",
                "⚡ Energy"
            ]
        )

        # ----------------------------------------------------
        # AIR QUALITY
        # ----------------------------------------------------

        if shap_domain == "🌫️ Air Quality":

            shap_dir = (
                ARTIFACTS_DIR
                / "shap"
                / "Air_Quality"
            )

            st.subheader("🌫️ Air Quality — SHAP Analysis")

            importance_file = (
                shap_dir
                / "Tables"
                / "air_proposed_hybrid_shap_importance.xlsx"
            )

            values_file = (
                shap_dir
                / "Tables"
                / "air_proposed_hybrid_shap_values.xlsx"
            )

            predictions_file = (
                shap_dir
                / "Tables"
                / "air_proposed_hybrid_shap_predictions.xlsx"
            )

            bar_image = (
                shap_dir
                / "Images"
                / "air_proposed_hybrid_shap_bar.png"
            )

            beeswarm_image = (
                shap_dir
                / "Images"
                / "air_proposed_hybrid_shap_beeswarm.png"
            )

            waterfall_image = (
                shap_dir
                / "Images"
                / "air_proposed_hybrid_shap_waterfall.png"
            )

        # ----------------------------------------------------
        # WEATHER
        # ----------------------------------------------------

        elif shap_domain == "🌦️ Weather":

            shap_dir = (
                ARTIFACTS_DIR
                / "shap"
                / "Weather"
            )

            st.subheader("🌦️ Weather — SHAP Analysis")

            importance_file = (
                shap_dir
                / "Tables"
                / "weather_proposed_hybrid_shap_importance.xlsx"
            )

            values_file = (
                shap_dir
                / "Tables"
                / "weather_proposed_hybrid_shap_values.xlsx"
            )

            predictions_file = (
                shap_dir
                / "Tables"
                / "weather_proposed_hybrid_shap_predictions.xlsx"
            )

            bar_image = (
                shap_dir
                / "Images"
                / "weather_proposed_hybrid_shap_bar.png"
            )

            beeswarm_image = (
                shap_dir
                / "Images"
                / "weather_proposed_hybrid_shap_beeswarm.png"
            )

            waterfall_image = (
                shap_dir
                / "Images"
                / "weather_proposed_hybrid_shap_waterfall.png"
            )

        # ----------------------------------------------------
        # ENERGY
        # ----------------------------------------------------

        else:

            shap_dir = (
                ARTIFACTS_DIR
                / "shap"
                / "Energy"
            )

            st.subheader("⚡ Energy — SHAP Analysis")

            importance_file = (
                shap_dir
                / "Tables"
                / "energy_proposed_hybrid_shap_importance.xlsx"
            )

            values_file = (
                shap_dir
                / "Tables"
                / "energy_proposed_hybrid_shap_values.xlsx"
            )

            predictions_file = (
                shap_dir
                / "Tables"
                / "energy_proposed_hybrid_shap_predictions.xlsx"
            )

            bar_image = (
                shap_dir
                / "Images"
                / "energy_proposed_hybrid_shap_bar.png"
            )

            beeswarm_image = (
                shap_dir
                / "Images"
                / "energy_proposed_hybrid_shap_beeswarm.png"
            )

            waterfall_image = (
                shap_dir
                / "Images"
                / "energy_proposed_hybrid_shap_waterfall.png"
            )

        # ----------------------------------------------------
        # SHAP IMPORTANCE TABLE
        # ----------------------------------------------------

        st.markdown("### 📋 SHAP Feature Importance")

        if importance_file.exists():

            importance_df = pd.read_excel(
                importance_file
            )

            st.dataframe(
                importance_df,
                use_container_width=True
            )

        else:

            st.warning(
                "SHAP importance file not found."
            )

        # ----------------------------------------------------
        # SHAP BAR
        # ----------------------------------------------------

        st.markdown("### 📊 Feature Importance")

        if bar_image.exists():

            st.image(
                str(bar_image),
                use_container_width=True
            )

        # ----------------------------------------------------
        # SHAP BEESWARM
        # ----------------------------------------------------

        st.markdown("### 🐝 SHAP Beeswarm Plot")

        if beeswarm_image.exists():

            st.image(
                str(beeswarm_image),
                use_container_width=True
            )

        # ----------------------------------------------------
        # SHAP WATERFALL
        # ----------------------------------------------------

        st.markdown("### 💧 SHAP Waterfall Plot")

        if waterfall_image.exists():

            st.image(
                str(waterfall_image),
                use_container_width=True
            )

        # ----------------------------------------------------
        # ADDITIONAL TABLES
        # ----------------------------------------------------

        with st.expander("📄 SHAP Values"):

            if values_file.exists():

                shap_values_df = pd.read_excel(
                    values_file
                )

                st.dataframe(
                    shap_values_df,
                    use_container_width=True
                )

        with st.expander("🔮 SHAP Predictions"):

            if predictions_file.exists():

                predictions_df = pd.read_excel(
                    predictions_file
                )

                st.dataframe(
                    predictions_df,
                    use_container_width=True
                )

        st.info(
            "The global SHAP analysis summarizes feature contributions "
            "across the selected explanation dataset."
        )
    # ============================================================
    # LIVE PREDICTION EXPLANATION
    # ============================================================

    elif shap_section == "🔎 Prediction Explanation":

        st.subheader("🔎 Prediction Explanation")

        st.write(
            """
            This section explains an individual prediction using SHAP.
            Enter a new observation and the dashboard identifies the
            features that contribute most strongly to the prediction.
            """
        )

        explanation_domain = st.selectbox(
            "Select Domain",
            [
                "🌫️ Air Quality",
                "🌦️ Weather",
                "⚡ Energy"
            ],
            key="explanation_domain"
        )

        # ========================================================
        # AIR QUALITY
        # ========================================================

        if explanation_domain == "🌫️ Air Quality":

            st.info(
                "Enter Air Quality values to generate an "
                "instance-level explanation."
            )

            col1, col2 = st.columns(2)

            with col1:

                pm25 = st.number_input(
                    "PM2.5",
                    value=50.0
                )

                pm10 = st.number_input(
                    "PM10",
                    value=80.0
                )

                nox = st.number_input(
                    "NOx",
                    value=30.0
                )

                no2 = st.number_input(
                    "NO2",
                    value=20.0
                )

                so2 = st.number_input(
                    "SO2",
                    value=10.0
                )

                vocs = st.number_input(
                    "VOCs",
                    value=15.0
                )

                co = st.number_input(
                    "CO",
                    value=1.0
                )

            with col2:

                co2 = st.number_input(
                    "CO2",
                    value=400.0
                )

                ch4 = st.number_input(
                    "CH4",
                    value=2.0
                )

                temperature = st.number_input(
                    "Temperature",
                    value=25.0
                )

                humidity = st.number_input(
                    "Humidity",
                    value=60.0
                )

                wind_direction = st.number_input(
                    "Wind Direction",
                    value=180.0
                )

                location_type = st.text_input(
                    "Location Type",
                    value="Urban"
                )

                source_label = st.text_input(
                    "Source Label",
                    value="Sensor"
                )

            if st.button(
                "🔍 Explain Air Quality Prediction",
                key="explain_air"
            ):

                try:

                    # --------------------------------------------
                    # Load models
                    # --------------------------------------------

                    preprocessor = joblib.load(
                        ARTIFACTS_DIR
                        / "preprocessors"
                        / "air"
                        / "preprocessor_air_hybrid.joblib"
                    )

                    lr_model = joblib.load(
                        ARTIFACTS_DIR
                        / "models"
                        / "air"
                        / "lr_final_air.joblib"
                    )

                    cb_model = joblib.load(
                        ARTIFACTS_DIR
                        / "models"
                        / "air"
                        / "cb_final_air.joblib"
                    )

                    gb_model = joblib.load(
                        ARTIFACTS_DIR
                        / "models"
                        / "air"
                        / "gb_final_air.joblib"
                    )

                    meta_model = joblib.load(
                        ARTIFACTS_DIR
                        / "models"
                        / "air"
                        / "meta_lr_air.joblib"
                    )

                    # --------------------------------------------
                    # Create input
                    # --------------------------------------------

                    air_input = pd.DataFrame([{
                        "PM2.5": pm25,
                        "PM10": pm10,
                        "NOx": nox,
                        "NO2": no2,
                        "SO2": so2,
                        "VOCs": vocs,
                        "CO": co,
                        "CO2": co2,
                        "CH4": ch4,
                        "Temperature": temperature,
                        "Humidity": humidity,
                        "Wind_Direction": wind_direction,
                        "Location_Type": location_type,
                        "Source_Label": source_label
                    }])

                    # --------------------------------------------
                    # Feature engineering
                    # --------------------------------------------

                    air_input["PM_Ratio"] = (
                        air_input["PM2.5"]
                        / air_input["PM10"].replace(0, np.nan)
                    )

                    air_input["NOx_NO2_Ratio"] = (
                        air_input["NOx"]
                        / air_input["NO2"].replace(0, np.nan)
                    )

                    air_input["CO_CO2_Ratio"] = (
                        air_input["CO"]
                        / air_input["CO2"].replace(0, np.nan)
                    )

                    air_input = air_input.replace(
                        [np.inf, -np.inf],
                        np.nan
                    ).fillna(0)

                    # --------------------------------------------
                    # Transform
                    # --------------------------------------------

                    X_transformed = preprocessor.transform(
                        air_input
                    )

                    if hasattr(
                        X_transformed,
                        "toarray"
                    ):
                        X_transformed = X_transformed.toarray()

                    # --------------------------------------------
                    # Hybrid prediction
                    # --------------------------------------------

                    pred_lr = lr_model.predict(
                        X_transformed
                    )[0]

                    pred_cb = cb_model.predict(
                        X_transformed
                    )[0]

                    pred_gb = gb_model.predict(
                        X_transformed
                    )[0]

                    meta_input = np.array([[
                        pred_lr,
                        pred_cb,
                        pred_gb
                    ]])

                    final_prediction = meta_model.predict(
                        meta_input
                    )[0]

                    # --------------------------------------------
                    # Display prediction
                    # --------------------------------------------

                    st.success(
                        f"Predicted Pollution Score: "
                        f"{final_prediction:.4f}"
                    )

                    # --------------------------------------------
                    # SHAP wrapper
                    # --------------------------------------------

                    def hybrid_predict(X):

                        pred_lr = lr_model.predict(X)

                        pred_cb = cb_model.predict(X)

                        pred_gb = gb_model.predict(X)

                        meta_input = np.column_stack([
                            pred_lr,
                            pred_cb,
                            pred_gb
                        ])

                        return meta_model.predict(
                            meta_input
                        )

                    # --------------------------------------------
                    # SHAP background
                    # --------------------------------------------

                    background = X_transformed

                    # --------------------------------------------
                    # SHAP explainer
                    # --------------------------------------------

                    explainer = shap.PermutationExplainer(
                        hybrid_predict,
                        background
                    )

                    shap_result = explainer(
                        X_transformed
                    )

                    shap_values = np.asarray(
                        shap_result.values
                    )

                    if shap_values.ndim == 3:

                        shap_values = shap_values[:, :, 0]

                    # --------------------------------------------
                    # Feature names
                    # --------------------------------------------

                    feature_names = (
                        preprocessor
                        .get_feature_names_out()
                    )

                    contribution_values = (
                        shap_values[0]
                    )

                    explanation_df = pd.DataFrame({
                        "Feature": feature_names,
                        "SHAP_Value": contribution_values,
                        "Absolute_SHAP": np.abs(
                            contribution_values
                        )
                    })

                    explanation_df = (
                        explanation_df
                        .sort_values(
                            "Absolute_SHAP",
                            ascending=False
                        )
                        .reset_index(drop=True)
                    )

                    # --------------------------------------------
                    # Display explanation
                    # --------------------------------------------

                    st.markdown(
                        "### 🔍 Top Feature Contributions"
                    )

                    st.dataframe(
                        explanation_df.head(10),
                        use_container_width=True
                    )

                    # --------------------------------------------
                    # Positive / Negative contributions
                    # --------------------------------------------

                    positive = explanation_df[
                        explanation_df["SHAP_Value"] > 0
                    ].head(5)

                    negative = explanation_df[
                        explanation_df["SHAP_Value"] < 0
                    ].head(5)

                    col1, col2 = st.columns(2)

                    with col1:

                        st.markdown(
                            "### 🔺 Features Increasing Prediction"
                        )

                        if len(positive) > 0:

                            st.dataframe(
                                positive[
                                    [
                                        "Feature",
                                        "SHAP_Value"
                                    ]
                                ],
                                use_container_width=True
                            )

                        else:

                            st.write(
                                "No positive contributions."
                            )

                    with col2:

                        st.markdown(
                            "### 🔻 Features Decreasing Prediction"
                        )

                        if len(negative) > 0:

                            st.dataframe(
                                negative[
                                    [
                                        "Feature",
                                        "SHAP_Value"
                                    ]
                                ],
                                use_container_width=True
                            )

                        else:

                            st.write(
                                "No negative contributions."
                            )

                    st.info(
                        """
                        Positive SHAP values indicate features that
                        push the prediction higher, while negative SHAP
                        values indicate features that push the prediction
                        lower relative to the model's expected prediction.
                        """
                    )

                except Exception as e:

                    st.error(
                        f"Prediction explanation failed: {e}"
                    )

            # ========================================================
            # WEATHER
            # ========================================================

        elif explanation_domain == "🌦️ Weather":

            st.info(
                "Enter Weather values to generate an "
                "instance-level SHAP explanation."
            )

            col1, col2 = st.columns(2)

            with col1:

                summary = st.text_input(
                    "Summary",
                    value="Partly Cloudy",
                    key="shap_weather_summary"
                )

                precip_type = st.selectbox(
                    "Precip Type",
                    ["rain", "snow"],
                    key="shap_weather_precip"
                )

                apparent_temperature = st.number_input(
                    "Apparent Temperature (C)",
                    value=20.0,
                    key="shap_weather_apparent_temp"
                )

                humidity = st.number_input(
                    "Humidity",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.70,
                    key="shap_weather_humidity"
                )

                wind_speed = st.number_input(
                    "Wind Speed (km/h)",
                    value=10.0,
                    key="shap_weather_wind_speed"
                )

                wind_bearing = st.number_input(
                    "Wind Bearing (degrees)",
                    min_value=0.0,
                    max_value=360.0,
                    value=180.0,
                    key="shap_weather_wind_bearing"
                )

                visibility = st.number_input(
                    "Visibility (km)",
                    value=10.0,
                    key="shap_weather_visibility"
                )

            with col2:

                pressure = st.number_input(
                    "Pressure (millibars)",
                    value=1010.0,
                    key="shap_weather_pressure"
                )

                year = st.number_input(
                    "Year",
                    min_value=2000,
                    max_value=2100,
                    value=2015,
                    step=1,
                    key="shap_weather_year"
                )

                month = st.number_input(
                    "Month",
                    min_value=1,
                    max_value=12,
                    value=6,
                    step=1,
                    key="shap_weather_month"
                )

                hour = st.number_input(
                    "Hour",
                    min_value=0,
                    max_value=23,
                    value=12,
                    step=1,
                    key="shap_weather_hour"
                )

                day_of_week = st.selectbox(
                    "DayOfWeek",
                    [
                        "Monday",
                        "Tuesday",
                        "Wednesday",
                        "Thursday",
                        "Friday",
                        "Saturday",
                        "Sunday"
                    ],
                    key="shap_weather_day"
                )

                season = st.selectbox(
                    "Season",
                    [
                        "Winter",
                        "Spring",
                        "Summer",
                        "Autumn"
                    ],
                    key="shap_weather_season"
                )

                wind_chill = st.selectbox(
                    "Wind_Chill_Indicator",
                    [
                        "No",
                        "Yes"
                    ],
                    key="shap_weather_wind_chill"
                )

            if st.button(
                "🔍 Explain Weather Prediction",
                key="explain_weather"
            ):

                try:

                    # --------------------------------------------
                    # Load saved models
                    # --------------------------------------------

                    preprocessor = joblib.load(
                        ARTIFACTS_DIR
                        / "preprocessors"
                        / "weather"
                        / "preprocessor_weather_hybrid.joblib"
                    )

                    lr_model = joblib.load(
                        ARTIFACTS_DIR
                        / "models"
                        / "weather"
                        / "lr_final_weather.joblib"
                    )

                    cb_model = joblib.load(
                        ARTIFACTS_DIR
                        / "models"
                        / "weather"
                        / "cb_final_weather.joblib"
                    )

                    gb_model = joblib.load(
                        ARTIFACTS_DIR
                        / "models"
                        / "weather"
                        / "gb_final_weather.joblib"
                    )

                    meta_model = joblib.load(
                        ARTIFACTS_DIR
                        / "models"
                        / "weather"
                        / "meta_lr_weather.joblib"
                    )

                    # --------------------------------------------
                    # Exact Weather model input
                    # --------------------------------------------

                    weather_input = pd.DataFrame([{
                        "Summary": summary,
                        "Precip Type": precip_type,
                        "Apparent Temperature (C)": apparent_temperature,
                        "Humidity": humidity,
                        "Wind Speed (km/h)": wind_speed,
                        "Wind Bearing (degrees)": wind_bearing,
                        "Visibility (km)": visibility,
                        "Pressure (millibars)": pressure,
                        "Year": year,
                        "Month": month,
                        "Hour": hour,
                        "DayOfWeek": day_of_week,
                        "Season": season,
                        "Wind_Chill_Indicator": wind_chill
                    }])

                    # --------------------------------------------
                    # Correct data types
                    # --------------------------------------------

                    numeric_columns = [
                        "Apparent Temperature (C)",
                        "Humidity",
                        "Wind Speed (km/h)",
                        "Wind Bearing (degrees)",
                        "Visibility (km)",
                        "Pressure (millibars)",
                        "Year",
                        "Month",
                        "Hour"
                    ]

                    categorical_columns = [
                        "Summary",
                        "Precip Type",
                        "DayOfWeek",
                        "Season",
                        "Wind_Chill_Indicator"
                    ]

                    for col in numeric_columns:

                        weather_input[col] = pd.to_numeric(
                            weather_input[col],
                            errors="coerce"
                        )

                    for col in categorical_columns:

                        weather_input[col] = (
                            weather_input[col]
                            .astype(str)
                        )

                    # --------------------------------------------
                    # Transform input
                    # --------------------------------------------

                    X_transformed = preprocessor.transform(
                        weather_input
                    )

                    if hasattr(
                        X_transformed,
                        "toarray"
                    ):
                        X_transformed = X_transformed.toarray()

                    # --------------------------------------------
                    # Hybrid prediction
                    # --------------------------------------------

                    pred_lr = lr_model.predict(
                        X_transformed
                    )[0]

                    pred_cb = cb_model.predict(
                        X_transformed
                    )[0]

                    pred_gb = gb_model.predict(
                        X_transformed
                    )[0]

                    meta_input = np.array([[
                        pred_lr,
                        pred_cb,
                        pred_gb
                    ]], dtype=float)

                    final_prediction = meta_model.predict(
                        meta_input
                    )[0]

                    st.success(
                        f"Predicted Temperature: "
                        f"{final_prediction:.4f} °C"
                    )

                    # --------------------------------------------
                    # Hybrid prediction wrapper for SHAP
                    # --------------------------------------------

                    def hybrid_predict(X):

                        pred_lr = lr_model.predict(X)

                        pred_cb = cb_model.predict(X)

                        pred_gb = gb_model.predict(X)

                        meta_input = np.column_stack([
                            pred_lr,
                            pred_cb,
                            pred_gb
                        ])

                        return meta_model.predict(
                            meta_input
                        )

                    # --------------------------------------------
                    # SHAP background
                    # --------------------------------------------

                    background = X_transformed

                    # --------------------------------------------
                    # Permutation SHAP
                    # --------------------------------------------

                    explainer = shap.PermutationExplainer(
                        hybrid_predict,
                        background
                    )

                    shap_result = explainer(
                        X_transformed
                    )

                    shap_values = np.asarray(
                        shap_result.values
                    )

                    if shap_values.ndim == 3:

                        shap_values = shap_values[:, :, 0]

                    # --------------------------------------------
                    # Feature names
                    # --------------------------------------------

                    feature_names = (
                        preprocessor
                        .get_feature_names_out()
                    )

                    contribution_values = (
                        shap_values[0]
                    )

                    explanation_df = pd.DataFrame({
                        "Feature": feature_names,
                        "SHAP_Value": contribution_values,
                        "Absolute_SHAP": np.abs(
                            contribution_values
                        )
                    })

                    explanation_df = (
                        explanation_df
                        .sort_values(
                            "Absolute_SHAP",
                            ascending=False
                        )
                        .reset_index(drop=True)
                    )

                    # --------------------------------------------
                    # Display top contributions
                    # --------------------------------------------

                    st.markdown(
                        "### 🔍 Top Feature Contributions"
                    )

                    st.dataframe(
                        explanation_df.head(10),
                        use_container_width=True
                    )

                    # --------------------------------------------
                    # Positive / negative contributions
                    # --------------------------------------------

                    positive = explanation_df[
                        explanation_df["SHAP_Value"] > 0
                    ].head(5)

                    negative = explanation_df[
                        explanation_df["SHAP_Value"] < 0
                    ].head(5)

                    col1, col2 = st.columns(2)

                    with col1:

                        st.markdown(
                            "### 🔺 Increasing Temperature"
                        )

                        if len(positive) > 0:

                            st.dataframe(
                                positive[
                                    [
                                        "Feature",
                                        "SHAP_Value"
                                    ]
                                ],
                                use_container_width=True
                            )

                        else:

                            st.write(
                                "No positive contributions."
                            )

                    with col2:

                        st.markdown(
                            "### 🔻 Decreasing Temperature"
                        )

                        if len(negative) > 0:

                            st.dataframe(
                                negative[
                                    [
                                        "Feature",
                                        "SHAP_Value"
                                    ]
                                ],
                                use_container_width=True
                            )

                        else:

                            st.write(
                                "No negative contributions."
                            )

                    st.info(
                        """
                        Positive SHAP values push the predicted temperature
                        higher, while negative SHAP values push it lower
                        relative to the model's expected prediction.
                        """
                    )

                except Exception as e:

                    st.error(
                        f"Weather prediction explanation failed: {e}"
                    )

            # ========================================================
    # ENERGY
    # ========================================================

        else:

            st.info(
                "Enter Energy values to generate an "
                "instance-level SHAP explanation."
            )

        col1, col2 = st.columns(2)

        with col1:

            temperature = st.number_input(
                "Temperature",
                value=25.0,
                key="shap_energy_temperature"
            )

            humidity = st.number_input(
                "Humidity",
                value=0.60,
                key="shap_energy_humidity"
            )

            square_footage = st.number_input(
                "SquareFootage",
                value=2000.0,
                key="shap_energy_squarefootage"
            )

            occupancy = st.number_input(
                "Occupancy",
                value=20.0,
                key="shap_energy_occupancy"
            )

            hvac_usage = st.selectbox(
                "HVACUsage",
                ["Low", "Medium", "High"],
                key="shap_energy_hvac"
            )

            lighting_usage = st.selectbox(
                "LightingUsage",
                ["Low", "Medium", "High"],
                key="shap_energy_lighting"
            )

            renewable_energy = st.number_input(
                "RenewableEnergy",
                value=10.0,
                key="shap_energy_renewable"
            )

        with col2:

            day_of_week = st.selectbox(
                "DayOfWeek",
                [
                    "Monday",
                    "Tuesday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                    "Saturday",
                    "Sunday"
                ],
                key="shap_energy_day"
            )

            holiday = st.selectbox(
                "Holiday",
                ["No", "Yes"],
                key="shap_energy_holiday"
            )

            hour = st.number_input(
                "Hour",
                min_value=0,
                max_value=23,
                value=12,
                step=1,
                key="shap_energy_hour"
            )

            month = st.number_input(
                "Month",
                min_value=1,
                max_value=12,
                value=6,
                step=1,
                key="shap_energy_month"
            )

            weekend = st.selectbox(
                "Weekend",
                ["No", "Yes"],
                key="shap_energy_weekend"
            )

            hvac_status = st.number_input(
                "HVAC_Status",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                key="shap_energy_hvac_status"
            )

            lighting_status = st.number_input(
                "Lighting_Status",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                key="shap_energy_lighting_status"
            )

        if st.button(
            "🔍 Explain Energy Prediction",
            key="explain_energy"
        ):

            try:

                # --------------------------------------------
                # Load saved models
                # --------------------------------------------

                preprocessor = joblib.load(
                    ARTIFACTS_DIR
                    / "preprocessors"
                    / "energy"
                    / "energy_preprocessor_c14.joblib"
                )

                lr_model = joblib.load(
                    ARTIFACTS_DIR
                    / "models"
                    / "energy"
                    / "lr_final_lrcbgb_c14.joblib"
                )

                cb_model = joblib.load(
                    ARTIFACTS_DIR
                    / "models"
                    / "energy"
                    / "cb_final_lrcbgb_c14.joblib"
                )

                gb_model = joblib.load(
                    ARTIFACTS_DIR
                    / "models"
                    / "energy"
                    / "gb_final_lrcbgb_c14.joblib"
                )

                meta_model = joblib.load(
                    ARTIFACTS_DIR
                    / "models"
                    / "energy"
                    / "meta_lr_lrcbgb_c14.joblib"
                )

                # --------------------------------------------
                # Exact 14 Energy input features
                # --------------------------------------------

                energy_input = pd.DataFrame([{
                    "Temperature": temperature,
                    "Humidity": humidity,
                    "SquareFootage": square_footage,
                    "Occupancy": occupancy,
                    "HVACUsage": hvac_usage,
                    "LightingUsage": lighting_usage,
                    "RenewableEnergy": renewable_energy,
                    "DayOfWeek": day_of_week,
                    "Holiday": holiday,
                    "Hour": hour,
                    "Month": month,
                    "Weekend": weekend,
                    "HVAC_Status": hvac_status,
                    "Lighting_Status": lighting_status
                }])

                # --------------------------------------------
                # Correct data types
                # --------------------------------------------

                numerical_columns = [
                    "Temperature",
                    "Humidity",
                    "SquareFootage",
                    "Occupancy",
                    "RenewableEnergy",
                    "Hour",
                    "Month",
                    "HVAC_Status",
                    "Lighting_Status"
                ]

                categorical_columns = [
                    "HVACUsage",
                    "LightingUsage",
                    "DayOfWeek",
                    "Holiday",
                    "Weekend"
                ]

                for col in numerical_columns:

                    energy_input[col] = pd.to_numeric(
                        energy_input[col],
                        errors="coerce"
                    )

                for col in categorical_columns:

                    energy_input[col] = (
                        energy_input[col]
                        .astype(str)
                    )

                # --------------------------------------------
                # Transform
                # --------------------------------------------

                X_transformed = preprocessor.transform(
                    energy_input
                )

                if hasattr(
                    X_transformed,
                    "toarray"
                ):
                    X_transformed = X_transformed.toarray()

                # --------------------------------------------
                # Hybrid prediction
                # --------------------------------------------

                pred_lr = lr_model.predict(
                    X_transformed
                )[0]

                pred_cb = cb_model.predict(
                    X_transformed
                )[0]

                pred_gb = gb_model.predict(
                    X_transformed
                )[0]

                meta_input = np.array([[
                    pred_lr,
                    pred_cb,
                    pred_gb
                ]], dtype=float)

                final_prediction = meta_model.predict(
                    meta_input
                )[0]

                st.success(
                    f"Predicted Energy Consumption: "
                    f"{final_prediction:.4f}"
                )

                # --------------------------------------------
                # Hybrid prediction wrapper
                # --------------------------------------------

                def hybrid_predict(X):

                    pred_lr = lr_model.predict(X)

                    pred_cb = cb_model.predict(X)

                    pred_gb = gb_model.predict(X)

                    meta_input = np.column_stack([
                        pred_lr,
                        pred_cb,
                        pred_gb
                    ])

                    return meta_model.predict(
                        meta_input
                    )

                # --------------------------------------------
                # SHAP background
                # --------------------------------------------

                background = X_transformed

                # --------------------------------------------
                # Permutation SHAP
                # --------------------------------------------

                explainer = shap.PermutationExplainer(
                    hybrid_predict,
                    background
                )

                shap_result = explainer(
                    X_transformed
                )

                shap_values = np.asarray(
                    shap_result.values
                )

                if shap_values.ndim == 3:

                    shap_values = shap_values[:, :, 0]

                # --------------------------------------------
                # Feature names
                # --------------------------------------------

                feature_names = (
                    preprocessor
                    .get_feature_names_out()
                )

                contribution_values = (
                    shap_values[0]
                )

                explanation_df = pd.DataFrame({
                    "Feature": feature_names,
                    "SHAP_Value": contribution_values,
                    "Absolute_SHAP": np.abs(
                        contribution_values
                    )
                })

                explanation_df = (
                    explanation_df
                    .sort_values(
                        "Absolute_SHAP",
                        ascending=False
                    )
                    .reset_index(drop=True)
                )

                # --------------------------------------------
                # Top contributions
                # --------------------------------------------

                st.markdown(
                    "### 🔍 Top Feature Contributions"
                )

                st.dataframe(
                    explanation_df.head(10),
                    use_container_width=True
                )

                # --------------------------------------------
                # Positive / negative contributions
                # --------------------------------------------

                positive = explanation_df[
                    explanation_df["SHAP_Value"] > 0
                ].head(5)

                negative = explanation_df[
                    explanation_df["SHAP_Value"] < 0
                ].head(5)

                col1, col2 = st.columns(2)

                with col1:

                    st.markdown(
                        "### 🔺 Increasing Energy Consumption"
                    )

                    if len(positive) > 0:

                        st.dataframe(
                            positive[
                                [
                                    "Feature",
                                    "SHAP_Value"
                                ]
                            ],
                            use_container_width=True
                        )

                    else:

                        st.write(
                            "No positive contributions."
                        )

                with col2:

                    st.markdown(
                        "### 🔻 Decreasing Energy Consumption"
                    )

                    if len(negative) > 0:

                        st.dataframe(
                            negative[
                                [
                                    "Feature",
                                    "SHAP_Value"
                                ]
                            ],
                            use_container_width=True
                        )

                    else:

                        st.write(
                            "No negative contributions."
                        )

                st.info(
                    """
                    Positive SHAP values push the predicted energy
                    consumption higher, while negative SHAP values
                    push it lower relative to the model's expected
                    prediction.
                    """
                )

            except Exception as e:

                st.error(
                    f"Energy prediction explanation failed: {e}"
                )


# ============================================================
# CROSS-DOMAIN ANALYTICS
# ============================================================

elif page == "🔗 Cross-Domain Analytics":

    st.title("🔗 Cross-Domain Analytics")

    st.write(
        """
        This section examines relationships across the Air Quality,
        Weather, and Energy domains to provide broader smart-city
        context. These relationships represent associations and
        contextual evidence, not causal relationships.
        """
    )

    cross_section = st.radio(
        "Select Analysis",
        [
            "🌦️ Weather → Energy",
            "🌫️ Air Quality ↔ Weather",
            "🌍 Three-Domain Relationships",
            "💡 Smart-City Insights",
            "🎯 Scenario Analysis"
        ],
        horizontal=True
    )

    # ========================================================
    # WEATHER → ENERGY
    # ========================================================

    if cross_section == "🌦️ Weather → Energy":

        st.subheader("🌦️ Weather → ⚡ Energy")

        st.write(
            """
            Weather and energy data were aligned using Month and Hour
            to investigate contextual relationships between weather
            conditions and energy consumption.
            """
        )

        tables_dir = (
            ARTIFACTS_DIR
            / "cross_domain"
            / "tables"
        )

        images_dir = (
            ARTIFACTS_DIR
            / "cross_domain"
            / "images"
        )

        # ----------------------------------------------------
        # Correlations
        # ----------------------------------------------------

        st.markdown("### 📊 Weather–Energy Correlations")

        file = (
            tables_dir
            / "weather_energy_correlations.xlsx"
        )

        if file.exists():

            df = pd.read_excel(file)

            st.dataframe(
                df,
                use_container_width=True
            )

        # ----------------------------------------------------
        # Temperature Analysis
        # ----------------------------------------------------

        st.markdown("### 🌡️ Temperature vs Energy")

        file = (
            tables_dir
            / "temperature_energy_analysis.xlsx"
        )

        if file.exists():

            df = pd.read_excel(file)

            st.dataframe(
                df,
                use_container_width=True
            )

        # ----------------------------------------------------
        # Humidity Analysis
        # ----------------------------------------------------

        st.markdown("### 💧 Humidity vs Energy")

        file = (
            tables_dir
            / "humidity_energy_analysis.xlsx"
        )

        if file.exists():

            df = pd.read_excel(file)

            st.dataframe(
                df,
                use_container_width=True
            )

        # ----------------------------------------------------
        # Regression
        # ----------------------------------------------------

        st.markdown("### 📈 Weather → Energy Regression")

        file = (
            tables_dir
            / "weather_energy_regression_results.xlsx"
        )

        if file.exists():

            df = pd.read_excel(file)

            st.dataframe(
                df,
                use_container_width=True
            )

        # ----------------------------------------------------
        # Graph
        # ----------------------------------------------------

        graph = (
            images_dir
            / "weather_energy_correlation.png"
        )

        if graph.exists():

            st.markdown(
                "### 📊 Correlation Visualization"
            )

            st.image(
                str(graph),
                use_container_width=True
            )

        st.info(
            """
            Interpretation: Weather variables show relatively weak
            relationships with energy consumption. Therefore, weather
            conditions should be considered together with operational
            and building-related factors rather than used alone for
            energy prediction.
            """
        )

    # ========================================================
    # AIR QUALITY ↔ WEATHER
    # ========================================================

    elif cross_section == "🌫️ Air Quality ↔ Weather":

        st.subheader(
            "🌫️ Air Quality ↔ 🌦️ Weather"
        )

        st.write(
            """
            This analysis investigates relationships between selected
            weather variables and Pollution Score.
            """
        )

        tables_dir = (
            ARTIFACTS_DIR
            / "cross_domain"
            / "tables"
        )

        images_dir = (
            ARTIFACTS_DIR
            / "cross_domain"
            / "images"
        )

        # ----------------------------------------------------
        # Correlations
        # ----------------------------------------------------

        st.markdown(
            "### 📊 Air–Weather Correlations"
        )

        file = (
            tables_dir
            / "air_weather_pollution_correlations.xlsx"
        )

        if file.exists():

            df = pd.read_excel(file)

            st.dataframe(
                df,
                use_container_width=True
            )

        # ----------------------------------------------------
        # Temperature
        # ----------------------------------------------------

        st.markdown(
            "### 🌡️ Temperature vs Pollution"
        )

        file = (
            tables_dir
            / "air_temperature_pollution_analysis.xlsx"
        )

        if file.exists():

            df = pd.read_excel(file)

            st.dataframe(
                df,
                use_container_width=True
            )

        # ----------------------------------------------------
        # Humidity
        # ----------------------------------------------------

        st.markdown(
            "### 💧 Humidity vs Pollution"
        )

        file = (
            tables_dir
            / "air_humidity_pollution_analysis.xlsx"
        )

        if file.exists():

            df = pd.read_excel(file)

            st.dataframe(
                df,
                use_container_width=True
            )

        # ----------------------------------------------------
        # Visualization
        # ----------------------------------------------------

        graph = (
            images_dir
            / "air_weather_pollution_correlation.png"
        )

        if graph.exists():

            st.markdown(
                "### 📊 Correlation Visualization"
            )

            st.image(
                str(graph),
                use_container_width=True
            )

        st.info(
            """
            Interpretation: Temperature and humidity show notable
            associations with Pollution Score in the analyzed data.
            These relationships should be interpreted as statistical
            associations rather than direct causal effects.
            """
        )

    # ========================================================
    # THREE-DOMAIN RELATIONSHIPS
    # ========================================================

    elif cross_section == "🌍 Three-Domain Relationships":

        st.subheader(
            "🌍 Three-Domain Relationships"
        )

        st.write(
            """
            The three-domain analysis brings together important
            relationships identified across Air Quality, Weather,
            and Energy.
            """
        )

        tables_dir = (
            ARTIFACTS_DIR
            / "cross_domain"
            / "tables"
        )

        images_dir = (
            ARTIFACTS_DIR
            / "cross_domain"
            / "images"
        )

        file = (
            tables_dir
            / "three_domain_relationships.xlsx"
        )

        if file.exists():

            df = pd.read_excel(file)

            st.dataframe(
                df,
                use_container_width=True
            )

        graph = (
            images_dir
            / "three_domain_relationships.png"
        )

        if graph.exists():

            st.image(
                str(graph),
                use_container_width=True
            )

    # ========================================================
    # SMART-CITY INSIGHTS
    # ========================================================

    elif cross_section == "💡 Smart-City Insights":

        st.subheader(
            "💡 Smart-City Insights"
        )

        st.write(
            """
            These insights summarize the major cross-domain
            relationships identified from the analysis.
            """
        )

        tables_dir = (
            ARTIFACTS_DIR
            / "cross_domain"
            / "tables"
        )

        file = (
            tables_dir
            / "smart_city_insights.xlsx"
        )

        if file.exists():

            insights_df = pd.read_excel(file)

            st.dataframe(
                insights_df,
                use_container_width=True
            )

            st.success(
                "Cross-domain evidence successfully loaded."
            )

    # ========================================================
    # SCENARIO ANALYSIS
    # ========================================================

    else:

        st.subheader(
            "🎯 Scenario Analysis"
        )

        tables_dir = (
            ARTIFACTS_DIR
            / "cross_domain"
            / "tables"
        )

        images_dir = (
            ARTIFACTS_DIR
            / "cross_domain"
            / "images"
        )

        scenario = st.selectbox(
            "Select Scenario",
            [
                "🌫️ Pollution Scenario",
                "⚡ Energy Scenario",
                "🏙️ Smart-City Actions"
            ]
        )

        # ----------------------------------------------------
        # Pollution
        # ----------------------------------------------------

        if scenario == "🌫️ Pollution Scenario":

            file = (
                tables_dir
                / "pollution_scenario_analysis.xlsx"
            )

            if file.exists():

                df = pd.read_excel(file)

                st.dataframe(
                    df,
                    use_container_width=True
                )

            graph = (
                images_dir
                / "pollution_scenario_analysis.png"
            )

            if graph.exists():

                st.image(
                    str(graph),
                    use_container_width=True
                )

        # ----------------------------------------------------
        # Energy
        # ----------------------------------------------------

        elif scenario == "⚡ Energy Scenario":

            file = (
                tables_dir
                / "energy_scenario_analysis.xlsx"
            )

            if file.exists():

                df = pd.read_excel(file)

                st.dataframe(
                    df,
                    use_container_width=True
                )

            graph = (
                images_dir
                / "energy_scenario_analysis.png"
            )

            if graph.exists():

                st.image(
                    str(graph),
                    use_container_width=True
                )

        # ----------------------------------------------------
        # Actions
        # ----------------------------------------------------

        else:

            file = (
                tables_dir
                / "smart_city_scenario_actions.xlsx"
            )

            if file.exists():

                df = pd.read_excel(file)

                st.dataframe(
                    df,
                    use_container_width=True
                )

                st.success(
                    "Smart-city action recommendations loaded."
                )


# ============================================================
# SMART-CITY DECISION SUPPORT
# ============================================================

elif page == "🏙️ Smart-City Decision Support":

    st.title("🏙️ Smart-City Decision Support")

    st.write(
        """
        This module connects machine-learning predictions,
        explainability results, cross-domain relationships, and
        scenario-based evidence to support smart-city interpretation
        and operational decision-making.
        """
    )

    st.markdown(
        """
        ### 🔄 Decision-Support Workflow

        **User Input → Prediction → SHAP Explanation → Scenario
        Identification → Cross-Domain Evidence → Interpretation
        → Recommended Action**
        """
    )

    st.divider()

    # ========================================================
    # SELECT DOMAIN
    # ========================================================

    decision_domain = st.selectbox(
        "Select Smart-City Domain",
        [
            "🌫️ Air Quality",
            "🌦️ Weather",
            "⚡ Energy"
        ]
    )

    tables_dir = (
        ARTIFACTS_DIR
        / "cross_domain"
        / "tables"
    )

    images_dir = (
        ARTIFACTS_DIR
        / "cross_domain"
        / "images"
    )

    # ========================================================
    # AIR QUALITY
    # ========================================================

    if decision_domain == "🌫️ Air Quality":

        st.subheader(
            "🌫️ Air Quality Decision Support"
        )

        st.write(
            """
            Air-quality decision support combines pollution prediction
            with the observed relationships between pollution and
            weather-related variables.
            """
        )

        # ----------------------------------------------------
        # Scenario
        # ----------------------------------------------------

        st.markdown("### 🎯 Scenario")

        pollution_file = (
            tables_dir
            / "pollution_scenario_analysis.xlsx"
        )

        if pollution_file.exists():

            pollution_df = pd.read_excel(
                pollution_file
            )

            st.dataframe(
                pollution_df,
                use_container_width=True
            )

        # ----------------------------------------------------
        # Cross-domain evidence
        # ----------------------------------------------------

        st.markdown(
            "### 🔗 Cross-Domain Evidence"
        )

        air_weather_file = (
            tables_dir
            / "air_weather_pollution_correlations.xlsx"
        )

        if air_weather_file.exists():

            air_weather_df = pd.read_excel(
                air_weather_file
            )

            st.dataframe(
                air_weather_df,
                use_container_width=True
            )

        # ----------------------------------------------------
        # Interpretation
        # ----------------------------------------------------

        st.markdown(
            "### 🧠 Interpretation"
        )

        st.info(
            """
            The analysis identified associations between selected
            weather variables and Pollution Score. Temperature and
            humidity are useful contextual indicators when interpreting
            pollution conditions.

            These relationships represent statistical associations
            observed in the analyzed data and should not be interpreted
            as direct causal relationships.
            """
        )

        # ----------------------------------------------------
        # Recommended action
        # ----------------------------------------------------

        st.markdown(
            "### 🚦 Recommended Smart-City Action"
        )

        st.success(
            """
            Increase monitoring attention during conditions associated
            with elevated pollution levels, combine pollution
            predictions with weather context, and prioritize
            location-specific air-quality monitoring where risk
            indicators are elevated.
            """
        )

    # ========================================================
    # WEATHER
    # ========================================================

    elif decision_domain == "🌦️ Weather":

        st.subheader(
            "🌦️ Weather Decision Support"
        )

        st.write(
            """
            Weather predictions can provide contextual information
            for energy planning and broader smart-city operations.
            """
        )

        # ----------------------------------------------------
        # Weather-Energy evidence
        # ----------------------------------------------------

        st.markdown(
            "### 🔗 Weather → Energy Evidence"
        )

        weather_energy_file = (
            tables_dir
            / "weather_energy_correlations.xlsx"
        )

        if weather_energy_file.exists():

            weather_energy_df = pd.read_excel(
                weather_energy_file
            )

            st.dataframe(
                weather_energy_df,
                use_container_width=True
            )

        # ----------------------------------------------------
        # Regression evidence
        # ----------------------------------------------------

        st.markdown(
            "### 📈 Weather–Energy Regression"
        )

        regression_file = (
            tables_dir
            / "weather_energy_regression_results.xlsx"
        )

        if regression_file.exists():

            regression_df = pd.read_excel(
                regression_file
            )

            st.dataframe(
                regression_df,
                use_container_width=True
            )

        # ----------------------------------------------------
        # Interpretation
        # ----------------------------------------------------

        st.markdown(
            "### 🧠 Interpretation"
        )

        st.info(
            """
            The weather–energy analysis found relatively weak overall
            explanatory power for weather variables alone. Therefore,
            weather predictions should be treated as contextual
            information rather than as the sole basis for energy
            management decisions.
            """
        )

        # ----------------------------------------------------
        # Recommended action
        # ----------------------------------------------------

        st.markdown(
            "### 🚦 Recommended Smart-City Action"
        )

        st.success(
            """
            Combine weather forecasts with building occupancy,
            HVAC operation, lighting usage, and other operational
            variables when planning energy-management strategies.
            """
        )

    # ========================================================
    # ENERGY
    # ========================================================

    else:

        st.subheader(
            "⚡ Energy Decision Support"
        )

        st.write(
            """
            Energy decision support combines energy prediction with
            weather context and scenario-based analysis.
            """
        )

        # ----------------------------------------------------
        # Energy scenario
        # ----------------------------------------------------

        st.markdown(
            "### 🎯 Energy Scenario"
        )

        energy_file = (
            tables_dir
            / "energy_scenario_analysis.xlsx"
        )

        if energy_file.exists():

            energy_df = pd.read_excel(
                energy_file
            )

            st.dataframe(
                energy_df,
                use_container_width=True
            )

        # ----------------------------------------------------
        # Weather-energy evidence
        # ----------------------------------------------------

        st.markdown(
            "### 🔗 Weather–Energy Evidence"
        )

        weather_energy_file = (
            tables_dir
            / "weather_energy_correlations.xlsx"
        )

        if weather_energy_file.exists():

            weather_energy_df = pd.read_excel(
                weather_energy_file
            )

            st.dataframe(
                weather_energy_df,
                use_container_width=True
            )

        # ----------------------------------------------------
        # Smart-city actions
        # ----------------------------------------------------

        st.markdown(
            "### 🚦 Smart-City Actions"
        )

        action_file = (
            tables_dir
            / "smart_city_scenario_actions.xlsx"
        )

        if action_file.exists():

            action_df = pd.read_excel(
                action_file
            )

            st.dataframe(
                action_df,
                use_container_width=True
            )

        # ----------------------------------------------------
        # Interpretation
        # ----------------------------------------------------

        st.markdown(
            "### 🧠 Interpretation"
        )

        st.info(
            """
            Weather variables alone explain only a limited portion of
            energy-consumption variation. Energy-management decisions
            should therefore consider operational variables such as
            occupancy, HVAC usage, lighting usage, and renewable-energy
            availability alongside weather context.
            """
        )

        # ----------------------------------------------------
        # Recommended action
        # ----------------------------------------------------

        st.success(
            """
            Use energy predictions together with operational
            conditions to identify periods requiring closer monitoring,
            optimize HVAC and lighting usage, and consider renewable
            energy availability when planning consumption.
            """
        )

    st.divider()

    st.caption(
        """
        Decision-support recommendations are based on model outputs,
        statistical associations, and scenario analysis. They are
        intended as analytical guidance rather than automated control
        decisions.
        """
    )


# ============================================================
# ABOUT PROJECT
# ============================================================

elif page == "ℹ️ About Project":

    st.title("ℹ️ About Project")

    st.subheader(
        "Explainable Data Analytics for Sustainable Smart City Applications"
    )

    st.write(
        """
        This project develops an explainable and adaptable data
        analytics framework for smart-city applications across
        three heterogeneous domains: Air Quality, Weather, and Energy.
        """
    )

    st.divider()

    # ========================================================
    # OBJECTIVE
    # ========================================================

    st.markdown("### 🎯 Project Objective")

    st.write(
        """
        The objective is to develop a machine-learning based
        analytical framework that can predict important smart-city
        outcomes while also providing understandable explanations
        for model predictions.
        """
    )

    # ========================================================
    # DOMAINS
    # ========================================================

    st.markdown("### 🌍 Application Domains")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.info(
            """
            ### 🌫️ Air Quality

            Prediction of Pollution Score with explainable
            feature contributions.
            """
        )

    with col2:

        st.info(
            """
            ### 🌦️ Weather

            Prediction of Temperature using weather-related
            observations and temporal features.
            """
        )

    with col3:

        st.info(
            """
            ### ⚡ Energy

            Prediction of Energy Consumption using environmental
            and operational building information.
            """
        )

    st.divider()

    # ========================================================
    # METHODOLOGY
    # ========================================================

    st.markdown("### ⚙️ Methodology")

    methodology = [
        "Data Collection and Preparation",
        "Exploratory Data Analysis",
        "Feature Engineering",
        "Feature Selection",
        "Baseline Model Development",
        "Proposed Hybrid Model Development",
        "Model Evaluation",
        "SHAP-based Explainability",
        "Cross-Domain Analytics",
        "Smart-City Decision Support"
    ]

    for i, step in enumerate(
        methodology,
        start=1
    ):

        st.write(
            f"**{i}.** {step}"
        )

    # ========================================================
    # PROPOSED MODEL
    # ========================================================

    st.markdown("### 🤖 Proposed Hybrid Model")

    st.code(
        """
Input Features
      ↓
Linear Regression
      +
CatBoost
      +
Gradient Boosting
      ↓
5-Fold OOF Predictions
      ↓
Linear Regression Meta-Learner
      ↓
Final Prediction
        """,
        language="text"
    )

    st.write(
        """
        The proposed approach combines predictions from three
        complementary base learners using out-of-fold predictions
        and a Linear Regression meta-learner.
        """
    )

    # ========================================================
    # EXPLAINABILITY
    # ========================================================

    st.markdown("### 🔍 Explainability")

    st.write(
        """
        SHAP is used to understand feature contributions at both
        global and individual prediction levels.

        Global explanations identify important features across the
        explanation dataset, while instance-level explanations show
        which features contribute to a particular prediction.
        """
    )

    # ========================================================
    # CROSS-DOMAIN ANALYTICS
    # ========================================================

    st.markdown("### 🔗 Cross-Domain Analytics")

    st.write(
        """
        Cross-domain analysis examines relationships among Air
        Quality, Weather, and Energy. The analysis provides contextual
        evidence for smart-city interpretation and does not claim
        direct causal relationships.
        """
    )

    # ========================================================
    # DECISION SUPPORT
    # ========================================================

    st.markdown("### 🏙️ Smart-City Decision Support")

    st.write(
        """
        The dashboard connects predictions, explanations,
        cross-domain evidence, scenario analysis, and recommended
        actions to support analytical decision-making.
        """
    )

    st.code(
        """
User Input
    ↓
Prediction
    ↓
SHAP Explanation
    ↓
Scenario Identification
    ↓
Cross-Domain Evidence
    ↓
Smart-City Interpretation
    ↓
Recommended Action
        """,
        language="text"
    )

    # ========================================================
    # IMPORTANT RESEARCH NOTE
    # ========================================================

    st.markdown("### 📌 Research Note")

    st.warning(
        """
        The proposed hybrid model is not expected to outperform every
        baseline model on every dataset or evaluation metric.
        Performance is domain-dependent, and all results are reported
        without artificially forcing the proposed model to be superior.
        """
    )

    # ========================================================
    # LIMITATIONS
    # ========================================================

    st.markdown("### ⚠️ Limitations")

    limitations = [
        "The datasets represent specific analytical settings and may not capture every real-world smart-city condition.",
        "Cross-domain relationships represent statistical associations rather than causal effects.",
        "Air Quality data does not provide a suitable timestamp for exact temporal alignment with Weather data.",
        "Model performance may vary across domains because the underlying data characteristics differ.",
        "The dashboard is intended for analytical decision support rather than autonomous control."
    ]

    for limitation in limitations:

        st.write(
            f"• {limitation}"
        )

    # ========================================================
    # PROJECT HIGHLIGHT
    # ========================================================

    st.markdown("### ⭐ Project Highlight")

    st.success(
        """
        The key contribution of this project is an end-to-end,
        explainable and cross-domain smart-city analytics framework
        that integrates machine learning, hybrid modeling, SHAP
        explainability, cross-domain analysis, and decision support
        within a single interactive dashboard.
        """
    )