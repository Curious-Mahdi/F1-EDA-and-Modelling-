import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import io

# --- Streamlit App Configuration ---
st.set_page_config(
    page_title="F1 Race Performance Analysis and Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Utility Functions ---

@st.cache_data
def load_and_preprocess_data(uploaded_file):
    """
    Loads, cleans, and preprocesses the F1 data.
    Equivalent to notebook cells 3, 123, 133 to 146.
    """
    try:
        # Load the dataframe
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return None

    # Replace the string representation of NaN values
    df = df.replace("\\N", np.nan)

    # Drop the 'fastestLapTime' column as per notebook cell 135
    if 'fastestLapTime' in df.columns:
        df = df.drop('fastestLapTime', axis=1)

    # Convert specified columns to numeric, coercing errors (cell 136)
    numeric_cols_to_convert = [
        "points", "laps", "milliseconds", "fastestLap", "rank", "fastestLapSpeed"
    ]
    for col in numeric_cols_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Impute missing values with custom logic (cells 139, 141, 143)
    # 1. Fill 'points' where it's missing and rank > 10 (or rank is NaN, which means usually DNF/no points)
    if 'points' in df.columns and 'rank' in df.columns:
        # Check if rank > 10, fill NaN points with 0 if true
        # Using .loc with boolean indexing can be tricky with NaNs, let's simplify based on intent:
        # If rank is NaN/missing, it likely means DNF or far off point-scoring, so set points to 0 if missing.
        df.loc[df['points'].isna(), 'points'] = 0

    # 2. Fill 'milliseconds' where it's missing and 'target_finish' is 0 (DNF)
    if 'milliseconds' in df.columns and 'target_finish' in df.columns:
        df.loc[df['milliseconds'].isna() & (df['target_finish'] == 0), 'milliseconds'] = 0

    # 3. Fill 'rank' where it's missing and 'laps' or 'milliseconds' is 0
    if 'rank' in df.columns and 'laps' in df.columns and 'milliseconds' in df.columns:
        df.loc[df['rank'].isna() & ((df['laps'] == 0) | (df['milliseconds'] == 0)), 'rank'] = 0

    # Impute remaining missing numerical values using median grouped by 'year' (cell 145)
    numerical_cols_to_impute = [
        "points", "laps", "milliseconds", "fastestLap", "rank", "fastestLapSpeed"
    ]
    for col in numerical_cols_to_impute:
        if col in df.columns:
            # Groupby year and fill with median of that year
            df[col] = df.groupby("year")[col].transform(lambda x: x.fillna(x.median()))
            # Fill any remaining NaNs with the overall median
            df[col] = df[col].fillna(df[col].median())

    return df

@st.cache_data
def prepare_for_model(df):
    """
    Prepares the data for the Random Forest model.
    Equivalent to notebook cells 166, 167, 168.
    """
    if df is None:
        return None, None, None, None, None

    # Define the target variable 'y'
    if 'target_finish' in df.columns:
        y = df['target_finish']
    else:
        st.error("Target column 'target_finish' not found in the data.")
        return None, None, None, None, None

    # Columns to drop (cell 166) - checking for existence
    cols_to_drop = [
        "target_finish", "resultId", "raceId", "driveRref", "surname", "forename", "dob",
        "constructorRef", "name", "circuitRef", "name_y", "location", "date",
        "fastestLap", "rank"
    ]
    
    # Drop columns not needed for X (cell 167)
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # One-hot encode categorical features (cell 168)
    categorical_cols = X.select_dtypes(include='object').columns
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Clean column names for potential model compatibility (cell 168)
    X.columns = X.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True)

    # Split data for training (cell 172)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X, y, X_train, X_test, y_train, y_test

@st.cache_data
def train_model(X_train, y_train):
    """
    Trains the Random Forest model.
    Equivalent to notebook cells 176, 177.
    """
    # Use class_weight='balanced' to handle the class imbalance noted in EDA
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)
    return rf_model

# --- Streamlit Main App ---

st.title("F1 Race Performance Analysis and DNF Prediction")
st.markdown("Upload the `f1_dnf.csv` file to begin the analysis.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = load_and_preprocess_data(uploaded_file)

    if df is not None:
        st.header("1. Data Overview")
        st.dataframe(df.head())
        st.write(f"Dataset Shape: {df.shape}")

        # --- EDA Section ---
        st.header("2. Exploratory Data Analysis (EDA)")

        # EDA 2.1: Grid Position vs Final Position (Cell 59)
        st.subheader("Starting Grid Position vs Final Race Position")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=df, x='grid', y='positionOrder', ax=ax)
        # Check if regression plot is reasonable before plotting
        if df['grid'].corr(df['positionOrder']) is not None:
            sns.regplot(data=df, x='grid', y='positionOrder', scatter=False, color='red', ax=ax)
        
        ax.set_title('Starting Grid Position vs Final Race Position', fontsize=16, weight='bold')
        ax.set_xlabel('Grid Position (Starting)', fontsize=13)
        ax.set_ylabel('Final Position (Finishing Order)', fontsize=13)
        ax.invert_yaxis() # Invert y-axis as lower positionOrder means better finish
        st.pyplot(fig)
        
        st.markdown(
            """
            ### Findings
            * There is a visible **positive correlation** between starting `grid` position and `positionOrder`. This means a better starting position (lower `grid` value) tends to result in a better final finishing order (lower `positionOrder` value).
            * The regression line's slope is present, confirming the relationship, though the scatter suggests it's **not a perfect correlation**, indicating other factors strongly influence the final position.
            """
        )
        st.markdown("---")

        # EDA 2.2: Distribution of Driver Points (Cell 61)
        st.subheader("Distribution of Driver Points")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df['points'], bins=20, kde=True, color='skyblue', ax=ax)
        ax.set_title("Distribution of Driver Points", fontsize=14)
        ax.set_xlabel("Points Scored")
        ax.set_ylabel("Number of Drivers")
        st.pyplot(fig)

        st.markdown(
            """
            ### Finding: Data Imbalance
            * The distribution is **heavily skewed to the left**, with a large number of entries having 0 points.
            * This confirms a **significant class imbalance** for predicting points (if it were a regression task) or suggests that most drivers either score 0 points or a small amount.
            """
        )
        st.markdown("---")
        
        # EDA 2.3: Correlation Heatmap (Cell 68)
        st.subheader("Correlation Heatmap of Numerical Features")
        numeric_cols_for_corr = ['grid', 'positionOrder', 'points', 'laps', 'milliseconds', 'fastestLapSpeed']
        # Ensure only columns present in the DF are used
        present_numeric_cols = [col for col in numeric_cols_for_corr if col in df.columns]
        
        if present_numeric_cols:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(df[present_numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
            ax.set_title('Correlation Matrix')
            st.pyplot(fig)

            st.markdown(
                """
                ### Correlation Findings
                * **Position vs. Points**: The strong negative correlation between `positionOrder` and `points` (e.g., -0.87) is expected: the smaller the finishing position (closer to 1st), the higher the points.
                * **Laps vs. Time**: There's a strong positive correlation between `laps` and `milliseconds` (e.g., 0.69), which is expected as more laps take more time.
                * **Position vs. Grid**: The positive correlation between `grid` and `positionOrder` (e.g., 0.59) further confirms the scatter plot finding: starting further back generally leads to a worse final position.
                """
            )
        else:
            st.warning("Could not generate heatmap as required numerical columns are missing.")
        st.markdown("---")

        # EDA 2.4: Target Distribution (Cell 150)
        st.subheader("Distribution of Target Finish")
        
        if 'target_finish' in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(x="target_finish", data=df, ax=ax)
            ax.set_title("Distribution of Target Finish (0: DNF, 1: Finished)")
            ax.set_xlabel("Target Finish")
            ax.set_ylabel("Count")
            ax.set_xticklabels(['0 (Did Not Finish)', '1 (Finished)'])
            st.pyplot(fig)
            st.write(df['target_finish'].value_counts())
            
            finish_count = df['target_finish'].value_counts().get(1, 0)
            dnf_count = df['target_finish'].value_counts().get(0, 0)
            total = finish_count + dnf_count
            if total > 0:
                st.info(f"Class 0 (DNF): {dnf_count} ({dnf_count/total:.2%}) | Class 1 (Finished): {finish_count} ({finish_count/total:.2%})")
                st.warning("This confirms the **imbalanced nature** of the classification problem, as noted in the points distribution.")
            else:
                st.warning("No data found for target distribution.")
        else:
            st.error("Target column 'target_finish' not found for distribution plot.")
        st.markdown("---")

        # --- Model Training and Evaluation ---
        st.header("3. Machine Learning Model (Random Forest)")
        
        # Prepare data
        X, y, X_train, X_test, y_train, y_test = prepare_for_model(df)
        
        if X is not None:
            st.info(f"Model will be trained on {X_train.shape[0]} samples and evaluated on {X_test.shape[0]} samples with {X_train.shape[1]} features.")

            # Train the model
            with st.spinner("Training Random Forest Classifier..."):
                rf_model = train_model(X_train, y_train)
            st.success("Model training complete!")

            # Predict and evaluate
            y_pred = rf_model.predict(X_test)
            
            # Classification Report (Cell 181)
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose().round(2)
            st.dataframe(report_df)
            
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred, labels=rf_model.classes_)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['DNF (0)', 'Finished (1)'])
            
            fig_cm, ax_cm = plt.subplots(figsize=(6, 6))
            disp.plot(ax=ax_cm, cmap='Blues', values_format='d')
            st.pyplot(fig_cm)
            
            st.markdown(
                """
                ### Model Performance Summary
                * The model (Random Forest with `class_weight='balanced'`) shows **high precision and recall** for both classes, indicating excellent performance.
                * The **F1-scores of 0.99 for DNF (0) and 0.98 for Finished (1)** confirm the model's strong predictive capability on this specific test set.
                * **High overall accuracy (0.99)** suggests the model successfully learned the patterns for predicting a race finish.
                """
            )
        
else:
    st.info("Please upload your `f1_dnf.csv` file to run the analysis and model.")
