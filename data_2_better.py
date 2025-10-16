# =====================================================================
# Formula 1 Race Performance Prediction & Analytical Insights Dashboard
# Handles Missing Values Automatically
# =====================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

# ---------------------------------------------------------------------
# Streamlit configuration
# ---------------------------------------------------------------------
st.set_page_config(page_title="F1 Race Analysis & Prediction", layout="wide")

st.markdown("""
# Formula 1 Race Performance Dashboard
Professional analysis of driver, constructor, and race performance with predictive modeling.
""")

# Sidebar navigation
page = st.sidebar.radio(
    "Navigation",
    ["Data Overview", "Model Training", "Model Evaluation & Insights"]
)

# ---------------------------------------------------------------------
# PAGE 1: DATA OVERVIEW
# ---------------------------------------------------------------------
if page == "Data Overview":
    st.subheader("Dataset Overview")

    uploaded_file = st.file_uploader("Upload your Formula 1 dataset (CSV)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df

        st.write(f"**Dataset Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

        with st.expander("Preview Data"):
            st.dataframe(df.head(10), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.write("### Summary Statistics")
            st.dataframe(df.describe().T)
        with col2:
            st.write("### Missing Values")
            missing = df.isnull().sum()
            st.dataframe(missing[missing > 0])

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            st.markdown("### Distribution of Numeric Columns")
            cols = st.columns(min(4, len(numeric_cols)))
            for i, col in enumerate(numeric_cols[:4]):
                with cols[i]:
                    fig = px.histogram(df, x=col, nbins=30, title=col)
                    fig.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0),
                                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)

        if "constructorId" in df.columns and "points" in df.columns:
            st.markdown("### Constructor Points Distribution")
            fig = px.box(df, x="constructorId", y="points", color="constructorId",
                         points="all", title="Points Distribution by Constructor")
            fig.update_layout(height=400, margin=dict(t=50),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------
# PAGE 2: MODEL TRAINING
# ---------------------------------------------------------------------
elif page == "Model Training":
    st.subheader("Model Training")

    if "df" not in st.session_state:
        st.warning("Please upload a dataset first.")
    else:
        df = st.session_state.df.copy()

        if "positionOrder" not in df.columns:
            st.error("Target column 'positionOrder' not found in dataset.")
            st.stop()

        target = "positionOrder"
        features = [col for col in df.columns if col != target]

        # Encode categorical columns
        for col in df.select_dtypes(exclude=np.number).columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

        X = df[features]
        y = df[target]

        # Handle missing values using SimpleImputer
        imputer = SimpleImputer(strategy="mean")
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )

        # Train Gradient Boosting model
        model = GradientBoostingRegressor(random_state=42)
        model.fit(X_train, y_train)

        st.session_state.model = model
        st.session_state.X_train, st.session_state.X_test = X_train, X_test
        st.session_state.y_train, st.session_state.y_test = y_train, y_test

        st.success("Model training completed successfully.")

        st.markdown("### Cross-Validation Performance")
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        mae_cv = -cross_val_score(model, X, y, scoring="neg_mean_absolute_error", cv=cv)
        r2_cv = cross_val_score(model, X, y, scoring="r2", cv=cv)

        col1, col2 = st.columns(2)
        col1.metric("Average MAE", f"{mae_cv.mean():.2f}")
        col2.metric("Average R²", f"{r2_cv.mean():.3f}")

        fig = go.Figure(data=[
            go.Box(y=mae_cv, name="MAE", boxmean="sd", marker_color="#00509E"),
            go.Box(y=r2_cv, name="R²", boxmean="sd", marker_color="#00B4D8")
        ])
        fig.update_layout(
            title="Cross-Validation Distribution",
            height=400, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------
# PAGE 3: MODEL EVALUATION & INSIGHTS
# ---------------------------------------------------------------------
elif page == "Model Evaluation & Insights":
    st.subheader("Model Evaluation & Insights")

    if "model" not in st.session_state:
        st.warning("Train the model first.")
    else:
        model = st.session_state.model
        X_train, X_test = st.session_state.X_train, st.session_state.X_test
        y_train, y_test = st.session_state.y_train, st.session_state.y_test

        # Predictions
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        within_3 = np.mean(np.abs(y_pred - y_test) <= 3) * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Mean Absolute Error", f"{mae:.2f}")
        col2.metric("R² Score", f"{r2:.3f}")
        col3.metric("Within ±3 Positions", f"{within_3:.1f}%")

        # Actual vs Predicted
        st.markdown("### Actual vs Predicted Finishing Positions")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_test, y=y_pred, mode="markers",
            marker=dict(color="#0077B6", size=7, opacity=0.8),
            name="Predictions"
        ))
        fig.add_trace(go.Scatter(
            x=[y_test.min(), y_test.max()],
            y=[y_test.min(), y_test.max()],
            mode="lines", line=dict(color="red", dash="dash"),
            name="Perfect"
        ))
        fig.update_layout(height=450, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        # Error distribution
        st.markdown("### Prediction Error Distribution")
        errors = y_pred - y_test
        fig = px.histogram(errors, nbins=25, title="Prediction Error Distribution",
                           color_discrete_sequence=["#0096C7"])
        fig.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        # Feature importance
        st.markdown("### Feature Importance")
        importance = pd.DataFrame({
            "Feature": X_train.columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        fig = px.bar(importance.head(12), x="Feature", y="Importance",
                     title="Top Features Influencing Predictions",
                     color="Importance", color_continuous_scale="Blues")
        fig.update_layout(height=450, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        # -----------------------------------------------------------------
        # Analytical Insights
        # -----------------------------------------------------------------
        st.markdown("## Analytical Insights")

        if "grid" in X_train.columns:
            corr_grid = np.corrcoef(st.session_state.df["grid"], st.session_state.df["positionOrder"])[0, 1]
            st.markdown(f"**Grid Position Effect:** Correlation with finishing = `{corr_grid:.2f}`. "
                        "Drivers starting higher usually finish better.")

        if "constructorId" in X_train.columns:
            avg_pos = st.session_state.df.groupby("constructorId")["positionOrder"].mean().sort_values()
            fig = px.bar(avg_pos, title="Average Finishing Position by Constructor",
                         color=avg_pos, color_continuous_scale="Tealgrn")
            st.plotly_chart(fig, use_container_width=True)

        if "driverId" in X_train.columns:
            driver_avg = st.session_state.df.groupby("driverId")["positionOrder"].mean().sort_values()
            fig = px.bar(driver_avg.head(15), title="Top Consistent Drivers",
                         color=driver_avg.head(15), color_continuous_scale="Blues")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        ---
        This dashboard integrates historical Formula 1 data with predictive modeling to provide
        insights into driver, constructor, and race performance trends. The model captures key
        performance patterns and offers interpretable analysis for future race outcomes.
        """)

