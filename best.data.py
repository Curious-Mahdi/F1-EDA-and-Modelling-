# =====================================================================
# Formula 1 Performance Prediction and Analytical Insights Dashboard
# Refined UI Edition - Professional Layout
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

# ---------------------------------------------------------------------
# Streamlit Page Config
# ---------------------------------------------------------------------
st.set_page_config(page_title="F1 Race Analysis & Prediction", layout="wide")

st.markdown("""
# Formula 1 Race Performance Analysis  
Gain insights into driver, constructor, and race performance trends — powered by machine learning.
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

    uploaded_file = st.file_uploader("Upload F1 dataset (CSV)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df

        st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

        with st.expander("🔍 Data Preview"):
            st.dataframe(df.head(10), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.write("### Summary Statistics")
            st.write(df.describe().T)
        with col2:
            st.write("### Missing Values")
            st.write(df.isnull().sum()[df.isnull().sum() > 0])

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            st.markdown("### Distribution Overview")
            num_to_show = min(4, len(numeric_cols))
            cols = st.columns(num_to_show)
            for i, col in enumerate(numeric_cols[:num_to_show]):
                with cols[i]:
                    fig = px.histogram(df, x=col, nbins=30, title=col)
                    fig.update_layout(
                        showlegend=False, margin=dict(l=0, r=0, t=30, b=0),
                        height=250, paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(fig, use_container_width=True)

        if "constructorId" in df.columns and "points" in df.columns:
            st.markdown("### Constructor Points Distribution")
            fig = px.box(df, x="constructorId", y="points", color="constructorId",
                         title="Points Distribution by Constructor", points="all")
            fig.update_layout(
                showlegend=False, height=400, margin=dict(t=50),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
            )
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

        df = df.dropna(subset=["positionOrder"])
        target = "positionOrder"
        features = [col for col in df.columns if col != target]

        for col in df.select_dtypes(exclude=np.number).columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )

        model = GradientBoostingRegressor(random_state=42)
        model.fit(X_train, y_train)

        st.session_state.model = model
        st.session_state.X_train, st.session_state.X_test = X_train, X_test
        st.session_state.y_train, st.session_state.y_test = y_train, y_test

        st.success("Model training completed successfully.")

        st.markdown("### Cross-Validation Results")
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        mae_cv = -cross_val_score(model, X, y, scoring="neg_mean_absolute_error", cv=cv)
        r2_cv = cross_val_score(model, X, y, scoring="r2", cv=cv)

        c1, c2 = st.columns(2)
        c1.metric("Mean Absolute Error (CV avg)", f"{mae_cv.mean():.2f}")
        c2.metric("R² Score (CV avg)", f"{r2_cv.mean():.3f}")

        fig = go.Figure(data=[
            go.Box(y=mae_cv, name="MAE", boxmean="sd", marker_color="#00509E"),
            go.Box(y=r2_cv, name="R²", boxmean="sd", marker_color="#00B4D8")
        ])
        fig.update_layout(
            title="Cross-Validation Performance Distribution",
            height=400, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------
# PAGE 3: MODEL EVALUATION AND INSIGHTS
# ---------------------------------------------------------------------
elif page == "Model Evaluation & Insights":
    st.subheader("Model Evaluation and Insights")

    if "model" not in st.session_state:
        st.warning("Train the model first.")
    else:
        model = st.session_state.model
        X_test, y_test = st.session_state.X_test, st.session_state.y_test
        X_train, y_train = st.session_state.X_train, st.session_state.y_train

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        within_3 = np.mean(np.abs(y_pred - y_test) <= 3) * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("Mean Absolute Error", f"{mae:.2f}")
        c2.metric("R² Score", f"{r2:.3f}")
        c3.metric("Within ±3 Positions", f"{within_3:.1f}%")

        st.markdown("### Actual vs Predicted Finishing Positions")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_test, y=y_pred, mode="markers",
            name="Predicted vs Actual",
            marker=dict(color="#0077B6", size=7, opacity=0.8)
        ))
        fig.add_trace(go.Scatter(
            x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()],
            mode="lines", name="Perfect Prediction",
            line=dict(color="red", dash="dash")
        ))
        fig.update_layout(
            xaxis_title="Actual Position", yaxis_title="Predicted Position",
            height=450, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Error Distribution")
        errors = y_pred - y_test
        fig = px.histogram(errors, nbins=25, title="Prediction Error Distribution",
                           color_discrete_sequence=["#0096C7"])
        fig.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Feature Importance")
        importance = pd.DataFrame({
            "Feature": X_train.columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        fig = px.bar(importance.head(12), x="Feature", y="Importance",
                     title="Top 12 Influencing Features",
                     color="Importance", color_continuous_scale="Blues")
        fig.update_layout(
            height=450, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)

        # -----------------------------------------------------------------
        # Insights Section
        # -----------------------------------------------------------------
        st.markdown("## Analytical Insights")

        if "grid" in X_train.columns:
            grid_corr = np.corrcoef(df["grid"], df["positionOrder"])[0, 1]
            st.markdown(f"**Grid Position Influence:** Correlation with finish = `{grid_corr:.2f}`. "
                        "Drivers starting higher on the grid show a significant advantage.")
        if "constructorId" in X_train.columns:
            avg_pos = df.groupby("constructorId")["positionOrder"].mean().sort_values()
            fig = px.bar(avg_pos, title="Average Finishing Position by Constructor",
                         color=avg_pos, color_continuous_scale="Tealgrn")
            st.plotly_chart(fig, use_container_width=True)
        if "driverId" in X_train.columns:
            driver_avg = df.groupby("driverId")["positionOrder"].mean().sort_values()
            fig = px.bar(driver_avg.head(15), title="Top Consistent Drivers",
                         color=driver_avg.head(15), color_continuous_scale="Blues")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        ---
        This dashboard integrates historical Formula 1 data with predictive modeling  
        to interpret performance factors affecting race outcomes — from grid start to constructor reliability.
        """)
