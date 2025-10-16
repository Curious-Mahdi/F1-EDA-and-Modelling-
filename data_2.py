import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


st.set_page_config(page_title="F1 Race Finish Prediction", layout="wide")
pages = ["Data Upload", "Feature Engineering", "Model Training", "Insights"]
page = st.sidebar.selectbox("Select Page", pages)


if page == "Data Upload":
    st.header("Upload Dataset")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.data = df
        st.write("Dataset Preview")
        st.dataframe(df.head(10))
        st.write("Shape:", df.shape)
    elif "data" in st.session_state:
        df = st.session_state.data
        st.dataframe(df.head(10))
    else:
        st.warning("Please upload a dataset to proceed.")


def engineer_features(df):
    df_new = df.copy()
    
    
    drop_cols = ['resultId', 'raceId', 'driverRef', 'surname', 'forename',
                 'constructorRef', 'name', 'circuitRef', 'name_y', 'location',
                 'date', 'nationality_x', 'nationality_y', 'lat', 'lng', 'alt',
                 'fastestLapTime', 'fastestLap', 'rank', 'fastestLapSpeed']
    df_new.drop(columns=[col for col in drop_cols if col in df_new.columns], inplace=True)
    
    
    df_new.fillna(df_new.median(numeric_only=True), inplace=True)
    
    
    driver_counts = df.groupby('driverRef')['year'].count().to_dict()
    df_new['driver_experience'] = df['driverRef'].map(driver_counts)
    
    
    constructor_points = df.groupby('constructorRef')['points'].sum().to_dict()
    constructor_avg_finish = df.groupby('constructorRef')['positionOrder'].mean().to_dict()
    df_new['constructor_total_points'] = df['constructorRef'].map(constructor_points)
    df_new['constructor_avg_finish'] = df['constructorRef'].map(constructor_avg_finish)
    
    
    df_new['years_since_start'] = df['year'] - df['year'].min()
    df_new['is_early_season'] = df['round'] <= 5
    df_new['is_late_season'] = df['round'] >= (df['round'].max() - 3)
    
    
    cat_cols = df_new.select_dtypes(include='object').columns
    df_new = pd.get_dummies(df_new, columns=cat_cols, drop_first=True)
    
    return df_new

if page == "Feature Engineering":
    if "data" not in st.session_state:
        st.warning("Upload data first.")
    else:
        st.header("Feature Engineering & EDA")
        df = st.session_state.data
        
        
        st.subheader("Basic Info & Missing Values")
        st.write(df.info())
        st.write(df.isnull().sum())
        
        st.subheader("Summary Statistics")
        st.write(df.describe())
        
        
        st.subheader("Correlation Heatmap")
        plt.figure(figsize=(10, 6))
        numeric_cols = df.select_dtypes(include=np.number).columns
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
        st.pyplot(plt.gcf())
        plt.clf()
        
        
        df_feat = engineer_features(df)
        st.session_state.df_feat = df_feat
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Constructor Features")
            st.markdown("""
            - `constructor_avg_finish` - Average finish position
            - `constructor_total_points` - Total points scored
            """)
            
        with col2:
            st.markdown("#### Temporal Features")
            st.markdown("""
            - `years_since_start` - Years since dataset start
            - `is_early_season` - First 5 races flag
            - `is_late_season` - Last 3 races flag
            """)
        
        st.subheader("Feature Preview")
        new_features = [col for col in df_feat.columns if col not in df.columns]
        st.dataframe(df_feat[new_features].head(10))
        st.info(f"Created {len(new_features)} new features")
        
        
        st.subheader("Additional Visualizations")
        fig, axes = plt.subplots(1, 2, figsize=(12,4))
        sns.boxplot(x='grid', y='positionOrder', data=df, ax=axes[0])
        axes[0].set_title("Grid vs Finish Position")
        sns.scatterplot(x='driver_experience', y='positionOrder', data=df, ax=axes[1])
        axes[1].set_title("Driver Experience vs Finish Position")
        st.pyplot(fig)
        plt.clf()


if page == "Model Training":
    if "df_feat" not in st.session_state:
        st.warning("Complete feature engineering first.")
    else:
        st.header("Model Training & Evaluation")
        df_feat = st.session_state.df_feat
        
        X = df_feat.drop(columns=['target_finish'])
        y = df_feat['target_finish']
        
        
        split_idx = int(len(X)*0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
       
        scaler = StandardScaler()
        X_train[X_train.select_dtypes(include=np.number).columns] = scaler.fit_transform(X_train.select_dtypes(include=np.number))
        X_test[X_test.select_dtypes(include=np.number).columns] = scaler.transform(X_test.select_dtypes(include=np.number))
        
        if st.button("Train Random Forest"):
            model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
           
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            within_3 = np.mean(np.abs(y_test - y_pred) <= 3) * 100
            
            st.subheader("Model Performance")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MAE", f"{mae:.2f}")
            col2.metric("RMSE", f"{rmse:.2f}")
            col3.metric("R²", f"{r2:.2f}")
            col4.metric("Within ±3", f"{within_3:.2f}%")
            
           
            st.subheader("Predictions vs Actual")
            plt.figure(figsize=(8,6))
            plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
            plt.plot([0, 25], [0, 25], color='red', linestyle='--')
            plt.xlabel("Actual Finish")
            plt.ylabel("Predicted Finish")
            plt.title("Predicted vs Actual Finish Positions")
            st.pyplot(plt.gcf())
            plt.clf()
            
            st.session_state.model = model
            st.success("Random Forest model trained successfully.")

if page == "Insights":
    if "df_feat" not in st.session_state or "model" not in st.session_state:
        st.warning("Complete previous steps first.")
    else:
        st.header("Key Insights & Recommendations")
        df_feat = st.session_state.df_feat
        model = st.session_state.model
        
        X = df_feat.drop(columns=['target_finish'])
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top Features Influencing Finish Position")
            st.write(importances.head(10))
        
        with col2:
            st.subheader("Observations")
            st.markdown("""
            - Grid position is the strongest predictor of race finish.
            - Driver experience and constructor performance significantly influence results.
            - Temporal features (early/late season) have smaller but noticeable impact.
            - Further improvements could include weather, tire strategy, and race incidents.
            """)
