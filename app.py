import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, LassoCV, ElasticNetCV, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
@st.cache
def load_data():
    df = pd.read_csv('AmesHousing.csv')
    return df

# Define functions
def missing_percent(df):
    nan_percent = 100 * (df.isnull().sum() / len(df))
    nan_percent = nan_percent[nan_percent > 0].sort_values()
    return nan_percent

# Main function to run the app
def main():
    st.title("Ames Housing Price Prediction")
    
    df = load_data()
    
    st.header("Dataset Overview")
    st.write(df.head())
    
    selected_year = st.selectbox("Select a year", df['Year Built'].unique())

    filtered_df = df[df['Year Built'] == selected_year]
    
    st.write(filtered_df.head())

    # Fill missing values
    bsmt_num_cols = ['BsmtFin SF 1', 'BsmtFin SF 2', 'Bsmt Unf SF','Total Bsmt SF' ,'Bsmt Full Bath', 'Bsmt Half Bath']
    df[bsmt_num_cols] = df[bsmt_num_cols].fillna(0)
    bsmt_str_cols = ['Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin Type 2']
    df[bsmt_str_cols] = df[bsmt_str_cols].fillna('None')
    df['Mas Vnr Type'] = df['Mas Vnr Type'].fillna('None')
    df['Mas Vnr Area'] = df['Mas Vnr Area'].fillna(0)
    Gar_str_cols = ['Garage Type', 'Garage Finish', 'Garage Qual', 'Garage Cond']
    df[Gar_str_cols] = df[Gar_str_cols].fillna('None')
    df['Garage Yr Blt'] = df['Garage Yr Blt'].fillna(0)
    df['Fireplace Qu'] = df['Fireplace Qu'].fillna('None')
    df['Lot Frontage'] = df.groupby('Neighborhood')['Lot Frontage'].transform(lambda val: val.fillna(val.mean()))
    df = df.drop(['Fence', 'Alley', 'Misc Feature','Pool QC'], axis=1)
    df = df.dropna(axis=0, subset=['Electrical', 'Garage Area'])
    
    st.write("Missing values filled and unnecessary columns dropped.")
    
    nan_percent = missing_percent(df)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=nan_percent.index, y=nan_percent)
    plt.xticks(rotation=90)
    st.pyplot(plt)
    
    st.header("Model Training and Evaluation")
    
    # Preprocessing
    df['MS SubClass'] = df['MS SubClass'].apply(str)
    df_num = df.select_dtypes(exclude='object')
    df_obj = df.select_dtypes(include='object')
    df_obj = pd.get_dummies(df_obj, drop_first=True)
    Final_df = pd.concat([df_num, df_obj], axis=1)
    
    X = Final_df.drop(['SalePrice'], axis=1)
    y = Final_df['SalePrice']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    
    # Standard Scaling
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train models
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    ridge_model = Ridge(alpha=10)
    ridge_model.fit(X_train, y_train)
    
    ridge_cv_model = RidgeCV(alphas=(0.5, 1.0, 10.0), scoring='neg_mean_absolute_error')
    ridge_cv_model.fit(X_train, y_train)
    
    lasso_cv_model = LassoCV(eps=0.01, n_alphas=100, cv=5)
    lasso_cv_model.fit(X_train, y_train)
    
    elastic_model = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1], cv=5, max_iter=10000)
    elastic_model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_ridge = ridge_cv_model.predict(X_test)
    y_pred_lasso = lasso_cv_model.predict(X_test)
    y_pred_elastic = elastic_model.predict(X_test)
    
    # Evaluation
    MAE = mean_absolute_error(y_test, y_pred)
    MSE = mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(MSE)
    
    MAE_ridge = mean_absolute_error(y_test, y_pred_ridge)
    MSE_ridge = mean_squared_error(y_test, y_pred_ridge)
    RMSE_ridge = np.sqrt(MSE_ridge)
    
    MAE_Lasso = mean_absolute_error(y_test, y_pred_lasso)
    MSE_Lasso = mean_squared_error(y_test, y_pred_lasso)
    RMSE_Lasso = np.sqrt(MSE_Lasso)
    
    MAE_Elastic = mean_absolute_error(y_test, y_pred_elastic)
    MSE_Elastic = mean_squared_error(y_test, y_pred_elastic)
    RMSE_Elastic = np.sqrt(MSE_Elastic)
    
    st.write("Model Coefficients:")
    st.write(pd.DataFrame({'metrics':[MAE, MSE, RMSE],'ridge':[MAE_ridge ,MSE_ridge, RMSE_ridge] ,'LassoCV':[MAE_Lasso, MSE_Lasso, RMSE_Lasso], 'ElasticNetCV':[MAE_Elastic, MSE_Elastic, RMSE_Elastic] }, index=['MAE', 'MSE', 'RMSE']))
    
    st.write("Ridge Model Coefficients:")
    st.write(ridge_cv_model.coef_)
    
    st.write("Lasso Model Coefficients:")
    st.write(lasso_cv_model.coef_)
    
    st.write("ElasticNet Model Coefficients:")
    st.write(elastic_model.coef_)

if __name__ == '__main__':
    main()
