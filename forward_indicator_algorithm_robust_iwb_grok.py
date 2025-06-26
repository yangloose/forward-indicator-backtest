import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA
from fredapi import Fred
import streamlit as st
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

# FRED API setup
try:
    fred = Fred(api_key='72c7447b0ec7963e4192b8692142bc28')
except KeyError:
    raise ValueError("FRED_API_KEY not found. Ensure it’s set correctly.")

# Data mappings: FRED series to algorithm variables
FRED_MAPPINGS = {
    'Lagged_New_Orders': 'NEWORDER',
    '10Y_Treasury_Rate': 'DGS10',
    'Dollar_Index': 'DTWEXBGS',
    'Credit_Spread': 'BAA10Y',
    'Commodity_Index': 'PALLFNFINDEXQ',
    'Fed_Survey_Expectations': 'FRBSF',
    'Consumer_Expectations_Index': 'UMCSENT',
    'Global_M2': 'M2SL',
    'CPI': 'CPIAUCSL',
    'PPI': 'PPIACO',
    'ISM': 'NAPM',
    'Dallas_Fed': 'TXMFG',
    'Philly_New_Orders': 'PHILFRBNO',
    'NY_Fed': 'NYMFG',
    'Richmond_Fed': 'RICHMANU'
}

# Placeholder for X sentiment (requires API integration)
def fetch_x_sentiment(start_date, end_date):
    analyzer = SentimentIntensityAnalyzer()
    sample_posts = [
        "Manufacturing sector looks strong, PMI up! #economy",
        "Dollar weakening, good for stocks? #markets",
        "Recession fears overblown, New Orders rising #macro"
    ]
    sentiment_scores = [analyzer.polarity_scores(post)['compound'] for post in sample_posts]
    dates = pd.date_range(start=start_date, end=end_date, freq='M')
    return pd.Series(np.mean(sentiment_scores) * np.ones(len(dates)), index=dates, name='X_Sentiment')

# Placeholder for S&P Global PMI (requires API)
def fetch_sp_global_pmi(start_date, end_date):
    dates = pd.date_range(start=start_date, end=end_date, freq='M')
    return pd.Series(np.random.uniform(45, 55, len(dates)), index=dates, name='S&P_Global_Manufacturing_PMI')

# Fetch FRED data
def fetch_fred_data(series_id, start_date='2010-01-01', end_date='2025-06-17'):
    try:
        data = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
        return data.resample('M').last().to_frame(name=series_id)
    except Exception as e:
        print(f"Error fetching {series_id}: {e}")
        return pd.DataFrame()

# Load and prepare data
def load_data(start_date='2010-01-01'):
    try:
        iwb_data = pd.read_csv('IWB_data.csv', parse_dates=['Date'])
        iwb_data = iwb_data[['Date', 'IWB_Return']].dropna()
    except FileNotFoundError:
        st.error("IWB_data.csv not found in repository. Please upload it.")
        raise FileNotFoundError("IWB_data.csv not found.")
    
    iwb_data['Date'] = pd.to_datetime(iwb_data['Date'])

    fred_data = pd.DataFrame()
    for var, series_id in FRED_MAPPINGS.items():
        series_data = fetch_fred_data(series_id, start_date)
        if not series_data.empty:
            series_data = series_data.rename(columns={series_id: var})  # Rename to avoid overlap
            fred_data = fred_data.join(series_data, how='outer') if not fred_data.empty else series_data

    x_sentiment = fetch_x_sentiment(start_date, '2025-06-17')
    sp_global_pmi = fetch_sp_global_pmi(start_date, '2025-06-17')
    fred_data = fred_data.join(x_sentiment, how='outer').join(sp_global_pmi, how='outer')

    fred_data = fred_data.reset_index().rename(columns={'index': 'Date'})
    fred_data['Date'] = pd.to_datetime(fred_data['Date'])

    data = pd.merge(iwb_data, fred_data, on='Date', how='inner').set_index('Date').dropna()
    return data

# Robust normalization
def rolling_normalize(series, window=12):
    scaler = MinMaxScaler()
    normalized = np.zeros(len(series))
    for i in range(len(series)):
        start = max(0, i - window + 1)
        window_data = series[start:i+1].values.reshape(-1, 1)
        if len(window_data) > 1:
            normalized[i] = scaler.fit_transform(window_data)[-1, 0]
    return normalized

# Dynamic weights with PCA
def dynamic_weights(data):
    cpi_vol = data['CPI'].pct_change().rolling(12).std().iloc[-1]
    regional_vol = data[['Dallas_Fed', 'Philly_New_Orders']].std().mean()
    
    fin_features = ['Inverted_10Y', 'Inverted_DXY', 'Inverted_Credit_Spread', 'Commodity_Index']
    if all(f in data.columns for f in fin_features):
        X = data[fin_features].dropna()
        if len(X) > 1:
            pca = PCA(n_components=1)
            pca.fit(X)
            weights = pca.components_[0] / np.sum(np.abs(pca.components_[0]))
        else:
            weights = [0.25] * 4
    else:
        weights = [0.25] * 4
    
    base_weights = {
        'Predicted_ISM': 0.20,
        'Financial_Conditions': 0.20,
        'Macro_Sentiment': 0.15,
        'Global_M2': 0.15,
        'CPI': 0.10,
        'Business_Cycle': 0.10,
        'Regional_Sentiment': 0.10
    }
    if cpi_vol > data['CPI'].pct_change().rolling(12).std().quantile(0.75):
        base_weights['CPI'] += 0.05
        base_weights['Financial_Conditions'] -= 0.05
    if regional_vol > data[['Dallas_Fed', 'Philly_New_Orders']].std().quantile(0.75):
        base_weights['Regional_Sentiment'] += 0.05
        base_weights['Predicted_ISM'] -= 0.05
    return base_weights, weights

# Custom lead indicator targeting ISM New Orders
def calculate_lead_index(data):
    features = ['Lagged_New_Orders', 'Philly_New_Orders', 'Dollar_Index', '10Y_Treasury_Rate', 'Financial_Conditions_Raw']
    X = data[features].fillna(method='ffill')
    y = data['Lagged_New_Orders'].shift(-3).fillna(method='ffill')
    
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X.iloc[:-3], y.iloc[:-3])
    lead_pred = model.predict(X)
    
    data['Lead_Index'] = lead_pred
    data['Lead_Index_Score'] = rolling_normalize(data['Lead_Index']) * dynamic_weights(data)[0]['Predicted_ISM']
    return data

# Financial conditions index with PCA weights
def calculate_financial_conditions(data):
    data['Inverted_10Y'] = np.where(data['10Y_Treasury_Rate'] > 0, 1 / data['10Y_Treasury_Rate'], np.nan)
    data['Inverted_DXY'] = np.where(data['Dollar_Index'] > 0, 1 / data['Dollar_Index'], np.nan)
    data['Inverted_Credit_Spread'] = np.where(data['Credit_Spread'] > 0, 1 / data['Credit_Spread'], np.nan)
    fin_weights = dynamic_weights(data)[1]
    data['Financial_Conditions_Raw'] = (
        fin_weights[0] * data['Inverted_10Y'] +
        fin_weights[1] * data['Inverted_DXY'] +
        fin_weights[2] * data['Inverted_Credit_Spread'] +
        fin_weights[3] * data['Commodity_Index']
    )
    data['Financial_Conditions_Score'] = rolling_normalize(data['Financial_Conditions_Raw']) * dynamic_weights(data)[0]['Financial_Conditions']
    return data

# Monte Carlo simulation for regime robustness
def monte_carlo_simulation(data, n_simulations=1000):
    returns = data['Strategy_Returns'].dropna()
    mean = returns.mean()
    std = returns.std()
    simulations = np.random.normal(mean, std, (n_simulations, len(returns)))
    sim_cum_returns = np.cumprod(1 + simulations, axis=1) - 1
    sim_max