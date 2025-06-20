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
    'Lagged_NOI_Spread': 'M2SL',
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
    'S&P_Global_Manufacturing_PMI': 'NAPM',  # Placeholder; replace with API
    'Dallas_Fed': 'TXMFG',
    'Philly_New_Orders': 'PHILFRBNO',
    'NY_Fed': 'NYMFG',
    'Richmond_Fed': 'RICHMANU'
}

# Placeholder for X sentiment (requires API integration)
def fetch_x_sentiment(start_date, end_date):
    # Simulate sentiment with VADER (replace with actual X API data)
    analyzer = SentimentIntensityAnalyzer()
    # Example macro-related posts (replace with API fetch)
    sample_posts = [
        "Manufacturing sector looks strong, PMI up! #economy",
        "Dollar weakening, good for stocks? #markets",
        "Recession fears overblown, New Orders rising #macro"
    ]
    sentiment_scores = [analyzer.polarity_scores(post)['compound'] for post in sample_posts]
    dates = pd.date_range(start=start_date, end=end_date, freq='M')
    return pd.Series(np.mean(sentiment_scores) * np.ones(len(dates)), index=dates, name='X_Sentiment')

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
            fred_data = fred_data.join(series_data, how='outer') if not fred_data.empty else series_data

    x_sentiment = fetch_x_sentiment(start_date, '2025-06-17')
    fred_data = fred_data.join(x_sentiment, how='outer')

    fred_data = fred_data.reset_index().rename(columns={'index': 'Date'})
    fred_data['Date'] = pd.to_datetime(fred_data['Date'])

    data = pd.merge(iwb_data, fred_data, on='Date', how='inner').set_index('Date').dropna()
    data = data.rename(columns={v: k for k, v in FRED_MAPPINGS.items()})
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
    
    # PCA for financial conditions weights
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
    sim_max_drawdown = np.min(sim_cum_returns - np.maximum.accumulate(sim_cum_returns), axis=1)
    return np.mean(sim_max_drawdown)

# Calculate forward indicator
def calculate_forward_indicator(data):
    data = calculate_lead_index(data)
    data = calculate_financial_conditions(data)

    data['Regional_Sentiment'] = data[['Dallas_Fed', 'Philly_New_Orders', 'NY_Fed', 'Richmond_Fed']].mean(axis=1)
    data['Regional_Sentiment_Score'] = rolling_normalize(data['Regional_Sentiment']) * dynamic_weights(data)[0]['Regional_Sentiment']

    data['Fed_Sentiment_Score'] = rolling_normalize(data['Fed_Survey_Expectations'])
    data['Consumer_Sentiment_Score'] = rolling_normalize(data['Consumer_Expectations_Index'])
    data['X_Sentiment_Score'] = rolling_normalize(data['X_Sentiment'])
    data['Macro_Sentiment_Composite'] = (0.4 * data['Fed_Sentiment_Score'] + 0.3 * data['Consumer_Sentiment_Score'] + 0.3 * data['X_Sentiment_Score'])
    data['Macro_Sentiment_Score'] = data['Macro_Sentiment_Composite'] * dynamic_weights(data)[0]['Macro_Sentiment']

    data['Global_M2_Score'] = rolling_normalize(data['Global_M2']) * dynamic_weights(data)[0]['Global_M2']

    data['Inverted_CPI'] = np.where(data['CPI'] > 0, 1 / data['CPI'], np.nan)
    data['CPI_Score'] = rolling_normalize(data['Inverted_CPI']) * dynamic_weights(data)[0]['CPI']

    data['PMI_Composite'] = (data['ISM'] + data['S&P_Global_Manufacturing_PMI']) / 2
    data['Business_Cycle_Score'] = rolling_normalize(data['PMI_Composite']) * dynamic_weights(data)[0]['Business_Cycle']

    data['Smoothed_Forward_Score'] = (
        data['Lead_Index_Score'] +
        data['Financial_Conditions_Score'] +
        data['Macro_Sentiment_Score'] +
        data['Global_M2_Score'] +
        data['CPI_Score'] +
        data['Business_Cycle_Score'] +
        data['Regional_Sentiment_Score']
    )

    def classify_regime(score):
        if score >= 70:
            return "Bullish (Risk-On)"
        elif 40 <= score < 70:
            return "Neutral / Transition"
        else:
            return "Defensive (Risk-Off)"

    data['Market_Regime'] = data['Smoothed_Forward_Score'].apply(classify_regime)
    
    # Stop-loss signal
    data['Volatility'] = data['IWB_Return'].rolling(12).std()
    data['Stop_Loss'] = np.where(data['Volatility'] > data['Volatility'].quantile(0.9), -1, 0)
    
    return data

# Asset allocation with drawdown limits
def calculate_allocation(regime, volatility):
    max_drawdown_limit = -0.15
    if volatility > 0.05:  # High volatility threshold
        return {"Equities": 0.10, "Crypto": 0.05, "Bonds": 0.60, "Cash": 0.25}
    if regime == "Bullish (Risk-On)":
        return {"Equities": 0.60, "Crypto": 0.20, "Bonds": 0.10, "Cash": 0.10}
    elif regime == "Neutral / Transition":
        return {"Equities": 0.40, "Crypto": 0.10, "Bonds": 0.30, "Cash": 0.20}
    else:
        return {"Equities": 0.20, "Crypto": 0.05, "Bonds": 0.50, "Cash": 0.25}

# Calculate regime performance
def calculate_regime_performance(df):
    summary = df.groupby("Market_Regime")["IWB_Return"].agg([
        ("Avg Monthly Return", np.mean),
        ("Volatility", np.std),
        ("Win Rate", lambda x: (x > 0).mean()),
        ("Annualized Return", lambda x: (1 + np.mean(x)) ** 12 - 1)
    ])
    return summary

# Backtesting with cross-validation
def backtest_algorithm(data, n_splits=10):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []

    for train_idx, test_idx in tscv.split(data):
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx].copy()

        test_data = calculate_forward_indicator(test_data)

        test_data['Signal'] = np.where(test_data['Market_Regime'] == "Bullish (Risk-On)", 1,
                                      np.where(test_data['Market_Regime'] == "Defensive (Risk-Off)", -1, 0))
        test_data['Signal'] = np.where(test_data['Stop_Loss'] == -1, 0, test_data['Signal'])
        test_data['Strategy_Returns'] = test_data['Signal'].shift(1) * test_data['IWB_Return']

        cumulative_returns = (1 + test_data['Strategy_Returns'].dropna()).cumprod() - 1
        sharpe_ratio = (test_data['Strategy_Returns'].mean() / test_data['Strategy_Returns'].std()) * np.sqrt(12)
        max_drawdown = (test_data['Strategy_Returns'].cumsum() - test_data['Strategy_Returns'].cumsum().cummax()).min()
        regime_perf = calculate_regime_performance(test_data)

        results.append({
            'Cumulative_Returns': cumulative_returns.iloc[-1] if not cumulative_returns.empty else np.nan,
            'Sharpe_Ratio': sharpe_ratio,
            'Max_Drawdown': max_drawdown,
            'Regime_Performance': regime_perf
        })

    results_df = pd.DataFrame({
        'Cumulative_Returns': [r['Cumulative_Returns'] for r in results],
        'Sharpe_Ratio': [r['Sharpe_Ratio'] for r in results],
        'Max_Drawdown': [r['Max_Drawdown'] for r in results]
    })
    return results_df

# Visualizations
def plot_forward_score(data):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x=data.index, y=data['Smoothed_Forward_Score'], ax=ax, label='Forward Score')
    sns.lineplot(x=data.index, y=data['Financial_Conditions_Score'], ax=ax, label='Financial Conditions')
    ax.set_title('AlphaEdge Macro Indicator Trends')
    ax.set_ylabel('Score (0-100)')
    ax.legend()
    st.pyplot(fig)

# Generate PDF report
def generate_pdf_report(data, results, allocation):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica", 12)
    c.drawString(100, 750, "AlphaEdge Macro Report")
    c.drawString(100, 730, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d')}")
    
    c.drawString(100, 700, "Latest Forward Indicator Scores:")
    for i, (date, score) in enumerate(data['Smoothed_Forward_Score'].tail().items()):
        c.drawString(120, 680 - i*20, f"{date.strftime('%Y-%m-%d')}: {score:.2f}")
    
    c.drawString(100, 600, "Latest Market Regimes:")
    for i, (date, regime) in enumerate(data['Market_Regime'].tail().items()):
        c.drawString(120, 580 - i*20, f"{date.strftime('%Y-%m-%d')}: {regime}")
    
    c.drawString(100, 500, "Cross-Validation Results:")
    c.drawString(120, 480, f"Avg Cumulative Returns: {results['Cumulative_Returns'].mean():.4f}")
    c.drawString(120, 460, f"Avg Sharpe Ratio: {results['Sharpe_Ratio'].mean():.4f}")
    c.drawString(120, 440, f"Avg Max Drawdown: {results['Max_Drawdown'].mean():.4f}")
    
    c.drawString(100, 400, "Suggested Allocation:")
    for asset, weight in allocation.items():
        c.drawString(120, 380 - list(allocation.keys()).index(asset)*20, f"{asset}: {weight:.2%}")
    
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# Main execution
if __name__ == "__main__":
    st.title("AlphaEdge Macro Indicator")
    data = load_data()
    data = calculate_forward_indicator(data)
    results = backtest_algorithm(data)

    st.write("### Forward Indicator Scores (Last 5)")
    st.dataframe(data['Smoothed_Forward_Score'].tail())

    st.write("### Market Regimes (Last 5)")
    st.dataframe(data['Market_Regime'].tail())

    st.write("### Regime Performance")
    st