import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from fredapi import Fred
import yfinance as yf
import streamlit as st
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# FRED API setup
try:
    fred = Fred(api_key='72c7447b0ec7963e4192b8692142bc28')  # Your FRED API key
except KeyError:
    raise ValueError("FRED_API_KEY not found. Ensure it’s set correctly.")

# Data mappings: FRED series to algorithm variables
FRED_MAPPINGS = {
    'Lagged_New_Orders': 'NEWORDER',  # New Orders: Manufacturing
    'Lagged_NOI_Spread': 'M2SL',      # Proxy for NOI Spread (Money Supply M2)
    '10Y_Treasury_Rate': 'DGS10',     # 10-Year Treasury Constant Maturity Rate
    'Dollar_Index': 'DTWEXBGS',       # Trade Weighted U.S. Dollar Index
    'Credit_Spread': 'BAA10Y',        # Moody's Baa Corporate Bond Yield vs 10Y Treasury
    'Commodity_Index': 'PALLFNFINDEXQ', # All Commodities Price Index
    'Fed_Survey_Expectations': 'FRBSF', # Fed Survey (proxy)
    'Consumer_Expectations_Index': 'UMCSENT', # Univ. of Michigan Consumer Sentiment
    'Global_M2': 'M2SL',              # Proxy for Global M2 (US M2)
    'CPI': 'CPIAUCSL',                # Consumer Price Index
    'ISM': 'NAPM',                    # ISM Manufacturing Index
    'S&P_Global_Manufacturing_PMI': 'NAPM'  # Proxy (ISM used for S&P Global PMI)
}

# Fetch IWB monthly returns (from backtest_with_iwb.py)
def fetch_iwb_monthly_returns(start_date='2010-01-01'):
    try:
        iwb = yf.download("IWB", start=start_date, interval="1mo", auto_adjust=True)
        iwb = iwb[["Adj Close"]].rename(columns={"Adj Close": "IWB_Adj_Close"})
        iwb["IWB_Return"] = iwb["IWB_Adj_Close"].pct_change()
        iwb = iwb.dropna()
        iwb.index = iwb.index.to_period("M").to_timestamp("M")
        return iwb[["IWB_Return"]].reset_index().rename(columns={"index": "Date"})
    except Exception as e:
        raise ValueError(f"Error fetching IWB data: {e}")

# Fetch FRED data with error handling
def fetch_fred_data(series_id, start_date='2010-01-01', end_date='2025-06-17'):
    try:
        data = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
        return data.resample('M').last().to_frame(name=series_id)
    except Exception as e:
        print(f"Error fetching {series_id}: {e}")
        return pd.DataFrame()

# Load and prepare data
def load_data(start_date='2010-01-01'):
    # Fetch IWB data
    iwb_data = fetch_iwb_monthly_returns(start_date)
    iwb_data['Date'] = pd.to_datetime(iwb_data['Date'])

    # Fetch FRED data
    fred_data = pd.DataFrame()
    for var, series_id in FRED_MAPPINGS.items():
        series_data = fetch_fred_data(series_id, start_date)
        if not series_data.empty:
            fred_data = fred_data.join(series_data, how='outer') if not fred_data.empty else series_data

    # Convert FRED data index to match IWB monthly format
    fred_data = fred_data.reset_index().rename(columns={'index': 'Date'})
    fred_data['Date'] = pd.to_datetime(fred_data['Date'])

    # Merge IWB and FRED data
    data = pd.merge(iwb_data, fred_data, on='Date', how='inner').set_index('Date').dropna()

    # Rename columns to match algorithm variables
    data = data.rename(columns={v: k for k, v in FRED_MAPPINGS.items()})
    return data

# Robust normalization with rolling window
def rolling_normalize(series, window=12):  # 12 months for monthly data
    scaler = MinMaxScaler()
    normalized = np.zeros(len(series))
    for i in range(len(series)):
        start = max(0, i - window + 1)
        window_data = series[start:i+1].values.reshape(-1, 1)
        if len(window_data) > 1:
            normalized[i] = scaler.fit_transform(window_data)[-1, 0]
    return normalized

# Dynamic weights based on market conditions
def dynamic_weights(data):
    cpi_volatility = data['CPI'].pct_change().rolling(12).std().iloc[-1]
    base_weights = {
        'Predicted_ISM': 0.20,
        'Financial_Conditions': 0.25,
        'Macro_Sentiment': 0.15,
        'Global_M2': 0.10,
        'CPI': 0.10,
        'Business_Cycle': 0.05
    }
    if cpi_volatility > data['CPI'].pct_change().rolling(12).std().quantile(0.75):
        base_weights['CPI'] += 0.05
        base_weights['Financial_Conditions'] -= 0.05
    return base_weights

# Calculate forward indicator
def calculate_forward_indicator(data):
    # --- 1. Predicted ISM Component ---
    data['Predicted_ISM'] = (-0.0014 * data['Lagged_New_Orders']) + (0.988 * data['Lagged_NOI_Spread']) + 47.249
    data['Predicted_ISM_Score'] = rolling_normalize(data['Predicted_ISM']) * dynamic_weights(data)['Predicted_ISM']

    # --- 2. Financial Conditions Score ---
    data['Inverted_10Y'] = np.where(data['10Y_Treasury_Rate'] > 0, 1 / data['10Y_Treasury_Rate'], np.nan)
    data['Inverted_DXY'] = np.where(data['Dollar_Index'] > 0, 1 / data['Dollar_Index'], np.nan)
    data['Inverted_Credit_Spread'] = np.where(data['Credit_Spread'] > 0, 1 / data['Credit_Spread'], np.nan)
    data['BCOM'] = data['Commodity_Index']
    data['Financial_Conditions_Raw'] = (data['Inverted_10Y'] + data['Inverted_DXY'] + 
                                      data['Inverted_Credit_Spread'] + data['BCOM']) / 4
    data['Financial_Conditions_Score'] = rolling_normalize(data['Financial_Conditions_Raw']) * dynamic_weights(data)['Financial_Conditions']

    # --- 3. Macro Sentiment Score ---
    data['Fed_Sentiment_Score'] = rolling_normalize(data['Fed_Survey_Expectations'])
    data['Consumer_Sentiment_Score'] = rolling_normalize(data['Consumer_Expectations_Index'])
    data['Macro_Sentiment_Composite'] = (0.6 * data['Fed_Sentiment_Score']) + (0.4 * data['Consumer_Sentiment_Score'])
    data['Macro_Sentiment_Score'] = data['Macro_Sentiment_Composite'] * dynamic_weights(data)['Macro_Sentiment']

    # --- 4. Global M2 Score ---
    data['Global_M2_Score'] = rolling_normalize(data['Global_M2']) * dynamic_weights(data)['Global_M2']

    # --- 5. CPI Score ---
    data['Inverted_CPI'] = np.where(data['CPI'] > 0, 1 / data['CPI'], np.nan)
    data['CPI_Score'] = rolling_normalize(data['Inverted_CPI']) * dynamic_weights(data)['CPI']

    # --- 6. Business Cycle Confirmation ---
    data['PMI_Composite'] = (data['ISM'] + data['S&P_Global_Manufacturing_PMI']) / 2
    data['Business_Cycle_Score'] = rolling_normalize(data['PMI_Composite']) * dynamic_weights(data)['Business_Cycle']

    # --- Final Composite Score ---
    data['Smoothed_Forward_Score'] = (
        data['Predicted_ISM_Score'] +
        data['Financial_Conditions_Score'] +
        data['Macro_Sentiment_Score'] +
        data['Global_M2_Score'] +
        data['CPI_Score'] +
        data['Business_Cycle_Score']
    )

    # --- Regime Classification ---
    def classify_regime(score):
        if score >= 70:
            return "Bullish (Risk-On)"
        elif 40 <= score < 70:
            return "Neutral / Transition"
        else:
            return "Defensive (Risk-Off)"

    data['Market_Regime'] = data['Smoothed_Forward_Score'].apply(classify_regime)
    return data

# Calculate regime performance (from backtest_with_iwb.py)
def calculate_regime_performance(df):
    summary = df.groupby("Market_Regime")["IWB_Return"].agg([
        ("Avg Monthly Return", lambda x: np.mean(x)),
        ("Volatility", lambda x: np.std(x)),
        ("Win Rate", lambda x: (x > 0).mean()),
        ("Annualized Return", lambda x: (1 + np.mean(x)) ** 12 - 1)
    ])
    return summary

# Backtesting with cross-validation
def backtest_algorithm(data, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []

    for train_idx, test_idx in tscv.split(data):
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx].copy()

        # Calculate indicator
        test_data = calculate_forward_indicator(test_data)

        # Generate signals
        test_data['Signal'] = np.where(test_data['Market_Regime'] == "Bullish (Risk-On)", 1,
                                      np.where(test_data['Market_Regime'] == "Defensive (Risk-Off)", -1, 0))
        test_data['Strategy_Returns'] = test_data['Signal'].shift(1) * test_data['IWB_Return']

        # Performance metrics
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

    # Aggregate results
    results_df = pd.DataFrame({
        'Cumulative_Returns': [r['Cumulative_Returns'] for r in results],
        'Sharpe_Ratio': [r['Sharpe_Ratio'] for r in results],
        'Max_Drawdown': [r['Max_Drawdown'] for r in results]
    })
    print("Cross-Validation Results:")
    print(f"Avg Cumulative Returns: {results_df['Cumulative_Returns'].mean():.4f}")
    print(f"Avg Sharpe Ratio: {results_df['Sharpe_Ratio'].mean():.4f}")
    print(f"Avg Max Drawdown: {results_df['Max_Drawdown'].mean():.4f}")
    print("\nRegime Performance (Last Fold):")
    print(results[-1]['Regime_Performance'])

    return results_df

# Main execution
if __name__ == "__main__":
    data = load_data()
    data = calculate_forward_indicator(data)
    results = backtest_algorithm(data)

    # Streamlit output
    st.write("Forward Indicator Scores:", data['Smoothed_Forward_Score'].tail())
    st.write("Market Regimes:", data['Market_Regime'].tail())
    st.write("Regime Performance:", calculate_regime_performance(data))