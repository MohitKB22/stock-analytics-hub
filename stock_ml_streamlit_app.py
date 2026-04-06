import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="Stock Price Prediction", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 0rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# =====================================================
# FUNCTION DEFINITIONS
# =====================================================

@st.cache_data
def load_data(file_paths):
    """Load and concatenate CSV files"""
    dfs = []
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path)
            dfs.append(df)
        except Exception as e:
            st.warning(f"Could not load {file_path}: {e}")
    
    if dfs:
        combined_df = pd.concat(dfs, axis=0, ignore_index=True)
        if 'Unnamed: 0' in combined_df.columns:
            combined_df = combined_df.drop('Unnamed: 0', axis=1)
        combined_df['Date'] = pd.to_datetime(combined_df['Date'])
        combined_df = combined_df.sort_values('Date').reset_index(drop=True)
        return combined_df
    return None

def engineer_features(df):
    """Create technical indicators and features for ML"""
    df = df.copy()
    
    # Basic price features
    df['Price_Change'] = df['Close'].diff()
    df['Price_Change_Pct'] = df['Close'].pct_change() * 100
    df['Daily_Return'] = df['Close'].pct_change()
    
    # Moving Averages
    df['MA_7'] = df['Close'].rolling(window=7).mean()
    df['MA_14'] = df['Close'].rolling(window=14).mean()
    df['MA_30'] = df['Close'].rolling(window=30).mean()
    
    # Exponential Moving Average
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Momentum
    df['Momentum'] = df['Close'] - df['Close'].shift(10)
    
    # Volatility (Standard Deviation)
    df['Volatility'] = df['Daily_Return'].rolling(window=14).std() * 100
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    
    # High-Low Range
    df['HL_Range'] = df['High'] - df['Low']
    df['HL_Ratio'] = df['HL_Range'] / df['Close']
    
    # Volume features
    if 'Volume' in df.columns:
        df['Volume_MA'] = df['Volume'].rolling(window=7).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
    
    # Price position relative to MA
    df['Price_to_MA7'] = df['Close'] / df['MA_7']
    df['Price_to_MA30'] = df['Close'] / df['MA_30']
    
    # Drop NaN values created by rolling windows
    df = df.dropna()
    
    return df

def prepare_ml_data(df, lookback=30):
    """Prepare data for machine learning with lookback window"""
    features = ['Open', 'Close', 'High', 'Low', 'Price_Change', 'MA_7', 'MA_14', 
                'MA_30', 'EMA_12', 'EMA_26', 'MACD', 'Momentum', 'Volatility', 
                'RSI', 'BB_Upper', 'BB_Lower', 'HL_Range', 'Price_to_MA7', 'Price_to_MA30']
    
    # Check which features exist
    available_features = [f for f in features if f in df.columns]
    
    X = df[available_features].values
    y = df['Close'].values
    
    # Normalize features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create sequences for time series
    X_sequences = []
    y_sequences = []
    
    for i in range(len(X_scaled) - lookback):
        X_sequences.append(X_scaled[i:i+lookback].flatten())
        y_sequences.append(y[i+lookback])
    
    return np.array(X_sequences), np.array(y_sequences), scaler, available_features

def train_models(X_train, y_train, X_test, y_test):
    """Train multiple ML models"""
    models = {}
    results = {}
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    models['Linear Regression'] = lr
    results['Linear Regression'] = {
        'MAE': mean_absolute_error(y_test, lr_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, lr_pred)),
        'R2': r2_score(y_test, lr_pred),
        'predictions': lr_pred
    }
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    models['Random Forest'] = rf
    results['Random Forest'] = {
        'MAE': mean_absolute_error(y_test, rf_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)),
        'R2': r2_score(y_test, rf_pred),
        'predictions': rf_pred
    }
    
    # Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_test)
    models['Gradient Boosting'] = gb
    results['Gradient Boosting'] = {
        'MAE': mean_absolute_error(y_test, gb_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, gb_pred)),
        'R2': r2_score(y_test, gb_pred),
        'predictions': gb_pred
    }
    
    # Neural Network
    nn = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    nn.fit(X_train, y_train)
    nn_pred = nn.predict(X_test)
    models['Neural Network'] = nn
    results['Neural Network'] = {
        'MAE': mean_absolute_error(y_test, nn_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, nn_pred)),
        'R2': r2_score(y_test, nn_pred),
        'predictions': nn_pred
    }
    
    return models, results

def plot_predictions(y_test, results, title="Model Predictions Comparison"):
    """Plot actual vs predicted values"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for idx, (model_name, result) in enumerate(results.items()):
        axes[idx].plot(y_test, label='Actual', linewidth=2, marker='o', markersize=4)
        axes[idx].plot(result['predictions'], label='Predicted', linewidth=2, marker='s', markersize=4, alpha=0.7)
        axes[idx].set_title(f"{model_name}\nR² = {result['R2']:.4f}", fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Sample')
        axes[idx].set_ylabel('Price')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_historical_data(df, selected_stock):
    """Plot historical stock price"""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df['Date'], df['Close'], linewidth=2, label='Close Price', color='#1f77b4')
    ax.fill_between(df['Date'], df['Close'], alpha=0.3)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Price', fontsize=11)
    ax.set_title(f'Historical Price - {selected_stock}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_technical_indicators(df):
    """Plot technical indicators"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Price with Moving Averages
    axes[0].plot(df['Date'], df['Close'], label='Close Price', linewidth=2)
    axes[0].plot(df['Date'], df['MA_7'], label='MA7', linewidth=1.5, alpha=0.7)
    axes[0].plot(df['Date'], df['MA_30'], label='MA30', linewidth=1.5, alpha=0.7)
    axes[0].set_ylabel('Price', fontsize=10)
    axes[0].set_title('Price with Moving Averages', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # RSI
    axes[1].plot(df['Date'], df['RSI'], label='RSI', color='orange', linewidth=2)
    axes[1].axhline(y=70, color='r', linestyle='--', linewidth=1, label='Overbought (70)')
    axes[1].axhline(y=30, color='g', linestyle='--', linewidth=1, label='Oversold (30)')
    axes[1].set_ylabel('RSI', fontsize=10)
    axes[1].set_title('Relative Strength Index', fontsize=12, fontweight='bold')
    axes[1].set_ylim([0, 100])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # MACD
    axes[2].plot(df['Date'], df['MACD'], label='MACD', linewidth=2)
    axes[2].plot(df['Date'], df['Signal_Line'], label='Signal Line', linewidth=1.5, alpha=0.7)
    axes[2].bar(df['Date'], df['MACD'] - df['Signal_Line'], label='Histogram', alpha=0.3)
    axes[2].set_ylabel('MACD', fontsize=10)
    axes[2].set_xlabel('Date', fontsize=10)
    axes[2].set_title('MACD Indicator', fontsize=12, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

# =====================================================
# MAIN APP
# =====================================================

def main():
    st.title("📈 Stock Price Prediction with Machine Learning")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        st.markdown("---")
        
        # File upload option
        st.subheader("📁 Data Upload")
        uploaded_files = st.file_uploader(
            "Upload CSV files (or use default path)",
            type=['csv'],
            accept_multiple_files=True,
            help="Upload your stock data CSV files"
        )
        
        st.markdown("---")
        
        # Model parameters
        st.subheader("⚙️ Model Parameters")
        test_size = st.slider(
            "Test Size (%)",
            min_value=10,
            max_value=50,
            value=20,
            help="Percentage of data to use for testing"
        )
        
        lookback = st.slider(
            "Lookback Window",
            min_value=7,
            max_value=60,
            value=30,
            help="Number of days to look back for sequence creation"
        )
    
    # Main content
    if uploaded_files:
        # Load data from uploaded files
        file_paths = [file.name for file in uploaded_files]
        dfs = []
        for uploaded_file in uploaded_files:
            df_temp = pd.read_csv(uploaded_file)
            if 'Unnamed: 0' in df_temp.columns:
                df_temp = df_temp.drop('Unnamed: 0', axis=1)
            dfs.append(df_temp)
        
        df = pd.concat(dfs, axis=0, ignore_index=True)
    else:
        # Default file paths
        st.info("📌 Upload CSV files above to use custom data. Default paths will be attempted.")
        file_paths = [
            '../yfinance_projects/stock_H.1.csv',
            '../yfinance_projects/stock_H.csv',
            '../yfinance_projects/stock_I.csv',
            '../yfinance_projects/stock_In.csv',
            '../yfinance_projects/stock_Ind.csv',
            '../yfinance_projects/stock_L1.csv',
            '../yfinance_projects/stock_L2.csv',
            '../yfinance_projects/stock_L3.csv',
            '../yfinance_projects/stock_USA.csv'
        ]
        
        dfs = []
        for file_path in file_paths:
            try:
                df_temp = pd.read_csv(file_path)
                if 'Unnamed: 0' in df_temp.columns:
                    df_temp = df_temp.drop('Unnamed: 0', axis=1)
                dfs.append(df_temp)
            except FileNotFoundError:
                st.warning(f"⚠️ Could not load {file_path}")
                continue
            except Exception as e:
                st.warning(f"⚠️ Error loading {file_path}: {str(e)}")
                continue
        
        if len(dfs) == 0:
            st.error("❌ No data files found. Please upload CSV files to get started.")
            st.info("""
            **Required CSV Format:**
            - Columns: Date, Open, Close, High, Low, country, stock
            - Example: 2023-01-01, 150.00, 152.50, 153.00, 149.50, USA, AAPL
            """)
            st.stop()
        
        df = pd.concat(dfs, axis=0, ignore_index=True)
    
    if df is not None and len(df) > 0:
        try:
            # Data preparation
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            
            # Check if required columns exist
            if 'country' not in df.columns or 'stock' not in df.columns:
                st.error("❌ Error: Your CSV file must contain 'country' and 'stock' columns.")
                st.info("""
                **Required columns:**
                - Date (YYYY-MM-DD format)
                - Open, Close, High, Low (prices)
                - country (country code, e.g., USA, IND)
                - stock (stock symbol, e.g., AAPL, TCS)
                """)
                st.stop()
            
            # Select country
            st.subheader("🌍 Select Country & Stock")
            col1, col2 = st.columns(2)
            
            with col1:
                countries = sorted(df['country'].unique())
                if len(countries) == 0:
                    st.error("❌ No country data found. Please check your CSV file.")
                    st.stop()
                selected_country = st.selectbox(
                    "Select Country",
                    countries,
                    key='country_select'
                )
            
            # Filter by country
            country_data = df[df['country'] == selected_country]
            
            with col2:
                stocks = sorted(country_data['stock'].unique())
                if len(stocks) == 0:
                    st.error(f"❌ No stocks found for {selected_country}")
                    st.stop()
                selected_stock = st.selectbox(
                    "Select Stock",
                    stocks,
                    key='stock_select'
                )
            
            # Filter by stock
            stock_data = country_data[country_data['stock'] == selected_stock].copy()
        
            if len(stock_data) > lookback + 10:
                st.markdown("---")
                
                # Display data info
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Records", len(stock_data))
                with col2:
                    st.metric("Current Price", f"${stock_data['Close'].iloc[-1]:.2f}")
                with col3:
                    price_change = stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[0]
                    st.metric("Change", f"${price_change:.2f}", delta=f"{(price_change/stock_data['Close'].iloc[0]*100):.2f}%")
                with col4:
                    st.metric("Volatility", f"{stock_data['Close'].pct_change().std()*100:.2f}%")
                
                st.markdown("---")
                
                # Tabs for different sections
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "📊 Historical Data", 
                    "📈 Technical Indicators", 
                    "🤖 ML Models", 
                    "📋 Feature Analysis",
                    "📉 Data Overview"
                ])
                
                # Tab 1: Historical Data
                with tab1:
                    st.subheader(f"Historical Price Data - {selected_stock}")
                    fig = plot_historical_data(stock_data, selected_stock)
                    st.pyplot(fig)
                
                # Tab 2: Technical Indicators
                with tab2:
                    st.subheader(f"Technical Indicators - {selected_stock}")
                    stock_data_engineered = engineer_features(stock_data)
                    if len(stock_data_engineered) > 0:
                        fig = plot_technical_indicators(stock_data_engineered)
                        st.pyplot(fig)
                    else:
                        st.warning("Not enough data to calculate indicators")
                
                # Tab 3: ML Models
                with tab3:
                    st.subheader("🤖 Machine Learning Models")
                    
                    if st.button("Train Models", use_container_width=True):
                        with st.spinner("Engineering features and training models..."):
                            # Feature engineering
                            stock_data_engineered = engineer_features(stock_data)
                            
                            if len(stock_data_engineered) > lookback + 10:
                                # Prepare data
                                X, y, scaler, features = prepare_ml_data(stock_data_engineered, lookback=lookback)
                                
                                # Split data
                                X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size=test_size/100, shuffle=False
                                )
                                
                                # Train models
                                models, results = train_models(X_train, y_train, X_test, y_test)
                                
                                # Display results
                                st.success("✅ Models trained successfully!")
                                
                                # Model comparison
                                st.markdown("### Model Performance Comparison")
                                
                                results_df = pd.DataFrame({
                                    'Model': list(results.keys()),
                                    'MAE': [results[m]['MAE'] for m in results.keys()],
                                    'RMSE': [results[m]['RMSE'] for m in results.keys()],
                                    'R² Score': [results[m]['R2'] for m in results.keys()]
                                })
                                
                                st.dataframe(results_df, use_container_width=True)
                                
                                # Best model
                                best_model = max(results.keys(), key=lambda x: results[x]['R2'])
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Best Model", best_model)
                                with col2:
                                    st.metric("Best R² Score", f"{results[best_model]['R2']:.4f}")
                                with col3:
                                    st.metric("Best RMSE", f"${results[best_model]['RMSE']:.2f}")
                                
                                # Plot predictions
                                st.markdown("### Predictions Comparison")
                                fig = plot_predictions(y_test, results)
                                st.pyplot(fig)
                            else:
                                st.warning("❌ Not enough data to train models with selected lookback window")
                
                # Tab 4: Feature Analysis
                with tab4:
                    st.subheader("📊 Feature Engineering Analysis")
                    stock_data_engineered = engineer_features(stock_data)
                    
                    if len(stock_data_engineered) > 0:
                        # Display engineered features
                        st.markdown("### Available Features")
                        
                        feature_cols = [col for col in stock_data_engineered.columns 
                                       if col not in ['Date', 'country', 'stock', 'Open', 'Close', 'High', 'Low']]
                        
                        col_pairs = st.multiselect(
                            "Select features to visualize",
                            feature_cols,
                            default=['MA_7', 'MA_30', 'RSI'],
                            max_selections=3
                        )
                        
                        if col_pairs:
                            fig, axes = plt.subplots(len(col_pairs), 1, figsize=(14, 4*len(col_pairs)))
                            if len(col_pairs) == 1:
                                axes = [axes]
                            
                            for idx, feature in enumerate(col_pairs):
                                axes[idx].plot(stock_data_engineered['Date'], stock_data_engineered[feature], 
                                              linewidth=2, color='#1f77b4')
                                axes[idx].set_ylabel(feature, fontsize=10)
                                axes[idx].set_title(f'{feature} Over Time', fontsize=12, fontweight='bold')
                                axes[idx].grid(True, alpha=0.3)
                            
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                        # Feature statistics
                        st.markdown("### Feature Statistics")
                        stats_df = stock_data_engineered[feature_cols].describe()
                        st.dataframe(stats_df, use_container_width=True)
                
                # Tab 5: Data Overview
                with tab5:
                    st.subheader("📋 Data Overview")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("### First Few Rows")
                        st.dataframe(stock_data.head(10), use_container_width=True)
                    
                    with col2:
                        st.markdown("### Data Statistics")
                        st.dataframe(stock_data[['Open', 'Close', 'High', 'Low']].describe(), use_container_width=True)
                    
                    st.markdown("### Data Types")
                    st.info(f"**Shape:** {stock_data.shape[0]} rows × {stock_data.shape[1]} columns")
                    st.dataframe(pd.DataFrame({
                        'Column': stock_data.columns,
                        'Data Type': stock_data.dtypes
                    }), use_container_width=True)
            else:
                st.error(f"❌ Not enough data for {selected_stock}. Need at least {lookback + 10} records.")
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
    else:
        st.error("❌ Unable to load data. Please check the file paths or upload CSV files.")

if __name__ == "__main__":
    main()
