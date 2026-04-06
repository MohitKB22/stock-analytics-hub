# 📈 Stock Price Prediction with Machine Learning & Streamlit

A comprehensive machine learning-powered web application for stock price prediction and technical analysis using Streamlit.

## 🎯 Overview

This application converts your original stock analysis code into a production-ready Streamlit app with advanced machine learning models, technical indicators, and interactive visualizations.

### Key Features
- **Interactive Web Interface** - Built with Streamlit for easy use
- **Multiple ML Algorithms** - Linear Regression, Random Forest, Gradient Boosting, Neural Networks
- **Advanced Feature Engineering** - 20+ technical indicators
- **Real-time Predictions** - Predict stock prices with trained models
- **Technical Analysis** - Moving averages, RSI, MACD, Bollinger Bands
- **Performance Metrics** - MAE, RMSE, R² Score for model evaluation
- **Visual Analytics** - Interactive charts and comparisons

---

## 📊 Machine Learning Features

### Feature Engineering (20+ Features)
1. **Basic Features**
   - Price Change & Percentage Change
   - Daily Returns
   - High-Low Range

2. **Moving Averages**
   - Simple MA (7, 14, 30 days)
   - Exponential MA (12, 26 days)

3. **Momentum Indicators**
   - MACD (Moving Average Convergence Divergence)
   - Signal Line
   - Momentum (10-day)

4. **Volatility Measures**
   - Rolling Standard Deviation
   - Bollinger Bands (Upper, Middle, Lower, Width)

5. **Strength Indicators**
   - RSI (Relative Strength Index)
   - Price-to-MA Ratios

6. **Volume Features**
   - Volume Moving Average
   - Volume Ratio

### Machine Learning Algorithms

#### 1. Linear Regression
- Simple yet effective baseline model
- Best for linear relationships
- Fast training and prediction

#### 2. Random Forest
- Ensemble method with 100 decision trees
- Captures non-linear patterns
- Provides feature importance
- Robust to outliers

#### 3. Gradient Boosting
- Sequential tree building
- Reduces bias and variance
- Strong predictive power
- Best for complex patterns

#### 4. Neural Network (MLP)
- Deep learning approach
- Architecture: Input → Hidden(100) → Hidden(50) → Output
- Captures complex non-linear relationships
- Great for high-dimensional data

### Evaluation Metrics
- **MAE** (Mean Absolute Error) - Average prediction error
- **RMSE** (Root Mean Squared Error) - Penalizes large errors
- **R² Score** - Proportion of variance explained (0-1 scale)

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Step 1: Clone or Download
```bash
# Navigate to your project directory
cd your_project_folder
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Prepare Your Data
Ensure your CSV files have the following columns:
```
Date, Open, Close, High, Low, Volume (optional), country, stock
```

---

## 💻 Usage

### Running the App
```bash
streamlit run stock_ml_streamlit_app.py
```

The app will open in your default browser at `http://localhost:8501`

### Using the Application

#### 1. **Data Upload (Sidebar)**
   - Click "Upload CSV files" to upload your stock data
   - Or use default file paths in the code

#### 2. **Configure Parameters (Sidebar)**
   - **Test Size**: Percentage of data for testing (10-50%)
   - **Lookback Window**: Number of days for sequence creation (7-60)

#### 3. **Select Stock**
   - Choose Country from dropdown
   - Choose Stock from dropdown

#### 4. **View Tabs**

##### 📊 Historical Data
- Line chart of closing prices over time
- Shows overall trend and volatility

##### 📈 Technical Indicators
- Price with Moving Averages (7, 30-day)
- RSI (Relative Strength Index)
  - > 70: Overbought signal
  - < 30: Oversold signal
- MACD with Signal Line and Histogram

##### 🤖 ML Models
- Click "Train Models" button
- View performance metrics comparison table
- See predictions vs actual values
- Identifies best performing model

##### 📋 Feature Analysis
- Select features to visualize
- View engineered features over time
- Statistical summary of features

##### 📉 Data Overview
- First 10 rows of data
- Statistical summary
- Data types and shape

---

## 📈 Model Architecture

### Data Processing Pipeline
```
Raw Stock Data
    ↓
Feature Engineering (20+ features)
    ↓
Data Normalization (MinMaxScaler)
    ↓
Sequence Creation (Lookback window)
    ↓
Train-Test Split (default 80-20)
    ↓
Model Training (4 algorithms)
    ↓
Evaluation & Prediction
```

### Training Process
1. **Feature Scaling**: MinMaxScaler normalizes features to 0-1 range
2. **Lookback Window**: Creates sequences using previous N days
3. **Train-Test Split**: 80% training, 20% testing (configurable)
4. **Parallel Training**: Models train simultaneously for efficiency
5. **Performance Evaluation**: Compare metrics across all models

---

## 🎨 UI/UX Features

### Sidebar Configuration
- Clean, organized parameter selection
- Real-time validation
- Helpful tooltips and descriptions

### Main Dashboard
- **Metric Cards**: Display key statistics
  - Total records
  - Current price
  - Price change
  - Volatility

- **Tabbed Interface**: Easy navigation
  - Organized information
  - No cluttered layout
  - Focused analysis per tab

### Visualizations
- **High-quality Charts**: Matplotlib & Seaborn
- **Interactive Dropdowns**: Feature selection
- **Data Tables**: Searchable and sortable
- **Color-coded Results**: Easy interpretation

---

## 📝 Example Workflow

### Step 1: Upload Data
Upload your stock CSV files through the sidebar

### Step 2: Select Stock
1. Choose a country from dropdown
2. Choose a stock symbol
3. View current price and metrics

### Step 3: Analyze Technical Indicators
1. Go to "Technical Indicators" tab
2. Review RSI, MACD, and Moving Averages
3. Identify trends and signals

### Step 4: Train ML Models
1. Navigate to "ML Models" tab
2. Click "Train Models" button
3. Wait for training to complete

### Step 5: Review Results
1. Check model performance table
2. Compare predictions vs actual values
3. Identify best performing model
4. Use for future predictions

---

## 🔧 Configuration Guide

### Adjusting Model Parameters

#### In `stock_ml_streamlit_app.py`:

**Random Forest**
```python
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
```
- Increase `n_estimators` for more trees (better but slower)
- Adjust `max_depth` to prevent overfitting

**Gradient Boosting**
```python
gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
```
- Adjust `learning_rate` for step size (default 0.1)
- Increase `max_depth` for complexity

**Neural Network**
```python
nn = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
```
- Modify hidden layer sizes: `(100, 50, 25)` for deeper network
- Increase `max_iter` for more training iterations

---

## 📊 CSV File Format

Your CSV files should have this structure:

```
Date,Open,Close,High,Low,Volume,country,stock
2023-01-01,150.00,152.50,153.00,149.50,1000000,USA,AAPL
2023-01-02,152.50,155.00,156.00,152.00,1200000,USA,AAPL
...
```

### Required Columns:
- `Date` - Trading date
- `Open` - Opening price
- `Close` - Closing price
- `High` - Highest price of the day
- `Low` - Lowest price of the day
- `country` - Country code
- `stock` - Stock symbol

### Optional Columns:
- `Volume` - Trading volume
- `Unnamed: 0` - Will be removed automatically

---

## 🐛 Troubleshooting

### Issue: "Not enough data"
**Solution**: Make sure you have at least 50+ records per stock and adjust lookback window

### Issue: Models not training
**Solution**: 
- Check data format and column names
- Ensure dates are properly formatted
- Verify no missing values in critical columns

### Issue: Slow performance
**Solution**:
- Reduce number of records
- Lower number of estimators in ensemble models
- Reduce neural network size

### Issue: High RMSE/MAE values
**Solution**:
- Increase lookback window
- Add more training data
- Try different model parameters
- Check data quality

---

## 📚 Feature Explanations

### Technical Indicators

**Moving Average (MA)**
- Simple average of closing prices over N days
- Smooths out price fluctuations
- Shows trend direction

**Exponential Moving Average (EMA)**
- Weighted average giving more weight to recent prices
- Responds faster to price changes than SMA

**MACD (Moving Average Convergence Divergence)**
- Shows momentum and trend changes
- Combination of two EMAs
- Signal line confirms trades

**RSI (Relative Strength Index)**
- Measures momentum on 0-100 scale
- > 70: Potentially overbought
- < 30: Potentially oversold

**Bollinger Bands**
- Consists of moving average and standard deviation bands
- Shows price volatility
- Price touching bands may indicate extremes

---

## 🎓 Learning Resources

### Machine Learning Concepts
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)

### Stock Market Analysis
- [Technical Analysis Guide](https://www.investopedia.com/terms/t/technicalanalysis.asp)
- [Stock Indicators Explained](https://www.investopedia.com/terms/t/technicalanalysis.asp)

---

## 📄 License

This project is open source and available under the MIT License.

---

## 🤝 Contributing

Contributions are welcome! Here are some enhancement ideas:
- Add more ML models (XGBoost, LightGBM)
- Implement LSTM for time series
- Add sentiment analysis
- Create prediction API
- Add backtesting functionality
- Support for real-time data

---

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review your data format
3. Check Streamlit and scikit-learn documentation
4. Ensure all dependencies are installed correctly

---

## 🎯 Next Steps

1. **Prepare your data** in CSV format
2. **Install requirements**: `pip install -r requirements.txt`
3. **Run the app**: `streamlit run stock_ml_streamlit_app.py`
4. **Upload data** and explore
5. **Train models** and analyze predictions
6. **Iterate** with different parameters

---

## 📊 Model Performance Tips

For best results:
- ✅ Use at least 6 months of historical data
- ✅ Ensure consistent data quality
- ✅ Try different lookback windows (14, 30, 60)
- ✅ Monitor both training and test metrics
- ✅ Use ensemble models for better accuracy
- ✅ Consider volatility of the stock
- ❌ Don't rely solely on historical performance
- ❌ Don't use for high-frequency trading without optimization

---

**Last Updated**: April 2026
**Version**: 1.0.0
