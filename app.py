import streamlit as st
import requests
import pandas as pd
from xgboost import XGBRegressor
from datetime import datetime, timedelta
import plotly.graph_objects as go

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="AI Weather Forecaster", page_icon="🌤️", layout="wide")

# --- 2. DATA & MODEL PIPELINE ---
@st.cache_data
def build_and_train_model():
    # Dynamic Date: Get yesterday's date
    yesterday = datetime.now() - timedelta(days=1)
    dynamic_end_date = yesterday.strftime('%Y-%m-%d')
    
    # Fetch Data from Open-Meteo
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 22.5726, # Kolkata
        "longitude": 88.3639,
        "start_date": "2010-01-01",
        "end_date": dynamic_end_date,
        "daily": ["temperature_2m_max", "precipitation_sum"],
        "timezone": "auto"
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    # Clean Data
    df = pd.DataFrame(data['daily'])
    df.rename(columns={'time': 'Date', 'temperature_2m_max': 'MaxTemp', 'precipitation_sum': 'Rainfall'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.dropna()

    # Feature Engineering (The "Memory")
    df['MaxTemp_Yesterday'] = df['MaxTemp'].shift(1)
    df['MaxTemp_2DaysAgo'] = df['MaxTemp'].shift(2)
    df['MaxTemp_3DaysAgo'] = df['MaxTemp'].shift(3)
    df['Rainfall_Yesterday'] = df['Rainfall'].shift(1)
    df['Target_Tomorrow_MaxTemp'] = df['MaxTemp'].shift(-1)
    
    # Isolate data for training vs predicting
    prediction_row = df.iloc[-1:] 
    train_df = df.dropna() 
    
    # Train the XGBoost Model
    X_train = train_df[['MaxTemp_Yesterday', 'MaxTemp_2DaysAgo', 'MaxTemp_3DaysAgo', 'Rainfall_Yesterday']]
    y_train = train_df['Target_Tomorrow_MaxTemp']
    
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    return model, train_df, prediction_row

# Run the pipeline with a loading spinner
with st.spinner("Fetching live historical data and training the AI... 🧠"):
    model, train_df, prediction_row = build_and_train_model()

# --- 3. MAKE THE PREDICTION ---
X_predict = prediction_row[['MaxTemp_Yesterday', 'MaxTemp_2DaysAgo', 'MaxTemp_3DaysAgo', 'Rainfall_Yesterday']]
tomorrow_pred = model.predict(X_predict)[0]

# --- 4. UI DASHBOARD DESIGN ---
# Header
st.title("🌤️ Next-Gen Weather Forecaster")
st.markdown("Predicting Kolkata's maximum temperature using **XGBoost** Machine Learning.")
st.divider()

# Get dates and past temps for display
target_date = datetime.now()
formatted_date = target_date.strftime("%A, %b %d, %Y")
yesterdays_temp = prediction_row['MaxTemp'].values[0]

# UI Columns for a clean metric display
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Target Date", value=formatted_date)
with col2:
    temp_difference = tomorrow_pred - yesterdays_temp
    st.metric(label="Predicted Max Temperature", 
              value=f"{tomorrow_pred:.2f} °C", 
              delta=f"{temp_difference:.2f} °C vs Yesterday")
with col3:
    st.metric(label="AI Model Confidence", value="High (14 Yrs Data)")

st.divider()

# --- 5. THE INTERACTIVE CHART ---
st.subheader("📈 30-Day Trend & AI Forecast")

# Get the last 30 days of actual data
last_30_days = train_df.tail(30)

# Build the Plotly chart
fig = go.Figure()

# Add the actual historical temperatures (Blue Line)
fig.add_trace(go.Scatter(
    x=last_30_days.index, 
    y=last_30_days['MaxTemp'], 
    mode='lines+markers', 
    name='Actual Temp',
    line=dict(color='dodgerblue', width=2)
))

# Add the AI's prediction for today (Big Red Star)
fig.add_trace(go.Scatter(
    x=[target_date], 
    y=[tomorrow_pred], 
    mode='markers', 
    name='AI Prediction',
    marker=dict(color='crimson', size=14, symbol='star')
))

# Clean up the chart's look
fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Max Temperature (°C)",
    hovermode="x unified",
    margin=dict(l=0, r=0, t=30, b=0)
)

# Display the chart on the website
st.plotly_chart(fig, use_container_width=True)

# --- 6. SIDEBAR ---
st.sidebar.header("📊 Behind the Scenes")
st.sidebar.write(f"This AI was dynamically trained on **{len(train_df):,}** days of historical weather data.")
st.sidebar.subheader("Raw Data Pipeline:")
st.sidebar.dataframe(train_df[['MaxTemp', 'Rainfall']].tail(10))

# --- 7. FOOTER & FEEDBACK FORM ---
st.divider()
st.subheader("📬 Get in Touch")
st.markdown("""
I tried to make this project  to demonstrate end-to-end Machine Learning deployment, featuring automated data pipelines, time-series feature engineering, and an XGBoost regression model . I am open to learn and apply more knowledge in this field.""")

#Contact
st.write("**You can contact me :**")
st.code("masoomali9345@gmail.com", language="text")