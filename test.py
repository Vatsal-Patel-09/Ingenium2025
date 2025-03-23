import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Set Seaborn style for better visuals
sns.set(style="whitegrid")

# Dummy data generation function
def generate_dummy_data(company):
    dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(90)]
    prices = np.random.normal(loc=100, scale=10, size=len(dates)).cumsum()
    returns = np.diff(prices) / prices[:-1]
    dummy_data = pd.DataFrame({
        'Date': dates,
        'Price': prices,
        'Returns': np.append(0, returns),  # Add 0 for first return
        'Company': company
    })
    # Dummy predictions
    dummy_data['Predicted_Price'] = dummy_data['Price'] * (1 + np.random.normal(0, 0.02, len(dates)))
    # Dummy real market data for comparison
    dummy_data['Real_Price'] = dummy_data['Price'] * (1 + np.random.normal(0, 0.01, len(dates)))
    return dummy_data

# Dummy performance metrics
def calculate_metrics(returns):
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized Sharpe
    roi = (returns + 1).prod() - 1  # Cumulative ROI
    max_drawdown = (np.maximum.accumulate(returns + 1) - (returns + 1)).max()
    return sharpe_ratio, roi, max_drawdown

# List of companies (expandable)
companies = ["Apple", "Google", "Tesla", "Microsoft", "Amazon"]

# Streamlit UI
def main():
    # Theme toggle in sidebar
    st.sidebar.header("Settings")
    theme = st.sidebar.selectbox("Theme", ["Light", "Dark"])
    if theme == "Dark":
        st.markdown(
            """
            <style>
            body {
                background-color: #1E1E1E;
                color: #FFFFFF;
            }
            .stApp {
                background-color: #1E1E1E;
                color: #FFFFFF;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <style>
            body {
                background-color: #F5F5F5;
                color: #333333;
            }
            .stApp {
                background-color: #F5F5F5;
                color: #333333;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

    # Title and description
    st.title("DRL-Based Algorithmic Trading Dashboard")
    st.write("Explore trading strategies with Deep Reinforcement Learning (DRL) for selected companies.")

    # Company selection and search
    st.sidebar.subheader("Company Selection")
    search_query = st.sidebar.text_input("Search Company", "")
    filtered_companies = [c for c in companies if search_query.lower() in c.lower()]
    selected_company = st.sidebar.selectbox("Select Company", filtered_companies if filtered_companies else companies)

    # Time frame selection
    time_frame = st.sidebar.selectbox("Time Frame", ["1 Month", "3 Months", "6 Months", "1 Year"])

    # Generate data for selected company
    df = generate_dummy_data(selected_company)

    # Filter data based on time frame
    if time_frame == "1 Month":
        df = df.tail(30)
    elif time_frame == "3 Months":
        df = df.tail(90)
    elif time_frame == "6 Months":
        df = df.tail(180)
    else:
        df = df.tail(365)

    # Main content
    st.header(f"Market Data Overview - {selected_company}")
    st.dataframe(df.tail(10))

    # Performance Metrics
    st.header("Performance Metrics")
    sharpe, roi, max_drawdown = calculate_metrics(df['Returns'])
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Sharpe Ratio", f"{sharpe:.2f}", delta_color="off")
    with col2:
        st.metric("ROI (%)", f"{roi * 100:.2f}", delta_color="off")
    with col3:
        st.metric("Max Drawdown (%)", f"{max_drawdown * 100:.2f}", delta_color="off")

    # Visualizations
    st.header("Visualizations")

    # Price Trend with Prediction
    st.subheader("Price Trend with DRL Prediction")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.lineplot(x='Date', y='Price', data=df, ax=ax1, label='Actual Price', color='blue')
    sns.lineplot(x='Date', y='Predicted_Price', data=df, ax=ax1, label='Predicted Price', color='orange', linestyle='--')
    ax1.set_title(f"Price Trend vs Prediction - {selected_company}")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price")
    plt.xticks(rotation=45)
    plt.legend()
    st.pyplot(fig1)

    # Comparison with Real Market Data
    st.subheader("Comparison with Real Market Data")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.lineplot(x='Date', y='Predicted_Price', data=df, ax=ax2, label='DRL Prediction', color='orange', linestyle='--')
    sns.lineplot(x='Date', y='Real_Price', data=df, ax=ax2, label='Real Market Price', color='green')
    ax2.set_title(f"DRL Prediction vs Real Market Data - {selected_company}")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Price")
    plt.xticks(rotation=45)
    plt.legend()
    st.pyplot(fig2)

    # Returns Distribution
    st.subheader("Returns Distribution")
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    sns.histplot(df['Returns'], bins=30, kde=True, ax=ax3, color='green')
    ax3.set_title("Distribution of Returns")
    ax3.set_xlabel("Returns")
    ax3.set_ylabel("Frequency")
    st.pyplot(fig3)

    # Market Adaptability Insights
    st.header("Market Adaptability Insights")
    st.write(f"""
    This section will analyze how the DRL model adapts to market conditions for {selected_company}. 
    Currently, it uses dummy data; real insights will be added post-model training.
    """)
    st.line_chart(df['Returns'])

    # Legal Considerations
    st.header("Legal & Ethical Considerations")
    st.write("""
    The trading model adheres to ethical financial practices and complies with regulations (to be detailed with real data).
    """)

    # Future Features
    st.sidebar.write("Future Features:")
    st.sidebar.write("- Real-time market data integration")
    st.sidebar.write("- Trained DRL model predictions")
    st.sidebar.write("- Advanced risk analysis")

if __name__ == "__main__":
    main()




## old
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Set Seaborn style for better visuals
sns.set(style="whitegrid")

# Dummy data generation function
def generate_dummy_data():
    dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(90)]
    prices = np.random.normal(loc=100, scale=10, size=len(dates)).cumsum()
    returns = np.diff(prices) / prices[:-1]
    dummy_data = pd.DataFrame({
        'Date': dates,
        'Price': prices,
        'Returns': np.append(0, returns)  # Add 0 for first return
    })
    return dummy_data

# Dummy performance metrics
def calculate_metrics(returns):
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized Sharpe
    roi = (returns + 1).prod() - 1  # Cumulative ROI
    max_drawdown = (np.maximum.accumulate(returns + 1) - (returns + 1)).max()
    return sharpe_ratio, roi, max_drawdown

# Streamlit UI
def main():
    # Title and description
    st.title("DRL-Based Algorithmic Trading Dashboard")
    st.write("""
    This dashboard visualizes the performance of a Deep Reinforcement Learning (DRL)-based trading model. 
    Currently, it uses dummy data, but it’s designed to integrate real or simulated market data once the model is trained.
    """)

    # Sidebar for settings
    st.sidebar.header("Settings")
    data_source = st.sidebar.selectbox("Data Source", ["Dummy Data", "Upload Data (Coming Soon)"])
    time_frame = st.sidebar.selectbox("Time Frame", ["1 Month", "3 Months", "6 Months", "1 Year"])
    
    # Generate or load data
    if data_source == "Dummy Data":
        df = generate_dummy_data()
    else:
        st.sidebar.write("Upload functionality will be added once the model is trained.")
        df = generate_dummy_data()  # Fallback to dummy data for now

    # Filter data based on time frame (dummy implementation)
    if time_frame == "1 Month":
        df = df.tail(30)
    elif time_frame == "3 Months":
        df = df.tail(90)
    elif time_frame == "6 Months":
        df = df.tail(180)
    else:
        df = df.tail(365)

    # Main content
    st.header("Market Data Overview")
    st.dataframe(df.tail(10))  # Show last 10 rows

    # Performance Metrics
    st.header("Performance Metrics")
    sharpe, roi, max_drawdown = calculate_metrics(df['Returns'])
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
    with col2:
        st.metric("ROI (%)", f"{roi * 100:.2f}")
    with col3:
        st.metric("Max Drawdown (%)", f"{max_drawdown * 100:.2f}")

    # Visualizations
    st.header("Visualizations")

    # Price Trend
    st.subheader("Price Trend")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.lineplot(x='Date', y='Price', data=df, ax=ax1, color='blue')
    ax1.set_title("Price Trend Over Time")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price")
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    # Returns Distribution
    st.subheader("Returns Distribution")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.histplot(df['Returns'], bins=30, kde=True, ax=ax2, color='green')
    ax2.set_title("Distribution of Returns")
    ax2.set_xlabel("Returns")
    ax2.set_ylabel("Frequency")
    st.pyplot(fig2)

    # Cumulative Returns
    st.subheader("Cumulative Returns")
    df['Cumulative_Returns'] = (df['Returns'] + 1).cumprod() - 1
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    sns.lineplot(x='Date', y='Cumulative_Returns', data=df, ax=ax3, color='purple')
    ax3.set_title("Cumulative Returns Over Time")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Cumulative Returns")
    plt.xticks(rotation=45)
    st.pyplot(fig3)

    # Market Adaptability Insights
    st.header("Market Adaptability Insights")
    st.write("""
    This section will analyze how the model adapts to changing market conditions once real data is integrated. 
    For now, it shows dummy insights based on random fluctuations.
    """)
    st.line_chart(df['Returns'])

    # Legal Considerations
    st.header("Legal & Ethical Considerations")
    st.write("""
    The trading model adheres to ethical financial practices and complies with regulations (to be detailed once real data is used).
    """)

    # Placeholder for future dynamic data integration
    st.sidebar.write("Future Features:")
    st.sidebar.write("- Upload real market data")
    st.sidebar.write("- Connect trained DRL model outputs")
    st.sidebar.write("- Advanced risk analysis")

if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Set Seaborn style for better visuals
sns.set(style="whitegrid")

# Dummy data generation function
def generate_dummy_data(company):
    dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(90)]
    prices = np.random.normal(loc=100, scale=10, size=len(dates)).cumsum()
    returns = np.diff(prices) / prices[:-1]
    dummy_data = pd.DataFrame({
        'Date': dates,
        'Price': prices,
        'Returns': np.append(0, returns),  # Add 0 for first return
        'Company': company
    })
    # Dummy predictions
    dummy_data['Predicted_Price'] = dummy_data['Price'] * (1 + np.random.normal(0, 0.02, len(dates)))
    # Dummy real market data for comparison
    dummy_data['Real_Price'] = dummy_data['Price'] * (1 + np.random.normal(0, 0.01, len(dates)))
    return dummy_data

# Dummy performance metrics
def calculate_metrics(returns):
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized Sharpe
    roi = (returns + 1).prod() - 1  # Cumulative ROI
    max_drawdown = (np.maximum.accumulate(returns + 1) - (returns + 1)).max()
    return sharpe_ratio, roi, max_drawdown

# List of companies (expandable)
companies = ["Apple", "Google", "Tesla", "Microsoft", "Amazon"]

# Streamlit UI
def main():
    # Theme toggle in sidebar
    st.sidebar.header("Settings")
    theme = st.sidebar.selectbox("Theme", ["Light", "Dark"])
    if theme == "Dark":
        st.markdown(
            """
            <style>
            body {
                background-color: #1E1E1E;
                color: #FFFFFF;
            }
            .stApp {
                background-color: #1E1E1E;
                color: #FFFFFF;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <style>
            body {
                background-color: #F5F5F5;
                color: #333333;
            }
            .stApp {
                background-color: #F5F5F5;
                color: #333333;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

    # Title and description
    st.title("DRL-Based Algorithmic Trading Dashboard")
    st.write("Explore trading strategies with Deep Reinforcement Learning (DRL) for selected companies.")

    # Company selection and search
    st.sidebar.subheader("Company Selection")
    search_query = st.sidebar.text_input("Search Company", "")
    filtered_companies = [c for c in companies if search_query.lower() in c.lower()]
    selected_company = st.sidebar.selectbox("Select Company", filtered_companies if filtered_companies else companies)

    # Time frame selection
    time_frame = st.sidebar.selectbox("Time Frame", ["1 Month", "3 Months", "6 Months", "1 Year"])

    # Generate data for selected company
    df = generate_dummy_data(selected_company)

    # Filter data based on time frame
    if time_frame == "1 Month":
        df = df.tail(30)
    elif time_frame == "3 Months":
        df = df.tail(90)
    elif time_frame == "6 Months":
        df = df.tail(180)
    else:
        df = df.tail(365)

    # Main content
    st.header(f"Market Data Overview - {selected_company}")
    st.dataframe(df.tail(10))

    # Performance Metrics
    st.header("Performance Metrics")
    sharpe, roi, max_drawdown = calculate_metrics(df['Returns'])
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Sharpe Ratio", f"{sharpe:.2f}", delta_color="off")
    with col2:
        st.metric("ROI (%)", f"{roi * 100:.2f}", delta_color="off")
    with col3:
        st.metric("Max Drawdown (%)", f"{max_drawdown * 100:.2f}", delta_color="off")

    # Visualizations
    st.header("Visualizations")

    # Price Trend with Prediction
    st.subheader("Price Trend with DRL Prediction")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.lineplot(x='Date', y='Price', data=df, ax=ax1, label='Actual Price', color='blue')
    sns.lineplot(x='Date', y='Predicted_Price', data=df, ax=ax1, label='Predicted Price', color='orange', linestyle='--')
    ax1.set_title(f"Price Trend vs Prediction - {selected_company}")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price")
    plt.xticks(rotation=45)
    plt.legend()
    st.pyplot(fig1)

    # Comparison with Real Market Data
    st.subheader("Comparison with Real Market Data")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.lineplot(x='Date', y='Predicted_Price', data=df, ax=ax2, label='DRL Prediction', color='orange', linestyle='--')
    sns.lineplot(x='Date', y='Real_Price', data=df, ax=ax2, label='Real Market Price', color='green')
    ax2.set_title(f"DRL Prediction vs Real Market Data - {selected_company}")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Price")
    plt.xticks(rotation=45)
    plt.legend()
    st.pyplot(fig2)

    # Returns Distribution
    st.subheader("Returns Distribution")
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    sns.histplot(df['Returns'], bins=30, kde=True, ax=ax3, color='green')
    ax3.set_title("Distribution of Returns")
    ax3.set_xlabel("Returns")
    ax3.set_ylabel("Frequency")
    st.pyplot(fig3)

    # Market Adaptability Insights
    st.header("Market Adaptability Insights")
    st.write(f"""
    This section will analyze how the DRL model adapts to market conditions for {selected_company}. 
    Currently, it uses dummy data; real insights will be added post-model training.
    """)
    st.line_chart(df['Returns'])

    # Legal Considerations
    st.header("Legal & Ethical Considerations")
    st.write("""
    The trading model adheres to ethical financial practices and complies with regulations (to be detailed with real data).
    """)

    # Future Features
    st.sidebar.write("Future Features:")
    st.sidebar.write("- Real-time market data integration")
    st.sidebar.write("- Trained DRL model predictions")
    st.sidebar.write("- Advanced risk analysis")

if __name__ == "__main__":
    main()




## old
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Set Seaborn style for better visuals
sns.set(style="whitegrid")

# Dummy data generation function
def generate_dummy_data():
    dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(90)]
    prices = np.random.normal(loc=100, scale=10, size=len(dates)).cumsum()
    returns = np.diff(prices) / prices[:-1]
    dummy_data = pd.DataFrame({
        'Date': dates,
        'Price': prices,
        'Returns': np.append(0, returns)  # Add 0 for first return
    })
    return dummy_data

# Dummy performance metrics
def calculate_metrics(returns):
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized Sharpe
    roi = (returns + 1).prod() - 1  # Cumulative ROI
    max_drawdown = (np.maximum.accumulate(returns + 1) - (returns + 1)).max()
    return sharpe_ratio, roi, max_drawdown

# Streamlit UI
def main():
    # Title and description
    st.title("DRL-Based Algorithmic Trading Dashboard")
    st.write("""
    This dashboard visualizes the performance of a Deep Reinforcement Learning (DRL)-based trading model. 
    Currently, it uses dummy data, but it’s designed to integrate real or simulated market data once the model is trained.
    """)

    # Sidebar for settings
    st.sidebar.header("Settings")
    data_source = st.sidebar.selectbox("Data Source", ["Dummy Data", "Upload Data (Coming Soon)"])
    time_frame = st.sidebar.selectbox("Time Frame", ["1 Month", "3 Months", "6 Months", "1 Year"])
    
    # Generate or load data
    if data_source == "Dummy Data":
        df = generate_dummy_data()
    else:
        st.sidebar.write("Upload functionality will be added once the model is trained.")
        df = generate_dummy_data()  # Fallback to dummy data for now

    # Filter data based on time frame (dummy implementation)
    if time_frame == "1 Month":
        df = df.tail(30)
    elif time_frame == "3 Months":
        df = df.tail(90)
    elif time_frame == "6 Months":
        df = df.tail(180)
    else:
        df = df.tail(365)

    # Main content
    st.header("Market Data Overview")
    st.dataframe(df.tail(10))  # Show last 10 rows

    # Performance Metrics
    st.header("Performance Metrics")
    sharpe, roi, max_drawdown = calculate_metrics(df['Returns'])
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
    with col2:
        st.metric("ROI (%)", f"{roi * 100:.2f}")
    with col3:
        st.metric("Max Drawdown (%)", f"{max_drawdown * 100:.2f}")

    # Visualizations
    st.header("Visualizations")

    # Price Trend
    st.subheader("Price Trend")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.lineplot(x='Date', y='Price', data=df, ax=ax1, color='blue')
    ax1.set_title("Price Trend Over Time")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price")
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    # Returns Distribution
    st.subheader("Returns Distribution")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.histplot(df['Returns'], bins=30, kde=True, ax=ax2, color='green')
    ax2.set_title("Distribution of Returns")
    ax2.set_xlabel("Returns")
    ax2.set_ylabel("Frequency")
    st.pyplot(fig2)

    # Cumulative Returns
    st.subheader("Cumulative Returns")
    df['Cumulative_Returns'] = (df['Returns'] + 1).cumprod() - 1
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    sns.lineplot(x='Date', y='Cumulative_Returns', data=df, ax=ax3, color='purple')
    ax3.set_title("Cumulative Returns Over Time")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Cumulative Returns")
    plt.xticks(rotation=45)
    st.pyplot(fig3)

    # Market Adaptability Insights
    st.header("Market Adaptability Insights")
    st.write("""
    This section will analyze how the model adapts to changing market conditions once real data is integrated. 
    For now, it shows dummy insights based on random fluctuations.
    """)
    st.line_chart(df['Returns'])

    # Legal Considerations
    st.header("Legal & Ethical Considerations")
    st.write("""
    The trading model adheres to ethical financial practices and complies with regulations (to be detailed once real data is used).
    """)

    # Placeholder for future dynamic data integration
    st.sidebar.write("Future Features:")
    st.sidebar.write("- Upload real market data")
    st.sidebar.write("- Connect trained DRL model outputs")
    st.sidebar.write("- Advanced risk analysis")

if __name__ == "__main__":
    main()


## test 2.0 chatgpt

import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Dummy Data
data = {
    'Date': pd.date_range(start='1/1/2023', periods=50, freq='D'),
    'Price': [100 + i + (i % 5) * 2 for i in range(50)],
    'Signal': ['Buy' if i % 10 == 0 else 'Sell' if i % 15 == 0 else 'Hold' for i in range(50)]
}
df = pd.DataFrame(data)

# Streamlit UI
st.title("DRL-Based Algorithmic Trading Dashboard")

# Sidebar for controls
st.sidebar.header("Controls")
data_source = st.sidebar.selectbox("Select Data Source", ["Dummy Data", "Real-Time Data"])
model_status = st.sidebar.radio("Model Status", ["Not Trained", "Training", "Trained"])

st.sidebar.subheader("Settings")
threshold = st.sidebar.slider("Risk Threshold", 0.0, 1.0, 0.5)

def plot_price_chart():
    st.subheader("Price Trend")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=df, x='Date', y='Price', ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

def plot_signal_distribution():
    st.subheader("Trading Signal Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x=df['Signal'], palette="viridis", ax=ax)
    st.pyplot(fig)

# Display data
st.subheader("Trading Data")
st.dataframe(df)

# Display Graphs
plot_price_chart()
plot_signal_distribution()

# Placeholder for Results
st.subheader("Performance Metrics")
st.write("Sharpe Ratio: --")
st.write("Return on Investment (ROI): --")
st.write("Maximum Drawdown: --")

st.subheader("Future Work")
st.write("Once the model is trained, real-time data integration will be enabled.")
