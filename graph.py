import streamlit as st
import pandas as pd
import requests
from textblob import TextBlob
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.linear_model import LinearRegression
import numpy as np
import altair as alt
import openai

# Configure NLTK
nltk.download('vader_lexicon')

# Configure Streamlit
st.set_page_config(page_icon="💬", layout="wide", page_title="Stock Prediction & Groq Chat Dashboard")


client = openai.Client(
    api_key='Your_api',
    base_url='Base_url'
)

# Function to analyze sentiment
def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment

# Function to fetch stock data
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Function to perform sentiment analysis
def perform_sentiment_analysis(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment
    return sentiment.polarity, sentiment.subjectivity

# Function to get stock market data
@st.cache_data
def get_stock_market_data(stock_name):
    API_ENDPOINT = "End_point"

    headers = {
        "name": stock_name,
        "Authorization": "Auth_api" 
    }

    response = requests.request("GET", API_ENDPOINT, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        return {}

# Function to get current price
@st.cache_data
def current_price(stock_name):
    API_ENDPOINT = "APi_Endpoint"

    headers = {
        "name": stock_name,
        "Authorization": "APi_auth" 
    }

    response = requests.request("GET", API_ENDPOINT, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        return {}

# Function to get key metrics
@st.cache_data
def key_metrics(stock_name):
    return get_stock_market_data(stock_name)

# Function to get chat completion
def get_chat_response(user_input):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a chatbot assistant."},
            {"role": "user", "content": user_input}
        ]
    )
    return completion.choices[0].message.content

# Streamlit app
def main():
    st.sidebar.title("Explore")
    options = st.sidebar.radio("Go to", ["Home", "Sentiment Analysis", "Stock Price Prediction", "Current Price", "Chat", "Key Metrics"])

    if options == "Home":
        st.title("Stock Prediction Website & Chat Assistant")
        st.write("Welcome to our Stock Prediction Website and Chat Assistant! Our platform is designed to empower users with comprehensive stock market knowledge and predictive insights. By leveraging advanced machine learning algorithms and real-time data analysis, we provide accurate stock predictions and valuable market trends. Our interactive chat assistant is always available to answer your questions, guide you through complex financial concepts, and offer personalized recommendations. Whether you're a novice investor or a seasoned trader, our website is your go-to resource for making informed decisions and staying ahead in the stock market. Join us today and take control of your financial future!")

    elif options == "Sentiment Analysis":
        st.title("Sentiment Analysis")

        new_stock = st.text_input("Enter the stock name to get recent news")
        if new_stock:
            news_results = key_metrics(new_stock).get("recentNews", [])
            if news_results:
                st.subheader("Recent News Results and Sentimental Analysis Result")
                for item in news_results:
                    headline = item['headline']
                    polarity, subjectivity = perform_sentiment_analysis(headline)
                    sentiment_classification = "Good" if polarity > 0 else "Bad" if polarity < 0 else "Neutral"
                    st.write(f"*ID:* {item['id']}")
                    st.write(f"*Headline:* {headline}")
                    st.write(f"*Date:* {item['date']}")
                    st.write(f"*Link:* [Read more]({item['url']})")
                    st.image(item['listimage'], caption=headline)
                    st.write(f"*Sentiment Polarity:* {polarity}")
                    st.write(f"*Sentiment Subjectivity:* {subjectivity}")
                    st.write(f"*Sentiment Classification:* {sentiment_classification}")
                    st.write("---")
            else:
                st.error("Failed to fetch recent news. Please try again later.")

        user_text = st.text_area("Enter text for sentiment analysis:")
        if st.button("Analyze Sentiment"):
            polarity, subjectivity = perform_sentiment_analysis(user_text)
            sentiment_classification = "Good" if polarity > 0 else "Bad" if polarity < 0 else "Neutral"
            st.write(f"Sentiment Polarity: {polarity}")
            st.write(f"Sentiment Subjectivity: {subjectivity}")
            st.write(f"Sentiment Classification: {sentiment_classification}")

    elif options == "Stock Price Prediction":
        st.title("Stock Price Prediction")

        ticker = st.text_input('Enter stock ticker for prediction:')
        days_to_predict = st.slider('Days to predict into the future:', 1, 60, 7)

        if st.button('Predict Stock Prices'):
            end_date = datetime.today().date()
            start_date = end_date - timedelta(days=365)

            stock_data = get_stock_data(ticker, start_date, end_date)

            if not stock_data.empty:
                stock_data['Date'] = stock_data.index
                stock_data['Date'] = stock_data['Date'].map(datetime.toordinal)

                X = stock_data[['Date']].values
                y = stock_data['Close'].values

                model = LinearRegression()
                model.fit(X, y)

                future_dates = np.arange(end_date.toordinal() + 1, end_date.toordinal() + days_to_predict + 1)
                predictions = model.predict(future_dates.reshape(-1, 1))

                variation = np.random.normal(0, 50, size=len(predictions))
                predictions_with_variation = predictions + variation

                prediction_df = pd.DataFrame({
                    'Date': [datetime.fromordinal(date) for date in future_dates],
                    'Predicted Close': predictions_with_variation
                })

                st.write(prediction_df)

                with st.spinner('Generating stock prediction chart...'):
                    fig, ax = plt.subplots()
                    ax.plot(prediction_df['Date'], prediction_df['Predicted Close'], label='Predicted', linestyle='-')
                    ax.set_xlabel('Date')
                    plt.xticks(rotation=90)
                    ax.set_ylabel('Stock Price')
                    ax.legend()
                    st.pyplot(fig)
            else:
                st.write(f'No stock data found for ticker: {ticker}')

    elif options == "Current Price":
        st.title("Pie Chart for Current Price")
        name = st.text_input("Enter stock name 1")
        name1 = st.text_input("Enter stock name 2")
        name2 = st.text_input("Enter stock name 3")

        if name and name1 and name2:
            news_results1 = current_price(name).get('currentPrice', {})
            news_results2 = current_price(name1).get('currentPrice', {})
            news_results3 = current_price(name2).get('currentPrice', {})

            price1 = news_results1.get('NSE', 'N/A')
            price2 = news_results2.get('NSE', 'N/A')
            price3 = news_results3.get('NSE', 'N/A')

            st.write(f"Current price of {name}: {price1}")
            st.write(f"Current price of {name1}: {price2}")
            st.write(f"Current price of {name2}: {price3}")

            data = {
                'Stock': [name, name1, name2],
                'Price': [price1, price2, price3]
            }

            df = pd.DataFrame(data)

            pie_chart = alt.Chart(df).mark_arc().encode(
                theta=alt.Theta(field='Price', type='quantitative'),
                color=alt.Color(field='Stock', type='nominal'),
                tooltip=['Stock', 'Price']
            ).properties(
                width=600,
                height=400
            )

            text = pie_chart.mark_text(radius=120, size=12).encode(
                text=alt.Text('Price:Q'),
                color=alt.value('black')
            )

            chart = pie_chart + text

            st.altair_chart(chart, use_container_width=True)

    elif options == "Chat":
        st.title("Chat Section")
        query = st.text_input("Chatbot Query")
        if query:
            response = get_chat_response(query)
            st.write(response)

    elif options == "Key Metrics":
        st.title("Key Metrics")
        name = st.text_input("Enter stock name for key metrics")
        if name:
            news_results1 = current_price(name).get('companyProfile', {})
            if news_results1:
                st.write(f"*Topic:* {news_results1.get('companyDescription')}")
                st.write(f"*Management:* {news_results1.get('mgIndustry')}")
                st.write(f"*Officers:* {news_results1.get('officers')}")
            else:
                st.error("Failed to fetch current price. Please try again later.")

if __name__ == "__main__":
    main()

