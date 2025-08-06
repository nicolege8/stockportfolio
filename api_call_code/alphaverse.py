from dotenv import load_dotenv
import os, requests, json

load_dotenv()  # reads .env into environment

API_KEY = '9DRAJDN4DCWHUDG5'
base_url = "https://www.alphavantage.co/query"
params = {
    "function":   "NEWS_SENTIMENT",
    "tickers":    "MSFT",
    "topics": "technology",
    "time_from":  "20240101T0000",
    "time_to":    "20240101T0554",
    "limit":      "1000",
    "apikey":     API_KEY,
}

r = requests.get(base_url, params=params)
data = r.json()
# print(data)

with open("msft_sentiment_34.json", "w") as f:
    json.dump(data, f, indent=2)

print("Saved", len(data.get("feed", [])), "articles to msft_sentiment_34.json")


# import requests

# # replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
# url = 'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=MSFT&time_from=20240101T0000&time_to=20250629T2359&limit=1000&apikey='
# r = requests.get(url)
# data = r.json()

# print(data)