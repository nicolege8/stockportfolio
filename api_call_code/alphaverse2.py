from dotenv import load_dotenv
import os, requests, json

load_dotenv()  # reads .env into environment

API_KEY = '0ZA4ALIL661ZO90Z'
base_url = "https://www.alphavantage.co/query"
params = {
    "function":   "EARNINGS_CALL_TRANSCRIPT",
    "symbol":    "WBD",
    "quarter":  "2025Q2",
    "apikey":     API_KEY,
}

r = requests.get(base_url, params=params)
data = r.json()
# print(data)

with open("wbd_q2_2025.json", "w") as f:
    json.dump(data, f, indent=2)    

print("Saved transcript to wbd_q2_2025.json")