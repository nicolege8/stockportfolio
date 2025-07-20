# hit rate limit... but have json files with 1000 articles within time span! without sentiment
from polygon import RESTClient
from polygon.rest.models import (
    TickerNews,
)

import os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("POLYGON_API_KEY")
client = RESTClient(API_KEY)

news = []
for n in client.list_ticker_news(
	ticker="MSFT",
	published_utc_gte="2024-01-01",
	published_utc_lte="2025-06-29",
	order="asc",
	limit="10",
	sort="published_utc",
	):
    news.append(n)

print(news)




