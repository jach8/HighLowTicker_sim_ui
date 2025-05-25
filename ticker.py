import random
import math
import asyncio
import json
import logging
import numpy as np 
import time
from collections import defaultdict


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simulation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HighLowTicker:
    def __init__(self):
        logger.info("Initializing HighLowTicker")
        self.new_highs = defaultdict(int)
        self.new_lows = defaultdict(int)
        self.last_high = {}
        self.last_low = {}
        self.last_price = {}
        self.last_pct_change = {}
        self.current_week52_highs = set()
        self.current_week52_lows = set()
        self.week52_high_prices = {}
        self.week52_low_prices = {}
        self.price_ranges = {"low": 0, "mid": 0, "high": 0}
        self.message_count = 0
        self.initialized_symbols = set()
        self.high_timestamps = []
        self.low_timestamps = []

    def process_stock(self, stock):
        symbol = stock.get('key')
        price = stock.get('LAST_PRICE')
        daily_high = stock.get('HIGH_PRICE')
        daily_low = stock.get('LOW_PRICE')
        week52_high = stock.get('HIGH_PRICE_52_WEEK')
        week52_low = stock.get('LOW_PRICE_52_WEEK')
        percent_change = stock.get('NET_CHANGE_PERCENT')
        current_time = time.time()

        if not symbol or price is None or price == 0:
            logger.warning(f"Invalid stock data: {stock}")
            return False

        self.last_price[symbol] = price

        daily_high = daily_high if daily_high is not None else self.last_high.get(symbol, price)
        daily_low = daily_low if daily_low is not None else self.last_low.get(symbol, price)
        week52_high = week52_high if week52_high is not None else self.week52_high_prices.get(symbol, 1e19)
        week52_low = week52_low if week52_low is not None else self.week52_low_prices.get(symbol, -1)

        if symbol not in self.week52_high_prices:
            self.week52_high_prices[symbol] = week52_high
        if symbol not in self.week52_low_prices:
            self.week52_low_prices[symbol] = week52_low

        if symbol not in self.initialized_symbols:
            self.last_high[symbol] = daily_high
            self.last_low[symbol] = daily_low
            self.last_pct_change[symbol] = percent_change if percent_change is not None else 0
            self.initialized_symbols.add(symbol)
            if daily_high >= week52_high:
                self.current_week52_highs.add(symbol)
            if daily_low <= week52_low:
                self.current_week52_lows.add(symbol)
        else:
            self.last_pct_change[symbol] = percent_change if percent_change is not None else self.last_pct_change.get(symbol, 0)
            if daily_high > self.last_high[symbol]:
                self.new_highs[symbol] += 1
                self.last_high[symbol] = daily_high
                self.high_timestamps.append((symbol, current_time))
                if daily_high > self.week52_high_prices[symbol]:
                    logger.info(f"{symbol} NEW 52-WEEK HIGH: {daily_high}")
                    self.week52_high_prices[symbol] = daily_high
                    self.current_week52_highs.add(symbol)
                elif daily_high >= self.week52_high_prices[symbol]:
                    self.current_week52_highs.add(symbol)

            if daily_low < self.last_low[symbol]:
                self.new_lows[symbol] += 1
                self.last_low[symbol] = daily_low
                self.low_timestamps.append((symbol, current_time))
                if daily_low < self.week52_low_prices[symbol]:
                    logger.info(f"{symbol} NEW 52-WEEK LOW: {daily_low}")
                    self.week52_low_prices[symbol] = daily_low
                    self.current_week52_lows.add(symbol)
                elif daily_low <= self.week52_low_prices[symbol]:
                    self.current_week52_lows.add(symbol)

        if price < 10:
            self.price_ranges["low"] += 1
        elif 10 <= price <= 50:
            self.price_ranges["mid"] += 1
        else:
            self.price_ranges["high"] += 1

        self.message_count += 1
        return True

    def get_state(self):
        current_time = time.time()
        high_counts = {
            "30s": sum(1 for _, ts in self.high_timestamps if current_time - ts <= 30),
            "5m": sum(1 for _, ts in self.high_timestamps if current_time - ts <= 360),
            "20m": sum(1 for _, ts in self.high_timestamps if current_time - ts <= 1680),
        }
        low_counts = {
            "30s": sum(1 for _, ts in self.low_timestamps if current_time - ts <= 30),
            "5m": sum(1 for _, ts in self.low_timestamps if current_time - ts <= 360),
            "20m": sum(1 for _, ts in self.low_timestamps if current_time - ts <= 1680),
        }
        self.high_timestamps = [(sym, ts) for sym, ts in self.high_timestamps if current_time - ts <= 300]
        self.low_timestamps = [(sym, ts) for sym, ts in self.low_timestamps if current_time - ts <= 300]

        return {
            "newHighs": dict(self.new_highs),
            "newLows": dict(self.new_lows),
            "lastHigh": self.last_high,
            "lastLow": self.last_low,
            "week52Highs": list(self.current_week52_highs),
            "week52Lows": list(self.current_week52_lows),
            "priceRanges": self.price_ranges,
            "messageCount": self.message_count,
            "highCounts": high_counts,
            "lowCounts": low_counts,
            "percentChange": self.last_pct_change,
            "content": [{"key": symbol, "LAST_PRICE": self.last_price.get(symbol, 0)} for symbol in self.initialized_symbols]
        }
