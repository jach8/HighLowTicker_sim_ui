import random
import math
import asyncio
import json
import logging
import numpy as np
from collections import defaultdict
import sys
import time
from ticker import HighLowTicker

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


class StockMarketSimulator:
    def __init__(self, queue_size=100):
        logger.info("Initializing StockMarketSimulator")
        self.symbols = self.load_symbols()
        self.queue = asyncio.Queue(queue_size)
        self.ticker = HighLowTicker()
        self.simulated_data = {}
        self.price_histories = {}
        self.stock_params = {}
        self.previous_prices = {}
        self.seconds_in_day = 6.5 * 60 * 60  # 6.5 hours
        self.current_time = 0
        self.speed_multiplier = 1.0
        self.volatility_adjustment = 1.0
        self.initialize_simulation()

    def load_symbols(self):
        try:
            with open('tickers.json', 'r') as f:
                data = json.load(f)
                symbols = list(set(data['symbols']))
                logger.info(f"Loaded {len(symbols)} unique symbols")
                return symbols
        except Exception as e:
            logger.error(f"Failed to load tickers.json: {e}, using default symbols")
            return ["SPY", "TSLA", "AAPL"]

    def initialize_simulation(self):
        for i, symbol in enumerate(self.symbols):
            if i % 3 == 0:
                start_price = random.uniform(5, 9)
                volatility = 0.5
            elif i % 3 == 1:
                start_price = random.uniform(15, 45)
                volatility = 0.3
            else:
                start_price = random.uniform(100, 400)
                volatility = 0.2

            trend_type = random.choice(['bullish', 'bearish', 'neutral'])
            drift = 0.05 if trend_type == 'bullish' else -0.05 if trend_type == 'bearish' else 0.0

            self.stock_params[symbol] = {'volatility': volatility, 'drift': drift}

            price_history = []
            current_price = start_price
            for day in range(252):
                daily_return = (drift - 0.5 * volatility**2) / 252 + volatility * math.sqrt(1/252) * random.gauss(0, 1)
                current_price *= math.exp(daily_return)
                price_history.append(current_price)

            self.price_histories[symbol] = price_history

            self.simulated_data[symbol] = {
                "key": symbol,
                "LAST_PRICE": start_price,
                "HIGH_PRICE": start_price,
                "LOW_PRICE": start_price,
                "HIGH_PRICE_52_WEEK": max(price_history),
                "LOW_PRICE_52_WEEK": min(price_history),
                "NET_CHANGE_PERCENT": 0.0
            }
            self.previous_prices[symbol] = start_price

        logger.info(f"Initialized simulation with {len(self.symbols)} symbols")

    @staticmethod
    def update_rate():
        return np.random.uniform(0.15, 0.30)

    async def read_stdin(self):
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)

        while True:
            try:
                line = await reader.readline()
                if not line:
                    await asyncio.sleep(0.1)
                    continue
                line = line.decode().strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                    if message.get('type') == 'setSpeed':
                        self.speed_multiplier = max(0.1, min(2.0, float(message.get('value', 1.0))))
                        logger.info(f"Updated speed multiplier to {self.speed_multiplier}")
                    elif message.get('type') == 'setVolatility':
                        self.volatility_adjustment = max(0.5, min(2.0, float(message.get('value', 1.0))))
                        logger.info(f"Updated volatility adjustment to {self.volatility_adjustment}")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse stdin message: {line}, error: {e}")
                except Exception as e:
                    logger.error(f"Error processing stdin message: {e}")
            except Exception as e:
                logger.error(f"Error reading from stdin: {e}")
                await asyncio.sleep(1)

    async def simulate_stream(self):
        logger.info("Starting simulated stream loop")
        dt = 1 / self.seconds_in_day
        seconds_per_minute = 60
        spike_probability_per_minute = 0.1
        last_spike_check = 0
        last_direction = None

        while True:
            try:
                simulated_msg = {"content": []}
                self.current_time = (self.current_time + 1) % self.seconds_in_day

                minutes_since_open = self.current_time / 60
                base_volatility_multiplier = 1.0
                if minutes_since_open <= 30:
                    base_volatility_multiplier = 2.0
                elif 360 <= minutes_since_open <= 390:
                    base_volatility_multiplier = 2.0
                elif 330 <= minutes_since_open <= 390:
                    base_volatility_multiplier = 1.5
                volatility_multiplier = base_volatility_multiplier * self.volatility_adjustment

                if (self.current_time - last_spike_check) >= seconds_per_minute:
                    last_spike_check = self.current_time
                    spike_probability = spike_probability_per_minute / len(self.symbols)

                num_stocks_to_update = max(1, len(self.symbols) // 5)
                stocks_to_update = random.sample(self.symbols, num_stocks_to_update)

                for symbol in stocks_to_update:
                    stock = self.simulated_data[symbol]
                    params = self.stock_params[symbol]
                    volatility = params['volatility'] * volatility_multiplier
                    drift = params['drift']

                    mu = drift - 0.5 * volatility**2
                    z = random.gauss(0, 1)
                    if last_direction == 'high':
                        log_return = mu * dt + volatility * math.sqrt(dt) * z - 0.01
                        last_direction = 'low'
                    else:
                        log_return = mu * dt + volatility * math.sqrt(dt) * z + 0.01
                        last_direction = 'high'

                    new_price = stock["LAST_PRICE"] * math.exp(log_return)

                    if (self.current_time - last_spike_check) >= seconds_per_minute and random.random() < spike_probability:
                        spike = 1.1 if random.random() > 0.5 else 0.9
                        new_price *= spike
                        logger.info(f"Price spike for {symbol}: {spike}x")

                    previous_price = self.previous_prices[symbol]
                    if previous_price != 0:
                        percent_change = ((new_price - previous_price) / previous_price) * 100
                    else:
                        percent_change = 0.0
                    self.previous_prices[symbol] = new_price

                    new_high = max(stock["HIGH_PRICE"], new_price)
                    new_low = min(stock["LOW_PRICE"], new_price)

                    self.price_histories[symbol].append(new_price)
                    if len(self.price_histories[symbol]) > 252 * int(self.seconds_in_day):
                        self.price_histories[symbol] = self.price_histories[symbol][-252 * int(self.seconds_in_day):]

                    stock["LAST_PRICE"] = new_price
                    stock["HIGH_PRICE"] = new_high
                    stock["LOW_PRICE"] = new_low
                    stock["HIGH_PRICE_52_WEEK"] = max(self.price_histories[symbol])
                    stock["LOW_PRICE_52_WEEK"] = min(self.price_histories[symbol])
                    stock["NET_CHANGE_PERCENT"] = percent_change

                    simulated_msg["content"].append(stock)

                await self.queue.put(simulated_msg)
                logger.debug(f"Added simulated message to queue, queue size: {self.queue.qsize()}")

                base_delay = self.update_rate() 
                sleep_delay_time = base_delay / self.speed_multiplier
                logger.debug(f"Sleep delay: {sleep_delay_time} seconds (speed multiplier: {self.speed_multiplier})")
                await asyncio.sleep(sleep_delay_time)

            except Exception as e:
                logger.error(f"Error in simulate_stream: {e}")
                await asyncio.sleep(1)

    async def handle_queue(self):
        last_message_time = time.time()
        while True:
            try:
                msg = await asyncio.wait_for(self.queue.get(), timeout=10.0)
                if 'content' not in msg:
                    logger.warning("Message missing 'content' field")
                    continue

                for stock in msg['content']:
                    self.ticker.process_stock(stock)
                    print(json.dumps(self.ticker.get_state()))

                last_message_time = time.time()
            except asyncio.TimeoutError:
                logger.warning("No messages in queue after 10 seconds")
                if time.time() - last_message_time > 60:
                    logger.warning("No messages received in 60 seconds")
                continue
            except Exception as e:
                logger.error(f"Queue processing error: {e}")

    async def stream(self):
        asyncio.ensure_future(self.handle_queue())
        asyncio.ensure_future(self.simulate_stream())
        asyncio.ensure_future(self.read_stdin())
        await asyncio.Future()