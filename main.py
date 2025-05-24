import asyncio
import logging
from simulator import StockMarketSimulator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

async def main():
    simulator = StockMarketSimulator()
    await simulator.stream()

if __name__ == '__main__':
    asyncio.run(main())