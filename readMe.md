# High Low Ticker Counter and Market Simulator

A Python-based stock market simulator that generates synthetic stock price data and tracks new 52-week highs and lows for a list of symbols, with a real-time web UI to display the results.

## Overview

This project simulates a stock market by generating price movements for a set of stock symbols using a geometric Brownian motion model. It tracks daily highs/lows, 52-week highs/lows, and categorizes stocks by price ranges. The simulation includes features like:
- Volatility adjustments based on market hours (e.g., higher volatility at open/close).
- Random price spikes to mimic real-world market events.
- A real-time web UI to display new highs and lows, including percentage price changes.
- Sortable tables with animated highlights for new entries.
- Dual line charts under the "New Highs" and "New Lows" sections to visualize price trends for selected stocks during the current trading session (8:00 AM to 8:00 PM EDT).
- Sliders to adjust simulation speed (0.0x to 2.0x, default 0.0x) and volatility (0.5x to 2.0x).

## Prerequisites

- **Python 3.7+**: Required to run the simulation backend.
- **Node.js 14+ and npm**: Required to run the WebSocket server and serve the UI.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/jach8/HighLowTicker_sim_ui.git
   cd HighLowTicker_sim_ui
   ```

2. install python dependencies: 
    ```bash
    pip install -r requirements.txt
    ```
3. Install Node.js dependencies:
    ```bash
   npm install
   ```

## Usage
- **Option 1:** (Recommended) Run the simulator with a web UI:
  - Start the webSocket server which will also run the Python simulation. 
  ```bash
  npm start 
  ```
  - Open `index.html` in your web browser to view the real-time simulation.
    - Alternatively you can use a simplt HTTP server to serve the file: 
    ```bash
      npx http-server -p 3000
    ```
      - Then navigate to `http://localhost:3000` in your web browser.

The UI will display (simulated) real-time updates of new highs and lows, including counts over different time periods and a table of symbols with their latest prices.

- **Option 2:** Run the simulator without a web UI:
  - You can run the simulator directly in Python to see the output in the console.

     ```bash
    python simulator.py
    ```
    - This simulator will ooutput JSON data to the console, showing the state of the market including:
      - New highs/lows for each symbol.
      - 52-week highs/lows.
      - Price range distribution (low: < $10, mid: $10-$50, high: > $50).
      - Counts of new highs/lows over different time windows (30s, 5m, 20m).
    - Logs are written to `simulation.log` for debugging.

#### The simulator will run indefinitely, generating new price data every second. You can stop the simulation by pressing `Ctrl+C`.

## Project Structure
`simulator.py`: Core simulation logic and stock price tracking.
`main.py`: Entry point to run the simulation.
`tickers.json`: List of stock symbols to simulate.
`requirements.txt`: Python dependencies.
`simulation.log`: Log file for debugging.


## Features
- Simulates stock price movements using a geometric Brownian motion model.
- Adjusts volatility based on market hours.
- Introduces random price spikes to simulate market events.
- Tracks 52-week highs/lows and daily price movements.
- Outputs data in JSON format for easy integration with other tools.
- Uses REact and ReactDom via CDN for rendering the UI 
- Uses Babel for JSX transpilation 
- uses Tailwind CSS for styling the UI 

## Limitations
- This is a simplified simulation and does not reflect real market dynamics.
- Does not include real-time data or connectivity to actual market APIs.
- The UI is basic and could be enhanced with additional visualizations (e.g., charts).



License
MIT License




----
### Example of the Ticker in Action 

![til](HighLowTicker.gif)