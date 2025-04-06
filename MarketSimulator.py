import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from datamodel import TradingState, OrderDepth, Order, Observation
from alg1 import Trader

class MarketSimulator:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path, delimiter=';')
        self.trader = Trader()
        self.timestamps = sorted(self.df['timestamp'].unique())
        self.position_limit = 50
        self.positions = defaultdict(int)           # symbol → net position
        self.cash = defaultdict(float)              # symbol → net cash flow
        self.pnl_over_time = defaultdict(list)      # symbol → list of P&L snapshots
        self.time_points = []

    def simulate(self):
        for timestamp in self.timestamps:
            df_time = self.df[self.df['timestamp'] == timestamp]
            state = self._build_trading_state(df_time, timestamp)
            orders, conversions, trader_data = self.trader.run(state)

            print(f"\n--- Timestamp {timestamp} ---")
            for symbol, order_list in orders.items():
                for order in order_list:
                    new_position = self.positions[symbol] + order.quantity
                    if abs(new_position) <= self.position_limit:
                        self.positions[symbol] = new_position
                        # Realized P&L logic:
                        self.cash[symbol] -= order.price * order.quantity  # Buy is negative, Sell is positive
                        print(f"{order} -> ✅ Executed | Pos: {self.positions[symbol]} | Cash: {self.cash[symbol]}")
                    else:
                        print(f"{order} -> ❌ Skipped | Would exceed limit")

            # Snapshot P&L
            self.time_points.append(timestamp)
            for symbol in self.positions:
                # Realized P&L is negative cash
                realized_pnl = -self.cash[symbol]
                self.pnl_over_time[symbol].append(realized_pnl)

        self.plot_pnl()

    def _build_trading_state(self, df: pd.DataFrame, timestamp: int) -> TradingState:
        order_depths = {}
        listings = {}
        own_trades = {}
        market_trades = {}
        plain_obs = {}
        conv_obs = {}

        for _, row in df.iterrows():
            symbol = row["product"]
            depth = OrderDepth()

            for i in range(1, 4):
                price = row[f"buy_p{i}"]
                volume = row[f"buy_v{i}"]
                if pd.notna(price) and pd.notna(volume):
                    depth.buy_orders[int(price)] = int(volume)

            for i in range(1, 4):
                price = row[f"sell_p{i}"]
                volume = row[f"sell_v{i}"]
                if pd.notna(price) and pd.notna(volume):
                    depth.sell_orders[int(price)] = -int(volume)

            order_depths[symbol] = depth

        observation = Observation(plain_obs, conv_obs)

        return TradingState(
            traderData="",
            timestamp=timestamp,
            listings=listings,
            order_depths=order_depths,
            own_trades=own_trades,
            market_trades=market_trades,
            position=self.positions.copy(),
            observations=observation
        )

    def plot_pnl(self):
        plt.figure(figsize=(10, 5))
        for symbol, pnl_series in self.pnl_over_time.items():
            plt.plot(self.time_points, pnl_series, label=symbol)
        plt.title("Realized P&L Over Time")
        plt.xlabel("Timestamp")
        plt.ylabel("P&L")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()