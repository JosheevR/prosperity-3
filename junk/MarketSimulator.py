import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from datamodel import TradingState, OrderDepth, Order, Observation
from junk.alg1 import Trader

class MarketSimulator:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path, delimiter=';')
        self.products= self.df["product"].unique().tolist()
        self.trader = Trader()
        self.timestamps = sorted(self.df['timestamp'].unique())
        self.log = pd.DataFrame(columns=["timestamp", "symbol", "price", "quantity", "position", "profit_loss", "realized_profit_loss"])

        self.profit_loss = 0
        self.realized_profit_loss = 0
        self.position = {product: 0 for product in self.products}
        self.inventory_cost = {product: 0.0 for product in self.products}

    def simulate(self):
        for timestamp in self.timestamps:
            print(f"\n--- Timestamp {timestamp} ---")
            trading_state = self.build_trading_state(timestamp)
            orders, conversions, trader_data = self.trader.run(trading_state)

            for symbol, order_list in orders.items():
                for order in order_list:
                    price = order.price
                    quantity = order.quantity
                    current_position = self.position[symbol]
                    new_position = current_position + quantity
                    trade_value = -price * quantity
                    new_profit_loss = self.profit_loss + trade_value

                    if abs(new_position) <= 50:
                        if quantity > 0:  # BUY
                            total_cost = self.inventory_cost[symbol] * current_position
                            total_cost += price * quantity
                            self.position[symbol] = new_position
                            self.profit_loss = new_profit_loss
                            self.inventory_cost[symbol] = total_cost / new_position if new_position != 0 else 0
                            print(f"BUY {symbol}:  price={price}, quantity={quantity}")

                        elif quantity < 0:  # SELL
                            avg_cost = self.inventory_cost[symbol]
                            realized_profit = (price - avg_cost) * (-quantity)
                            self.profit_loss = new_profit_loss
                            self.realized_profit_loss += realized_profit
                            self.position[symbol] = new_position
                            print(f"SELL {symbol}:  price={price}, quantity={-quantity} | realized P&L={realized_profit:.2f}")

                        self.log.loc[len(self.log)] = {
                            "timestamp": timestamp,
                            "symbol": symbol,
                            "price": price,
                            "quantity": quantity,
                            "position": self.position[symbol],
                            "profit_loss": self.profit_loss,
                            "realized_profit_loss": self.realized_profit_loss
                        }
        return self.log
        

    def build_trading_state(self, timestamp):
        traderData = None
        timestamp = timestamp
        listings = {}
        order_depths = self.build_order_depths(timestamp)
        own_trades = {}
        market_trades = {}
        position = self.position
        observations = None

        trading_state = TradingState(
                 traderData,
                 timestamp,
                 listings,
                 order_depths,
                 own_trades,
                 market_trades,
                 position,
                 observations)
        
        return trading_state

    def build_order_depths(self, timestamp):
        order_depths = {}
        df_time = self.df[self.df["timestamp"] == timestamp]

        for _, row in df_time.iterrows():
            product = row["product"]

            if product not in order_depths:
                order_depths[product] = OrderDepth()

            for i in range(1, 4):
                bid_price = row.get(f"bid_price_{i}")
                bid_volume = row.get(f"bid_volume_{i}")
                if pd.notna(bid_price) and pd.notna(bid_volume):
                    order_depths[product].buy_orders[int(bid_price)] = int(bid_volume)

                ask_price = row.get(f"ask_price_{i}")
                ask_volume = row.get(f"ask_volume_{i}")
                if pd.notna(ask_price) and pd.notna(ask_volume):
                    order_depths[product].sell_orders[int(ask_price)] = int(ask_volume)

        return order_depths