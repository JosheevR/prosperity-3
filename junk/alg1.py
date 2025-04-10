from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order

class Trader:
    def __init__(self):
        self.acceptable_price = 10000

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {}

        for product in state.order_depths.keys():
            if product == "KELP":
                continue
            
            order_depth: OrderDepth = state.order_depths[product]
            orders: list[Order] = []

            # Get current position for the product
            current_position = state.position.get(product, 0)

            # Buy logic (if price is good)
            if len(order_depth.sell_orders) > 0:
                best_ask = min(order_depth.sell_orders.keys())
                best_ask_volume = order_depth.sell_orders[best_ask]
                if best_ask < self.acceptable_price:
                    self.acceptable_price = best_ask
                    orders.append(Order(product, best_ask, best_ask_volume))  # BUY

            # Sell logic (only if we hold enough product)
            if len(order_depth.buy_orders) > 0:
                best_bid = max(order_depth.buy_orders.keys())
                best_bid_volume = order_depth.buy_orders[best_bid]

                # Only sell up to what we have
                if best_bid > self.acceptable_price and current_position > 0:
                    sell_quantity = min(best_bid_volume, current_position)
                    orders.append(Order(product, best_bid, -sell_quantity))  # SELL

            result[product] = orders

        traderData = "SAMPLE"
        conversions = 1 
        return result, conversions, traderData
