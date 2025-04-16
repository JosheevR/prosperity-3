from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle
import numpy as np
import math

class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"


PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        # for making
        "disregard_edge": 1,  # disregards orders for joining or pennying within this value from fair
        "join_edge": 2,  # joins orders within this edge
        "default_edge": 4,
        "soft_position_limit": 10,
    },
    Product.KELP: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.229,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
    },
    Product.SQUID_INK: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 10,
        "reversion_beta": .1,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
    },
    Product.CROISSANTS: {
        "take_width": 1.75,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 250,
        "reversion_beta": 0,
        "disregard_edge": 10,
        "join_edge": 10,
        "default_edge": 5,
    },
    Product.JAMS: {
        "take_width": 2,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 35, # optimized
        "reversion_beta": 0.2,  # optimized
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
    },
    #     Product.JAMS: {
    #     "disregard_edge": 1,
    #     "join_edge": 1,
    #     "default_edge": 3,
    #     "soft_position_limit": 30
    # },
    Product.DJEMBES: {
        "disregard_edge": 1,
        "join_edge": 1,
        "default_edge": 3,
        "soft_position_limit": 10
    },
    Product.PICNIC_BASKET1: {
        "disregard_edge": 1,
        "join_edge": 1,
        "default_edge": 3,
        "soft_position_limit": 10
    },
    Product.PICNIC_BASKET2: {
        "disregard_edge": 10,
        "join_edge": 10,
        "default_edge": 5,
        "soft_position_limit": 40,
        "reversion_beta": 5 # 2016 @ 10
    }
}


class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {
            Product.RAINFOREST_RESIN: 50,
            Product.KELP: 50,
            Product.SQUID_INK: 50,
            Product.CROISSANTS: 250,
            Product.JAMS: 350,
            Product.DJEMBES: 60,
            Product.PICNIC_BASKET1: 60,
            Product.PICNIC_BASKET2: 100
        }

    def take_best_orders(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (int, int):
        position_limit = self.LIMIT[product]

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(
                        best_ask_amount, position_limit - position
                    )  # max amt to buy
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]

            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(
                        best_bid_amount, position_limit + position
                    )  # should be the max we can sell
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))  # Buy order

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))  # Sell order
        return buy_order_volume, sell_order_volume

    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> List[Order]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            # Aggregate volume from all buy orders with price greater than fair_for_ask
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            # Aggregate volume from all sell orders with price lower than fair_for_bid
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def KELP_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params[Product.KELP]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params[Product.KELP]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if traderObject.get("KELP_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["KELP_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("KELP_last_price", None) != None:
                last_price = traderObject["KELP_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params[Product.KELP]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["KELP_last_price"] = mmmid_price
            return fair
        return None


    def SQUID_INK_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params[Product.SQUID_INK]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params[Product.SQUID_INK]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if traderObject.get("SQUID_INK_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["SQUID_INK_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("SQUID_INK_last_price", None) != None:
                last_price = traderObject["SQUID_INK_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params[Product.SQUID_INK]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["SQUID_INK_last_price"] = mmmid_price
            return fair
        return None
    
    def CROISSANTS_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params[Product.CROISSANTS]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params[Product.CROISSANTS]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if traderObject.get("CROISSANTS_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["CROISSANTS_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("CROISSANTS_last_price", None) != None:
                last_price = traderObject["CROISSANTS_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params[Product.CROISSANTS]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["CROISSANTS_last_price"] = mmmid_price
            return fair
        return None
    
    def JAMS_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params[Product.JAMS]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params[Product.JAMS]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if traderObject.get("JAMS_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["JAMS_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("JAMS_last_price", None) != None:
                last_price = traderObject["JAMS_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params[Product.JAMS]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["JAMS_last_price"] = mmmid_price
            return fair
        return None

    def picnic_basket1_fair_value(self, state):
        croissant_book = state.order_depths.get(Product.CROISSANTS)
        jam_book = state.order_depths.get(Product.JAMS)
        djembe_book = state.order_depths.get(Product.DJEMBES)

        def mid_price(book):
            if book and book.buy_orders and book.sell_orders:
                return (max(book.buy_orders) + min(book.sell_orders)) / 2
            return None

        croissant_mid = mid_price(croissant_book)
        jam_mid = mid_price(jam_book)
        djembe_mid = mid_price(djembe_book)

        if croissant_mid and jam_mid and djembe_mid:
            return 6 * croissant_mid + 3 * jam_mid + 1 * djembe_mid
        return None
    
    def PICNIC_BASKET2_fair_value(self, state, traderObject) -> float:
        croissant_book = state.order_depths.get(Product.CROISSANTS)
        jam_book = state.order_depths.get(Product.JAMS)

        def mid_price(book):
            if book and book.buy_orders and book.sell_orders:
                return (max(book.buy_orders) + min(book.sell_orders)) / 2
            return None

        croissant_mid = mid_price(croissant_book)
        jam_mid = mid_price(jam_book)

        if croissant_mid and jam_mid:
            mid = 4 * croissant_mid + 2 * jam_mid
            last_price = traderObject.get("PICNIC_BASKET2_last_price", None)

            if last_price is not None:
                last_return = (mid - last_price) / last_price
                beta = self.params[Product.PICNIC_BASKET2]["reversion_beta"]
                pred_return = last_return * beta
                fair = mid + (mid * pred_return)
            else:
                fair = mid

            traderObject["PICNIC_BASKET2_last_price"] = mid
            return fair
        return None



    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        buy_order_volume, sell_order_volume = self.take_best_orders(
            product,
            fair_value,
            take_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
            prevent_adverse,
            adverse_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def make_orders(
        self,
        product,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,  # disregard trades within this edge for pennying or joining
        join_edge: float,  # join trades within this edge
        default_edge: float,  # default edge to request if there are no levels to penny or join
        manage_position: bool = False,
        soft_position_limit: int = 0,
        # will penny all other levels with higher edge
    ):
        orders: List[Order] = []
        asks_above_fair = [
            price
            for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        bids_below_fair = [
            price
            for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge
        ]

        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        ask = round(fair_value + default_edge)
        if best_ask_above_fair != None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair  # join
            else:
                ask = best_ask_above_fair - 1  # penny

        bid = round(fair_value - default_edge)
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -1 * soft_position_limit:
                bid += 1

        buy_order_volume, sell_order_volume = self.market_make(
            product,
            orders,
            bid,
            ask,
            position,
            buy_order_volume,
            sell_order_volume,
        )

        return orders, buy_order_volume, sell_order_volume

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}

        if Product.RAINFOREST_RESIN in self.params and Product.RAINFOREST_RESIN in state.order_depths:
            rainforest_resin_position = (
                state.position[Product.RAINFOREST_RESIN]
                if Product.RAINFOREST_RESIN in state.position
                else 0
            )
            rainforest_resin_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.RAINFOREST_RESIN,
                    state.order_depths[Product.RAINFOREST_RESIN],
                    self.params[Product.RAINFOREST_RESIN]["fair_value"],
                    self.params[Product.RAINFOREST_RESIN]["take_width"],
                    rainforest_resin_position,
                )
            )
            rainforest_resin_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.RAINFOREST_RESIN,
                    state.order_depths[Product.RAINFOREST_RESIN],
                    self.params[Product.RAINFOREST_RESIN]["fair_value"],
                    self.params[Product.RAINFOREST_RESIN]["clear_width"],
                    rainforest_resin_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            rainforest_resin_make_orders, _, _ = self.make_orders(
                Product.RAINFOREST_RESIN,
                state.order_depths[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                rainforest_resin_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.RAINFOREST_RESIN]["disregard_edge"],
                self.params[Product.RAINFOREST_RESIN]["join_edge"],
                self.params[Product.RAINFOREST_RESIN]["default_edge"],
                True,
                self.params[Product.RAINFOREST_RESIN]["soft_position_limit"],
            )
            result[Product.RAINFOREST_RESIN] = (
                rainforest_resin_take_orders + rainforest_resin_clear_orders + rainforest_resin_make_orders
            )

        if Product.KELP in self.params and Product.KELP in state.order_depths:
            KELP_position = (
                state.position[Product.KELP]
                if Product.KELP in state.position
                else 0
            )
            KELP_fair_value = self.KELP_fair_value(
                state.order_depths[Product.KELP], traderObject
            )
            KELP_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    KELP_fair_value,
                    self.params[Product.KELP]["take_width"],
                    KELP_position,
                    self.params[Product.KELP]["prevent_adverse"],
                    self.params[Product.KELP]["adverse_volume"],
                )
            )
            KELP_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    KELP_fair_value,
                    self.params[Product.KELP]["clear_width"],
                    KELP_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            KELP_make_orders, _, _ = self.make_orders(
                Product.KELP,
                state.order_depths[Product.KELP],
                KELP_fair_value,
                KELP_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.KELP]["disregard_edge"],
                self.params[Product.KELP]["join_edge"],
                self.params[Product.KELP]["default_edge"],
            )
            result[Product.KELP] = (
                KELP_take_orders + KELP_clear_orders + KELP_make_orders
            )

        if Product.CROISSANTS in self.params and Product.CROISSANTS in state.order_depths:
            CROISSANTS_position = (
                state.position[Product.CROISSANTS]
                if Product.CROISSANTS in state.position
                else 0
            )
            CROISSANTS_fair_value = self.CROISSANTS_fair_value(
                state.order_depths[Product.CROISSANTS], traderObject
            )
            CROISSANTS_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.CROISSANTS,
                    state.order_depths[Product.CROISSANTS],
                    CROISSANTS_fair_value,
                    self.params[Product.CROISSANTS]["take_width"],
                    CROISSANTS_position,
                    self.params[Product.CROISSANTS]["prevent_adverse"],
                    self.params[Product.CROISSANTS]["adverse_volume"],
                )
            )
            CROISSANTS_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.CROISSANTS,
                    state.order_depths[Product.CROISSANTS],
                    CROISSANTS_fair_value,
                    self.params[Product.CROISSANTS]["clear_width"],
                    CROISSANTS_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            CROISSANTS_make_orders, _, _ = self.make_orders(
                Product.CROISSANTS,
                state.order_depths[Product.CROISSANTS],
                CROISSANTS_fair_value,
                CROISSANTS_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.CROISSANTS]["disregard_edge"],
                self.params[Product.CROISSANTS]["join_edge"],
                self.params[Product.CROISSANTS]["default_edge"],
            )
            result[Product.CROISSANTS] = (
                CROISSANTS_take_orders + CROISSANTS_clear_orders + CROISSANTS_make_orders
            )

            basket1_fair = self.picnic_basket1_fair_value(state)

            if basket1_fair:
                basket_book = state.order_depths[Product.PICNIC_BASKET1]
                basket_pos = state.position.get(Product.PICNIC_BASKET1, 0)
                croissant_pos = state.position.get(Product.CROISSANTS, 0)
                jam_pos = state.position.get(Product.JAMS, 0)
                djembe_pos = state.position.get(Product.DJEMBES, 0)

                result[Product.PICNIC_BASKET1] = []
                buy_order_volume = 0
                sell_order_volume = 0

                buy_order_volume, sell_order_volume = self.take_best_orders(
                    Product.PICNIC_BASKET1, basket1_fair, 1, result[Product.PICNIC_BASKET1], basket_book,
                    basket_pos, buy_order_volume, sell_order_volume
                )

                buy_order_volume, sell_order_volume = self.clear_position_order(
                    Product.PICNIC_BASKET1, basket1_fair, 1, result[Product.PICNIC_BASKET1], basket_book,
                    basket_pos, buy_order_volume, sell_order_volume
                )

                mm_orders, _, _ = self.make_orders(
                    Product.PICNIC_BASKET1, basket_book, basket1_fair, basket_pos,
                    buy_order_volume, sell_order_volume,
                    1, 1, 3, True, 10
                )
                result[Product.PICNIC_BASKET1].extend(mm_orders)
            
            if Product.SQUID_INK in self.params:
                SQUID_INK_position = (
                    state.position[Product.SQUID_INK]
                    if Product.SQUID_INK in state.position
                    else 0
                )
                SQUID_INK_fair_value = self.SQUID_INK_fair_value(
                    state.order_depths[Product.SQUID_INK], traderObject
                )
                SQUID_INK_take_orders, buy_order_volume, sell_order_volume = (
                    self.take_orders(
                        Product.SQUID_INK,
                        state.order_depths[Product.SQUID_INK],
                        SQUID_INK_fair_value,
                        self.params[Product.SQUID_INK]["take_width"],
                        SQUID_INK_position,
                        self.params[Product.SQUID_INK]["prevent_adverse"],
                        self.params[Product.SQUID_INK]["adverse_volume"],
                    )
                )
                SQUID_INK_clear_orders, buy_order_volume, sell_order_volume = (
                    self.clear_orders(
                        Product.SQUID_INK,
                        state.order_depths[Product.SQUID_INK],
                        SQUID_INK_fair_value,
                        self.params[Product.SQUID_INK]["clear_width"],
                        SQUID_INK_position,
                        buy_order_volume,
                        sell_order_volume,
                    )
                )
                SQUID_INK_make_orders, _, _ = self.make_orders(
                    Product.SQUID_INK,
                    state.order_depths[Product.SQUID_INK],
                    SQUID_INK_fair_value,
                    SQUID_INK_position,
                    buy_order_volume,
                    sell_order_volume,
                    self.params[Product.SQUID_INK]["disregard_edge"],
                    self.params[Product.SQUID_INK]["join_edge"],
                    self.params[Product.SQUID_INK]["default_edge"],
                )
                result[Product.SQUID_INK] = (
                    SQUID_INK_take_orders + SQUID_INK_clear_orders + SQUID_INK_make_orders
                )

            # if Product.JAMS in self.params:
            #     JAMS_position = (
            #         state.position[Product.JAMS]
            #         if Product.JAMS in state.position
            #         else 0
            #     )
            #     JAMS_fair_value = self.JAMS_fair_value(
            #         state.order_depths[Product.JAMS], traderObject
            #     )
            #     JAMS_take_orders, buy_order_volume, sell_order_volume = (
            #         self.take_orders(
            #             Product.JAMS,
            #             state.order_depths[Product.JAMS],
            #             JAMS_fair_value,
            #             self.params[Product.JAMS]["take_width"],
            #             JAMS_position,
            #             self.params[Product.JAMS]["prevent_adverse"],
            #             self.params[Product.JAMS]["adverse_volume"],
            #         )
            #     )
            #     JAMS_clear_orders, buy_order_volume, sell_order_volume = (
            #         self.clear_orders(
            #             Product.JAMS,
            #             state.order_depths[Product.JAMS],
            #             JAMS_fair_value,
            #             self.params[Product.JAMS]["clear_width"],
            #             JAMS_position,
            #             buy_order_volume,
            #             sell_order_volume,
            #         )
            #     )
            #     JAMS_make_orders, _, _ = self.make_orders(
            #         Product.JAMS,
            #         state.order_depths[Product.JAMS],
            #         JAMS_fair_value,
            #         JAMS_position,
            #         buy_order_volume,
            #         sell_order_volume,
            #         self.params[Product.JAMS]["disregard_edge"],
            #         self.params[Product.JAMS]["join_edge"],
            #         self.params[Product.JAMS]["default_edge"],
            #     )
            #     result[Product.JAMS] = (
            #         JAMS_take_orders + JAMS_clear_orders + JAMS_make_orders
            #     )

        # Option Strategy Handling
        option_strikes = [9500, 9750, 10000, 10250, 10500]
        for strike in option_strikes:
            product = f"VOLCANIC_ROCK_VOUCHER_{strike}"
            if product not in state.order_depths:
                continue

            order_depth = state.order_depths[product]
            position = state.position.get(product, 0)
            orders = []
            buy_vol = sell_vol = 0

            # === 1. Buying cheap deep OTM options ===
            if strike in [9500, 9750]:
                # Market might misprice deep OTM if IV is low → accumulate small positions
                bid_price = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
                ask_price = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else 9999
                mid_price = (bid_price + ask_price) / 2
                cheap_threshold = 30  # define your "cheap" threshold
                if mid_price < cheap_threshold:
                    buy_qty = min(10, self.LIMIT.get(product, 20) - position)
                    orders.append(Order(product, ask_price, buy_qty))

            # === 2. Focus on ATM 10000 ===
            elif strike == 10000:
                # Provide liquidity on both sides (delta/gamma scalping)
                mid = (max(order_depth.buy_orders) + min(order_depth.sell_orders)) / 2 if order_depth.buy_orders and order_depth.sell_orders else None
                if mid:
                    buy_price = int(mid - 2)
                    sell_price = int(mid + 2)
                    orders.append(Order(product, buy_price, 5))
                    orders.append(Order(product, sell_price, -5))

            # === 3. Sell volatility at 9500 or 9750 if IV spiked ===
            elif strike in [9500, 9750]:
                # Sell if prices have risen sharply (mimicking IV spike)
                recent_price = max(order_depth.buy_orders) if order_depth.buy_orders else 0
                if recent_price > 100:  # arbitrary high IV threshold
                    sell_qty = min(5, self.LIMIT.get(product, 10) + position)
                    orders.append(Order(product, recent_price, -sell_qty))

            # === 4. Directional trades for higher strikes ===
            elif strike in [10250, 10500]:
                # Simple trend-following breakout — buy if consistently rising
                prev_price = traderObject.get(f"{product}_last", None)
                current_price = (max(order_depth.buy_orders) + min(order_depth.sell_orders)) / 2 if order_depth.buy_orders and order_depth.sell_orders else None
                if prev_price and current_price and current_price > prev_price * 1.01:
                    buy_qty = min(5, self.LIMIT.get(product, 10) - position)
                    orders.append(Order(product, min(order_depth.sell_orders), buy_qty))
                if current_price:
                    traderObject[f"{product}_last"] = current_price

            if orders:
                result[product] = orders
     


        # if Product.SQUID_INK in self.params and Product.SQUID_INK in state.order_depths:
        #     SQUID_INK_position = (
        #         state.position[Product.SQUID_INK]
        #         if Product.SQUID_INK in state.position
        #         else 0
        #     )
        #     SQUID_INK_fair_value = self.SQUID_INK_fair_value(
        #         state.order_depths[Product.SQUID_INK], traderObject
        #     )
        #     SQUID_INK_take_orders, buy_order_volume, sell_order_volume = (
        #         self.take_orders(
        #             Product.SQUID_INK,
        #             state.order_depths[Product.SQUID_INK],
        #             SQUID_INK_fair_value,
        #             self.params[Product.SQUID_INK]["take_width"],
        #             SQUID_INK_position,
        #             self.params[Product.SQUID_INK]["prevent_adverse"],
        #             self.params[Product.SQUID_INK]["adverse_volume"],
        #         )
        #     )
        #     SQUID_INK_clear_orders, buy_order_volume, sell_order_volume = (
        #         self.clear_orders(
        #             Product.SQUID_INK,
        #             state.order_depths[Product.SQUID_INK],
        #             SQUID_INK_fair_value,
        #             self.params[Product.SQUID_INK]["clear_width"],
        #             SQUID_INK_position,
        #             buy_order_volume,
        #             sell_order_volume,
        #         )
        #     )
        #     SQUID_INK_make_orders, _, _ = self.make_orders(
        #         Product.SQUID_INK,
        #         state.order_depths[Product.SQUID_INK],
        #         SQUID_INK_fair_value,
        #         SQUID_INK_position,
        #         buy_order_volume,
        #         sell_order_volume,
        #         self.params[Product.SQUID_INK]["disregard_edge"],
        #         self.params[Product.SQUID_INK]["join_edge"],
        #         self.params[Product.SQUID_INK]["default_edge"],
        #     )
        #     result[Product.SQUID_INK] = (
        #         SQUID_INK_take_orders + SQUID_INK_clear_orders + SQUID_INK_make_orders
        #     )

        basket1_fair = self.picnic_basket1_fair_value(state)

        if basket1_fair and Product.PICNIC_BASKET1 in state.order_depths:
            basket_book = state.order_depths[Product.PICNIC_BASKET1]
            basket_pos = state.position.get(Product.PICNIC_BASKET1, 0)
            croissant_pos = state.position.get(Product.CROISSANTS, 0)
            jam_pos = state.position.get(Product.JAMS, 0)
            djembe_pos = state.position.get(Product.DJEMBES, 0)

            result[Product.PICNIC_BASKET1] = []
            buy_order_volume = 0
            sell_order_volume = 0

            buy_order_volume, sell_order_volume = self.take_best_orders(
                Product.PICNIC_BASKET1, basket1_fair, 1, result[Product.PICNIC_BASKET1], basket_book,
                basket_pos, buy_order_volume, sell_order_volume
            )

            buy_order_volume, sell_order_volume = self.clear_position_order(
                Product.PICNIC_BASKET1, basket1_fair, 1, result[Product.PICNIC_BASKET1], basket_book,
                basket_pos, buy_order_volume, sell_order_volume
            )

            mm_orders, _, _ = self.make_orders(
                Product.PICNIC_BASKET1, basket_book, basket1_fair, basket_pos,
                buy_order_volume, sell_order_volume,
                1, 1, 3, True, 10
            )
            result[Product.PICNIC_BASKET1].extend(mm_orders)

        basket2_fair = self.PICNIC_BASKET2_fair_value(state, traderObject)

        if basket2_fair and Product.PICNIC_BASKET2 in state.order_depths:
            basket2_book = state.order_depths[Product.PICNIC_BASKET2]
            basket2_pos = state.position.get(Product.PICNIC_BASKET2, 0)

            result[Product.PICNIC_BASKET2] = []
            buy_order_volume = 0
            sell_order_volume = 0

            buy_order_volume, sell_order_volume = self.take_best_orders(
                Product.PICNIC_BASKET2, basket2_fair, 1, result[Product.PICNIC_BASKET2], basket2_book,
                basket2_pos, buy_order_volume, sell_order_volume
            )

            buy_order_volume, sell_order_volume = self.clear_position_order(
                Product.PICNIC_BASKET2, basket2_fair, 1, result[Product.PICNIC_BASKET2], basket2_book,
                basket2_pos, buy_order_volume, sell_order_volume
            )

            mm_orders, _, _ = self.make_orders(
                Product.PICNIC_BASKET2, basket2_book, basket2_fair, basket2_pos,
                buy_order_volume, sell_order_volume,
                self.params[Product.PICNIC_BASKET2]["disregard_edge"], self.params[Product.PICNIC_BASKET2]["join_edge"], self.params[Product.PICNIC_BASKET2]["default_edge"], True, self.params[Product.PICNIC_BASKET2]["soft_position_limit"]
            )
            result[Product.PICNIC_BASKET2].extend(mm_orders)


        conversions = 1
        traderData = jsonpickle.encode(traderObject)

        return result, conversions, traderData
