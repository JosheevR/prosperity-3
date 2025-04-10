import pandas as pd
import itertools
import jsonpickle
import os
import copy
from tqdm import tqdm  # For the loading bar

from typing import Dict, Any
from datamodel import OrderDepth, TradingState, Listing, Order
from round_1_v8 import Trader, Product  # Adjust this import according to your project structure

# -----------------------------------------------------------------------------
# Baseline parameters (your original/default settings)
# -----------------------------------------------------------------------------
baseline_params = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 1,
        "join_edge": 2,
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
        "reversion_beta": 0.1,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
    },
}

# -----------------------------------------------------------------------------
# Define a (potentially larger) candidate grid for each parameter per asset.
# You can add more candidates or extend ranges as needed.
# -----------------------------------------------------------------------------
candidate_grid = {
    Product.RAINFOREST_RESIN: {
        "fair_value": [9990, 9995, 9998, 10000, 10002, 10005, 10010],
        "take_width": [1, 2, 3],
        "clear_width": [0, 1],
        "disregard_edge": [0, 1, 2, 3],
        "join_edge": [1, 2, 3],
        "default_edge": [3, 4, 5, 6],
        "soft_position_limit": [5, 10, 15, 20],
    },
    Product.KELP: {
        "take_width": [1, 2, 3],
        "clear_width": [0, 1],
        "prevent_adverse": [True, False],
        "adverse_volume": [10, 15, 20, 25],
        "reversion_beta": [-0.25, -0.229, -0.2, -0.15],
        "disregard_edge": [0, 1, 2],
        "join_edge": [0, 1, 2],
        "default_edge": [1, 2, 3],
    },
    Product.SQUID_INK: {
        "take_width": [1, 2],
        "clear_width": [0, 1, 2],
        "prevent_adverse": [True, False],
        "adverse_volume": [5, 10, 15, 20],
        "reversion_beta": [0.05, 0.1, 0.15, 0.2],
        "disregard_edge": [0, 1],
        "join_edge": [0, 1, 2],
        "default_edge": [1, 2, 3],
    },
}

# -----------------------------------------------------------------------------
# Function to load market data from the CSV files in prosperity2bt/resources/round1/
# -----------------------------------------------------------------------------
def load_market_data() -> pd.DataFrame:
    """
    Reads CSVs for day -2, -1, and 0,
    concatenates them, and returns a single DataFrame.
    Assumes columns: day;timestamp;product;...;profit_and_loss
    """
    folder = "prosperity2bt/resources/round1/"
    
    # Read day -2
    path_neg2 = os.path.join(folder, "prices_round_1_day_-2.csv")
    df_day_neg2 = pd.read_csv(path_neg2, sep=';')
    if 'day' not in df_day_neg2.columns:
        df_day_neg2['day'] = -2

    # Read day -1
    path_neg1 = os.path.join(folder, "prices_round_1_day_-1.csv")
    df_day_neg1 = pd.read_csv(path_neg1, sep=';')
    if 'day' not in df_day_neg1.columns:
        df_day_neg1['day'] = -1

    # Read day 0
    path_0 = os.path.join(folder, "prices_round_1_day_0.csv")
    df_day_0 = pd.read_csv(path_0, sep=';')
    if 'day' not in df_day_0.columns:
        df_day_0['day'] = 0

    # Concatenate data from all days and ensure proper type conversion
    df = pd.concat([df_day_neg2, df_day_neg1, df_day_0], ignore_index=True)
    df['day'] = df['day'].astype(int)
    df['timestamp'] = df['timestamp'].astype(int)
    return df

# -----------------------------------------------------------------------------
# Simulation function (as before)
# -----------------------------------------------------------------------------
def simulate_for_params(df: pd.DataFrame, param_set: Dict[str, Any]) -> float:
    """
    Runs the Trader simulation over all rows in df with the given parameter set,
    and returns the performance metric (the total of the 'profit_and_loss' column).
    """
    trader = Trader(params=param_set)
    total_pnl = 0.0
    
    # Initial positions for the assets.
    current_position = {
        Product.RAINFOREST_RESIN: 0,
        Product.KELP: 0,
        Product.SQUID_INK: 0
    }
    
    traderData = ""
    df_sorted = df.sort_values(by=['day', 'timestamp'])
    
    for idx, row in df_sorted.iterrows():
        product = row['product']
        od = OrderDepth()
        
        # Build the bid side of the order book.
        bid_prices = [('bid_price_1', 'bid_volume_1'),
                      ('bid_price_2', 'bid_volume_2'),
                      ('bid_price_3', 'bid_volume_3')]
        for bp, bv in bid_prices:
            if pd.notnull(row[bp]) and pd.notnull(row[bv]) and row[bv] != 0:
                od.buy_orders[int(row[bp])] = int(row[bv])
        
        # Build the ask side of the order book.
        ask_prices = [('ask_price_1', 'ask_volume_1'),
                      ('ask_price_2', 'ask_volume_2'),
                      ('ask_price_3', 'ask_volume_3')]
        for ap, av in ask_prices:
            if pd.notnull(row[ap]) and pd.notnull(row[av]) and row[av] != 0:
                od.sell_orders[int(row[ap])] = -int(row[av])
        
        state = TradingState(
            timestamp=int(row['timestamp']),
            listings={product: Listing(symbol=product, product=product, denomination=product)},
            order_depths={product: od},
            position=current_position.copy(),
            traderData=traderData,
            own_trades={},
            market_trades={},
            observations={}
        )
        
        result, conversions, new_traderData = trader.run(state)
        traderData = new_traderData
        
        if pd.notnull(row['profit_and_loss']):
            total_pnl += float(row['profit_and_loss'])
    
    return total_pnl

# -----------------------------------------------------------------------------
# Function to optimize an asset’s parameter set using a full grid search.
# -----------------------------------------------------------------------------
def optimize_asset(asset: str, baseline: Dict[str, Any], candidate_grid: Dict[str, Any], df: pd.DataFrame):
    """
    For the given asset, perform a full Cartesian grid search over its candidate
    parameters (while keeping other assets fixed at baseline).
    
    Returns the optimized parameter set for that asset, the best candidate tuple,
    and its performance (PnL).
    """
    # Get the candidate dictionary for this asset.
    asset_candidates = candidate_grid[asset]
    # Define an order for the parameters (using the keys in insertion order).
    keys = list(asset_candidates.keys())
    
    # Create the full Cartesian product of candidate values.
    candidate_tuples = list(itertools.product(*(asset_candidates[k] for k in keys)))
    total_candidates = len(candidate_tuples)
    
    best_candidate = None
    best_pnl = -float('inf')
    
    # Use a single loading bar (tqdm) for all candidate combinations.
    for candidate in tqdm(candidate_tuples, desc=f"Optimizing {asset}", total=total_candidates):
        # Create a copy of the baseline for testing.
        test_params = copy.deepcopy(baseline)
        # For this asset, update parameters using the candidate tuple.
        for i, key in enumerate(keys):
            test_params[asset][key] = candidate[i]
        
        candidate_pnl = simulate_for_params(df, test_params)
        if candidate_pnl > best_pnl:
            best_pnl = candidate_pnl
            best_candidate = candidate
    
    # Update baseline for this asset with the best candidate found.
    optimized = copy.deepcopy(baseline)
    for i, key in enumerate(keys):
        optimized[asset][key] = best_candidate[i]
    
    return optimized[asset], best_candidate, best_pnl

# -----------------------------------------------------------------------------
# Main optimization routine that runs each asset independently.
# -----------------------------------------------------------------------------
def optimize_assets():
    df = load_market_data()
    optimized_params = copy.deepcopy(baseline_params)
    
    # Optimize for each asset individually.
    for asset in candidate_grid:
        print(f"\nStarting optimization for asset: {asset}")
        opt_params, best_candidate, pnl = optimize_asset(asset, baseline_params, candidate_grid, df)
        optimized_params[asset] = opt_params
        print(f"Best parameter set for {asset}:")
        for key, value in opt_params.items():
            print(f"  {key}: {value}")
        print(f"Achieved PnL: {pnl:.2f}")
    
    print("\nFinal optimized parameter set (for all assets):")
    for asset in optimized_params:
        print(f"{asset}: {optimized_params[asset]}")
    
    return optimized_params

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    final_params = optimize_assets()
    print("\nDone! Final optimized parameters:")
    print(final_params)
