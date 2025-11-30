import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import solve_ivp

# ============================
# 1. Black–Scholes functions
# ============================

def black_scholes_call(S, K, T, r, sigma):
    """
    Black–Scholes price for a European call option.
    S: spot price (array or scalar)
    K: strike price (scalar)
    T: time to maturity in years (array or scalar)
    r: risk-free rate (annual)
    sigma: volatility (annual)
    """
    S = np.array(S, dtype=float)
    T = np.array(T, dtype=float)
    
    # Avoid division by zero / log of zero
    mask = (T > 0) & (sigma > 0) & (S > 0) & (K > 0)
    price = np.full_like(S, np.nan, dtype=float)
    
    if np.isscalar(T):
        T_eff = np.full_like(S, T, dtype=float)
    else:
        T_eff = T
    
    d1 = np.zeros_like(S, dtype=float)
    d2 = np.zeros_like(S, dtype=float)
    
    d1[mask] = (np.log(S[mask] / K) + (r + 0.5 * sigma**2) * T_eff[mask]) / (sigma * np.sqrt(T_eff[mask]))
    d2[mask] = d1[mask] - sigma * np.sqrt(T_eff[mask])
    
    price[mask] = S[mask] * norm.cdf(d1[mask]) - K * np.exp(-r * T_eff[mask]) * norm.cdf(d2[mask])
    price[~mask & (T_eff == 0)] = np.maximum(S[~mask & (T_eff == 0)] - K, 0.0)  # at maturity
    
    return price


# ============================
# 2. Helper functions
# ============================

def load_price_data(csv_path):
    """
    Load CSV with columns: Date,Open,High,Low,Close,Volume
    Returns DataFrame with Date as datetime index.
    """
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df

def estimate_annual_volatility(close_prices, trading_days_per_year=252):
    """
    Estimate annualized volatility from daily close prices using log returns.
    """
    close = np.array(close_prices, dtype=float)
    log_returns = np.diff(np.log(close))
    daily_vol = np.std(log_returns, ddof=1)
    annual_vol = daily_vol * np.sqrt(trading_days_per_year)
    return annual_vol

def build_time_axis(num_points, trading_days_per_year=252):
    """
    Create a time axis in years from 0 to (num_points-1)/trading_days_per_year.
    """
    days = np.arange(num_points)
    t_years = days / trading_days_per_year
    return t_years


# ============================
# 3. ODE model for underlying
# ============================

def geometric_growth_ode(t, S, mu):
    """
    dS/dt = mu * S
    Simple deterministic geometric growth model.
    """
    return mu * S

def solve_price_ode(S0, t_span, t_eval, mu):
    """
    Solve dS/dt = mu S from t_span[0] to t_span[1] with initial S0.
    """
    sol = solve_ivp(
        fun=lambda t, S: geometric_growth_ode(t, S, mu),
        t_span=t_span,
        y0=[S0],
        t_eval=t_eval,
    )
    return sol.y[0]


# ============================
# 4. Main analysis + plotting
# ============================

def analyze_scenario(csv_path, scenario_name="Scenario"):
    # ---- Load data ----
    df = load_price_data(csv_path)
    close = df["Close"].values
    dates = df["Date"].values
    n = len(df)

    # ---- Build time in years ----
    trading_days_per_year = 252
    t = build_time_axis(n, trading_days_per_year=trading_days_per_year)

    # ---- Estimate parameters ----
    S0 = close[0]
    K = S0  # at-the-money strike
    r = 0.02  # 2% risk-free rate, adjust as you like
    sigma = estimate_annual_volatility(close)
    
    # Simple drift estimate from log returns mean
    log_returns = np.diff(np.log(close))
    daily_mu = np.mean(log_returns)
    mu_annual = daily_mu * trading_days_per_year

    print(f"--- {scenario_name} ---")
    print(f"S0    = {S0:.2f}")
    print(f"r     = {r:.4f}")
    print(f"sigma (annualized) = {sigma:.4f}")
    print(f"mu (annualized drift from data) = {mu_annual:.4f}")
    print()

    # ---- ODE model for underlying price ----
    t_span = (t[0], t[-1])
    S_ode = solve_price_ode(S0=S0, t_span=t_span, t_eval=t, mu=mu_annual)

    # ---- Black–Scholes call prices over time ----
    # Assume an option that matures some days after the last data point.
    extra_days_to_maturity = 30
    T_final = t[-1] + extra_days_to_maturity / trading_days_per_year  # maturity time (in years)

    # Time-to-maturity at each observation
    T = T_final - t
    T[T < 0] = 0.0  # after maturity, clamp to 0

    call_bs_from_market = black_scholes_call(S=close, K=K, T=T, r=r, sigma=sigma)
    call_bs_from_ode    = black_scholes_call(S=S_ode, K=K, T=T, r=r, sigma=sigma)

    # ============================
    # 5. Plotting
    # ============================

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # --- Plot underlying price & ODE model ---
    ax1 = axes[0]
    ax1.plot(dates, close, marker="o", label="Market Close Price (CSV)")
    ax1.plot(dates, S_ode, linestyle="--", label="ODE Model S(t) (dS/dt = μS)")
    ax1.set_ylabel("Underlying Price")
    ax1.set_title(f"Underlying Price vs ODE Model — {scenario_name}")
    ax1.legend()
    ax1.grid(True)
    
    # --- Plot Black–Scholes call prices ---
    ax2 = axes[1]
    ax2.plot(dates, call_bs_from_market, marker="o", label="BS Call (using market S)")
    ax2.plot(dates, call_bs_from_ode, linestyle="--", label="BS Call (using ODE S)")
    ax2.set_ylabel("Call Option Price")
    ax2.set_title("Black–Scholes Call Prices Over Time")
    ax2.legend()
    ax2.grid(True)

    plt.xlabel("Date")
    plt.tight_layout()
    plt.show()


# ============================
# 6. Run for one of your CSVs
# ============================

if __name__ == "__main__":
    # Change this to any of your CSV files:
    # e.g. "stable_market.csv", "random_walk.csv", "volatile_market.csv", ...
    
    ### A “normal” stock with a gentle upward trend and low volatility.
    #csv_file = "stable_market.csv"
    #analyze_scenario(csv_file, scenario_name="Stable Market")

    ### A stock drifting around with no strong up or down trend.
    #csv_file = "random_walk.csv"
    #analyze_scenario(csv_file, scenario_name="Random Walk Market")

    ### Big price swings but not necessarily crashing or pumping.
    #csv_file = "volatile_market.csv"
    #analyze_scenario(csv_file, scenario_name="High Volatility Market")

    ### A sharp drawdown with fear and forced selling.
    #csv_file = "market_crash.csv"
    #analyze_scenario(csv_file, scenario_name="Market Crash")

    ### A sudden, aggressive move up
    csv_file = "pump_scenario.csv"
    analyze_scenario(csv_file, scenario_name="Pump / Short Squeeze")
