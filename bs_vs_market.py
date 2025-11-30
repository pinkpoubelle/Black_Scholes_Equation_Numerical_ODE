import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from datetime import datetime
import matplotlib.pyplot as plt


def black_scholes_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return np.nan
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_put(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return np.nan
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def estimate_historical_vol(ticker, lookback_days=252):
    data = yf.Ticker(ticker).history(period=f"{lookback_days}d")
    data = data.dropna(subset=["Close"])
    data["log_return"] = np.log(data["Close"] / data["Close"].shift(1))
    daily_vol = data["log_return"].std()
    ann_vol = daily_vol * np.sqrt(252)
    return float(ann_vol)


def analyze_black_scholes_vs_market(
    ticker: str,
    risk_free_rate: float = 0.04,
    use_hist_vol: bool = True,
    custom_vol: float | None = None,
    option_type: str = "call",
    min_volume: int = 1,
):
    tk = yf.Ticker(ticker)
    spot = tk.history(period="1d")["Close"].iloc[-1]
    print(f"Ticker: {ticker}")
    print(f"Spot price (S): {spot:.2f}")

    expiries = tk.options
    if not expiries:
        print("No options data available for this ticker.")
        return None, None, None

    nearest_expiry_str = expiries[0]
    expiry_date = datetime.strptime(nearest_expiry_str, "%Y-%m-%d").date()
    today = datetime.utcnow().date()
    T = (expiry_date - today).days / 365.0

    print(f"Using nearest expiry: {nearest_expiry_str} (T = {T:.4f} years)")
    if T <= 0:
        print("Nearest expiry is today or in the past; choose a later expiry.")
        return None, None, None

    if use_hist_vol or custom_vol is None:
        sigma_est = estimate_historical_vol(ticker)
        print(f"Estimated historical annualized volatility: {sigma_est:.4f}")
        sigma = sigma_est if custom_vol is None else custom_vol
    else:
        sigma = custom_vol

    print(f"Using volatility (sigma): {sigma:.4f}")
    print(f"Using risk-free rate (r): {risk_free_rate:.4f}")
    print(f"Analyzing option type: {option_type.upper()}")

    opt_chain = tk.option_chain(nearest_expiry_str)
    if option_type.lower() == "call":
        df_opt = opt_chain.calls.copy()
    else:
        df_opt = opt_chain.puts.copy()

    df_opt = df_opt[df_opt["volume"] >= min_volume].copy()
    if df_opt.empty:
        print("No options with sufficient volume found.")
        return None, None, None

    df_opt["mid"] = (df_opt["bid"] + df_opt["ask"]) / 2
    df_opt = df_opt.dropna(subset=["mid"])

    prices_bs = []
    for _, row in df_opt.iterrows():
        K = float(row["strike"])
        if option_type.lower() == "call":
            theo = black_scholes_call(spot, K, T, risk_free_rate, sigma)
        else:
            theo = black_scholes_put(spot, K, T, risk_free_rate, sigma)
        prices_bs.append(theo)

    df_opt["bs_price"] = prices_bs
    df_opt["abs_error"] = df_opt["bs_price"] - df_opt["mid"]
    df_opt["abs_pct_error"] = df_opt["abs_error"].abs() / df_opt["mid"] * 100
    df_opt["moneyness"] = df_opt["strike"] / spot
    df_opt = df_opt.sort_values("moneyness")

    print("\nSample comparison (first 15 options by moneyness):")
    cols_to_show = [
        "strike",
        "lastPrice",
        "mid",
        "bs_price",
        "abs_error",
        "abs_pct_error",
        "volume",
        "openInterest",
    ]
    print(df_opt[cols_to_show].head(15).to_string(index=False, float_format=lambda x: f"{x:8.4f}"))

    mae = df_opt["abs_error"].abs().mean()
    mape = df_opt["abs_pct_error"].mean()
    print("\nOverall error statistics:")
    print(f"Mean absolute error (MAE): {mae:.4f}")
    print(f"Mean absolute percentage error (MAPE): {mape:.2f}%")

    # return df AND a few useful scalars for plotting
    return df_opt, spot, option_type

def plot_black_scholes_vs_market(df_opt, spot, option_type: str):
    """
    Make two plots:
    1) Market vs BS price vs strike
    2) Absolute % error vs strike
    """
    if df_opt is None or df_opt.empty:
        print("No data to plot.")
        return

    # Sort by strike for cleaner plots
    df_plot = df_opt.sort_values("strike")

    strikes = df_plot["strike"].values
    market_prices = df_plot["mid"].values
    bs_prices = df_plot["bs_price"].values
    errors_pct = df_plot["abs_pct_error"].values

    plt.figure()
    plt.plot(strikes, market_prices, marker="o", label="Market mid price")
    plt.plot(strikes, bs_prices, marker="x", label="Black–Scholes price")
    plt.axvline(spot, linestyle="--", label=f"Spot price S={spot:.2f}")
    plt.xlabel("Strike price K")
    plt.ylabel("Option price")
    plt.title(f"{option_type.capitalize()} options: Market vs Black–Scholes")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(strikes, errors_pct, marker="o")
    plt.axvline(spot, linestyle="--", label=f"Spot price S={spot:.2f}")
    plt.xlabel("Strike price K")
    plt.ylabel("Absolute % error")
    plt.title(f"{option_type.capitalize()} options: |Error| % vs Strike")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    df_opt, spot, opt_type = analyze_black_scholes_vs_market(
        ticker="AAPL",
        risk_free_rate=0.04,
        use_hist_vol=True,
        custom_vol=None,
        option_type="call",
        min_volume=10,
    )

    plot_black_scholes_vs_market(df_opt, spot, opt_type)
