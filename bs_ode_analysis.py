""" The purpose of this script is to analyze the Black-Scholes option pricing model.
    - compares the analytical solution with a numerical solution which uses the method of lines: the finite difference method in space and the Runge-Kutta 4th order method in time.
    - includes 7 functions:
        1. datasheet_graph: visualizes stock price data from a CSV datasheet.
        2. read_datasheet: reads parameters from a CSV datasheet and converts them into usable variables for the Black-Scholes model.
        3. bs_call_price_exact: computes the analytical Black-Scholes call option price.
        4. bs_call_price: computes the analytical Black-Scholes call option price for entire spatial grid.
        5. method_of_lines_RK4: implements the method of lines with RK4
        6. analyticalvsnumerical: visualizes the numerical and analytical solutions.
        7. error_graph: visualizes the error between numerical and analytical solutions.
   - the main function reads parameters from a datasheet, computes both numerical and analytical solutions, and visualizes the results.
"""

import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt

def datasheet_graph(filepath):
    """
    Visualize the stock price data from the datasheet (CSV file).
    inputs:
        filepath: path to the CSV file
    outputs:
        A plot of stock prices over time.
    """
    df = pd.read_csv(filepath)
    
    # convert 'Date' column to datetime format and filter for the last year
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    latest_date = df["Date"].max()
    one_year_ago = latest_date - pd.Timedelta(days=365)
    close_prices = df[df["Date"] >= one_year_ago]['Close'].astype(float).values

    plt.figure(figsize=(10, 6))
    plt.plot(close_prices, label='Close Prices')
    plt.xlabel('Time (Days)')
    plt.ylabel('Stock Price')
    plt.title("Stock Prices from Datasheet: " + filepath)
    plt.legend()
    plt.grid()
    plt.savefig(filepath + "_stock_prices.png")
    plt.show()

def read_datasheet(filepath):
    """
    Read parameters from a datasheet (CSV file) and convert them into the usable variables for the black-scholes model.
    The stock price S is taken as the 'Close' price on December 31, 2024 in the datasheet.
    With this function, we calculate volatility from the 'Close' prices in the datasheet in the time range of the year of 2024  and set other parameters (K, r, T) using estimated values(us national treasury interest rate).
    inputs:
        filepath: path to the CSV file
    outputs:
        factors: dictionary of black-scholes parameters: S, K, r, sigma, T
    """
    df = pd.read_csv(filepath)

    # convert 'Date' column to datetime format and filter for the last year to calculate volatility
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    latest_date = df["Date"].max()
    one_year_ago = latest_date - pd.Timedelta(days=365)
    close_prices = df[df["Date"] >= one_year_ago]['Close'].astype(float).values

    # calculate volatility as the standard deviation of log returns
    log_returns = np.log(close_prices[1:] / close_prices[:-1])
    sigma = np.std(log_returns) * np.sqrt(252)  # annualized volatility

    S = close_prices[-1]  # current stock price
    K = close_prices[-1] * 1.05  # example strike price set at 5% above last close
    r = 0.03  # example risk-free rate
    T = 30 / 252  # example time to maturity of 30 days
    factors = {
        'S': S,
        'K': K,
        'r': r,
        'sigma': sigma,
        'T': T
    }

    print("Using Stock Price (S):", close_prices[-1])
    print("Using Strike Price (K):", K)
    print("Using Risk-Free Rate (r):", r)
    print("Using Calculated Volatility (sigma):", sigma)
    print("Using Time to Maturity (T):", T)

    return factors

def bs_call_price_exact(S, K, r, sigma, T):
    """
    Analytical Black–Scholes formula for a European call option.
    inputs:
        S : stock price
        K : strike
        r : risk-free rate
        sigma : volatility
        T : time to maturity (years)

    outputs:
        C : call option price
    """
    if T == 0:
        return max(S - K, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    C = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return C

def bs_call_price(S, K, r, sigma, T):
    """
    Analytical Black Scholes formula for a European call option.
    inputs:
        S: stock price
        K: strike
        r: risk-free rate
        sigma: volatility
        T: time to maturity (years)
    outputs:
        C: call option price
    """

    S = np.array(S, dtype=float)
    S_safe = np.maximum(S, 1e-12) # avoid log(0) issues

    # if the time to maturity is zero, return the original value
    if T == 0:
        return max(S - K, 0)
    
    # Calculate the intermediate values d1 and d2
    # d1 the option's delta: expected benefit from acquiring the stock outright, weighted by a risk-adjusted probability that the option will be exercised (in-the-money)
    # d2 the risk-adjusted probability that the option will be exercised at expiration (P(S > K)), and is used t calculate the present value of the contingent exercise payment (K * exp(-rT) * N(d2)
    d1 = (np.log(S_safe / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T)) 
    d2 = d1 - sigma * np.sqrt(T) # risk-adjusted profitability measure

    # calculate the call option price
    C = S_safe * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return C

def method_of_lines_RK4(S_max, K, r, sigma, T, M, N):
    """
    Method of lines for the Black-Scholes PDE (simplifies the complex PDE problem by discretizing in space):
    Takes the finite difference in S to reduce it into a system of ODEs in t, then applies the RK4 method backward in time.
    inputs:
        S_max: max stock price on grid
        K: strike
        r: risk-free rate
        sigma: volatility, 
        T: time to maturity (years)
        M: number of spatial intervals, determined by the user
        N: number of RK4 time steps, determined by the user

    outputs:
        S: stock price grid (array length M+1)
        C: call option prices at t=0 (array length M+1)
    """
    # spatial grid
    dS = float(S_max) / M
    S = np.linspace(0, float(S_max), M + 1, dtype=float) # stock price grid from 0 to S_max with M+1 points
    # time step using interval T and N steps
    dt = float(T) / N

    # terminal condition (t = T)
    C = np.maximum(S - K, 0).astype(float)  # call option payoff at maturity

    # boundary conditions functions (depend on t which should be C(0, t) = 0, C(S_max, t) = S_max - K * exp(-r(T-t)))
    def BC_left(t):
        return 0.0  # C(0, t) = 0 for a call option
    
    def BC_right(t):
        # time remaining to maturity is (T - t)
        return S_max - K * np.exp(-r * (T - t))  # C(S_max, t)
    
    # the ODE function dC/dt = F(C, t)
    def F(C, t):
        Cb = C.copy().astype(float)
        # boundary conditions on the copy
        Cb[0] = BC_left(t)
        Cb[-1] = BC_right(t)

        dCdt = np.zeros_like(Cb, dtype=float)

        # Interior nodes i = 1 ... M-1
        for i in range(1, M):
            dC_dS  = (Cb[i+1] - Cb[i-1]) / (2 * dS)
            d2C_dS2 = (Cb[i+1] - 2*Cb[i] + Cb[i-1]) / (dS**2)

            # the PDE transformed into ODE form:
            dCdt[i] = (0.5*sigma**2*S[i]**2*d2C_dS2) + (r*S[i]*dC_dS) - (r*Cb[i])
        
        # edges so index 0 and M are just the boundary conditions already set by BC_left and BC_right
        return dCdt
    
    # RK4 time-stepping backward in time t = T to t = 0
    t = float(T)
    # time-stepping loop
    for n in range(N):
        k1 = F(C, t)
        k2 = F(C + 0.5 * dt * k1, t - 0.5 * dt)
        k3 = F(C + 0.5 * dt * k2, t - 0.5 * dt)
        k4 = F(C + dt * k3, t - dt)

        # update C backward in time
        C += (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        t -= dt

        # boundary conditions at new time level
        C[0] = BC_left(t)
        C[-1] = BC_right(t)
        
    return S, C # return to main function as S and C_numerical

def analyticalvsnumerical(filepath, S, C_analytical, C_numerical):
    """
    Visualize the numerical and analytical Black-Scholes call option prices.
    inputs:
        S: stock price grid
        C_numerical: numerical call option prices from method_of_lines_RK4
        K: strike
        r: risk-free rate
        sigma: volatility
        T: time to maturity (years)
    outputs:
        A plot comparing numerical and analytical solutions.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(S, C_numerical, label='Numerical (MOL + RK4)', linestyle='--')
    plt.plot(S, C_analytical, label='Analytical (Black-Scholes)', linestyle='-')
    plt.xlabel('Stock Price S')
    plt.ylabel('Call Option Price C')
    plt.title("Black-Scholes Call Option Price for file [" + filepath + "]: Numerical vs Analytical")
    plt.legend()
    plt.grid()
    # plt.savefig(filepath + f"bs_comparison.png")
    plt.show()

def error_graph(filepath, error, C_analytical, C_numerical, S):
    """
    Visualize the error between numerical and analytical solutions.
    inputs:
        C_analytical: analytical call option prices
        C_numerical: numerical call option prices
        S: stock price grid
    outputs:
        A plot of the absolute error.
    """
    
    plt.figure(figsize=(10, 6))
    plt.plot(S, error, label='Absolute Error', color='red')
    plt.xlabel('Stock Price S')
    plt.ylabel('Absolute Error |C_analytical - C_numerical|')
    plt.title('Error between Analytical and Numerical Black-Scholes Call Option Prices')
    plt.legend()
    plt.grid()
    # plt.savefig(filepath + "bs_error.png")
    plt.show() 

if __name__ == "__main__":
    """ Main function to run the Black-Scholes analysis. """
    # read parameters from datasheet using apple last 5 years of stock data
    factors = read_datasheet('tsla_2025.csv')

    # set parameters for numerical method and analytical solution
    S0 = factors['S'] # current stock price
    print(  "Current Stock Price S0:", S0)
    S_max = 3 * S0  # max stock price on grid
    K = factors['K'] # strike price
    r = factors['r'] # risk-free rate
    sigma = factors['sigma'] # volatility
    T = factors['T'] # time to maturity (years)

    grid_sizes_M = [25, 50, 100, 200] # number of spatial intervals
    N = 1500 # number of RK4 time steps

    S_grids = []
    C_numerical = [] 
    C_analytical = []
    Errors = [] 

    # compute numerical solution and compare with analytical solution for different grid sizes
    for i, M in enumerate(grid_sizes_M):

        print(f"\nComputing numerical solution with M = {M} spatial intervals...")
        S, C_num = method_of_lines_RK4(S_max, K, r, sigma, T, M, N)
        C_an = bs_call_price(S, K, r, sigma, T)

        error = np.abs(C_an - C_num)

        S_grids.append(S)
        C_numerical.append(C_num)
        C_analytical.append(C_an)
        Errors.append(error)

        # Grid size and error output
        print("Grid Size M =", M)
        print("   Error (L∞):", np.max(np.abs(C_an - C_num)))

        # Numerical price at S0
        C_rk_S0 = np.interp(S0, S, C_num)
        # Analytical black-scholes price at S0
        C_exact = bs_call_price_exact(S0, K, r, sigma, T)

        print("   Numerical Price at S0:", C_rk_S0)
        print("   Analytical Price at S0:", C_exact)
        print("   Absolute error at S0:", abs(C_exact - C_rk_S0))

        # visualize numerical vs analytical and error
        analyticalvsnumerical(f"tsla_2025.csv (M={M})", S_grids[i], C_analytical[i], C_numerical[i])
        error_graph(f"tsla_2025.csv (M={M})", Errors[i], C_analytical[i], C_numerical[i], S_grids[i])

    datasheet_graph("tsla_2025.csv")