"""
PCA-Based Statistical Factor Model on Nifty 50
================================================
Author: Soumya | IIT Bombay

Dependencies:
    pip install yfinance pandas numpy scikit-learn matplotlib seaborn scipy
Usage:
    python pca_factor_model.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import yfinance as yf

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────────────────────────────────────
START_DATE    = "2020-01-01"
END_DATE      = "2026-03-15"
# FIX 1: raised from 10 → 20 so cumulative variance can actually reach 80%
# with ~49 stocks you need roughly 18-22 PCs to cross the threshold
N_FACTORS     = 20
VAR_THRESHOLD = 0.80
SAVE_PLOTS    = True

NIFTY50_TICKERS = [
    "RELIANCE.NS",  "TCS.NS",       "HDFCBANK.NS",  "INFY.NS",      "HINDUNILVR.NS",
    "ICICIBANK.NS", "SBIN.NS",      "BHARTIARTL.NS","KOTAKBANK.NS", "LT.NS",
    "AXISBANK.NS",  "ASIANPAINT.NS","MARUTI.NS",    "BAJFINANCE.NS","HCLTECH.NS",
    "SUNPHARMA.NS", "TITAN.NS",     "WIPRO.NS",     "NESTLEIND.NS", "POWERGRID.NS",
    "NTPC.NS",      "ULTRACEMCO.NS","TECHM.NS",     "BAJAJFINSV.NS","ONGC.NS",
    "COALINDIA.NS", "TATAMOTORS.NS","ADANIENT.NS",  "M&M.NS",       "GRASIM.NS",
    "JSWSTEEL.NS",  "HINDALCO.NS",  "TATASTEEL.NS", "DIVISLAB.NS",  "DRREDDY.NS",
    "CIPLA.NS",     "EICHERMOT.NS", "BAJAJ-AUTO.NS","HEROMOTOCO.NS","BPCL.NS",
    "IOC.NS",       "BRITANNIA.NS", "TATACONSUM.NS","INDUSINDBK.NS","UPL.NS",
    "SHREECEM.NS",  "APOLLOHOSP.NS","SBILIFE.NS",   "HDFCLIFE.NS",  "ADANIPORTS.NS"
]
NIFTY_INDEX = "^NSEI"

# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA COLLECTION
# ─────────────────────────────────────────────────────────────────────────────
def download_data(tickers, start, end):
    print("Downloading price data from Yahoo Finance ...")
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False)["Close"]
    thresh = int(len(raw) * 0.90)
    raw = raw.dropna(axis=1, thresh=thresh)
    raw = raw.ffill().dropna()
    print(f"   -> {len(raw.columns)} stocks retained after cleaning")
    return raw

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()

# ─────────────────────────────────────────────────────────────────────────────
# 3. STANDARDISE
# ─────────────────────────────────────────────────────────────────────────────
def prepare_returns(returns: pd.DataFrame):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(returns)
    return scaled, scaler, returns

# ─────────────────────────────────────────────────────────────────────────────
# 4. PCA
# ─────────────────────────────────────────────────────────────────────────────
def run_pca(scaled_returns, n_components=N_FACTORS):
    pca    = PCA(n_components=n_components)
    scores = pca.fit_transform(scaled_returns)
    cols   = [f"PC{i+1}" for i in range(n_components)]
    factor_scores = pd.DataFrame(scores, columns=cols)
    loadings      = pd.DataFrame(pca.components_.T, columns=cols)
    return pca, factor_scores, loadings

# ─────────────────────────────────────────────────────────────────────────────
# 5. FAMA-FRENCH 3-FACTOR PROXIES  (improved)
# ─────────────────────────────────────────────────────────────────────────────
def build_ff3_proxies(prices: pd.DataFrame, returns: pd.DataFrame):
    """
    MKT = Nifty 50 index log-return
    SMB = bottom-tercile price stocks minus top-tercile  (size proxy)
    HML = bottom-tercile 3-yr momentum minus top         (value proxy)

    FIX 5a: split changed from quintile (20%) to tercile (33%) — more standard
    FIX 5b: HML now uses 3-year cumulative return for a more stable value proxy
    """
    print("Building Fama-French factor proxies ...")

    nifty   = yf.download(NIFTY_INDEX, start=START_DATE, end=END_DATE,
                          auto_adjust=True, progress=False)["Close"]
    mkt_ret = np.log(nifty / nifty.shift(1)).dropna()

    common_idx = returns.index.intersection(mkt_ret.index)
    returns_a  = returns.loc[common_idx]
    mkt_a      = mkt_ret.loc[common_idx]

    # SMB: tercile split on mean price
    mean_price   = prices.mean()
    n_split      = max(1, len(mean_price) // 3)
    small_stocks = mean_price.nsmallest(n_split).index
    big_stocks   = mean_price.nlargest(n_split).index
    smb = (returns_a[small_stocks].mean(axis=1)
           - returns_a[big_stocks].mean(axis=1))

    # HML: 3-year cumulative return as value proxy
    lookback      = min(756, len(returns_a))
    cum_3yr       = (1 + returns_a.iloc[-lookback:]).prod() - 1
    n_split2      = max(1, len(cum_3yr) // 3)
    value_stocks  = cum_3yr.nsmallest(n_split2).index
    growth_stocks = cum_3yr.nlargest(n_split2).index
    hml = (returns_a[value_stocks].mean(axis=1)
           - returns_a[growth_stocks].mean(axis=1))

    ff3 = pd.DataFrame({
        "MKT (Nifty)": mkt_a.squeeze(),
        "SMB proxy":   smb,
        "HML proxy":   hml,
    }, index=common_idx).dropna()

    return ff3, returns_a

# ─────────────────────────────────────────────────────────────────────────────
# 6. PCA vs FF3 CORRELATION
# ─────────────────────────────────────────────────────────────────────────────
def compute_correlations(factor_scores: pd.DataFrame,
                         ff3: pd.DataFrame,
                         returns: pd.DataFrame):
    # attach real dates (same length as factor_scores), then intersect with ff3
    fs_dated       = factor_scores.copy()
    fs_dated.index = returns.index

    common_idx = fs_dated.index.intersection(ff3.index)
    fs = fs_dated.loc[common_idx]
    f3 = ff3.loc[common_idx]

    rows = {}
    for pc_col in fs.columns:
        rows[pc_col] = {
            ff_col: round(pearsonr(fs[pc_col], f3[ff_col])[0], 3)
            for ff_col in f3.columns
        }
    return pd.DataFrame(rows).T, fs, f3

# ─────────────────────────────────────────────────────────────────────────────
# 7. SAFE k80 COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────
def compute_k80(pca, n_factors=N_FACTORS, threshold=VAR_THRESHOLD):
    """
    FIX 2: np.argmax returns 0 when NO element satisfies the condition,
    which made k80=1 silently wrong.  Now we check with mask.any() first.
    """
    evr   = pca.explained_variance_ratio_[:n_factors] * 100
    cumvr = np.cumsum(evr)
    mask  = cumvr >= threshold * 100
    if mask.any():
        return int(np.argmax(mask)) + 1          # correct 1-indexed answer
    else:
        print(f"  WARNING: {threshold*100:.0f}% threshold not reached within "
              f"{n_factors} PCs (max cumvar = {cumvr[-1]:.1f}%). "
              f"Increase N_FACTORS.")
        return n_factors                          # best available answer

# ─────────────────────────────────────────────────────────────────────────────
# 8. VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "text.color":       "#e6edf3",
    "axes.labelcolor":  "#e6edf3",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "grid.color":       "#21262d",
    "grid.linewidth":   0.5,
    "font.family":      "DejaVu Sans",
    "font.size":        11,
})

ACCENT_BLUE   = "#58a6ff"
ACCENT_GREEN  = "#3fb950"
ACCENT_ORANGE = "#f0883e"
ACCENT_PURPLE = "#bc8cff"
ACCENT_RED    = "#ff7b72"


def fig1_scree_and_cumvar(pca, k80, date_label,
                           n_factors=N_FACTORS, threshold=VAR_THRESHOLD):
    evr   = pca.explained_variance_ratio_[:n_factors] * 100
    cumvr = np.cumsum(evr)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor("#0d1117")

    # Scree
    ax = axes[0]
    ax.bar(range(1, n_factors + 1), evr,
           color=[ACCENT_BLUE if i < k80 else "#30363d" for i in range(n_factors)],
           edgecolor="#21262d", linewidth=0.5, zorder=3)
    ax.plot(range(1, n_factors + 1), evr, "o-",
            color=ACCENT_GREEN, markersize=4, linewidth=1.5, zorder=4)
    ax.set_xlabel("Principal Component", fontsize=12)
    ax.set_ylabel("Variance Explained (%)", fontsize=12)
    ax.set_title("Scree Plot - Nifty 50 Returns", fontsize=14, color="#e6edf3", pad=14)
    ax.set_xticks(range(1, n_factors + 1))
    ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    ax.annotate(f"PC1\n{evr[0]:.1f}%",
                xy=(1, evr[0]), xytext=(2, evr[0] + 1),
                fontsize=9, color=ACCENT_BLUE,
                arrowprops=dict(arrowstyle="->", color=ACCENT_BLUE, lw=0.8))

    # Cumulative
    ax = axes[1]
    ax.fill_between(range(1, n_factors + 1), cumvr, alpha=0.15, color=ACCENT_PURPLE)
    ax.plot(range(1, n_factors + 1), cumvr, "o-",
            color=ACCENT_PURPLE, markersize=5, linewidth=2)
    ax.axhline(threshold * 100, color=ACCENT_ORANGE, linestyle="--",
               linewidth=1.5, label=f"{int(threshold*100)}% variance")
    ax.axvline(k80, color=ACCENT_ORANGE, linestyle=":", linewidth=1.2, alpha=0.7)
    ax.scatter([k80], [cumvr[k80 - 1]], s=100, color=ACCENT_ORANGE,
               zorder=5, label=f"k={k80} factors")
    ax.annotate(f" {k80} factors\n needed",
                xy=(k80, cumvr[k80 - 1]),
                xytext=(k80 + 0.8, max(cumvr[k80 - 1] - 10, 5)),
                fontsize=9, color=ACCENT_ORANGE,
                arrowprops=dict(arrowstyle="->", color=ACCENT_ORANGE, lw=0.8))
    ax.set_xlabel("Number of Factors", fontsize=12)
    ax.set_ylabel("Cumulative Variance Explained (%)", fontsize=12)
    ax.set_title("Cumulative Variance - 80% Threshold", fontsize=14,
                 color="#e6edf3", pad=14)
    ax.set_xticks(range(1, n_factors + 1))
    ax.set_ylim(0, 105); ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    ax.legend(fontsize=10, framealpha=0.2)

    # FIX 3: dynamic date label passed in from main()
    plt.suptitle(f"PCA Factor Model  |  Nifty 50  |  {date_label}",
                 fontsize=16, color="#e6edf3", y=1.01, fontweight="bold")
    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig("fig1_scree.png", dpi=180, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print("   Saved -> fig1_scree.png")
    plt.show()


def fig2_factor_loadings_heatmap(loadings, returns, k80, date_label):
    n_show   = min(k80 + 2, loadings.shape[1])
    load_sub = loadings.iloc[:, :n_show].copy()
    load_sub.index = [t.replace(".NS", "") for t in returns.columns]

    fig, ax = plt.subplots(figsize=(n_show * 1.4 + 2, 14))
    fig.patch.set_facecolor("#0d1117")

    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(load_sub, ax=ax, cmap=cmap, center=0,
                vmin=-0.35, vmax=0.35,
                linewidths=0.3, linecolor="#21262d",
                cbar_kws={"label": "Factor loading", "shrink": 0.7})

    ax.set_title(f"PCA Factor Loadings - First {n_show} Factors  |  {date_label}",
                 fontsize=13, color="#e6edf3", pad=14)
    ax.set_xlabel("Principal Component", fontsize=12)
    ax.set_ylabel("NSE Stock", fontsize=12)
    ax.tick_params(axis="both", labelsize=9)
    ax.axvline(k80, color=ACCENT_ORANGE, lw=1.5, linestyle="--", alpha=0.7)
    ax.text(k80 + 0.1, -1,
            f"<- {k80} PCs = {VAR_THRESHOLD*100:.0f}%+ variance",
            fontsize=8, color=ACCENT_ORANGE, va="top")

    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig("fig2_loadings.png", dpi=180, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print("   Saved -> fig2_loadings.png")
    plt.show()


def fig3_ff3_correlation(corr_df, date_label):
    abs_corr = corr_df.abs()
    fig, ax  = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#0d1117")

    cmap = sns.light_palette(ACCENT_BLUE, as_cmap=True)
    sns.heatmap(abs_corr, ax=ax, cmap=cmap,
                vmin=0, vmax=1, annot=True, fmt=".2f",
                linewidths=0.4, linecolor="#21262d",
                cbar_kws={"label": "|Pearson r|", "shrink": 0.8},
                annot_kws={"size": 10})

    ax.set_title(f"PCA Factors vs Fama-French 3-Factor Proxies  |  {date_label}",
                 fontsize=13, color="#e6edf3", pad=14)
    ax.set_xlabel("Fama-French Factor", fontsize=12)
    ax.set_ylabel("PCA Factor", fontsize=12)
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig("fig3_ff3_corr.png", dpi=180, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print("   Saved -> fig3_ff3_corr.png")
    plt.show()


def fig4_factor_timeseries(fs, ff3, date_label, k_show=3):
    fig, axes = plt.subplots(k_show, 1, figsize=(14, k_show * 3), sharex=True)
    fig.patch.set_facecolor("#0d1117")

    colors = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_PURPLE, ACCENT_ORANGE, ACCENT_RED]
    mkt    = ff3["MKT (Nifty)"].rolling(5).mean()

    for i, ax in enumerate(axes):
        pc_col = f"PC{i+1}"
        if pc_col not in fs.columns:
            break
        score = fs[pc_col].rolling(5).mean()
        ax.plot(score.index, score.values,
                color=colors[i % len(colors)], lw=1.2,
                label=f"PC{i+1} (smoothed)", zorder=3)
        ax.fill_between(score.index, score.values, alpha=0.1,
                        color=colors[i % len(colors)])
        ax2 = ax.twinx()
        ax2.plot(mkt.index, mkt.values, color="#8b949e", lw=0.7,
                 linestyle="--", label="Nifty return", alpha=0.6)
        ax2.set_ylabel("MKT return", fontsize=8, color="#8b949e")
        ax2.tick_params(colors="#8b949e", labelsize=8)
        ax.set_ylabel(f"PC{i+1} score", fontsize=10, color=colors[i % len(colors)])
        ax.tick_params(colors=colors[i % len(colors)], labelsize=8)
        ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
        ax.legend(loc="upper left",  fontsize=8, framealpha=0.2)
        ax2.legend(loc="upper right", fontsize=8, framealpha=0.2)

    axes[-1].set_xlabel("Date", fontsize=11)
    # FIX 3: dynamic date label, no more hardcoded "2019-2024"
    plt.suptitle(f"PCA Factor Time-Series vs Market (Nifty 50)  |  {date_label}",
                 fontsize=14, color="#e6edf3", y=1.01, fontweight="bold")
    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig("fig4_timeseries.png", dpi=180, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print("   Saved -> fig4_timeseries.png")
    plt.show()


def print_summary(pca, loadings, returns, k80, corr_df, date_label):
    evr = pca.explained_variance_ratio_ * 100
    print("\n" + "=" * 60)
    print("  PCA FACTOR MODEL - SUMMARY")
    print("=" * 60)
    print(f"  Stocks analysed  : {len(returns.columns)}")
    print(f"  Date range       : {date_label}")
    print(f"  Obs (days)       : {len(returns)}")
    print(f"\n  Variance explained (first {N_FACTORS} PCs):")
    for i, v in enumerate(evr[:N_FACTORS]):
        bar = "X" * int(v / 1.5)
        print(f"    PC{i+1:>2}  {v:5.2f}%  {bar}")
    print(f"\n  -> {k80} PCs explain >={VAR_THRESHOLD*100:.0f}% of total variance")

    print(f"\n  Top 5 stocks by |loading| on PC1 (market factor):")
    top5 = loadings["PC1"].abs().nlargest(5)
    for tk, val in top5.items():
        name = tk.replace(".NS", "")
        sign = "+" if loadings.loc[tk, "PC1"] > 0 else "-"
        print(f"    {name:<18} {sign}{val:.3f}")

    print(f"\n  FF3 correlation (|r|) - best match per PC:")
    for pc in corr_df.index[:5]:
        best_ff = corr_df.abs().loc[pc].idxmax()
        val     = corr_df.abs().loc[pc, best_ff]
        print(f"    {pc} <-> {best_ff:<18}  |r| = {val:.3f}")
    print("=" * 60 + "\n")

# ─────────────────────────────────────────────────────────────────────────────
# 9. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Step 1 - Data
    prices  = download_data(NIFTY50_TICKERS, START_DATE, END_DATE)
    returns = compute_log_returns(prices)

    # FIX 3: build date label from actual data, not from config strings
    d0         = returns.index[0]
    d1         = returns.index[-1]
    date_label = f"{d0.strftime('%b %Y')} - {d1.strftime('%b %Y')}"
    print(f"   Actual date range: {date_label}")

    # Step 2 - Standardise
    scaled, scaler, returns = prepare_returns(returns)

    # Step 3 - PCA
    print("Running PCA ...")
    pca, factor_scores, loadings = run_pca(scaled, n_components=N_FACTORS)
    loadings.index = returns.columns

    # FIX 4: compute k80 HERE before print_summary and all plots
    k80 = compute_k80(pca)

    # Step 4 - FF3 proxies
    ff3, returns_aligned = build_ff3_proxies(prices, returns)

    # Step 5 - Correlations
    print("Computing PCA-FF3 correlations ...")
    corr_df, fs_dated, ff3_aligned = compute_correlations(
        factor_scores, ff3, returns)

    # Step 6 - Summary
    print_summary(pca, loadings, returns, k80, corr_df, date_label)

    # Step 7 - Plots
    print("Generating plots ...")
    fig1_scree_and_cumvar(pca, k80, date_label)
    fig2_factor_loadings_heatmap(loadings, returns, k80, date_label)
    fig3_ff3_correlation(corr_df, date_label)
    fig4_factor_timeseries(fs_dated, ff3_aligned, date_label)

    print("\nAll done.")
    print("Figures: fig1_scree.png  fig2_loadings.png  fig3_ff3_corr.png  fig4_timeseries.png")


if __name__ == "__main__":
    main()