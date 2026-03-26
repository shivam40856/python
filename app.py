"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         STOCK / CRYPTO ANALYSIS & PREDICTION TOOL                          ║
║         Python · Pandas · NumPy · Matplotlib · Scikit-learn · Streamlit    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Run with:  streamlit run app.py

BUGS FIXED vs original:
  1. Removed deprecated infer_datetime_format=True  (pandas 2.x FutureWarning)
  2. Removed unused  import matplotlib.dates as mdates
  3. Removed unused  PANEL = "#111827" constant
  4. Robust frequency inference — null-offset guard + proper fallback to "1B"
  5. Replaced broken fill_betweenx() with correct fill_between() for forecast shading
  6. RSI fill_between NaN fix — fillna(50) on plot series prevents broken shading
  7. Python 3.8-compatible type hints — List[Tuple] from typing instead of list[tuple]
  + Added: demo data loader, per-chart PNG download buttons, plt.close() memory fix
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")                          # must come before pyplot import
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import List, Tuple                 # FIX 7: Python 3.8-compatible hints
import io
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock/Crypto Analysis Tool",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# THEME CONSTANTS  (FIX 3: removed unused PANEL variable)
# ─────────────────────────────────────────────────────────────────────────────
BG     = "#0b0f19"
BORDER = "#1e2d45"
TEXT   = "#94a3b8"
ACCENT = "#38bdf8"
GREEN  = "#22c55e"
RED    = "#ef4444"
ORANGE = "#f59e0b"
PURPLE = "#a78bfa"

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
    background-color: {BG};
    color: #e2e8f0;
}}
h1, h2, h3 {{ font-family: 'Space Mono', monospace; }}
.stApp {{ background: {BG}; }}

div[data-testid="metric-container"] {{
    background: linear-gradient(135deg, #131929, #1a2035);
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 16px;
}}
div[data-testid="metric-container"] label {{ color: #64748b !important; font-size: 0.75rem; }}
div[data-testid="metric-container"] [data-testid="metric-value"] {{
    color: {ACCENT} !important; font-family: 'Space Mono', monospace;
}}
section[data-testid="stSidebar"] {{
    background: #0d1322;
    border-right: 1px solid {BORDER};
}}
.stButton>button {{
    background: linear-gradient(135deg, #0ea5e9, #6366f1);
    color: white; border: none; border-radius: 8px;
    font-family: 'Space Mono', monospace; font-size: 0.85rem;
    padding: 0.5rem 1.5rem;
}}
.stButton>button:hover {{ opacity: 0.85; transform: translateY(-1px); }}
.section-title {{
    font-family: 'Space Mono', monospace;
    color: {ACCENT};
    font-size: 1rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    border-bottom: 1px solid {BORDER};
    padding-bottom: 8px;
    margin: 24px 0 12px 0;
}}
.insight-box {{
    background: linear-gradient(135deg, #0f2027, #1a2a3a);
    border-left: 4px solid {ACCENT};
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 0.9rem;
    line-height: 1.6;
}}
.insight-up   {{ border-left-color: {GREEN}; }}
.insight-down {{ border-left-color: {RED}; }}
.insight-warn {{ border-left-color: {ORANGE}; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def load_and_clean(file) -> pd.DataFrame:
    """Load CSV, detect date column, sort, forward-fill missing values."""
    try:
        df = pd.read_csv(file, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(file)

    df.columns = [c.strip() for c in df.columns]

    date_col = next(
        (c for c in df.columns if c.lower() in ("date", "timestamp", "time", "datetime")),
        None
    )
    if date_col is None:
        st.error("❌ No 'Date' column found. CSV must have: Date, Open, High, Low, Close, Volume")
        st.stop()

    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)
    df = df.rename(columns={date_col: "Date"}).set_index("Date").sort_index()
    df.columns = [c.strip().title() for c in df.columns]

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"❌ Missing columns: {missing}")
        st.stop()

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.ffill().bfill().dropna(subset=required)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute MA10, MA50, Daily Returns, RSI(14), Bollinger Bands(20)."""
    df = df.copy()

    df["MA10"]         = df["Close"].rolling(10).mean()
    df["MA50"]         = df["Close"].rolling(50).mean()
    df["Daily_Return"] = df["Close"].pct_change() * 100

    # RSI (14-period Wilder method)
    delta     = df["Close"].diff()
    gain      = delta.clip(lower=0).rolling(14).mean()
    loss      = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    # Bollinger Bands (20-period, ±2 standard deviations)
    mid            = df["Close"].rolling(20).mean()
    std            = df["Close"].rolling(20).std()
    df["BB_Upper"] = mid + 2 * std
    df["BB_Lower"] = mid - 2 * std
    df["BB_Mid"]   = mid

    return df


def detect_outliers(df: pd.DataFrame, col: str = "Close") -> pd.DataFrame:
    """Return rows where the specified column is an IQR outlier (1.5× rule)."""
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr    = q3 - q1
    return df[(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)]


def run_prediction(df: pd.DataFrame, n_days: int):
    """
    Fit a Linear Regression on a numeric day index, then predict n_days ahead.
    Returns: future_dates, future_prices, metrics dict, in-sample y_fit, fit index.
    """
    data        = df[["Close"]].dropna().copy()
    data["Day"] = np.arange(len(data))

    X, y  = data[["Day"]].values, data["Close"].values
    model = LinearRegression().fit(X, y)
    y_fit = model.predict(X)

    metrics = {
        "MAE":  mean_absolute_error(y, y_fit),
        "RMSE": np.sqrt(mean_squared_error(y, y_fit)),
        "R²":   r2_score(y, y_fit),
    }

    last_day  = data["Day"].iloc[-1]
    last_date = data.index[-1]

    # FIX 4: safe frequency inference with None-guard and full fallback chain
    try:
        freq   = pd.infer_freq(data.index[-20:]) or "B"
        offset = pd.tseries.frequencies.to_offset(freq)
        if offset is None:
            raise ValueError("to_offset returned None")
    except Exception:
        freq   = "B"
        offset = pd.tseries.frequencies.to_offset("1B")

    future_X      = np.arange(last_day + 1, last_day + 1 + n_days).reshape(-1, 1)
    future_prices = model.predict(future_X)
    future_dates  = pd.date_range(start=last_date + offset, periods=n_days, freq=freq)

    return future_dates, future_prices, metrics, y_fit, data.index


def volatility_annualised(df: pd.DataFrame) -> float:
    """Annualised volatility = daily_return_std × √252."""
    return df["Daily_Return"].std() * np.sqrt(252)


# ─────────────────────────────────────────────────────────────────────────────
# DEMO DATA GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_demo_csv(name: str = "DEMO", start_price: float = 150.0,
                      n: int = 500, trend: float = 0.0003,
                      vol: float = 0.018, seed: int = 42) -> bytes:
    """
    Generate a realistic synthetic OHLCV dataset and return it as CSV bytes.
    This lets the app work with zero file uploads — useful for demos and testing.
    """
    np.random.seed(seed)
    dates   = pd.date_range("2022-01-01", periods=n, freq="B")
    closes  = [start_price]
    for _ in range(n - 1):
        closes.append(closes[-1] * (1 + trend + vol * np.random.randn()))
    closes  = np.array(closes)
    highs   = closes * (1 + np.abs(np.random.normal(0, 0.007, n)))
    lows    = closes * (1 - np.abs(np.random.normal(0, 0.007, n)))
    opens   = np.roll(closes, 1); opens[0] = start_price
    volumes = np.random.randint(800_000, 6_000_000, n).astype(int)

    demo_df = pd.DataFrame({
        "Date":   [d.strftime("%Y-%m-%d") for d in dates],
        "Open":   np.round(opens,  4),
        "High":   np.round(highs,  4),
        "Low":    np.round(lows,   4),
        "Close":  np.round(closes, 4),
        "Volume": volumes,
    })
    return demo_df.to_csv(index=False).encode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# CHART UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def fig_to_bytes(fig) -> bytes:
    """Serialize a matplotlib figure to PNG bytes for Streamlit download."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()


def show_chart(fig, label: str = "chart"):
    """
    Render a chart in Streamlit, add a PNG download button,
    then close the figure to prevent memory leaks in long sessions.
    """
    st.pyplot(fig, use_container_width=True)
    st.download_button(
        label=f"⬇️ Download {label}.png",
        data=fig_to_bytes(fig),
        file_name=f"{label}.png",
        mime="image/png",
        key=f"dl_{label}_{id(fig)}"
    )
    plt.close(fig)   # FIX: free matplotlib memory — critical for multi-chart pages


def _style_ax(ax):
    """Apply the dark finance theme to a matplotlib Axes in-place."""
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.grid(color=BORDER, linestyle="--", linewidth=0.4, alpha=0.8)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    return ax


def new_fig(w: float = 14, h: float = 4):
    """Create a new dark-themed (figure, axes) pair."""
    fig, ax = plt.subplots(figsize=(w, h), facecolor=BG)
    return fig, _style_ax(ax)


# ─────────────────────────────────────────────────────────────────────────────
# PLOT FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_close(df: pd.DataFrame):
    fig, ax = new_fig(14, 4)
    ax.plot(df.index, df["Close"], color=ACCENT, lw=1.5, label="Close")
    ax.set_title("Closing Price History", color=TEXT, fontsize=11)
    ax.set_ylabel("Price", color=TEXT)
    ax.legend(facecolor=BG, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)
    plt.tight_layout()
    return fig


def plot_ma(df: pd.DataFrame, show_vol: bool = True):
    rows = [3, 1] if show_vol else [1]
    fig  = plt.figure(figsize=(14, 6 if show_vol else 4), facecolor=BG)
    gs   = GridSpec(len(rows), 1, figure=fig, hspace=0.06, height_ratios=rows)

    ax1 = _style_ax(fig.add_subplot(gs[0]))
    ax1.plot(df.index, df["Close"],    color=ACCENT, lw=1.4, label="Close")
    ax1.plot(df.index, df["MA10"],     color=GREEN,  lw=1.1, ls="--", label="MA10")
    ax1.plot(df.index, df["MA50"],     color=ORANGE, lw=1.1, ls="--", label="MA50")
    ax1.plot(df.index, df["BB_Upper"], color=ACCENT, lw=0.6, ls=":",  alpha=0.5)
    ax1.plot(df.index, df["BB_Lower"], color=ACCENT, lw=0.6, ls=":",  alpha=0.5, label="BB Bands")
    ax1.fill_between(df.index, df["BB_Upper"], df["BB_Lower"], color=ACCENT, alpha=0.04)
    ax1.set_title("Price · MA10 · MA50 · Bollinger Bands", color=TEXT, fontsize=11)
    ax1.set_ylabel("Price", color=TEXT)
    ax1.legend(facecolor=BG, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)

    if show_vol:
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax2 = _style_ax(fig.add_subplot(gs[1], sharex=ax1))
        bar_colors = [GREEN if c >= o else RED for c, o in zip(df["Close"], df["Open"])]
        ax2.bar(df.index, df["Volume"], color=bar_colors, alpha=0.65, width=1)
        ax2.set_ylabel("Volume", color=TEXT, fontsize=8)

    plt.tight_layout()
    return fig


def plot_returns(df: pd.DataFrame):
    fig, ax = new_fig(14, 3)
    r = df["Daily_Return"].dropna()
    ax.bar(r.index, r.values,
           color=[GREEN if v >= 0 else RED for v in r.values],
           alpha=0.7, width=1)
    ax.axhline(0, color=TEXT, lw=0.6)
    ax.set_title("Daily Returns (%)", color=TEXT, fontsize=11)
    ax.set_ylabel("Return %", color=TEXT)
    plt.tight_layout()
    return fig


def plot_rsi(df: pd.DataFrame):
    fig, ax = new_fig(14, 3)
    # FIX 6: RSI has NaN for first 14 rows — fillna(50) prevents broken shading
    rsi_plot = df["RSI"].fillna(50)
    ax.plot(df.index, rsi_plot, color=PURPLE, lw=1.3, label="RSI (14)")
    ax.axhline(70, color=RED,   lw=0.8, ls="--", label="Overbought 70")
    ax.axhline(30, color=GREEN, lw=0.8, ls="--", label="Oversold 30")
    ax.fill_between(df.index, rsi_plot, 70, where=(rsi_plot >= 70), color=RED,   alpha=0.15)
    ax.fill_between(df.index, rsi_plot, 30, where=(rsi_plot <= 30), color=GREEN, alpha=0.15)
    ax.set_ylim(0, 100)
    ax.set_title("RSI – Relative Strength Index", color=TEXT, fontsize=11)
    ax.set_ylabel("RSI", color=TEXT)
    ax.legend(facecolor=BG, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)
    plt.tight_layout()
    return fig


def plot_corr(df: pd.DataFrame):
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume",
                         "MA10", "MA50", "Daily_Return", "RSI"] if c in df.columns]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG)
    _style_ax(ax)
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax).ax.tick_params(colors=TEXT, labelsize=7)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=7, color=TEXT)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols, fontsize=7, color=TEXT)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color="black" if abs(corr.values[i, j]) > 0.6 else TEXT)
    ax.set_title("Correlation Matrix", color=TEXT, fontsize=11)
    plt.tight_layout()
    return fig


def plot_prediction(df: pd.DataFrame, future_dates, future_prices, y_fit, fit_idx):
    fig, ax = new_fig(14, 5)
    ax.plot(df.index,     df["Close"],   color=ACCENT, lw=1.3, label="Historical Close")
    ax.plot(fit_idx,      y_fit,         color=ORANGE, lw=1.0, ls="--", alpha=0.7, label="LR Fit")
    ax.plot(future_dates, future_prices, color=GREEN,  lw=2.0, marker="o", ms=5, label="Forecast")
    ax.axvline(df.index[-1], color=TEXT, lw=0.8, ls=":")

    # FIX 5: was fill_betweenx() which does not accept datetime x-coords → use fill_between()
    ymin = df["Close"].min() * 0.95
    ymax = df["Close"].max() * 1.05
    ax.fill_between([df.index[-1], future_dates[-1]], ymin, ymax, color=GREEN, alpha=0.05)

    ax.set_title("Price Prediction – Linear Regression", color=TEXT, fontsize=11)
    ax.set_ylabel("Price", color=TEXT)
    ax.legend(facecolor=BG, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)
    plt.tight_layout()
    return fig


def plot_comparison(datasets: dict):
    fig, ax = new_fig(14, 5)
    palette = [ACCENT, GREEN, ORANGE, PURPLE, RED, "#ec4899"]
    for (name, df_cmp), color in zip(datasets.items(), palette):
        norm = df_cmp["Close"] / df_cmp["Close"].iloc[0] * 100
        ax.plot(df_cmp.index, norm, color=color, lw=1.4, label=name.replace(".csv", ""))
    ax.axhline(100, color=TEXT, lw=0.5, ls="--", alpha=0.5)
    ax.set_title("Normalised Price Comparison (Base = 100)", color=TEXT, fontsize=11)
    ax.set_ylabel("Normalised Price", color=TEXT)
    ax.legend(facecolor=BG, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# AUTO-INSIGHTS ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def generate_insights(df: pd.DataFrame) -> List[Tuple[str, str]]:
    """Return a list of (css_tag, markdown_text) insight tuples."""
    insights: List[Tuple[str, str]] = []
    close = df["Close"]

    # 1. Trend direction (last 20 sessions linear slope)
    recent = close.iloc[-20:].values
    slope  = np.polyfit(range(len(recent)), recent, 1)[0]
    if slope > 0:
        insights.append(("up",   f"📈 **Uptrend detected** – price rose ~{abs(slope):.4f}/day over the last 20 sessions."))
    else:
        insights.append(("down", f"📉 **Downtrend detected** – price fell ~{abs(slope):.4f}/day over the last 20 sessions."))

    # 2. All-time high / low
    insights.append(("info", f"🏆 **All-time high**: {close.max():.4f} on {close.idxmax().strftime('%Y-%m-%d')}"))
    insights.append(("info", f"🔻 **All-time low**: {close.min():.4f} on {close.idxmin().strftime('%Y-%m-%d')}"))

    # 3. Annualised volatility
    vol   = volatility_annualised(df)
    label = "High ⚠️" if vol > 50 else ("Moderate" if vol > 20 else "Low ✅")
    tag   = "warn" if vol > 50 else "info"
    insights.append((tag, f"⚡ **Annualised Volatility**: {vol:.1f}% — {label}"))

    # 4. RSI signal
    if "RSI" in df.columns:
        rsi = df["RSI"].dropna().iloc[-1]
        if rsi > 70:
            insights.append(("warn", f"🔴 **RSI = {rsi:.1f}** – asset may be **overbought**. Watch for a reversal."))
        elif rsi < 30:
            insights.append(("up",   f"🟢 **RSI = {rsi:.1f}** – asset may be **oversold**. Potential entry zone."))
        else:
            insights.append(("info", f"⚪ **RSI = {rsi:.1f}** – neutral momentum zone."))

    # 5. MA crossover signal
    valid = df[["MA10", "MA50"]].dropna()
    if not valid.empty:
        if valid["MA10"].iloc[-1] > valid["MA50"].iloc[-1]:
            insights.append(("up",   "✅ **Golden Cross** – MA10 above MA50 → short-term bullish momentum."))
        else:
            insights.append(("down", "⚠️ **Death Cross** – MA10 below MA50 → short-term bearish pressure."))

    # 6. Price range
    rng = close.max() - close.min()
    insights.append(("info", f"📊 **Price range**: {rng:.4f} ({rng / close.min() * 100:.1f}% of the low)"))

    # 7. Average volume
    avg_vol = df["Volume"].mean()
    insights.append(("info", f"📦 **Average daily volume**: {avg_vol:,.0f}"))

    return insights


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

def main():

    # ── Banner ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='text-align:center;padding:1.8rem 0 0.8rem'>
      <h1 style='font-family:"Space Mono",monospace;font-size:1.9rem;
                 background:linear-gradient(90deg,#38bdf8,#6366f1);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0'>
        📈 STOCK / CRYPTO ANALYSIS TOOL
      </h1>
      <p style='color:#475569;font-size:0.85rem;margin-top:6px'>
        Upload OHLCV CSV data &nbsp;·&nbsp; Explore trends &nbsp;·&nbsp; Predict prices
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Controls")
        st.markdown("---")

        uploaded = st.file_uploader(
            "Upload CSV file(s)", type=["csv"], accept_multiple_files=True,
            help="Columns needed: Date, Open, High, Low, Close, Volume"
        )

        st.markdown("**— or use demo data —**")
        demo_choice = st.selectbox(
            "Built-in demo dataset",
            ["None", "📈 Bull Stock", "📉 Bear Stock", "₿ Crypto (Volatile)"]
        )
        load_demo = st.button("▶ Load Demo Dataset", use_container_width=True)

        st.markdown("---")
        n_days   = st.slider("Forecast horizon (days)", 3, 30, 10)
        show_vol = st.checkbox("Show Volume bars", value=True)

        st.markdown("---")
        st.markdown("""
        <small style='color:#475569'>
        ⚠️ Predictions use <b>Linear Regression</b> on a time index.<br>
        For educational use only — not financial advice.
        </small>""", unsafe_allow_html=True)

    # ── Empty state ──────────────────────────────────────────────────────────
    if not uploaded and not load_demo:
        st.markdown("""
        <div style='text-align:center;padding:4rem 2rem;background:#0f1829;
                    border-radius:16px;border:1px dashed #1e3a5f;margin-top:2rem'>
          <div style='font-size:3rem'>📂</div>
          <h3 style='color:#38bdf8;font-family:"Space Mono",monospace'>
            Upload a CSV file to begin
          </h3>
          <p style='color:#475569'>
            Required columns: <code>Date · Open · High · Low · Close · Volume</code>
          </p>
          <p style='color:#334155;font-size:0.8rem;margin-top:0.5rem'>
            Or pick a <b>demo dataset</b> from the sidebar and click <b>Load Demo Dataset</b>.
          </p>
        </div>""", unsafe_allow_html=True)
        return

    # ── Load datasets ─────────────────────────────────────────────────────────
    datasets = {}

    # Demo data path
    if load_demo and demo_choice != "None":
        demo_map = {
            "📈 Bull Stock":        ("BULL_STOCK",  150.0,   0.0005, 0.015, 1),
            "📉 Bear Stock":        ("BEAR_STOCK",  200.0,  -0.0004, 0.016, 2),
            "₿ Crypto (Volatile)": ("CRYPTO_BTC",  30000.0, 0.0002, 0.035, 3),
        }
        dname, sp, tr, vl, sd = demo_map[demo_choice]
        csv_bytes = generate_demo_csv(dname, sp, 500, tr, vl, sd)
        with st.spinner(f"Loading demo: {demo_choice}…"):
            try:
                df_raw = load_and_clean(io.BytesIO(csv_bytes))
                datasets[f"{dname}.csv"] = add_features(df_raw)
                st.success(f"✅ Demo dataset **{demo_choice}** loaded — 500 rows of simulated OHLCV data.")
            except Exception as e:
                st.error(f"Demo load error: {e}")

    # Uploaded files path
    for f in uploaded:
        with st.spinner(f"Loading {f.name}…"):
            try:
                df_raw = load_and_clean(f)
                datasets[f.name] = add_features(df_raw)
            except Exception as e:
                st.error(f"Error loading **{f.name}**: {e}")

    if not datasets:
        return

    # ── Dataset selector ──────────────────────────────────────────────────────
    sel = (
        st.selectbox("Active dataset", list(datasets.keys()))
        if len(datasets) > 1
        else list(datasets.keys())[0]
    )
    df = datasets[sel]

    # ═════════════════════════════════════════════════════════════════════════
    # TABS
    # ═════════════════════════════════════════════════════════════════════════
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["🗂 Dataset", "🔍 EDA", "📊 Charts", "🔮 Prediction", "💡 Insights"]
    )

    # ── TAB 1 · DATASET ──────────────────────────────────────────────────────
    with tab1:
        st.markdown("<div class='section-title'>Overview</div>", unsafe_allow_html=True)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Rows",             f"{len(df):,}")
        c2.metric("Columns",          f"{len(df.columns)}")
        c3.metric("Date Range",       f"{(df.index[-1] - df.index[0]).days} days")
        c4.metric("Missing Values",   f"{df.isnull().sum().sum()}")
        c5.metric("Outliers (Close)", f"{len(detect_outliers(df))}")

        base_cols = ["Open", "High", "Low", "Close", "Volume"]

        st.markdown("<div class='section-title'>First 10 Rows</div>", unsafe_allow_html=True)
        st.dataframe(
            df[base_cols].head(10)
              .style.format("{:.4f}", subset=base_cols[:-1])
              .format("{:,.0f}", subset=["Volume"]),
            use_container_width=True
        )

        st.markdown("<div class='section-title'>Statistical Summary</div>", unsafe_allow_html=True)
        st.dataframe(df[base_cols].describe().style.format("{:.4f}"), use_container_width=True)

        st.markdown("<div class='section-title'>Last 5 Rows</div>", unsafe_allow_html=True)
        st.dataframe(
            df[base_cols].tail(5)
              .style.format("{:.4f}", subset=base_cols[:-1])
              .format("{:,.0f}", subset=["Volume"]),
            use_container_width=True
        )

        st.download_button(
            "⬇️ Download Cleaned CSV",
            data=df.reset_index().to_csv(index=False).encode("utf-8"),
            file_name=f"{sel}_cleaned.csv",
            mime="text/csv"
        )

    # ── TAB 2 · EDA ──────────────────────────────────────────────────────────
    with tab2:
        st.markdown("<div class='section-title'>Missing Values</div>", unsafe_allow_html=True)
        mv = df.isnull().sum().rename("Missing").to_frame()
        mv["% Missing"] = (mv["Missing"] / len(df) * 100).round(2)
        st.dataframe(mv, use_container_width=True)

        st.markdown("<div class='section-title'>Correlation Matrix</div>", unsafe_allow_html=True)
        show_chart(plot_corr(df), "correlation_matrix")

        st.markdown("<div class='section-title'>Outliers — IQR Method (Close Price)</div>",
                    unsafe_allow_html=True)
        outs = detect_outliers(df)
        if outs.empty:
            st.success("✅ No outliers detected in Close price.")
        else:
            st.warning(f"⚠️ {len(outs)} outlier(s) found:")
            st.dataframe(
                outs[["Open", "High", "Low", "Close", "Volume"]]
                  .style.format("{:.4f}", subset=["Open", "High", "Low", "Close"]),
                use_container_width=True
            )

        st.markdown("<div class='section-title'>Daily Returns Distribution</div>",
                    unsafe_allow_html=True)
        fig_d, ax_d = plt.subplots(figsize=(10, 3), facecolor=BG)
        _style_ax(ax_d)
        r = df["Daily_Return"].dropna()
        ax_d.hist(r, bins=60, color=ACCENT, alpha=0.75, edgecolor=BG)
        ax_d.axvline(r.mean(), color=ORANGE, lw=1.5, ls="--", label=f"Mean: {r.mean():.3f}%")
        ax_d.axvline(0, color=TEXT, lw=0.7)
        ax_d.set_xlabel("Daily Return %", color=TEXT)
        ax_d.set_ylabel("Frequency", color=TEXT)
        ax_d.legend(facecolor=BG, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)
        plt.tight_layout()
        show_chart(fig_d, "returns_distribution")

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("<div class='section-title'>Return Statistics</div>", unsafe_allow_html=True)
            ret_stats = pd.DataFrame({
                "Metric": ["Mean", "Std Dev", "Min", "Max", "Skewness", "Kurtosis"],
                "Value":  [f"{r.mean():.4f}%", f"{r.std():.4f}%",
                           f"{r.min():.4f}%",  f"{r.max():.4f}%",
                           f"{r.skew():.4f}",  f"{r.kurtosis():.4f}"]
            })
            st.dataframe(ret_stats, use_container_width=True, hide_index=True)

        with col_r:
            st.markdown("<div class='section-title'>Price Statistics</div>", unsafe_allow_html=True)
            c = df["Close"]
            price_stats = pd.DataFrame({
                "Metric": ["Current", "Mean", "Median", "Std Dev", "Volatility (Ann.)"],
                "Value":  [f"{c.iloc[-1]:.4f}", f"{c.mean():.4f}", f"{c.median():.4f}",
                           f"{c.std():.4f}",    f"{volatility_annualised(df):.2f}%"]
            })
            st.dataframe(price_stats, use_container_width=True, hide_index=True)

    # ── TAB 3 · CHARTS ───────────────────────────────────────────────────────
    with tab3:
        st.markdown("<div class='section-title'>Closing Price</div>", unsafe_allow_html=True)
        show_chart(plot_close(df), "closing_price")

        st.markdown("<div class='section-title'>Moving Averages + Bollinger Bands + Volume</div>",
                    unsafe_allow_html=True)
        show_chart(plot_ma(df, show_vol=show_vol), "moving_averages")

        st.markdown("<div class='section-title'>Daily Returns</div>", unsafe_allow_html=True)
        show_chart(plot_returns(df), "daily_returns")

        st.markdown("<div class='section-title'>RSI – Relative Strength Index</div>",
                    unsafe_allow_html=True)
        show_chart(plot_rsi(df), "rsi")

        if len(datasets) > 1:
            st.markdown("<div class='section-title'>Multi-Asset Comparison (Normalised)</div>",
                        unsafe_allow_html=True)
            show_chart(plot_comparison(datasets), "comparison")

    # ── TAB 4 · PREDICTION ───────────────────────────────────────────────────
    with tab4:
        st.markdown("<div class='section-title'>Linear Regression Forecast</div>",
                    unsafe_allow_html=True)

        with st.spinner("Running model…"):
            future_dates, future_prices, metrics, y_fit, fit_idx = run_prediction(df, n_days)

        m1, m2, m3 = st.columns(3)
        m1.metric("MAE",  f"{metrics['MAE']:.4f}")
        m2.metric("RMSE", f"{metrics['RMSE']:.4f}")
        m3.metric("R²",   f"{metrics['R²']:.4f}")

        show_chart(
            plot_prediction(df, future_dates, future_prices, y_fit, fit_idx),
            "price_prediction"
        )

        st.markdown("<div class='section-title'>Predicted Prices</div>", unsafe_allow_html=True)
        last_close = df["Close"].iloc[-1]
        pred_df = pd.DataFrame({
            "Date":            [d.strftime("%Y-%m-%d") for d in future_dates],
            "Predicted Price": [f"{p:.4f}" for p in future_prices],
            "Change vs Last":  [
                f"{p - last_close:+.4f}  ({(p - last_close) / last_close * 100:+.2f}%)"
                for p in future_prices
            ],
        })
        st.dataframe(pred_df, use_container_width=True, hide_index=True)

        direction  = "📈 Upward" if future_prices[-1] > last_close else "📉 Downward"
        change_pct = (future_prices[-1] - last_close) / last_close * 100
        st.info(
            f"**Model forecast direction**: {direction} &nbsp;|&nbsp; "
            f"Projected {n_days}-day change: **{change_pct:+.2f}%**"
        )

        st.markdown("""
        <small style='color:#475569'>
        ⚠️ Linear Regression assumes a linear price trend.
        Real markets are non-linear — use this as a learning tool, not trading advice.
        </small>""", unsafe_allow_html=True)

    # ── TAB 5 · INSIGHTS ─────────────────────────────────────────────────────
    with tab5:
        st.markdown("<div class='section-title'>Automated Market Insights</div>",
                    unsafe_allow_html=True)

        for tag, text in generate_insights(df):
            css = f"insight-box insight-{tag}" if tag in ("up", "down", "warn") else "insight-box"
            st.markdown(f"<div class='{css}'>{text}</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-title'>Period Performance</div>", unsafe_allow_html=True)
        periods   = {"1 Month": 21, "3 Months": 63, "6 Months": 126, "1 Year": 252}
        perf_data = []
        for label, days in periods.items():
            if len(df) > days:
                sp  = df["Close"].iloc[-days]
                ep  = df["Close"].iloc[-1]
                chg = (ep - sp) / sp * 100
                perf_data.append({
                    "Period":      label,
                    "Start Price": f"{sp:.4f}",
                    "End Price":   f"{ep:.4f}",
                    "Change %":    f"{chg:+.2f}%"
                })

        if perf_data:
            st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)
        else:
            st.info("Not enough data to compute period performance.")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
