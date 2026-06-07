"""
WhaleCrypto — Kalshi Crypto Trading Bot
=========================================
Trades Kalshi binary crypto markets (KXBTC / KXETH) by comparing the
market's implied probability against a fair-value model built from:
  • Price  : CF Benchmark RTI approximation (median of constituent exchanges)
  • Vol    : Realized 24-h volatility from 15-min Coinbase candles
  • Drift  : Short-term momentum from the last 2-h of 15-min candles

Why CF Benchmark prices?
  Kalshi settles KXBTC markets against the BRTI (Bitcoin Real Time Index)
  and KXETH against the ETHUSD_RTI — both published by CF Benchmarks.
  Using the same reference price removes the single-exchange price risk.

Strategy:
  For each open KXBTC / KXETH market resolving within MAX_MINUTES_TO_RESOLVE:
    1. Filter to near-the-money strikes (within MAX_OTM_PCT of RTI)
    2. Require a two-sided book (both YES and NO bids present)
    3. Compute fair P(YES) with drift-adjusted log-normal model
    4. If |fair_prob − market_ask| > EDGE_THRESHOLD (after fee), place limit order
    5. Sanity-check: skip markets where model and mid diverge > 35%

Risk controls:
  • DRY_RUN=true by default — no real orders placed until you opt in
  • MAX_POSITION_USD caps dollars at risk per trade
  • MAX_OPEN_POSITIONS caps concurrent bets
  • Only 2-sided books traded (no one-sided illiquid markets)
  • 35% model-vs-market divergence guard

Run:
  python3 bot.py                    # dry run (default)
  DRY_RUN=false python3 bot.py      # live trading
"""

import math
import time
import re
import os
import subprocess
import logging
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional
from pathlib import Path

from dotenv import load_dotenv

from kalshi_client import KalshiClient
from price_feed import CFBenchmarkFeed
import db

load_dotenv("env")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("whale_crypto")

# ─── Config (override via env vars) ──────────────────────────────────────────────

EDGE_THRESHOLD     = float(os.getenv("EDGE_THRESHOLD",      "0.10"))   # 10% min edge over ask price
MAX_OTM_PCT        = float(os.getenv("MAX_OTM_PCT",         "0.04"))   # ignore strikes >4% from RTI
MAX_DIVERGENCE     = float(os.getenv("MAX_DIVERGENCE",      "0.35"))   # skip if model vs mid > 35%
MAX_POSITION_USD   = float(os.getenv("MAX_POSITION_USD",    "50"))     # $ at risk per single order leg
MAX_CONTRACTS      = int(os.getenv("MAX_CONTRACTS",         "200"))    # hard cap per order
MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS",    "5"))
MAX_PORTFOLIO_PCT  = float(os.getenv("MAX_PORTFOLIO_PCT",   "0.10"))   # max % of balance to risk per session
MAX_PER_ASSET      = int(os.getenv("MAX_PER_ASSET",         "1"))      # max positions per asset per session
TREND_SKIP_PCT     = float(os.getenv("TREND_SKIP_PCT",      "0.015"))  # skip NO-above in up-trend / YES-above in down-trend
SCAN_INTERVAL      = int(os.getenv("SCAN_INTERVAL_SECONDS", "60"))
MIN_MINUTES        = float(os.getenv("MIN_MINUTES",         "60"))     # at least 1h to close
MAX_MINUTES        = float(os.getenv("MAX_MINUTES",         "1500"))   # at most ~25h to close
KALSHI_FEE         = 0.007                                             # ~0.7% taker fee

# All known Kalshi crypto series: asset symbol → Kalshi series ticker
# The bot will try each one and silently skip any with no open markets.
CRYPTO_SERIES: dict[str, str] = {
    "BTC":  "KXBTC",
    "ETH":  "KXETH",
    "SOL":  "KXSOL",
    "XRP":  "KXXRP",
    "DOGE": "KXDOGE",
    "AVAX": "KXAVAX",
    "LINK": "KXLINK",
    "LTC":  "KXLTC",
}

# Strike grid for each asset.
# Kalshi changes increments as price levels shift — verified from live ticker data.
# The check allows both exact multiples AND half-step offsets (e.g. BTC uses +50 offset).
# Any strike not matching either pattern is a ghost market with a stale orderbook.
STRIKE_STEP: dict[str, float] = {
    "BTC":  100,    # B70050 / B70150 / B70250 ... (+50 offset from round hundreds)
    "ETH":  20,     # B1840 / B1860 / B1880 ...
    "SOL":  1,      # $73 / $74 / $75 ...
    "XRP":  0.02,   # $1.3099500 / $1.3299500 ... (~0.01 offset from round cents)
    "DOGE": 0.005,  # typical DOGE increment
}

# trade_history.json kept for local dev; db.py swaps it for Postgres on Railway


# ─── Data models ─────────────────────────────────────────────────────────────────

@dataclass
class MarketInfo:
    ticker: str
    asset: str            # "BTC" or "ETH"
    threshold: float      # USD price in the market title
    above: bool           # True → YES wins if price > threshold
    close_time: datetime
    minutes_left: float
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float

    @property
    def yes_mid(self) -> float:
        return (self.yes_bid + self.yes_ask) / 2


@dataclass
class Signal:
    market: MarketInfo
    side: str             # "yes" or "no"
    fair_prob: float      # model probability that YES resolves
    price: float          # price we'll pay (the ask)
    edge: float           # fair_EV minus price minus fee
    contracts: int
    expected_profit: float
    # Model inputs at placement time — stored for post-hoc calibration analysis
    vol_used:          float = 0.0   # annualized realized vol fed to BS model
    drift_used:        float = 0.0   # annualized drift (momentum) fed to BS model
    spot_at_placement: float = 0.0   # RTI price when the order was placed


# ─── Math ────────────────────────────────────────────────────────────────────────

def norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erfc — no scipy needed."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def fair_prob_above(
    spot: float,
    strike: float,
    minutes: float,
    annual_vol: float,
    annual_drift: float = 0.0,
) -> float:
    """
    P(S_T > strike) under GBM with drift:
      d2 = [ln(S/K) + (μ − σ²/2)·T] / (σ·√T)
      P = N(d2)
    """
    T = minutes / (365 * 24 * 60)
    if T <= 0:
        return 1.0 if spot > strike else 0.0
    d2 = (math.log(spot / strike) + (annual_drift - annual_vol ** 2 / 2) * T) / (
        annual_vol * math.sqrt(T)
    )
    return norm_cdf(d2)


# ─── Market parsing ───────────────────────────────────────────────────────────────

def _parse_ticker(ticker: str) -> Optional[tuple[bool, float]]:
    """
    Extract (above, threshold) from ticker like KXBTC-26MAY2417-B76625.

    Range markets (e.g. KXETH-25MAY2417-R1730-T1769) must be excluded.
    Their ticker ends with -T<num> just like a plain "below" binary market,
    so we explicitly reject any ticker that contains the range indicator -R<num>.
    """
    # Kalshi range market tickers contain -R followed by digits (e.g. -R1730-T1769)
    if re.search(r"-R\d", ticker):
        return None
    m = re.search(r"-([BT])([\d.]+)$", ticker)
    if not m:
        return None
    above = m.group(1).upper() == "B"
    return above, float(m.group(2))


def _parse_title(title: str) -> Optional[tuple[bool, float]]:
    """Extract (above, threshold) from title like 'Will BTC be above $76,625 at ...'"""
    m = re.search(r"\b(above|below)\s+\$([0-9,]+(?:\.[0-9]+)?)", title, re.IGNORECASE)
    if not m:
        return None
    above = m.group(1).lower() == "above"
    return above, float(m.group(2).replace(",", ""))


def _parse_close_time(raw: dict) -> Optional[datetime]:
    for key in ("close_time", "expiration_time", "expiration_ts"):
        val = raw.get(key)
        if not val:
            continue
        if isinstance(val, (int, float)):
            return datetime.fromtimestamp(val, tz=timezone.utc)
        try:
            return datetime.fromisoformat(str(val).replace("Z", "+00:00"))
        except ValueError:
            continue
    return None


# ─── Kelly position sizing ───────────────────────────────────────────────────────

def kelly_contracts(edge: float, price: float, balance: float) -> int:
    """
    Quarter-Kelly position size for a binary bet.

    Full Kelly = edge / (1 − price)   ← fraction of bankroll to bet
    We use ¼-Kelly for safety, then cap at MAX_POSITION_USD and floor at $2.

    Example: edge=20%, price=10¢, balance=$275
      full_kelly = 0.20 / 0.90 = 22.2%
      quarter    = 5.6%  → $15.40
      cap to $10 → 100 contracts
    """
    if price <= 0 or price >= 1 or edge <= 0:
        return max(1, min(int(MAX_POSITION_USD / max(price, 0.01)), MAX_CONTRACTS))
    full_kelly    = edge / (1.0 - price)
    quarter_kelly = full_kelly / 4.0
    usd           = max(2.0, min(balance * quarter_kelly, MAX_POSITION_USD))
    return max(1, min(int(usd / price), MAX_CONTRACTS))


# ─── Bot ─────────────────────────────────────────────────────────────────────────

class WhaleCryptoBot:
    def __init__(self, kalshi: KalshiClient, feed: CFBenchmarkFeed, dry_run: bool = True):
        self.kalshi  = kalshi
        self.feed    = feed
        self.dry_run = dry_run
        self.history: list[dict] = []
        self._placed: set[str] = set()      # "ticker_side" keys already traded this session
        self._session_exposure: float = 0.0  # total dollars committed this session
        self._portfolio_limit: float = float("inf")  # set from live balance in run()
        self._session_date: str = datetime.now(timezone.utc).strftime("%Y-%m-%d")  # daily reset
        # Circuit breaker state — reset daily
        self._consecutive_losses: int = 0
        self._circuit_breaker_half_size: bool = False
        self._circuit_breaker_paused: bool = False
        self._circuit_breaker_last_check: float = 0.0
        # Outcome backfill — run once per hour
        self._last_outcome_backfill: float = 0.0
        self._load_history()

    # ── Persistence ──────────────────────────────────────────────────────────────

    def _load_history(self):
        try:
            self.history = db.load_trades()
            logger.info(f"Loaded {len(self.history)} past trades")
            db.seed_placed(self._placed)
            if self._placed:
                logger.info(f"Skipping {len(self._placed)} already-traded position(s) from prior run")
        except Exception as e:
            logger.warning(f"Could not load history: {e}")
            self.history = []

    def _seed_from_open_orders(self):
        """
        Seed _placed from live Kalshi open orders on startup.
        This prevents duplicates when the Railway worker restarts with an empty DB.
        """
        try:
            data   = self.kalshi._get("/portfolio/orders",
                                      params={"limit": 100, "status": "resting"})
            orders = data.get("orders", [])
            before = len(self._placed)
            for o in orders:
                ticker = o.get("ticker", "")
                side   = o.get("side", "")
                if ticker and side:
                    self._placed.add(f"{ticker}_{side}")
                    # Also block the asset (BTC/ETH/XRP) to enforce MAX_PER_ASSET
            added = len(self._placed) - before
            if added:
                logger.info(f"Seeded {added} open order(s) from Kalshi into dedup set")
        except Exception as e:
            logger.warning(f"Could not seed from open orders: {e}")

    def _save_trade(self, record: dict):
        try:
            db.save_trade(record)
            self.history.append(record)
        except Exception as e:
            logger.warning(f"Could not save trade: {e}")

    # ── Market data ──────────────────────────────────────────────────────────────

    def _fetch_markets(self, asset: str) -> list[dict]:
        series = CRYPTO_SERIES[asset]
        data = self.kalshi._get("/markets", params={"series_ticker": series, "status": "open", "limit": 200})
        return data.get("markets", [])

    def _parse_market(self, raw: dict, spot: float) -> Optional[MarketInfo]:
        ticker = raw.get("ticker", "")
        title  = raw.get("title", "")

        asset = next((a for a, s in CRYPTO_SERIES.items() if ticker.startswith(s)), None)
        if not asset:
            return None

        # ── Layer 1: reject non-binary market types ──────────────────────────
        # Kalshi exposes a market_type field; we only want plain binary markets.
        market_type = (raw.get("market_type") or raw.get("category") or "").lower()
        if market_type and market_type not in ("binary", ""):
            logger.debug(f"Skip {ticker}: non-binary market_type='{market_type}'")
            return None

        # ── Layer 2: reject range/bracket markets by title keyword ───────────
        # Range market titles always contain "range" or "between".
        # This is the most reliable signal and catches all known Kalshi range market formats.
        if re.search(r"\b(range|between|bracket)\b", title, re.IGNORECASE):
            logger.debug(f"Skip {ticker}: range market (title='{title[:60]}')")
            return None

        close_time = _parse_close_time(raw)
        if not close_time:
            return None
        now          = datetime.now(timezone.utc)
        minutes_left = (close_time - now).total_seconds() / 60
        if not (MIN_MINUTES <= minutes_left <= MAX_MINUTES):
            return None

        parsed = _parse_title(title) or _parse_ticker(ticker)
        if not parsed:
            return None
        above, threshold = parsed

        # Skip ghost markets sitting between real Kalshi strikes.
        # Allow both exact multiples (remainder=0) and half-step offsets (remainder=step/2)
        # because Kalshi uses offset grids on some assets (e.g. BTC: 70050, 70150, 70250).
        step = STRIKE_STEP.get(asset)
        if step:
            remainder = round(threshold % step, 6)
            half      = round(step / 2, 6)
            if remainder != 0 and abs(remainder - half) > 0.0001 and abs(remainder - step) > 0.001:
                logger.debug(f"Skip {ticker}: ${threshold} not on ${step} grid (ghost market)")
                return None

        # Skip deep OTM strikes — spreads are wide and model error dominates
        if abs(threshold - spot) / spot > MAX_OTM_PCT:
            return None

        try:
            prices = self.kalshi.get_best_prices(ticker)
        except Exception as e:
            logger.debug(f"Orderbook failed {ticker}: {e}")
            return None

        # All four prices must exist (two-sided market)
        yes_bid = prices.get("yes_bid")
        yes_ask = prices.get("yes_ask")
        no_bid  = prices.get("no_bid")
        no_ask  = prices.get("no_ask")
        if any(p is None for p in (yes_bid, yes_ask, no_bid, no_ask)):
            return None

        return MarketInfo(
            ticker=ticker,
            asset=asset,
            threshold=threshold,
            above=above,
            close_time=close_time,
            minutes_left=minutes_left,
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            no_bid=no_bid,
            no_ask=no_ask,
        )

    # ── Signal generation ────────────────────────────────────────────────────────

    def _find_signals(self, rtis: dict[str, float], threshold: float = None) -> tuple[list[Signal], int]:
        """Returns (signals, total_markets_scanned) for funnel diagnostics."""
        if threshold is None:
            threshold = EDGE_THRESHOLD
        signals: list[Signal] = []
        balance        = (self._portfolio_limit / MAX_PORTFOLIO_PCT) if MAX_PORTFOLIO_PCT > 0 else 275.0
        size_mult      = 0.5 if self._circuit_breaker_half_size else 1.0
        total_markets  = 0

        for asset, spot in rtis.items():
            raw_markets    = self._fetch_markets(asset)
            total_markets += len(raw_markets)
            vol   = self.feed.get_annualized_vol(asset)
            drift = self.feed.get_recent_drift(asset)

            # ── Priority 1: Trend filter ─────────────────────────────────────
            # Don't bet NO on "above" markets during a strong up-trend, and
            # don't bet YES on "above" markets during a strong down-trend.
            trend_6h = self.feed.get_return(asset, lookback_candles=24)
            skip_no_above  = trend_6h >  TREND_SKIP_PCT   # up-trend  → don't short
            skip_yes_above = trend_6h < -TREND_SKIP_PCT   # down-trend → don't go long
            if skip_no_above:
                logger.info(f"{asset} 6h trend +{trend_6h:.1%} → skipping NO-above bets (don't fight the trend)")
            elif skip_yes_above:
                logger.info(f"{asset} 6h trend {trend_6h:.1%} → skipping YES-above bets (don't fight the trend)")

            # ── Spot orderbook pressure ──────────────────────────────────────
            # Bid/ask dollar-volume ratio from Coinbase: < 0.70 = heavy sell
            # pressure, > 1.40 = heavy buy pressure.  Skip bets that swim
            # against a pronounced imbalance.
            pressure       = self.feed.get_orderbook_pressure(asset)
            heavy_sell_prs = pressure < 0.70
            heavy_buy_prs  = pressure > 1.40
            if heavy_sell_prs:
                logger.info(f"{asset} heavy sell pressure (ratio={pressure:.2f}) → skipping bullish bets")
            elif heavy_buy_prs:
                logger.info(f"{asset} heavy buy pressure (ratio={pressure:.2f}) → skipping bearish bets")

            # ── Parse all in-window markets first (needed for monotonicity) ──
            in_window_markets: list[MarketInfo] = []
            for raw in raw_markets:
                info = self._parse_market(raw, spot)
                if info:
                    in_window_markets.append(info)

            logger.info(f"{asset}: {len(raw_markets)} markets total, {len(in_window_markets)} in time window")

            # ── Priority 4: Monotonicity check ───────────────────────────────
            # For "above" markets at the same expiry, YES price must decrease
            # as the strike increases. Violations signal a ghost/stale orderbook.
            # Build a set of tickers that pass the check.
            valid_tickers: set[str] = set()
            by_expiry: dict[str, list[MarketInfo]] = {}
            for info in in_window_markets:
                key = f"{info.close_time.date()}_{info.above}"
                by_expiry.setdefault(key, []).append(info)

            for group in by_expiry.values():
                above_group = [m for m in group if m.above]
                above_group.sort(key=lambda m: m.threshold)
                monotonic = True
                for i in range(1, len(above_group)):
                    prev, curr = above_group[i - 1], above_group[i]
                    # Higher strike must have lower or equal YES mid
                    if curr.yes_mid > prev.yes_mid + 0.02:   # 2¢ tolerance for spread noise
                        logger.info(
                            f"Monotonicity violation: {prev.ticker} ({prev.yes_mid:.0%}) < "
                            f"{curr.ticker} ({curr.yes_mid:.0%}) — skipping both"
                        )
                        monotonic = False
                        # Mark both the violating pair as invalid
                        valid_tickers.discard(prev.ticker)
                        valid_tickers.discard(curr.ticker)
                    elif monotonic:
                        valid_tickers.add(prev.ticker)
                        valid_tickers.add(curr.ticker)
                # Single-market groups are always valid
                if len(above_group) == 1:
                    valid_tickers.add(above_group[0].ticker)
                # Below-direction markets aren't monotonicity-checked (rare)
                for m in group:
                    if not m.above:
                        valid_tickers.add(m.ticker)

            # ── Generate signals ─────────────────────────────────────────────
            for info in in_window_markets:
                if info.ticker not in valid_tickers:
                    continue

                p_yes = fair_prob_above(spot, info.threshold, info.minutes_left, vol, drift)
                if not info.above:
                    p_yes = 1.0 - p_yes

                # Sanity guard: skip if model and market mid diverge wildly
                if abs(p_yes - info.yes_mid) > MAX_DIVERGENCE:
                    logger.debug(
                        f"Skip {info.ticker}: model={p_yes:.0%} mid={info.yes_mid:.0%} "
                        f"divergence={abs(p_yes-info.yes_mid):.0%}"
                    )
                    continue

                # YES leg — bullish on "above" markets, bearish on "below" markets
                # Block if trend filter fires, or if spot pressure opposes the bet
                yes_bullish = info.above   # YES-above = bullish; YES-below = bearish
                yes_vs_pressure = (yes_bullish and heavy_sell_prs) or (not yes_bullish and heavy_buy_prs)
                if not skip_yes_above and not yes_vs_pressure and 0.02 < info.yes_ask < 0.98:
                    edge = p_yes - info.yes_ask - KALSHI_FEE
                    if edge >= threshold:
                        n = max(1, int(kelly_contracts(edge, info.yes_ask, balance) * size_mult))
                        signals.append(Signal(info, "yes", p_yes, info.yes_ask, edge, n, n * edge,
                                              vol_used=vol, drift_used=drift, spot_at_placement=spot))

                # NO leg — bearish on "above" markets, bullish on "below" markets
                no_bearish = info.above    # NO-above = bearish; NO-below = bullish
                no_vs_pressure = (no_bearish and heavy_buy_prs) or (not no_bearish and heavy_sell_prs)
                if not skip_no_above and not no_vs_pressure and 0.02 < info.no_ask < 0.98:
                    edge = (1.0 - p_yes) - info.no_ask - KALSHI_FEE
                    if edge >= threshold:
                        n = max(1, int(kelly_contracts(edge, info.no_ask, balance) * size_mult))
                        signals.append(Signal(info, "no", p_yes, info.no_ask, edge, n, n * edge,
                                              vol_used=vol, drift_used=drift, spot_at_placement=spot))

        signals.sort(key=lambda s: s.edge, reverse=True)
        return signals, total_markets

    # ── Notifications ────────────────────────────────────────────────────────────

    @staticmethod
    def _notify(title: str, message: str):
        """Send a macOS desktop notification (silent no-op on other platforms)."""
        try:
            script = f'display notification "{message}" with title "{title}" sound name "Ping"'
            subprocess.run(["osascript", "-e", script], capture_output=True, timeout=3)
        except Exception:
            pass  # Non-macOS or osascript unavailable — just skip

    # ── Execution ────────────────────────────────────────────────────────────────

    def _execute(self, sig: Signal) -> bool:
        tag = "[DRY RUN] " if self.dry_run else ""
        direction = "above" if sig.market.above else "below"
        logger.info(
            f"{tag}BUY {sig.contracts}x {sig.market.ticker} {sig.side.upper()} "
            f"@ {sig.price:.2%} | RTI-fair={sig.fair_prob:.2%} | edge={sig.edge:.2%} | "
            f"strike ${sig.market.threshold:,.0f} {direction} | "
            f"{sig.market.minutes_left:.0f}min | exp_profit=${sig.expected_profit:.2f}"
        )
        mode_tag = "📋 Dry Run" if self.dry_run else "⚡ LIVE"
        self._notify(
            title=f"WhaleCrypto {mode_tag}",
            message=(
                f"{sig.market.asset} {sig.side.upper()} ${sig.market.threshold:,.0f} {direction} | "
                f"edge {sig.edge:.0%} | {sig.contracts} contracts @ {sig.price:.0%}"
            ),
        )

        record: dict = {
            "ts":                datetime.now(timezone.utc).isoformat(),
            "ticker":            sig.market.ticker,
            "asset":             sig.market.asset,
            "threshold":         sig.market.threshold,
            "above":             sig.market.above,
            "side":              sig.side,
            "contracts":         sig.contracts,
            "price":             sig.price,
            "fair_prob":         round(sig.fair_prob, 4),
            "edge":              round(sig.edge, 4),
            "minutes_left":      round(sig.market.minutes_left, 1),
            "dry_run":           self.dry_run,
            "order_id":          None,
            # Model inputs — stored for post-hoc calibration and vol accuracy analysis
            "vol_used":          round(sig.vol_used, 4),
            "drift_used":        round(sig.drift_used, 4),
            "spot_at_placement": round(sig.spot_at_placement, 2),
        }

        if not self.dry_run:
            try:
                price_cents = round(sig.price * 100)
                order = self.kalshi.place_order(
                    ticker=sig.market.ticker,
                    side=sig.side,
                    action="buy",
                    count=sig.contracts,
                    order_type="limit",
                    **{f"{sig.side}_price": price_cents},
                )
                record["order_id"] = order.get("order", {}).get("order_id")
                logger.info(f"Order placed: {record['order_id']}")
            except Exception as e:
                logger.error(f"Order failed for {sig.market.ticker}: {e}")
                return False

        self._save_trade(record)
        return True

    # ── Main loop ─────────────────────────────────────────────────────────────────

    def _daily_reset(self):
        """Reset session state at UTC midnight so the bot trades fresh each day."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._session_date:
            logger.info(f"Daily reset: new UTC day {today} — clearing {len(self._placed)} placed positions")
            self._placed.clear()
            self._session_exposure = 0.0
            self._session_date = today
            self._consecutive_losses = 0
            self._circuit_breaker_half_size = False
            self._circuit_breaker_paused = False
            # Re-seed from whatever is currently resting on Kalshi
            if not self.dry_run:
                self._seed_from_open_orders()
            # Refresh portfolio limit with current balance
            try:
                balance = self.kalshi.get_balance()
                self._portfolio_limit = balance * MAX_PORTFOLIO_PCT
                logger.info(f"Daily reset: new portfolio cap ${self._portfolio_limit:.2f} (balance ${balance:.2f})")
            except Exception:
                pass

    def _cancel_stale_orders(self, stale_drift: float = 0.20):
        """
        Cancel resting orders whose market price has drifted more than stale_drift
        from the limit price. Frees capital and removes stale dedup entries.
        """
        if self.dry_run:
            return
        try:
            data   = self.kalshi._get("/portfolio/orders",
                                      params={"limit": 100, "status": "resting"})
            orders = data.get("orders", [])
        except Exception as e:
            logger.warning(f"Could not fetch resting orders for staleness check: {e}")
            return

        for o in orders:
            ticker   = o.get("ticker", "")
            side     = o.get("side", "")   # "yes" or "no"
            order_id = o.get("order_id", "")

            # Get the limit price (what we paid)
            if side == "yes":
                limit = float(o.get("yes_price_dollars") or o.get("yes_price", 0) / 100)
            else:
                limit = float(o.get("no_price_dollars") or o.get("no_price", 0) / 100)

            if not limit or not order_id:
                continue

            # Get current market price for the same side
            try:
                prices = self.kalshi.get_best_prices(ticker)
                current = prices.get(f"{side}_ask") or prices.get(f"{side}_bid")
                if current is None:
                    continue
            except Exception:
                continue

            drift = abs(current - limit)
            if drift > stale_drift:
                logger.info(
                    f"Cancelling stale {side.upper()} order on {ticker}: "
                    f"limit={limit:.0%} current={current:.0%} drift={drift:.0%}"
                )
                try:
                    self.kalshi.cancel_order(order_id)
                    # Free the dedup slot so the bot can re-evaluate
                    self._placed.discard(f"{ticker}_{side}")
                except Exception as e:
                    logger.warning(f"Could not cancel {order_id}: {e}")

    def _get_effective_threshold(self) -> float:
        """
        Raise edge requirement during US market hours (9am–5pm EDT / 13:00–21:00 UTC)
        when Kalshi is most liquid and prices are most efficiently arbitraged away.
        """
        utc_hour = datetime.now(timezone.utc).hour
        if 13 <= utc_hour < 21:
            return max(EDGE_THRESHOLD, 0.15)
        return EDGE_THRESHOLD

    def _update_circuit_breaker(self):
        """
        Count consecutive settled losses from live trade history and update breaker state.
        2 losses → half position size;  4 losses → pause trading for the rest of the day.
        Runs at most every 5 minutes to avoid excess Kalshi API calls.
        """
        if self.dry_run:
            return
        if time.time() - self._circuit_breaker_last_check < 300:
            return
        self._circuit_breaker_last_check = time.time()

        live_trades = sorted(
            [t for t in self.history if not t.get("dry_run") and t.get("order_id")],
            key=lambda t: t.get("ts", ""),
            reverse=True,
        )
        if not live_trades:
            return

        consecutive = 0
        for trade in live_trades[:6]:
            ticker   = trade.get("ticker", "")
            our_side = trade.get("side", "yes")
            if not ticker:
                continue
            try:
                market = self.kalshi.get_market(ticker)
            except Exception:
                continue
            if market.get("status") != "settled":
                continue
            result = (market.get("result") or "").lower()
            if not result:
                continue
            if result == our_side.lower():
                break           # we won — streak is over
            consecutive += 1

        old = self._consecutive_losses
        self._consecutive_losses      = consecutive
        self._circuit_breaker_half_size = consecutive >= 2
        self._circuit_breaker_paused    = consecutive >= 4

        if consecutive != old:
            if consecutive == 0:
                logger.info("Circuit breaker: streak cleared — full position size restored")
            elif consecutive >= 4:
                logger.warning(f"⛔ Circuit breaker TRIPPED ({consecutive} consecutive losses) — pausing today")
            elif consecutive >= 2:
                logger.warning(f"⚠  Circuit breaker ({consecutive} consecutive losses) — reducing to half size")
            else:
                logger.info(f"Circuit breaker: {consecutive} consecutive loss(es) tracked")

    def _backfill_outcomes(self):
        """
        For every trade with no outcome yet, check whether Kalshi has settled
        the market.  If settled, write back outcome / result / settled_pnl so
        the dashboard and circuit breaker can work from local data instead of
        re-fetching from Kalshi on every request.

        Includes dry-run trades so model calibration accumulates even in paper
        trading mode.  Runs at most once per hour.
        """
        if time.time() - self._last_outcome_backfill < 3600:
            return
        self._last_outcome_backfill = time.time()

        pending = [t for t in self.history if t.get("ticker") and not t.get("outcome")]
        if not pending:
            return

        logger.info(f"Outcome backfill: checking {len(pending)} unsettled trade(s)")
        updated = 0
        for trade in pending:
            ticker    = trade.get("ticker", "")
            our_side  = trade.get("side", "yes")
            order_id  = trade.get("order_id")       # None for dry-run
            contracts = int(trade.get("contracts", 0))
            price     = float(trade.get("price", 0.0))
            ts        = trade.get("ts", "")
            is_dry    = trade.get("dry_run", True)

            try:
                market = self.kalshi.get_market(ticker)
            except Exception as e:
                logger.debug(f"Backfill: could not fetch {ticker}: {e}")
                continue

            mkt_status = market.get("status", "unknown")
            if mkt_status != "settled":
                logger.info(f"Backfill: {ticker} status={mkt_status!r} (not settled yet)")
                continue

            result = (market.get("result") or "").lower()
            if not result:
                logger.warning(f"Backfill: {ticker} is settled but result field is empty — raw={market}")
                continue

            # For live orders verify fill count; dry-run assumes filled
            filled = contracts
            if not is_dry and order_id:
                try:
                    order  = self.kalshi.get_order(order_id)
                    filled = int(order.get("filled_count") or order.get("contracts_filled") or contracts)
                except Exception:
                    pass

            if filled == 0 and not is_dry:
                outcome     = "unfilled"
                settled_pnl = 0.0
            else:
                won         = result == our_side.lower()
                outcome     = "won" if won else "lost"
                settled_pnl = round(filled * (1.0 - price) if won else -filled * price, 2)

            try:
                db.update_trade_outcome(ts, ticker, outcome, result, settled_pnl)
                trade["outcome"]     = outcome
                trade["result"]      = result
                trade["settled_pnl"] = settled_pnl
                updated += 1
                tag = "[DRY] " if is_dry else ""
                logger.info(
                    f"{tag}Settled {ticker} {our_side.upper()} → {result.upper()} "
                    f"({outcome}) filled={filled}  P&L=${settled_pnl:+.2f}"
                )
            except Exception as e:
                logger.warning(f"Could not save outcome for {ticker} {ts}: {e}")

        if updated:
            logger.info(f"Outcome backfill: {updated} trade(s) resolved")

    def scan_once(self):
        self._daily_reset()
        self._backfill_outcomes()        # write settled outcomes back to DB (hourly)
        self._cancel_stale_orders()      # prune orders drifted >20% from limit
        self._update_circuit_breaker()
        if self._circuit_breaker_paused:
            logger.warning("⛔ Circuit breaker active — skipping scan (too many consecutive losses today)")
            return

        effective_threshold = self._get_effective_threshold()
        utc_hour = datetime.now(timezone.utc).hour
        if effective_threshold > EDGE_THRESHOLD:
            logger.info(f"US hours (UTC {utc_hour:02d}:xx) — edge threshold raised to {effective_threshold:.0%}")

        logger.info("─" * 64)
        try:
            rtis = self.feed.get_prices(list(CRYPTO_SERIES.keys()))
        except Exception as e:
            logger.error(f"Price feed error: {e}")
            return

        try:
            signals, markets_scanned = self._find_signals(rtis, threshold=effective_threshold)
        except Exception as e:
            logger.error(f"Signal scan error: {e}", exc_info=True)
            return

        logger.info(f"Funnel: {markets_scanned} markets → {len(signals)} signal(s) above {effective_threshold:.0%} threshold")
        if not signals:
            logger.info("No signals above threshold this scan")
            return

        logger.info(f"{'─'*20} {len(signals)} signal(s) {'─'*20}")
        for s in signals[:10]:
            direction = "above" if s.market.above else "below"
            logger.info(
                f"  {s.market.ticker:42s} {s.side.upper()} @ {s.price:.0%} | "
                f"fair={s.fair_prob:.0%} | mid={s.market.yes_mid:.0%} | "
                f"edge={s.edge:.0%} | ${s.market.threshold:,.0f} {direction} | "
                f"{s.market.minutes_left:.0f}min"
            )

        executed = 0
        for sig in signals:
            if executed >= MAX_OPEN_POSITIONS:
                break
            key          = f"{sig.market.ticker}_{sig.side}"
            opposite_key = f"{sig.market.ticker}_{'yes' if sig.side == 'no' else 'no'}"
            if key in self._placed:
                logger.info(f"  Skip {sig.market.ticker} {sig.side.upper()} — already traded this session")
                continue
            if opposite_key in self._placed:
                logger.info(f"  Skip {sig.market.ticker} {sig.side.upper()} — opposite side already traded (avoid wash)")
                continue

            # Priority 2: per-asset position limit — prevents 3× correlated BTC bets
            series_prefix  = CRYPTO_SERIES.get(sig.market.asset, "")
            asset_placed   = sum(1 for k in self._placed if k.startswith(series_prefix))
            if asset_placed >= MAX_PER_ASSET:
                logger.info(
                    f"  Skip {sig.market.ticker} {sig.side.upper()} — "
                    f"already have {asset_placed} {sig.market.asset} position(s) this session"
                )
                continue

            # Portfolio budget guard
            remaining = self._portfolio_limit - self._session_exposure
            if remaining <= 0:
                logger.info(
                    f"Portfolio limit reached — ${self._session_exposure:.2f} committed "
                    f"of ${self._portfolio_limit:.2f} allowed. Stopping."
                )
                break
            max_affordable = int(remaining / sig.price)
            if max_affordable < 1:
                logger.info(
                    f"  Skip {sig.market.ticker} {sig.side.upper()} — can't fit within remaining budget "
                    f"(${remaining:.2f} left, price {sig.price:.0%})"
                )
                continue
            sig.contracts = min(sig.contracts, max_affordable)
            sig.expected_profit = sig.contracts * sig.edge

            if self._execute(sig):
                self._placed.add(key)
                self._session_exposure += sig.contracts * sig.price
                executed += 1
                time.sleep(2)   # avoid Kalshi rate limit between orders

        if executed:
            logger.info(f"Funnel: {executed} order(s) placed this scan | session exposure ${self._session_exposure:.2f}")

    def run(self):
        mode = "DRY RUN — paper trading" if self.dry_run else "⚠ LIVE TRADING ⚠"

        try:
            balance = self.kalshi.get_balance()
            self._portfolio_limit = balance * MAX_PORTFOLIO_PCT
            balance_str = f"${balance:.2f}"
            limit_str   = f"${self._portfolio_limit:.2f}"
        except Exception as e:
            logger.warning(f"Could not fetch balance ({e}) — no portfolio cap applied")
            balance_str = "unknown"
            limit_str   = "unlimited"

        logger.info("=" * 64)
        logger.info(f"WhaleCrypto Bot  [{mode}]")
        logger.info(f"  Price source : CF Benchmark RTI (constituent-exchange median)")
        logger.info(f"  Assets       : {', '.join(CRYPTO_SERIES)}  ({len(CRYPTO_SERIES)} series)")
        logger.info(f"  Edge thresh  : {EDGE_THRESHOLD:.0%} (raised to 15% during US hours 13–21 UTC)")
        logger.info(f"  Circuit breaker: half-size after 2 losses, pause after 4 losses")
        logger.info(f"  Pressure filter: skip bets vs heavy spot imbalance (ratio <0.70 or >1.40)")
        logger.info(f"  OTM filter   : ±{MAX_OTM_PCT:.0%} of RTI")
        logger.info(f"  Max position : ${MAX_POSITION_USD:.0f} per trade")
        logger.info(f"  Portfolio cap: {MAX_PORTFOLIO_PCT:.0%} of balance = {limit_str} (balance: {balance_str})")
        logger.info(f"  Time window  : {MIN_MINUTES:.0f}–{MAX_MINUTES:.0f} min to close")
        logger.info(f"  Scan every   : {SCAN_INTERVAL}s")
        logger.info("=" * 64)

        # Seed dedup set from live Kalshi open orders (critical on Railway where DB starts empty)
        if not self.dry_run:
            self._seed_from_open_orders()

        try:
            while True:
                try:
                    self.scan_once()
                except Exception as e:
                    logger.error(f"Scan error: {e}", exc_info=True)
                logger.info(f"Sleeping {SCAN_INTERVAL}s…")
                time.sleep(SCAN_INTERVAL)
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        finally:
            live = [t for t in self.history if not t.get("dry_run")]
            dry  = [t for t in self.history if t.get("dry_run")]
            logger.info(f"Session total: {len(self.history)} trades ({len(live)} live, {len(dry)} dry-run)")


# ─── Entry point ─────────────────────────────────────────────────────────────────

def main():
    # ── Private key: file path (local) OR raw content env var (Railway) ──────
    import tempfile
    key_content  = os.getenv("KALSHI_PRIVATE_KEY_CONTENT", "")
    key_path_env = os.getenv("KALSHI_PRIVATE_KEY_PATH", "")

    # Also handle the common mistake of pasting key content into KALSHI_PRIVATE_KEY_PATH
    if not key_content and key_path_env.strip().startswith("-----BEGIN"):
        key_content = key_path_env

    if key_content:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".pem", delete=False)
        tmp.write(key_content.replace("\\n", "\n"))
        tmp.close()
        private_key_path = tmp.name
    else:
        private_key_path = key_path_env or "./kalshi_private_key.pem"

    dry_run    = os.getenv("DRY_RUN", "true").lower() != "false"
    api_key_id = os.getenv("KALSHI_API_KEY_ID")

    if not api_key_id:
        raise SystemExit("Missing KALSHI_API_KEY_ID in env")

    # ── Database init + JSON migration ────────────────────────────────────────
    db.init_db()
    db.migrate_from_json()

    kalshi = KalshiClient(api_key_id=api_key_id, private_key_path=private_key_path)
    feed   = CFBenchmarkFeed()

    WhaleCryptoBot(kalshi=kalshi, feed=feed, dry_run=dry_run).run()


if __name__ == "__main__":
    main()
