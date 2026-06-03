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

HISTORY_FILE = Path("trade_history.json")


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
    """Extract (above, threshold) from ticker like KXBTC-26MAY2417-B76625."""
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
        self._load_history()

    # ── Persistence ──────────────────────────────────────────────────────────────

    def _load_history(self):
        if HISTORY_FILE.exists():
            try:
                self.history = json.loads(HISTORY_FILE.read_text())
                logger.info(f"Loaded {len(self.history)} past trades from {HISTORY_FILE}")
                # Re-seed _placed so we don't re-enter positions that haven't expired yet.
                # Use the recorded ts + minutes_left to compute expiry — no ticker parsing needed.
                now = datetime.now(timezone.utc)
                for rec in self.history:
                    ticker       = rec.get("ticker", "")
                    side         = rec.get("side", "")
                    ts_str       = rec.get("ts", "")
                    minutes_left = rec.get("minutes_left", 0)
                    if not ts_str:
                        continue
                    try:
                        placed_at = datetime.fromisoformat(ts_str)
                        expiry    = placed_at + timedelta(minutes=minutes_left)
                        if expiry > now:
                            self._placed.add(f"{ticker}_{side}")
                    except (ValueError, TypeError):
                        pass
                if self._placed:
                    logger.info(f"Skipping {len(self._placed)} already-traded position(s) from prior run")
            except Exception:
                self.history = []

    def _save_history(self):
        try:
            HISTORY_FILE.write_text(json.dumps(self.history, indent=2))
        except Exception as e:
            logger.warning(f"Could not save history: {e}")

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

    def _find_signals(self, rtis: dict[str, float]) -> list[Signal]:
        signals: list[Signal] = []
        balance = (self._portfolio_limit / MAX_PORTFOLIO_PCT) if MAX_PORTFOLIO_PCT > 0 else 275.0

        for asset, spot in rtis.items():
            raw_markets = self._fetch_markets(asset)
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

                # YES leg — skip if down-trend
                if not skip_yes_above and 0.02 < info.yes_ask < 0.98:
                    edge = p_yes - info.yes_ask - KALSHI_FEE
                    if edge >= EDGE_THRESHOLD:
                        n = kelly_contracts(edge, info.yes_ask, balance)
                        signals.append(Signal(info, "yes", p_yes, info.yes_ask, edge, n, n * edge))

                # NO leg — skip if up-trend
                if not skip_no_above and 0.02 < info.no_ask < 0.98:
                    edge = (1.0 - p_yes) - info.no_ask - KALSHI_FEE
                    if edge >= EDGE_THRESHOLD:
                        n = kelly_contracts(edge, info.no_ask, balance)
                        signals.append(Signal(info, "no", p_yes, info.no_ask, edge, n, n * edge))

        signals.sort(key=lambda s: s.edge, reverse=True)
        return signals

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
            "ts":       datetime.now(timezone.utc).isoformat(),
            "ticker":   sig.market.ticker,
            "asset":    sig.market.asset,
            "threshold": sig.market.threshold,
            "above":    sig.market.above,
            "side":     sig.side,
            "contracts": sig.contracts,
            "price":    sig.price,
            "fair_prob": round(sig.fair_prob, 4),
            "edge":     round(sig.edge, 4),
            "minutes_left": round(sig.market.minutes_left, 1),
            "dry_run":  self.dry_run,
            "order_id": None,
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

        self.history.append(record)
        self._save_history()
        return True

    # ── Main loop ─────────────────────────────────────────────────────────────────

    def scan_once(self):
        logger.info("─" * 64)
        try:
            rtis = self.feed.get_prices(list(CRYPTO_SERIES.keys()))
        except Exception as e:
            logger.error(f"Price feed error: {e}")
            return

        try:
            signals = self._find_signals(rtis)
        except Exception as e:
            logger.error(f"Signal scan error: {e}", exc_info=True)
            return

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
        logger.info(f"  Edge thresh  : {EDGE_THRESHOLD:.0%}")
        logger.info(f"  OTM filter   : ±{MAX_OTM_PCT:.0%} of RTI")
        logger.info(f"  Max position : ${MAX_POSITION_USD:.0f} per trade")
        logger.info(f"  Portfolio cap: {MAX_PORTFOLIO_PCT:.0%} of balance = {limit_str} (balance: {balance_str})")
        logger.info(f"  Time window  : {MIN_MINUTES:.0f}–{MAX_MINUTES:.0f} min to close")
        logger.info(f"  Scan every   : {SCAN_INTERVAL}s")
        logger.info("=" * 64)

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
    dry_run          = os.getenv("DRY_RUN", "true").lower() != "false"
    api_key_id       = os.getenv("KALSHI_API_KEY_ID")
    private_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")

    if not api_key_id or not private_key_path:
        raise SystemExit("Missing KALSHI_API_KEY_ID or KALSHI_PRIVATE_KEY_PATH in env file")

    kalshi = KalshiClient(api_key_id=api_key_id, private_key_path=private_key_path)
    feed   = CFBenchmarkFeed()

    WhaleCryptoBot(kalshi=kalshi, feed=feed, dry_run=dry_run).run()


if __name__ == "__main__":
    main()
