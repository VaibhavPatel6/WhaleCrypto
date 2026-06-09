"""
CF Benchmarks RTI Price Feed
==============================
BTC and ETH: approximates the BRTI / ETHUSD_RTI by taking the median across
the exact CF Benchmark constituent exchanges (Bitstamp, Coinbase, Gemini,
Kraken, itBit/Paxos).  Within a few dollars of the official index.

All other assets (SOL, XRP, DOGE, …): Coinbase Exchange spot price.
Kalshi's settlement methodology for non-BTC/ETH markets varies, but
Coinbase is typically one of the primary reference exchanges.

Volatility + drift: realized from Coinbase 15-min candles for every asset
(same data quality as the BVOL/EVOL indices CF Benchmarks publishes).
"""

import math
import time
import logging
import httpx
from statistics import median

logger = logging.getLogger(__name__)

# 15-min periods per year
PERIODS_PER_YEAR = 35_040

# ── CF Benchmark constituent exchanges (BTC and ETH only) ────────────────────

BTC_SOURCES = [
    ("Coinbase", "https://api.exchange.coinbase.com/products/BTC-USD/ticker",  lambda r: float(r["price"])),
    ("Kraken",   "https://api.kraken.com/0/public/Ticker?pair=XBTUSD",         lambda r: float(r["result"]["XXBTZUSD"]["c"][0])),
    ("Bitstamp", "https://www.bitstamp.net/api/v2/ticker/btcusd/",             lambda r: float(r["last"])),
    ("Gemini",   "https://api.gemini.com/v1/pubticker/btcusd",                 lambda r: float(r["last"])),
    ("itBit",    "https://api.paxos.com/v2/markets/BTCUSD/ticker",             lambda r: float(r["last_execution"]["price"])),
]

ETH_SOURCES = [
    ("Coinbase", "https://api.exchange.coinbase.com/products/ETH-USD/ticker",  lambda r: float(r["price"])),
    ("Kraken",   "https://api.kraken.com/0/public/Ticker?pair=ETHUSD",         lambda r: float(r["result"]["ETHUSD"]["c"][0])),
    ("Bitstamp", "https://www.bitstamp.net/api/v2/ticker/ethusd/",             lambda r: float(r["last"])),
    ("Gemini",   "https://api.gemini.com/v1/pubticker/ethusd",                 lambda r: float(r["last"])),
    ("itBit",    "https://api.paxos.com/v2/markets/ETHUSD/ticker",             lambda r: float(r["last_execution"]["price"])),
]

CF_SOURCES = {"BTC": BTC_SOURCES, "ETH": ETH_SOURCES}

# ── Coinbase product IDs for spot price + candles (all supported assets) ─────

COINBASE_PAIRS: dict[str, str] = {
    "BTC":  "BTC-USD",
    "ETH":  "ETH-USD",
    "SOL":  "SOL-USD",
    "XRP":  "XRP-USD",
    "DOGE": "DOGE-USD",
    "AVAX": "AVAX-USD",
    "LINK": "LINK-USD",
    "LTC":  "LTC-USD",
    "BCH":  "BCH-USD",
    "UNI":  "UNI-USD",
    "AAVE": "AAVE-USD",
    "DOT":  "DOT-USD",
    "ADA":  "ADA-USD",
    "MATIC": "MATIC-USD",
    "ATOM": "ATOM-USD",
}

# Realistic annualized vol fallbacks per asset (used if candles unavailable)
FALLBACK_VOLS: dict[str, float] = {
    "BTC":  0.75,
    "ETH":  0.90,
    "SOL":  1.20,
    "XRP":  1.00,
    "DOGE": 1.30,
    "AVAX": 1.20,
    "LINK": 1.10,
    "LTC":  0.90,
    "BCH":  0.90,
    "UNI":  1.10,
    "AAVE": 1.10,
    "DOT":  1.10,
    "ADA":  1.00,
    "MATIC": 1.20,
    "ATOM": 1.10,
}

COINBASE_CANDLES = "https://api.exchange.coinbase.com/products/{pair}/candles"


class CFBenchmarkFeed:
    def __init__(self):
        self.client = httpx.Client(timeout=6.0, headers={"User-Agent": "WhaleCrypto/1.0"})
        self._vol_cache:      dict[str, tuple[float, float]] = {}  # asset -> (vol, ts)
        self._drift_cache:    dict[str, tuple[float, float]] = {}  # asset -> (drift, ts)
        self._pressure_cache: dict[str, tuple[float, float]] = {}  # asset -> (ratio, ts)

    # ── Spot price ───────────────────────────────────────────────────────────

    def get_rti(self, asset: str) -> float:
        """
        Return the best available spot price for `asset`.

        BTC / ETH  → median of all CF Benchmark constituent exchanges
        Everything else → Coinbase Exchange spot price
        """
        asset = asset.upper()

        if asset in CF_SOURCES:
            return self._cf_median(asset)
        else:
            return self._coinbase_spot(asset)

    def _cf_median(self, asset: str) -> float:
        """Median price across CF Benchmark constituent exchanges."""
        prices: list[float] = []
        for name, url, parser in CF_SOURCES[asset]:
            try:
                resp = self.client.get(url, timeout=4.0)
                resp.raise_for_status()
                prices.append(parser(resp.json()))
                logger.debug(f"  {name}: ${prices[-1]:,.4f}")
            except Exception as e:
                logger.debug(f"  {name} unavailable: {e}")

        if not prices:
            raise RuntimeError(f"All {asset} CF Benchmark sources failed")

        rti = median(prices)
        logger.info(f"{asset} RTI ≈ ${rti:,.4f}  (median of {len(prices)} exchange(s))")
        return rti

    def _coinbase_spot(self, asset: str) -> float:
        """Coinbase Exchange spot price for any supported asset."""
        pair = COINBASE_PAIRS.get(asset)
        if not pair:
            raise ValueError(f"No Coinbase pair configured for {asset}")
        resp = self.client.get(
            f"https://api.exchange.coinbase.com/products/{pair}/ticker",
            timeout=4.0,
        )
        resp.raise_for_status()
        price = float(resp.json()["price"])
        logger.info(f"{asset} spot ≈ ${price:,.4f}  (Coinbase)")
        return price

    def get_prices(self, assets: list[str]) -> dict[str, float]:
        """Fetch spot prices for a list of assets, skipping any that fail."""
        result: dict[str, float] = {}
        for asset in assets:
            try:
                result[asset] = self.get_rti(asset)
            except Exception as e:
                logger.warning(f"Price fetch failed for {asset}: {e}")
        return result

    # ── Volatility ───────────────────────────────────────────────────────────

    def get_annualized_vol(self, asset: str, lookback_periods: int = 96) -> float:
        """
        Realized vol from the last `lookback_periods` 15-min Coinbase candles.
        Cached 15 min.  lookback=96 → last 24 h.
        """
        asset = asset.upper()
        cached = self._vol_cache.get(asset)
        if cached and time.time() - cached[1] < 900:
            return cached[0]

        pair = COINBASE_PAIRS.get(asset)
        if not pair:
            return self._fallback_vol(asset)

        try:
            resp = self.client.get(
                COINBASE_CANDLES.format(pair=pair),
                params={"granularity": 900},
            )
            resp.raise_for_status()
            closes = [float(c[4]) for c in resp.json()[:lookback_periods + 1]]

            if len(closes) < 5:
                return self._fallback_vol(asset)

            log_returns = [math.log(closes[i] / closes[i + 1]) for i in range(len(closes) - 1)]
            mean     = sum(log_returns) / len(log_returns)
            variance = sum((r - mean) ** 2 for r in log_returns) / max(len(log_returns) - 1, 1)
            vol = math.sqrt(variance) * math.sqrt(PERIODS_PER_YEAR)

            self._vol_cache[asset] = (vol, time.time())
            logger.info(f"{asset} realized vol (24h 15-min): {vol:.1%}/yr")
            return vol

        except Exception as e:
            logger.warning(f"Vol fetch failed for {asset}: {e}")
            return self._fallback_vol(asset)

    # ── Drift (short-term momentum) ──────────────────────────────────────────

    def get_recent_drift(self, asset: str, lookback_candles: int = 8) -> float:
        """
        Raw log return over the last lookback_candles × 15-min periods (2 h default).
        NOT annualized — returns actual price change, e.g. -0.005 = fell 0.5%.
        Capped at ±0.05 to exclude flash crashes.  Cached 15 min.

        This is fed into fair_prob_above() which scales it to the remaining horizon
        and caps it at ±1σ√T, so a single volatile candle can't flip probability
        to an extreme.
        """
        asset = asset.upper()
        cached = self._drift_cache.get(asset)
        if cached and time.time() - cached[1] < 900:
            return cached[0]

        pair = COINBASE_PAIRS.get(asset)
        if not pair:
            return 0.0

        try:
            resp = self.client.get(
                COINBASE_CANDLES.format(pair=pair),
                params={"granularity": 900},
            )
            resp.raise_for_status()
            closes = [float(c[4]) for c in resp.json()[:lookback_candles + 1]]

            if len(closes) < 2:
                return 0.0

            # Total log return over the window (not per-period, not annualized)
            total_return = math.log(closes[0] / closes[-1])
            drift = max(-0.05, min(0.05, total_return))

            self._drift_cache[asset] = (drift, time.time())
            logger.info(f"{asset} 2h momentum: {drift:+.4f} ({drift*100:+.2f}%)")
            return drift

        except Exception as e:
            logger.debug(f"Drift fetch failed for {asset}: {e}")
            return 0.0

    # ── Trend ────────────────────────────────────────────────────────────────────

    def get_return(self, asset: str, lookback_candles: int = 24) -> float:
        """
        Simple price return over the last `lookback_candles` × 15-min periods.
        Default = 24 candles = 6 hours.  Positive = up, negative = down.
        Returns 0.0 on any error so callers get a safe neutral default.
        """
        asset = asset.upper()
        pair  = COINBASE_PAIRS.get(asset)
        if not pair:
            return 0.0
        try:
            resp = self.client.get(
                COINBASE_CANDLES.format(pair=pair),
                params={"granularity": 900},
            )
            resp.raise_for_status()
            closes = [float(c[4]) for c in resp.json()[:lookback_candles + 1]]
            if len(closes) < 2:
                return 0.0
            return (closes[0] - closes[-1]) / closes[-1]
        except Exception:
            return 0.0

    # ── Spot orderbook pressure ──────────────────────────────────────────────

    def get_orderbook_pressure(self, asset: str, depth: int = 10) -> float:
        """
        Coinbase spot orderbook bid/ask dollar-volume ratio.
        > 1.0 → more buy pressure;  < 1.0 → more sell pressure.
        Returns 1.0 (neutral) on any error.  Cached 2 min.
        """
        asset = asset.upper()
        cached = self._pressure_cache.get(asset)
        if cached and time.time() - cached[1] < 120:
            return cached[0]

        pair = COINBASE_PAIRS.get(asset)
        if not pair:
            return 1.0
        try:
            resp = self.client.get(
                f"https://api.exchange.coinbase.com/products/{pair}/book",
                params={"level": 2},
                timeout=5.0,
            )
            resp.raise_for_status()
            book = resp.json()
            bid_vol = sum(float(p) * float(q) for p, q, _ in book.get("bids", [])[:depth])
            ask_vol = sum(float(p) * float(q) for p, q, _ in book.get("asks", [])[:depth])
            ratio = bid_vol / ask_vol if ask_vol > 0 else 1.0
            self._pressure_cache[asset] = (ratio, time.time())
            logger.debug(f"{asset} orderbook pressure (bid/ask vol): {ratio:.2f}")
            return ratio
        except Exception as e:
            logger.debug(f"Orderbook pressure unavailable for {asset}: {e}")
            return 1.0

    @staticmethod
    def _fallback_vol(asset: str) -> float:
        return FALLBACK_VOLS.get(asset.upper(), 1.00)
