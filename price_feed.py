"""
CF Benchmarks RTI Price Feed
==============================
Approximates the BRTI (Bitcoin Real Time Index) and ETHUSD_RTI by averaging
prices from the exact constituent exchanges that CF Benchmarks uses:
  Bitstamp, Coinbase, Gemini, Kraken, itBit (Paxos)

CF Benchmarks methodology: partition-weighted median across these venues.
Our simple median/average is within a few dollars of the official index —
good enough since Kalshi's markets have $25–$250 strike intervals.

Volatility: realized vol from Coinbase 15-min candles (same data quality as
the BVOL/EVOL indices published by CF Benchmarks).
"""

import math
import time
import logging
import httpx
from statistics import median

logger = logging.getLogger(__name__)

# 15-min periods per year
PERIODS_PER_YEAR = 35_040

# CF Benchmark constituent exchange endpoints
# Format: (name, url, parser_fn)
BTC_SOURCES = [
    ("Coinbase",  "https://api.exchange.coinbase.com/products/BTC-USD/ticker",     lambda r: float(r["price"])),
    ("Kraken",    "https://api.kraken.com/0/public/Ticker?pair=XBTUSD",            lambda r: float(r["result"]["XXBTZUSD"]["c"][0])),
    ("Bitstamp",  "https://www.bitstamp.net/api/v2/ticker/btcusd/",                lambda r: float(r["last"])),
    ("Gemini",    "https://api.gemini.com/v1/pubticker/btcusd",                    lambda r: float(r["last"])),
    ("itBit",     "https://api.paxos.com/v2/markets/BTCUSD/ticker",                lambda r: float(r["last_execution"]["price"])),
]

ETH_SOURCES = [
    ("Coinbase",  "https://api.exchange.coinbase.com/products/ETH-USD/ticker",     lambda r: float(r["price"])),
    ("Kraken",    "https://api.kraken.com/0/public/Ticker?pair=ETHUSD",            lambda r: float(r["result"]["ETHUSD"]["c"][0])),
    ("Bitstamp",  "https://www.bitstamp.net/api/v2/ticker/ethusd/",                lambda r: float(r["last"])),
    ("Gemini",    "https://api.gemini.com/v1/pubticker/ethusd",                    lambda r: float(r["last"])),
    ("itBit",     "https://api.paxos.com/v2/markets/ETHUSD/ticker",                lambda r: float(r["last_execution"]["price"])),
]

SOURCES = {"BTC": BTC_SOURCES, "ETH": ETH_SOURCES}

# Coinbase candles endpoint for vol/drift (15-min OHLCV)
COINBASE_CANDLES = "https://api.exchange.coinbase.com/products/{pair}/candles"
COINBASE_PAIRS = {"BTC": "BTC-USD", "ETH": "ETH-USD"}


class CFBenchmarkFeed:
    def __init__(self):
        self.client = httpx.Client(timeout=6.0, headers={"User-Agent": "WhaleCrypto/1.0"})
        self._vol_cache: dict[str, tuple[float, float]] = {}     # asset -> (vol, ts)
        self._drift_cache: dict[str, tuple[float, float]] = {}   # asset -> (drift, ts)

    def get_rti(self, asset: str) -> float:
        """
        Fetch the CF Benchmark RTI approximation by taking the median of all
        available constituent exchange prices (same exchanges CF Benchmarks uses).
        Falls back gracefully if some sources are unavailable.
        """
        asset = asset.upper()
        prices: list[float] = []

        for name, url, parser in SOURCES[asset]:
            try:
                resp = self.client.get(url, timeout=4.0)
                resp.raise_for_status()
                prices.append(parser(resp.json()))
                logger.debug(f"  {name}: ${prices[-1]:,.2f}")
            except Exception as e:
                logger.debug(f"  {name} unavailable: {e}")

        if not prices:
            raise RuntimeError(f"All {asset} price sources failed")

        rti = median(prices)
        logger.info(f"{asset} RTI ≈ ${rti:,.2f}  (median of {len(prices)} exchange(s))")
        return rti

    def get_prices(self) -> dict[str, float]:
        return {asset: self.get_rti(asset) for asset in SOURCES}

    def get_annualized_vol(self, asset: str, lookback_periods: int = 96) -> float:
        """
        Realized vol from the last `lookback_periods` 15-min Coinbase candles.
        Result cached for 15 minutes. lookback=96 → last 24 h.
        """
        cached = self._vol_cache.get(asset)
        if cached and time.time() - cached[1] < 900:
            return cached[0]

        try:
            pair = COINBASE_PAIRS[asset.upper()]
            resp = self.client.get(
                COINBASE_CANDLES.format(pair=pair),
                params={"granularity": 900},
            )
            resp.raise_for_status()
            # Format: [[time, low, high, open, close, volume], ...] newest first
            closes = [float(c[4]) for c in resp.json()[:lookback_periods + 1]]

            if len(closes) < 5:
                return self._fallback_vol(asset)

            log_returns = [math.log(closes[i] / closes[i + 1]) for i in range(len(closes) - 1)]
            mean = sum(log_returns) / len(log_returns)
            variance = sum((r - mean) ** 2 for r in log_returns) / max(len(log_returns) - 1, 1)
            vol = math.sqrt(variance) * math.sqrt(PERIODS_PER_YEAR)

            self._vol_cache[asset] = (vol, time.time())
            logger.info(f"{asset} realized vol (24h 15-min): {vol:.1%}/yr")
            return vol

        except Exception as e:
            logger.warning(f"Vol fetch failed for {asset}: {e}")
            return self._fallback_vol(asset)

    def get_recent_drift(self, asset: str, lookback_candles: int = 8) -> float:
        """
        Annualized drift from the last `lookback_candles` 15-min candles (default = last 2h).
        Capped at ±200%/yr to prevent runaway momentum extrapolation.
        Cached for 15 minutes.
        """
        cached = self._drift_cache.get(asset)
        if cached and time.time() - cached[1] < 900:
            return cached[0]

        try:
            pair = COINBASE_PAIRS[asset.upper()]
            resp = self.client.get(
                COINBASE_CANDLES.format(pair=pair),
                params={"granularity": 900},
            )
            resp.raise_for_status()
            closes = [float(c[4]) for c in resp.json()[:lookback_candles + 1]]

            if len(closes) < 2:
                return 0.0

            log_returns = [math.log(closes[i] / closes[i + 1]) for i in range(len(closes) - 1)]
            mean_per_period = sum(log_returns) / len(log_returns)
            annualized = mean_per_period * PERIODS_PER_YEAR
            drift = max(-2.0, min(2.0, annualized))

            self._drift_cache[asset] = (drift, time.time())
            logger.info(f"{asset} drift (2h momentum): {drift:+.1%}/yr")
            return drift

        except Exception as e:
            logger.debug(f"Drift fetch failed for {asset}: {e}")
            return 0.0

    @staticmethod
    def _fallback_vol(asset: str) -> float:
        return {"BTC": 0.75, "ETH": 0.90}.get(asset.upper(), 0.80)
