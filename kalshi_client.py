"""
Kalshi API Client
Docs: https://trading-api.kalshi.com/trade-api/v2
Auth: RSA private key signing (recommended) or email/password
"""

import re
import time
import json
import base64
import hashlib
import logging
from typing import Optional
from datetime import datetime, timezone

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric.padding import PSS, MGF1
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)

KALSHI_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_DEMO_URL = "https://demo-api.kalshi.co/trade-api/v2"


class KalshiClient:
    """
    Kalshi REST API client.

    Authentication via RSA key signing (preferred):
      - Generate a key pair, upload public key to Kalshi dashboard
      - Set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH in .env

    Or email/password (simpler, less secure):
      - Set KALSHI_EMAIL and KALSHI_PASSWORD in .env
    """

    def __init__(
        self,
        api_key_id: Optional[str] = None,
        private_key_path: Optional[str] = None,
        email: Optional[str] = None,
        password: Optional[str] = None,
        demo: bool = False,
    ):
        self.base_url = KALSHI_DEMO_URL if demo else KALSHI_BASE_URL
        self.api_key_id = api_key_id
        self.private_key = None
        self.token = None  # For email/password auth

        if private_key_path and api_key_id:
            self._load_private_key(private_key_path)
            self.auth_mode = "rsa"
        elif email and password:
            self.email = email
            self.password = password
            self.auth_mode = "login"
        else:
            raise ValueError("Provide either (api_key_id + private_key_path) or (email + password)")

        self.client = httpx.Client(timeout=10.0)

    def _load_private_key(self, path: str):
        with open(path, "rb") as f:
            self.private_key = serialization.load_pem_private_key(
                f.read(), password=None, backend=default_backend()
            )

    def _sign_request(self, method: str, path: str, body: str = "") -> dict:
        """Generate RSA signature headers for Kalshi API.
        Kalshi signs: timestamp_ms + METHOD + full_path  (body excluded).
        """
        timestamp_ms = str(int(time.time() * 1000))
        full_path = "/trade-api/v2" + path
        msg_string = timestamp_ms + method.upper() + full_path
        signature = self.private_key.sign(
            msg_string.encode("utf-8"),
            PSS(mgf=MGF1(hashes.SHA256()), salt_length=PSS.MAX_LENGTH),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
        }

    def login(self):
        """Authenticate with email/password and store session token."""
        resp = self.client.post(
            f"{self.base_url}/login",
            json={"email": self.email, "password": self.password},
        )
        resp.raise_for_status()
        self.token = resp.json()["token"]
        logger.info("Kalshi login successful")

    def _headers(self, method: str = "GET", path: str = "", body: str = "") -> dict:
        base = {"Content-Type": "application/json"}
        if self.auth_mode == "rsa":
            base.update(self._sign_request(method, path, body))
        elif self.token:
            base["Authorization"] = f"Bearer {self.token}"
        return base

    def _get(self, path: str, params: dict = None) -> dict:
        url = f"{self.base_url}{path}"
        headers = self._headers("GET", path)
        resp = self.client.get(url, headers=headers, params=params)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, body: dict) -> dict:
        body_str = json.dumps(body)
        url = f"{self.base_url}{path}"
        headers = self._headers("POST", path, body_str)
        resp = self.client.post(url, headers=headers, content=body_str)
        resp.raise_for_status()
        return resp.json()

    # ─── Markets ────────────────────────────────────────────────────────────────

    def get_markets(
        self,
        status: str = "open",
        limit: int = 200,
        cursor: str = None,
        max_markets: int = 1000,
    ) -> list[dict]:
        """Fetch open markets up to max_markets. Handles pagination with rate limiting."""
        all_markets = []
        params = {"status": status, "limit": limit}
        if cursor:
            params["cursor"] = cursor

        while len(all_markets) < max_markets:
            data = self._get("/markets", params=params)
            markets = data.get("markets", [])
            all_markets.extend(markets)
            next_cursor = data.get("cursor")
            if not next_cursor or len(markets) < limit:
                break
            params["cursor"] = next_cursor
            time.sleep(0.3)  # Kalshi rate limit: ~3 req/s on market listing

        return all_markets[:max_markets]

    def get_market(self, ticker: str) -> dict:
        """Get a single market by ticker."""
        return self._get(f"/markets/{ticker}")["market"]

    def get_orderbook(self, ticker: str, depth: int = 10) -> dict:
        """
        Get the orderbook for a market.
        Handles both legacy 'orderbook' format (cents, integer qty)
        and the newer 'orderbook_fp' format (decimal dollars, dollar qty).
        """
        data = self._get(f"/markets/{ticker}/orderbook", params={"depth": depth})

        if "orderbook_fp" in data:
            ob = data["orderbook_fp"]
            # In orderbook_fp the field names are inverted vs. the legacy format:
            #   no_dollars  = YES bids (buy YES orders), ascending → best bid is last
            #   yes_dollars = NO bids (buy NO orders),  ascending → best bid is last
            raw_yes = [(float(p), float(q)) for p, q in (ob.get("no_dollars") or [])]
            raw_no  = [(float(p), float(q)) for p, q in (ob.get("yes_dollars") or [])]
        else:
            ob = data.get("orderbook", {})
            raw_yes = [(p / 100, q) for p, q in (ob.get("yes") or [])]
            raw_no  = [(p / 100, q) for p, q in (ob.get("no") or [])]

        return {
            "ticker": ticker,
            "yes_bids": raw_yes,   # [(price, qty), ...] ascending → last = best
            "no_bids":  raw_no,
            "raw": data,
        }

    def get_best_prices(self, ticker: str) -> dict:
        """Return best bid/ask for YES and NO as decimals (0–1)."""
        ob = self.get_orderbook(ticker, depth=10)
        yes_bids = ob["yes_bids"]
        no_bids  = ob["no_bids"]

        best_yes_bid = yes_bids[-1][0] if yes_bids else None
        best_no_bid  = no_bids[-1][0]  if no_bids  else None

        # YES ask = 1 − best NO bid (and vice versa) because YES + NO = $1 payout
        best_yes_ask = (1.0 - best_no_bid)  if best_no_bid  is not None else None
        best_no_ask  = (1.0 - best_yes_bid) if best_yes_bid is not None else None

        return {
            "ticker":   ticker,
            "yes_bid":  best_yes_bid,
            "yes_ask":  best_yes_ask,
            "no_bid":   best_no_bid,
            "no_ask":   best_no_ask,
        }

    # ─── Portfolio ───────────────────────────────────────────────────────────────

    def get_balance(self) -> float:
        """Return available balance in USD."""
        data = self._get("/portfolio/balance")
        return data.get("balance", 0) / 100  # cents → dollars

    def get_positions(self) -> list[dict]:
        return self._get("/portfolio/positions").get("market_positions", [])

    # ─── Orders ──────────────────────────────────────────────────────────────────

    def place_order(
        self,
        ticker: str,
        side: str,          # "yes" or "no"
        action: str,        # "buy" or "sell"
        count: int,         # number of contracts
        order_type: str = "limit",
        yes_price: Optional[int] = None,   # cents (1–99)
        no_price: Optional[int] = None,
        expiration_ts: Optional[int] = None,
    ) -> dict:
        """
        Place a limit or market order on Kalshi.

        For a limit BUY YES at $0.45:
            side="yes", action="buy", yes_price=45
        For a limit BUY NO at $0.38:
            side="no", action="buy", no_price=38
        """
        # client_order_id: alphanumeric + dashes + underscores only — strip dots
        safe_ticker = re.sub(r"[^A-Za-z0-9\-_]", "_", ticker)
        body = {
            "ticker": ticker,
            "client_order_id": f"arb_{safe_ticker}_{int(time.time()*1000)}",
            "type": order_type,
            "action": action,
            "side": side,
            "count": count,
        }
        if yes_price is not None:
            body["yes_price"] = yes_price
        if no_price is not None:
            body["no_price"] = no_price
        if expiration_ts:
            body["expiration_ts"] = expiration_ts

        logger.info(f"Kalshi placing order: {body}")
        return self._post("/portfolio/orders", body)

    def cancel_order(self, order_id: str) -> dict:
        return self._post(f"/portfolio/orders/{order_id}/decrease", {"reduce_by": 999999})

    def get_order(self, order_id: str) -> dict:
        return self._get(f"/portfolio/orders/{order_id}")["order"]
