"""
P&L Summary — WhaleCrypto
==========================
Reads trade_history.json, fetches each market's settlement result from Kalshi,
and prints a clear win/loss breakdown.

Usage:
  python3 pnl.py
"""

import os
import sys
from dotenv import load_dotenv
from kalshi_client import KalshiClient
import db

load_dotenv("env")

KALSHI_FEE = 0.007


def fetch_result(kalshi: KalshiClient, ticker: str) -> str | None:
    """Return 'yes', 'no', or None if the market hasn't settled yet."""
    try:
        market = kalshi.get_market(ticker)
        result = market.get("result")
        # Kalshi returns "" or None for unsettled markets
        return result if result in ("yes", "no") else None
    except Exception as e:
        print(f"  Warning: could not fetch {ticker}: {e}")
        return None


def main():
    history = db.load_trades()
    if not history:
        print("No trade history found — run the bot first.")
        sys.exit(0)

    api_key_id       = os.getenv("KALSHI_API_KEY_ID")
    private_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
    if not api_key_id or not private_key_path:
        raise SystemExit("Missing KALSHI_API_KEY_ID or KALSHI_PRIVATE_KEY_PATH in env file")

    kalshi = KalshiClient(api_key_id=api_key_id, private_key_path=private_key_path)

    # Cache results so we don't hit the API for the same ticker twice
    result_cache: dict[str, str | None] = {}

    settled, pending = [], []

    for rec in history:
        ticker = rec["ticker"]
        side   = rec["side"]
        price  = rec["price"]
        contracts = rec["contracts"]
        dry_run   = rec.get("dry_run", False)

        if ticker not in result_cache:
            print(f"Checking {ticker}…")
            result_cache[ticker] = fetch_result(kalshi, ticker)

        market_result = result_cache[ticker]

        if market_result is None:
            pending.append(rec)
            continue

        won = (market_result == side)

        if won:
            # Collect $1 per contract, pay fee on winnings
            pnl = contracts * (1.0 - price) - contracts * KALSHI_FEE
        else:
            pnl = -contracts * price

        settled.append({
            **rec,
            "market_result": market_result,
            "won": won,
            "pnl": round(pnl, 2),
        })

    # ── Print summary ────────────────────────────────────────────────────────────

    print()
    print("=" * 64)
    print("  WhaleCrypto P&L Summary")
    print("=" * 64)

    if settled:
        wins   = [t for t in settled if t["won"]]
        losses = [t for t in settled if not t["won"]]
        total_pnl   = sum(t["pnl"] for t in settled)
        total_spent = sum(t["contracts"] * t["price"] for t in settled)
        win_rate    = len(wins) / len(settled) * 100
        roi         = total_pnl / total_spent * 100 if total_spent else 0

        print(f"\n  Settled trades : {len(settled)}")
        print(f"  Wins / Losses  : {len(wins)} W  /  {len(losses)} L  ({win_rate:.0f}% win rate)")
        print(f"  Total spent    : ${total_spent:.2f}")
        print(f"  Net P&L        : ${total_pnl:+.2f}")
        print(f"  ROI            : {roi:+.1f}%")

        print()
        print(f"  {'Ticker':<44} {'Side':<5} {'Qty':>4} {'Price':>6} {'Result':<7} {'P&L':>8}")
        print(f"  {'-'*44} {'-'*5} {'-'*4} {'-'*6} {'-'*7} {'-'*8}")
        for t in settled:
            outcome = "WIN " if t["won"] else "LOSS"
            dry     = " [dry]" if t.get("dry_run") else ""
            print(
                f"  {t['ticker']:<44} {t['side'].upper():<5} {t['contracts']:>4} "
                f"{t['price']:>6.0%} {outcome:<7} ${t['pnl']:>+7.2f}{dry}"
            )
    else:
        print("\n  No settled trades yet.")

    if pending:
        print()
        print(f"  Pending (not yet settled): {len(pending)}")
        seen = set()
        for t in pending:
            key = f"{t['ticker']}_{t['side']}"
            if key not in seen:
                seen.add(key)
                dry = " [dry]" if t.get("dry_run") else ""
                print(f"    {t['ticker']}  {t['side'].upper()}{dry}")

    print()
    print("=" * 64)


if __name__ == "__main__":
    main()
