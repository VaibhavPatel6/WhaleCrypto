"""
Database layer — PostgreSQL on Railway, JSON file locally.

When DATABASE_URL is set (Railway injects this automatically when you add a
Postgres database), all trades are stored in Postgres and survive restarts.

When DATABASE_URL is not set (local dev), falls back to trade_history.json —
so nothing breaks on your Mac.
"""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

DATABASE_URL = os.getenv("DATABASE_URL")
HISTORY_FILE = Path("trade_history.json")


# ── PostgreSQL backend ────────────────────────────────────────────────────────

if DATABASE_URL:
    import psycopg2
    from psycopg2.extras import RealDictCursor

    def _conn():
        return psycopg2.connect(DATABASE_URL)

    def init_db():
        """Create the trades table if it doesn't exist. Run once at startup."""
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        id           SERIAL PRIMARY KEY,
                        ts           TEXT    NOT NULL,
                        ticker       TEXT    NOT NULL,
                        asset        TEXT    NOT NULL,
                        threshold    REAL    NOT NULL,
                        above        BOOLEAN NOT NULL,
                        side         TEXT    NOT NULL,
                        contracts    INTEGER NOT NULL,
                        price        REAL    NOT NULL,
                        fair_prob    REAL    NOT NULL,
                        edge         REAL    NOT NULL,
                        minutes_left REAL    NOT NULL,
                        dry_run      BOOLEAN NOT NULL DEFAULT false,
                        order_id     TEXT
                    )
                """)
                # Extra columns — added after initial schema; idempotent on existing tables
                for col_def in [
                    "outcome          TEXT",   # 'won', 'lost', 'unfilled'
                    "result           TEXT",   # 'yes' or 'no' (what Kalshi settled)
                    "settled_pnl      REAL",   # actual dollars won or lost
                    "vol_used         REAL",   # annualized vol fed to BS model at placement
                    "drift_used       REAL",   # annualized drift fed to BS model at placement
                    "spot_at_placement REAL",  # RTI price at placement
                ]:
                    cur.execute(f"ALTER TABLE trades ADD COLUMN IF NOT EXISTS {col_def}")
            conn.commit()

    def load_trades() -> list[dict]:
        """Return all trades ordered oldest-first."""
        with _conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM trades ORDER BY ts ASC")
                return [dict(r) for r in cur.fetchall()]

    def save_trade(record: dict):
        """Insert a single trade record."""
        # Normalise to guarantee all columns are present (old records may lack new fields)
        row = {
            "ts":                 record.get("ts"),
            "ticker":             record.get("ticker"),
            "asset":              record.get("asset"),
            "threshold":          record.get("threshold"),
            "above":              record.get("above"),
            "side":               record.get("side"),
            "contracts":          record.get("contracts"),
            "price":              record.get("price"),
            "fair_prob":          record.get("fair_prob"),
            "edge":               record.get("edge"),
            "minutes_left":       record.get("minutes_left"),
            "dry_run":            record.get("dry_run", False),
            "order_id":           record.get("order_id"),
            "vol_used":           record.get("vol_used"),
            "drift_used":         record.get("drift_used"),
            "spot_at_placement":  record.get("spot_at_placement"),
        }
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO trades
                        (ts, ticker, asset, threshold, above, side, contracts,
                         price, fair_prob, edge, minutes_left, dry_run, order_id,
                         vol_used, drift_used, spot_at_placement)
                    VALUES
                        (%(ts)s, %(ticker)s, %(asset)s, %(threshold)s, %(above)s,
                         %(side)s, %(contracts)s, %(price)s, %(fair_prob)s, %(edge)s,
                         %(minutes_left)s, %(dry_run)s, %(order_id)s,
                         %(vol_used)s, %(drift_used)s, %(spot_at_placement)s)
                """, row)
            conn.commit()

    def update_trade_outcome(ts: str, ticker: str, outcome: str, result: str, settled_pnl: float):
        """Write back the settlement outcome for a trade identified by (ts, ticker)."""
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE trades
                       SET outcome=%(outcome)s, result=%(result)s, settled_pnl=%(settled_pnl)s
                     WHERE ts=%(ts)s AND ticker=%(ticker)s
                    """,
                    {"ts": ts, "ticker": ticker, "outcome": outcome,
                     "result": result, "settled_pnl": settled_pnl},
                )
            conn.commit()

    def migrate_from_json():
        """One-time: import existing trade_history.json into the database."""
        if not HISTORY_FILE.exists():
            return
        try:
            trades = json.loads(HISTORY_FILE.read_text())
            existing = {r["ts"] + r["ticker"] for r in load_trades()}
            imported = 0
            for rec in trades:
                key = rec.get("ts", "") + rec.get("ticker", "")
                if key not in existing:
                    save_trade(rec)
                    imported += 1
            if imported:
                print(f"[db] Migrated {imported} trades from trade_history.json → Postgres")
        except Exception as e:
            print(f"[db] Migration skipped: {e}")


# ── Local JSON fallback ───────────────────────────────────────────────────────

else:
    def init_db():
        pass  # nothing to initialise for JSON

    def load_trades() -> list[dict]:
        if not HISTORY_FILE.exists():
            return []
        try:
            return json.loads(HISTORY_FILE.read_text())
        except Exception:
            return []

    def save_trade(record: dict):
        trades = load_trades()
        trades.append(record)
        try:
            HISTORY_FILE.write_text(json.dumps(trades, indent=2))
        except Exception as e:
            print(f"[db] Could not save trade: {e}")

    def update_trade_outcome(ts: str, ticker: str, outcome: str, result: str, settled_pnl: float):
        trades = load_trades()
        for t in trades:
            if t.get("ts") == ts and t.get("ticker") == ticker:
                t["outcome"]     = outcome
                t["result"]      = result
                t["settled_pnl"] = settled_pnl
                break
        HISTORY_FILE.write_text(json.dumps(trades, indent=2))

    def migrate_from_json():
        pass  # already IS the JSON


# ── Shared helpers ────────────────────────────────────────────────────────────

def seed_placed(placed: set[str]):
    """
    Re-populate the _placed deduplication set from DB/file on startup,
    skipping any markets that have already expired.
    """
    now = datetime.now(timezone.utc)
    for rec in load_trades():
        ticker       = rec.get("ticker", "")
        side         = rec.get("side", "")
        ts_str       = rec.get("ts", "")
        minutes_left = rec.get("minutes_left", 0)
        if not ts_str:
            continue
        try:
            placed_at = datetime.fromisoformat(ts_str)
            if placed_at + timedelta(minutes=minutes_left) > now:
                placed.add(f"{ticker}_{side}")
        except (ValueError, TypeError):
            pass
