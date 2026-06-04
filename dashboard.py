"""
WhaleCrypto Dashboard
=====================
FastAPI web dashboard for monitoring the trading bot.

Run:
  python3 dashboard.py
Then open: http://localhost:8000
"""

import json
import math
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import tempfile
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

load_dotenv("env")

# Private key: file path (local) OR raw content env var (Railway).
# Also handles the common mistake of pasting key content into KALSHI_PRIVATE_KEY_PATH.
_key_content  = os.getenv("KALSHI_PRIVATE_KEY_CONTENT", "")
_key_path_env = os.getenv("KALSHI_PRIVATE_KEY_PATH", "")
if not _key_content and _key_path_env.strip().startswith("-----BEGIN"):
    _key_content = _key_path_env
if _key_content:
    _tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".pem", delete=False)
    _tmp.write(_key_content.replace("\\n", "\n"))
    _tmp.close()
    os.environ["KALSHI_PRIVATE_KEY_PATH"] = _tmp.name

from kalshi_client import KalshiClient
import db

db.init_db()

KALSHI_FEE = 0.007

app = FastAPI(title="WhaleCrypto")

# Lazy Kalshi client — don't crash at startup if env vars aren't set yet
_kalshi: KalshiClient | None = None

def get_kalshi() -> KalshiClient | None:
    global _kalshi
    if _kalshi is None:
        key_id   = os.getenv("KALSHI_API_KEY_ID")
        key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
        if key_id and key_path:
            try:
                _kalshi = KalshiClient(api_key_id=key_id, private_key_path=key_path)
            except Exception:
                pass
    return _kalshi


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bot_running() -> bool:
    try:
        r = subprocess.run(["pgrep", "-f", "bot.py"], capture_output=True)
        return r.returncode == 0
    except (FileNotFoundError, OSError):
        return False  # pgrep not available (Railway)


def _load_history() -> list[dict]:
    return db.load_trades()


def _fetch_result(ticker: str) -> tuple[str | None, str]:
    """
    Returns (result, status) where result is 'yes'/'no'/None
    and status is the raw Kalshi market status string.
    """
    k = get_kalshi()
    if not k:
        return None, "unknown"
    try:
        market = k.get_market(ticker)
        status = market.get("status", "unknown")
        result = market.get("result")
        if result in ("yes", "no"):
            return result, status
        return None, status
    except Exception:
        return None, "unknown"


def _compute_pnl(history: list[dict]) -> dict:
    result_cache: dict[str, tuple] = {}   # ticker → (result, status)
    settled, pending = [], []
    dry_settled = []   # dry-run trades with stored outcomes (calibration only)

    for rec in history:
        ticker = rec["ticker"]
        is_dry = rec.get("dry_run", False)

        # Use pre-stored outcome when available (avoids a Kalshi API call per trade)
        if rec.get("outcome") and rec.get("result"):
            market_result = rec["result"]
            mkt_status    = "settled"
        else:
            if is_dry:
                continue   # dry-run with no outcome yet — skip
            if ticker not in result_cache:
                result_cache[ticker] = _fetch_result(ticker)
            market_result, mkt_status = result_cache[ticker]

        if market_result is None:
            if not is_dry:
                # Show expired-but-unresolved trades as pending so they're visible
                pending.append({**rec, "market_status": mkt_status})
            continue

        won = (market_result == rec["side"])

        if is_dry:
            # Dry-run: contribute to calibration but not P&L
            dry_settled.append({**rec, "won": won, "pnl": 0.0,
                                 "market_result": market_result})
            continue

        pnl = (rec["contracts"] * (1.0 - rec["price"]) - rec["contracts"] * KALSHI_FEE
               if won else -rec["contracts"] * rec["price"])
        settled.append({**rec, "won": won, "pnl": round(pnl, 2),
                        "market_result": market_result})

    wins     = [t for t in settled if t["won"]]
    losses   = [t for t in settled if not t["won"]]
    total_pnl   = sum(t["pnl"] for t in settled)
    total_spent = sum(t["contracts"] * t["price"] for t in settled)
    roi         = (total_pnl / total_spent * 100) if total_spent else 0

    # Calibration: bucket trades by fair_prob, compute win rate per bucket
    # Include dry-run trades (they have stored outcomes) for a larger sample.
    buckets: dict[str, dict] = {}
    for t in settled + dry_settled:
        fp    = t.get("fair_prob", 0.5)
        p_win = fp if t["side"] == "yes" else 1 - fp
        b     = f"{int(p_win * 10) * 10}-{int(p_win * 10) * 10 + 10}%"
        if b not in buckets:
            buckets[b] = {"predicted": round(p_win * 100, 1), "wins": 0, "total": 0,
                          "dry_total": 0}
        buckets[b]["total"] += 1
        if t.get("dry_run"):
            buckets[b]["dry_total"] += 1
        if t["won"]:
            buckets[b]["wins"] += 1

    for b in buckets.values():
        b["actual"] = round(b["wins"] / b["total"] * 100, 1) if b["total"] else 0

    # Cumulative P&L series (sorted by timestamp)
    sorted_settled = sorted(settled, key=lambda t: t.get("ts", ""))
    cumulative, running = [], 0.0
    for t in sorted_settled:
        running += t["pnl"]
        cumulative.append({"ts": t["ts"][:10], "pnl": round(running, 2),
                           "ticker": t["ticker"]})

    # Edge accuracy: avg predicted edge vs avg actual return per contract
    avg_edge = 0.0
    avg_return = 0.0
    if settled:
        edges = []
        returns = []
        for t in settled:
            fp  = t.get("fair_prob", 0.5)
            p_win = fp if t["side"] == "yes" else 1 - fp
            edge  = p_win - t["price"] - KALSHI_FEE
            edges.append(edge)
            returns.append(t["pnl"] / (t["contracts"] * t["price"]) if t["contracts"] > 0 else 0)
        avg_edge   = round(sum(edges) / len(edges) * 100, 1)
        avg_return = round(sum(returns) / len(returns) * 100, 1)

    return {
        "settled":      len(settled),
        "pending":      len(pending),
        "dry_settled":  len(dry_settled),   # paper trades with outcomes (calibration only)
        "wins":         len(wins),
        "losses":       len(losses),
        "win_rate":     round(len(wins) / len(settled) * 100, 1) if settled else 0,
        "total_spent":  round(total_spent, 2),
        "total_pnl":    round(total_pnl, 2),
        "roi":          round(roi, 1),
        "avg_edge":     avg_edge,
        "avg_return":   avg_return,
        "calibration":  buckets,
        "cumulative":   cumulative,
        "trades":       sorted_settled[-20:],  # last 20 settled
    }


# ── API routes ────────────────────────────────────────────────────────────────

@app.get("/api/status")
def api_status():
    try:
        k        = get_kalshi()
        balance  = k.get_balance() if k else 0.0
        dry_run  = os.getenv("DRY_RUN", "true").lower() != "false"
        running  = _bot_running()
        history  = _load_history()
        live     = [t for t in history if not t.get("dry_run")]
        dry      = [t for t in history if t.get("dry_run")]
        return {
            "running":    running,
            "dry_run":    dry_run,
            "balance":    round(balance, 2),
            "live_trades": len(live),
            "dry_trades":  len(dry),
            "last_scan":   history[-1]["ts"][:19].replace("T", " ") if history else "—",
            "credentials": k is not None,
            "railway":     bool(os.getenv("RAILWAY_ENVIRONMENT")),
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/orders")
def api_orders():
    try:
        k = get_kalshi()
        if not k:
            return {"orders": [], "error": "credentials not configured"}
        data   = k._get("/portfolio/orders", params={"limit": 20, "status": "resting"})
        orders = data.get("orders", [])
        out    = []
        for o in orders:
            price = float(o.get("no_price_dollars") or o.get("yes_price_dollars") or 0)
            side  = o.get("side", "")
            out.append({
                "ticker":    o.get("ticker", ""),
                "side":      side.upper(),
                "contracts": o.get("initial_count_fp", "?"),
                "price":     round(price * 100),
                "status":    o.get("status", ""),
                "placed":    o.get("created_time", "")[:16].replace("T", " "),
            })
        return {"orders": out}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/pnl")
def api_pnl():
    try:
        history = _load_history()
        return _compute_pnl(history)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/signals")
def api_signals():
    """Recent signals from trade history (last 15, newest first)."""
    history = _load_history()
    signals = []
    for rec in reversed(history[-30:]):
        signals.append({
            "ts":         rec.get("ts", "")[:16].replace("T", " "),
            "ticker":     rec.get("ticker", ""),
            "asset":      rec.get("asset", ""),
            "side":       rec.get("side", "").upper(),
            "price":      round(rec.get("price", 0) * 100),
            "fair_prob":  round(rec.get("fair_prob", 0) * 100),
            "edge":       round(rec.get("edge", 0) * 100),
            "contracts":  rec.get("contracts", 0),
            "dry_run":    rec.get("dry_run", True),
        })
    return {"signals": signals[:15]}


@app.post("/api/bot/stop")
def bot_stop():
    # On Railway the bot is a separate worker service — cannot stop it from here.
    if os.getenv("RAILWAY_ENVIRONMENT"):
        return {"status": "railway", "msg": "Bot runs as a Railway worker service. Stop it from the Railway dashboard."}
    subprocess.run(["pkill", "-f", "python3 bot.py"])
    return {"status": "stopped"}


@app.post("/api/bot/start")
def bot_start():
    # On Railway the bot is a separate worker service — cannot start it from here.
    if os.getenv("RAILWAY_ENVIRONMENT"):
        return {"status": "railway", "msg": "Bot runs as a Railway worker service and starts automatically. Manage it from the Railway dashboard."}
    subprocess.Popen(
        ["python3", "bot.py"],
        cwd=str(Path(__file__).parent),
        stdout=open("bot.log", "a"),
        stderr=subprocess.STDOUT,
    )
    time.sleep(1)
    return {"status": "started", "running": _bot_running()}


# ── Dashboard HTML ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTMLResponse(content=DASHBOARD_HTML)


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WhaleCrypto</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0a0a0f; color: #e2e8f0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; font-size: 14px; }

  header { background: #111827; border-bottom: 1px solid #1f2937; padding: 16px 24px; display: flex; align-items: center; justify-content: space-between; }
  header h1 { font-size: 20px; font-weight: 700; color: #fff; letter-spacing: -0.5px; }
  header h1 span { color: #3b82f6; }

  .badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; }
  .badge.live    { background: #064e3b; color: #34d399; }
  .badge.dry     { background: #1e3a5f; color: #60a5fa; }
  .badge.stopped { background: #3b1f1f; color: #f87171; }
  .badge .dot    { width: 7px; height: 7px; border-radius: 50%; background: currentColor; animation: pulse 2s infinite; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }

  main { padding: 24px; max-width: 1400px; margin: 0 auto; }

  .grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px; }

  .card { background: #111827; border: 1px solid #1f2937; border-radius: 12px; padding: 20px; }
  .card h2 { font-size: 12px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px; }
  .card .value { font-size: 28px; font-weight: 700; color: #fff; }
  .card .value.green { color: #34d399; }
  .card .value.red   { color: #f87171; }
  .card .sub { font-size: 12px; color: #6b7280; margin-top: 4px; }

  .card-body h2 { font-size: 15px; color: #e2e8f0; font-weight: 600; margin-bottom: 16px; text-transform: none; letter-spacing: 0; }

  table { width: 100%; border-collapse: collapse; }
  th { text-align: left; font-size: 11px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.6px; padding: 8px 12px; border-bottom: 1px solid #1f2937; }
  td { padding: 10px 12px; border-bottom: 1px solid #1a2333; font-size: 13px; }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: #1a2333; }

  .tag { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }
  .tag.yes   { background: #064e3b; color: #34d399; }
  .tag.no    { background: #4c1d1d; color: #f87171; }
  .tag.win   { background: #064e3b; color: #34d399; }
  .tag.loss  { background: #4c1d1d; color: #f87171; }
  .tag.dry   { background: #1e3a5f; color: #60a5fa; }
  .tag.live  { background: #3b1f06; color: #fb923c; }

  .pnl.pos { color: #34d399; }
  .pnl.neg { color: #f87171; }

  .btn { padding: 8px 16px; border-radius: 8px; border: none; cursor: pointer; font-size: 13px; font-weight: 600; transition: opacity 0.2s; }
  .btn:hover { opacity: 0.8; }
  .btn.start { background: #064e3b; color: #34d399; }
  .btn.stop  { background: #4c1d1d; color: #f87171; }

  .calib-row { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; font-size: 13px; }
  .calib-bar-wrap { flex: 1; background: #1f2937; border-radius: 4px; height: 8px; overflow: hidden; }
  .calib-bar { height: 100%; border-radius: 4px; }
  .calib-label { width: 80px; color: #9ca3af; }
  .calib-val   { width: 80px; text-align: right; }

  .empty { text-align: center; color: #4b5563; padding: 32px; }

  #refresh-ts { font-size: 11px; color: #4b5563; }
</style>
</head>
<body>

<header>
  <h1>Whale<span>Crypto</span></h1>
  <div style="display:flex;align-items:center;gap:12px;">
    <span id="refresh-ts">—</span>
    <div id="bot-badge" class="badge stopped"><div class="dot"></div> <span>—</span></div>
    <button id="toggle-btn" class="btn start" onclick="toggleBot()">Start Bot</button>
  </div>
</header>

<main>

  <!-- KPI row -->
  <div class="grid-4">
    <div class="card">
      <h2>Balance</h2>
      <div class="value" id="balance">—</div>
      <div class="sub" id="live-trades">— live trades</div>
    </div>
    <div class="card">
      <h2>Net P&L (live)</h2>
      <div class="value" id="total-pnl">—</div>
      <div class="sub" id="roi">ROI: —</div>
    </div>
    <div class="card">
      <h2>Win Rate</h2>
      <div class="value" id="win-rate">—</div>
      <div class="sub" id="win-loss">— settled trades</div>
    </div>
    <div class="card">
      <h2>Model Edge vs Realised</h2>
      <div class="value" id="avg-edge">—</div>
      <div class="sub" id="avg-return">avg realised: —</div>
    </div>
  </div>

  <!-- P&L chart + calibration -->
  <div class="grid-2">
    <div class="card card-body">
      <h2>Cumulative P&L</h2>
      <canvas id="pnl-chart" height="180"></canvas>
      <div class="empty" id="pnl-empty" style="display:none">No settled live trades yet</div>
    </div>
    <div class="card card-body">
      <h2>Model Calibration <span style="font-size:11px;color:#6b7280;font-weight:400">(predicted win rate vs actual)</span></h2>
      <div id="calib-rows"><div class="empty">No data yet — need 50+ settled trades</div></div>
    </div>
  </div>

  <!-- Open orders + signals -->
  <div class="grid-2">
    <div class="card card-body">
      <h2>Open Orders</h2>
      <table>
        <thead><tr><th>Market</th><th>Side</th><th>Qty</th><th>Limit</th><th>Placed</th></tr></thead>
        <tbody id="orders-body"><tr><td colspan="5" class="empty">Loading…</td></tr></tbody>
      </table>
    </div>
    <div class="card card-body">
      <h2>Recent Signals</h2>
      <table>
        <thead><tr><th>Time</th><th>Ticker</th><th>Side</th><th>Price</th><th>Edge</th><th>Mode</th></tr></thead>
        <tbody id="signals-body"><tr><td colspan="6" class="empty">Loading…</td></tr></tbody>
      </table>
    </div>
  </div>

</main>

<script>
let pnlChart = null;

function fmt(v, prefix='$') {
  if (v === null || v === undefined) return '—';
  const n = parseFloat(v);
  const s = Math.abs(n).toFixed(2);
  return (n >= 0 ? '+' : '−') + prefix + s;
}

async function loadStatus() {
  let d;
  try { d = await fetch('/api/status').then(r=>r.json()); }
  catch(e) { document.getElementById('balance').textContent = 'API error'; return; }
  document.getElementById('balance').textContent = d.balance > 0 ? '$' + d.balance.toFixed(2) : '—';
  document.getElementById('live-trades').textContent = d.live_trades + ' live · ' + d.dry_trades + ' dry';
  document.getElementById('refresh-ts').textContent = 'Last scan: ' + (d.last_scan || '—');

  const badge = document.getElementById('bot-badge');
  const btn   = document.getElementById('toggle-btn');

  // On Railway the worker runs independently — always show "Railway Worker" status
  if (d.railway) {
    badge.className = 'badge live';
    badge.querySelector('span').textContent = 'Railway Worker';
    btn.textContent = 'Managed by Railway';
    btn.className = 'btn start';
    btn.style.opacity = '0.5';
    btn.onclick = () => alert('The bot runs as a Railway worker service and starts automatically. To restart it, go to your Railway dashboard -> worker service -> Redeploy.');
    return;
  }

  if (d.running && !d.dry_run) {
    badge.className = 'badge live';
    badge.querySelector('span').textContent = 'Live';
    btn.className = 'btn stop'; btn.textContent = 'Stop Bot';
  } else if (d.running && d.dry_run) {
    badge.className = 'badge dry';
    badge.querySelector('span').textContent = 'Dry Run';
    btn.className = 'btn stop'; btn.textContent = 'Stop Bot';
  } else {
    badge.className = 'badge stopped';
    badge.querySelector('span').textContent = 'Stopped';
    btn.className = 'btn start'; btn.textContent = 'Start Bot';
  }
}

async function loadPnl() {
  let d;
  try { d = await fetch('/api/pnl').then(r=>r.json()); }
  catch(e) { return; }
  if (d.error) return;

  const pnlEl = document.getElementById('total-pnl');
  pnlEl.textContent = (d.total_pnl >= 0 ? '+$' : '-$') + Math.abs(d.total_pnl).toFixed(2);
  pnlEl.className = 'value ' + (d.total_pnl >= 0 ? 'green' : 'red');

  document.getElementById('roi').textContent = 'ROI: ' + (d.roi >= 0 ? '+' : '') + d.roi + '%';
  document.getElementById('win-rate').textContent = d.win_rate + '%';
  document.getElementById('win-loss').textContent = d.wins + 'W / ' + d.losses + 'L · ' + d.settled + ' settled';
  document.getElementById('avg-edge').textContent = (d.avg_edge >= 0 ? '+' : '') + d.avg_edge + '%';
  document.getElementById('avg-return').textContent = 'avg realised: ' + (d.avg_return >= 0 ? '+' : '') + d.avg_return + '%';

  // P&L chart
  if (d.cumulative && d.cumulative.length > 0) {
    document.getElementById('pnl-empty').style.display = 'none';
    const labels = d.cumulative.map(p => p.ts);
    const values = d.cumulative.map(p => p.pnl);
    const color  = values[values.length-1] >= 0 ? '#34d399' : '#f87171';
    if (pnlChart) pnlChart.destroy();
    pnlChart = new Chart(document.getElementById('pnl-chart'), {
      type: 'line',
      data: { labels, datasets: [{ data: values, borderColor: color, backgroundColor: color+'20',
        fill: true, tension: 0.3, pointRadius: 3, pointBackgroundColor: color }] },
      options: { plugins: { legend: { display: false } },
        scales: { x: { ticks: { color: '#6b7280' }, grid: { color: '#1f2937' } },
                  y: { ticks: { color: '#6b7280', callback: v => '$'+v }, grid: { color: '#1f2937' } } } }
    });
  } else {
    document.getElementById('pnl-empty').style.display = 'block';
  }

  // Calibration
  const calib = d.calibration || {};
  const rows  = document.getElementById('calib-rows');
  if (Object.keys(calib).length === 0) {
    rows.innerHTML = '<div class="empty">No data yet — need 50+ settled trades</div>';
  } else {
    rows.innerHTML = Object.entries(calib).sort().map(([b, v]) => {
      const pred = v.predicted; const act = v.actual;
      const diff = act - pred;
      const clr  = Math.abs(diff) < 5 ? '#34d399' : Math.abs(diff) < 15 ? '#fbbf24' : '#f87171';
      return `<div class="calib-row">
        <div class="calib-label">${b}</div>
        <div style="flex:1">
          <div class="calib-bar-wrap"><div class="calib-bar" style="width:${pred}%;background:#3b82f650"></div></div>
          <div class="calib-bar-wrap" style="margin-top:3px"><div class="calib-bar" style="width:${act}%;background:${clr}"></div></div>
        </div>
        <div class="calib-val" style="color:${clr}">${diff>=0?'+':''}${diff.toFixed(0)}%</div>
        <div style="color:#6b7280;font-size:11px">${v.total} trades</div>
      </div>`;
    }).join('');
  }
}

async function loadOrders() {
  try {
    const d = await fetch('/api/orders').then(r=>r.json());
    const tb = document.getElementById('orders-body');
    if (d.error) { tb.innerHTML = `<tr><td colspan="5" class="empty" style="color:#f87171">${d.error}</td></tr>`; return; }
    if (!d.orders || d.orders.length === 0) {
      tb.innerHTML = '<tr><td colspan="5" class="empty">No open orders</td></tr>'; return;
    }
    tb.innerHTML = d.orders.map(o => `<tr>
      <td style="max-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#9ca3af;font-size:12px">${o.ticker}</td>
      <td><span class="tag ${o.side.toLowerCase()}">${o.side}</span></td>
      <td>${o.contracts}</td>
      <td>${o.price}¢</td>
      <td style="color:#6b7280;font-size:12px">${o.placed}</td>
    </tr>`).join('');
  } catch(e) {
    document.getElementById('orders-body').innerHTML = '<tr><td colspan="5" class="empty" style="color:#f87171">Connection error</td></tr>';
  }
}

async function loadSignals() {
  try {
    const d = await fetch('/api/signals').then(r=>r.json());
    const tb = document.getElementById('signals-body');
    if (!d.signals || d.signals.length === 0) {
      tb.innerHTML = '<tr><td colspan="6" class="empty">No signals yet</td></tr>'; return;
    }
    tb.innerHTML = d.signals.map(s => `<tr>
      <td style="color:#6b7280;font-size:12px">${s.ts}</td>
      <td style="max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:12px;color:#9ca3af">${s.ticker}</td>
      <td><span class="tag ${s.side.toLowerCase()}">${s.side}</span></td>
      <td>${s.price}¢</td>
      <td style="color:#34d399">${s.edge}%</td>
      <td><span class="tag ${s.dry_run ? 'dry' : 'live'}">${s.dry_run ? 'dry' : 'live'}</span></td>
    </tr>`).join('');
  } catch(e) {
    document.getElementById('signals-body').innerHTML = '<tr><td colspan="6" class="empty" style="color:#f87171">Connection error</td></tr>';
  }
}

async function toggleBot() {
  const running = document.getElementById('toggle-btn').textContent === 'Stop Bot';
  await fetch('/api/bot/' + (running ? 'stop' : 'start'), { method: 'POST' });
  setTimeout(loadStatus, 1200);
}

async function refresh() {
  await Promise.all([loadStatus(), loadPnl(), loadOrders(), loadSignals()]);
}

refresh();
setInterval(refresh, 30000);
</script>
</body>
</html>"""


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))   # Railway sets PORT automatically
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
