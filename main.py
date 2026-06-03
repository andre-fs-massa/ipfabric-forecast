from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import anthropic
import json
import random
import math
import time
import os
from datetime import datetime, timedelta

app = FastAPI(title="IP Fabric — Predictive Incident Forecasting")
app.mount("/static", StaticFiles(directory="static"), name="static")

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

DEVICE_TEMPLATES = [
    {"id": "core-sw-01", "name": "Core Switch 01", "type": "switch", "role": "core", "vendor": "Cisco", "location": "DC-A"},
    {"id": "core-sw-02", "name": "Core Switch 02", "type": "switch", "role": "core", "vendor": "Cisco", "location": "DC-A"},
    {"id": "edge-rt-01", "name": "Edge Router 01", "type": "router", "role": "edge", "vendor": "Juniper", "location": "DC-A"},
    {"id": "edge-rt-02", "name": "Edge Router 02", "type": "router", "role": "edge", "vendor": "Juniper", "location": "DC-B"},
    {"id": "fw-01",      "name": "Firewall 01",    "type": "firewall", "role": "perimeter", "vendor": "Palo Alto", "location": "DC-A"},
    {"id": "fw-02",      "name": "Firewall 02",    "type": "firewall", "role": "perimeter", "vendor": "Palo Alto", "location": "DC-B"},
    {"id": "dist-sw-01", "name": "Dist Switch 01", "type": "switch", "role": "distribution", "vendor": "Arista", "location": "Floor-1"},
    {"id": "dist-sw-02", "name": "Dist Switch 02", "type": "switch", "role": "distribution", "vendor": "Arista", "location": "Floor-2"},
    {"id": "wan-rt-01",  "name": "WAN Router 01",  "type": "router", "role": "wan", "vendor": "Cisco", "location": "DC-A"},
    {"id": "lb-01",      "name": "Load Balancer 01","type": "loadbalancer", "role": "services", "vendor": "F5", "location": "DC-A"},
]

# Scenario seeds — each device gets a hidden "fate" to make demos interesting
SCENARIO_SEEDS = {
    "core-sw-01": {"trend": "degrading",  "severity": "high"},
    "edge-rt-01": {"trend": "degrading",  "severity": "medium"},
    "fw-01":      {"trend": "degrading",  "severity": "critical"},
    "dist-sw-01": {"trend": "stable",     "severity": "low"},
    "dist-sw-02": {"trend": "degrading",  "severity": "medium"},
    "core-sw-02": {"trend": "stable",     "severity": "low"},
    "edge-rt-02": {"trend": "stable",     "severity": "low"},
    "fw-02":      {"trend": "stable",     "severity": "low"},
    "wan-rt-01":  {"trend": "degrading",  "severity": "medium"},
    "lb-01":      {"trend": "stable",     "severity": "low"},
}


def generate_metric_history(device_id: str, metric: str, periods: int = 12):
    """Generate realistic historical metrics with trend baked in."""
    seed = SCENARIO_SEEDS.get(device_id, {"trend": "stable", "severity": "low"})
    rng = random.Random(device_id + metric)

    base_values = {
        "cpu": {"stable": 35, "degrading": 38},
        "errors": {"stable": 0.5, "degrading": 1.2},
        "uptime_drops": {"stable": 0.0, "degrading": 0.3},
        "latency": {"stable": 2.0, "degrading": 2.5},
    }

    severity_multipliers = {"low": 1.0, "medium": 1.4, "high": 1.8, "critical": 2.5}
    mult = severity_multipliers.get(seed["severity"], 1.0)

    base = base_values.get(metric, {}).get(seed["trend"], 20)
    history = []

    for i in range(periods):
        noise = rng.gauss(0, base * 0.08)
        if seed["trend"] == "degrading":
            growth = (base * 0.04 * mult * i)
            spike = rng.gauss(0, base * 0.06 * mult) if i > periods // 2 else 0
            val = base + growth + noise + spike
        else:
            val = base + noise

        # clamp
        if metric == "cpu":
            val = max(5, min(98, val))
        elif metric == "errors":
            val = max(0, val)
        elif metric == "uptime_drops":
            val = max(0, val)
        else:
            val = max(0.5, val)

        history.append(round(val, 2))

    return history


def build_device_snapshot():
    """Build a full telemetry snapshot for all devices."""
    devices = []
    now = datetime.utcnow()
    for tmpl in DEVICE_TEMPLATES:
        seed = SCENARIO_SEEDS.get(tmpl["id"], {"trend": "stable", "severity": "low"})
        cpu_hist    = generate_metric_history(tmpl["id"], "cpu")
        err_hist    = generate_metric_history(tmpl["id"], "errors")
        uptime_hist = generate_metric_history(tmpl["id"], "uptime_drops")
        lat_hist    = generate_metric_history(tmpl["id"], "latency")

        timestamps = [(now - timedelta(hours=11 - i)).strftime("%H:%M") for i in range(12)]

        devices.append({
            **tmpl,
            "trend": seed["trend"],
            "severity": seed["severity"],
            "metrics": {
                "cpu":          {"history": cpu_hist,    "current": cpu_hist[-1],    "unit": "%"},
                "errors":       {"history": err_hist,    "current": err_hist[-1],    "unit": "/min"},
                "uptime_drops": {"history": uptime_hist, "current": uptime_hist[-1], "unit": "events"},
                "latency":      {"history": lat_hist,    "current": lat_hist[-1],    "unit": "ms"},
            },
            "timestamps": timestamps,
            "last_seen": now.isoformat(),
        })
    return devices


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/api/snapshot")
async def get_snapshot():
    """Return current device telemetry snapshot."""
    return {"devices": build_device_snapshot(), "generated_at": datetime.utcnow().isoformat()}


@app.post("/api/forecast")
async def run_forecast():
    """Send snapshot to Claude and return a ranked risk forecast."""
    devices = build_device_snapshot()

    # Build a concise telemetry summary for the prompt
    telemetry_lines = []
    for d in devices:
        cpu     = d["metrics"]["cpu"]
        errors  = d["metrics"]["errors"]
        uptime  = d["metrics"]["uptime_drops"]
        latency = d["metrics"]["latency"]

        cpu_trend    = round(cpu["history"][-1]    - cpu["history"][0],    2)
        err_trend    = round(errors["history"][-1] - errors["history"][0], 2)
        uptime_trend = round(uptime["history"][-1] - uptime["history"][0], 2)
        lat_trend    = round(latency["history"][-1]- latency["history"][0],2)

        telemetry_lines.append(
            f"- {d['name']} ({d['vendor']} {d['type']}, {d['location']}): "
            f"CPU now={cpu['current']}% Δ12h={cpu_trend:+.1f}%, "
            f"errors now={errors['current']}/min Δ12h={err_trend:+.2f}, "
            f"uptime-drops Δ12h={uptime_trend:+.2f}, "
            f"latency now={latency['current']}ms Δ12h={lat_trend:+.2f}ms"
        )

    telemetry_text = "\n".join(telemetry_lines)

    prompt = f"""You are an AI network operations analyst for IP Fabric's Predictive Incident Forecasting module.

Below is a 12-hour telemetry snapshot (current values + delta over 12 hours) for {len(devices)} network devices.
Your job is to analyze the TRENDS, not just the current values, and forecast which devices are most at risk of causing an incident in the next 7–14 days.

TELEMETRY SNAPSHOT:
{telemetry_text}

Return ONLY valid JSON in this exact schema — no markdown, no explanation outside the JSON:
{{
  "forecast_generated_at": "<ISO timestamp>",
  "summary": "<2-3 sentence executive summary of the overall network health>",
  "risk_forecast": [
    {{
      "device_id": "<matches one of the device names above, lowercase-hyphenated>",
      "device_name": "<human readable name>",
      "risk_level": "<critical|high|medium|low>",
      "risk_score": <0-100 integer>,
      "days_to_incident": <estimated days as integer, null if low risk>,
      "primary_signal": "<the single most concerning metric trend>",
      "reasoning": "<2-3 sentences explaining WHY this device is at risk, referencing specific metric trends>",
      "recommended_action": "<specific, actionable recommendation>"
    }}
  ],
  "network_health_score": <0-100 integer, overall network health>,
  "critical_count": <number of critical devices>,
  "high_count": <number of high-risk devices>
}}

Order risk_forecast by risk_score descending. Include ALL {len(devices)} devices.
Device IDs to use: {[d['id'] for d in devices]}
"""

    try:
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = message.content[0].text.strip()
        # strip accidental markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        forecast = json.loads(raw)
        return forecast
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Claude returned invalid JSON: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
