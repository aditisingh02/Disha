"""
Dynamic explainability: generate human-readable explanations from actual outputs.
All text is data-driven so it changes with inputs and results.
"""

from datetime import datetime, timezone
from typing import Any


def explain_inputs(inputs: dict) -> list[str]:
    """Explain what the received inputs mean and how they affect the pipeline."""
    lines = []
    evt = inputs.get("event_context", "none")
    if evt == "none":
        lines.append("No event context: speeds use raw historical (road, hour) means.")
    else:
        factors = {"accident": "30%", "weather": "15%", "public_event": "20%"}
        lines.append("Event context '%s': predicted speeds reduced by %s (applied to all horizons)." % (evt, factors.get(evt, "?")))
    opt = inputs.get("optimization_preference", "fastest")
    lines.append("Optimization '%s': path shown is %s; comparison includes eco alternative (12%% longer, lower emissions)." % (opt, "shortest by distance" if opt == "fastest" else opt))
    return lines


def explain_forecast(forecast: dict, departure_time: str) -> list[str]:
    """Explain multi-horizon and congestion from actual forecast data."""
    lines = []
    mh = forecast.get("multi_horizon_forecasts", {})
    if mh:
        try:
            dep = datetime.fromisoformat(departure_time.replace("Z", "+00:00"))
        except Exception:
            dep = datetime.now(timezone.utc)
        h0 = dep.hour
        lines.append("Multi-horizon: each horizon uses historical (road, future_hour) mean. Departure hour = %d (UTC)." % h0)
        # Which future hours
        hour_labels = []
        for label in ["1h", "3h", "6h", "12h", "24h"]:
            if label not in mh:
                continue
            h = int(label.replace("h", ""))
            future_h = (h0 + h) % 24
            hour_labels.append("%s→hour %d" % (label, future_h))
        lines.append("Future hours: " + ", ".join(hour_labels) + ".")
        # Why some differ
        speeds_seen = {}
        for label, data in mh.items():
            s = data.get("speed_kmh")
            if s not in speeds_seen:
                speeds_seen[s] = []
            speeds_seen[s].append(label)
        if len(speeds_seen) > 1:
            diffs = ["%s km/h at %s" % (s, ", ".join(labels)) for s, labels in speeds_seen.items()]
            lines.append("Different speeds by time-of-day: " + "; ".join(diffs) + ".")
        else:
            lines.append("All horizons show similar speed (historical means for those hours are close).")
    cong = forecast.get("congestion_classification", {})
    if cong:
        level = cong.get("level", "—")
        score = cong.get("score", 0)
        delay = cong.get("delay_vs_freeflow_min")
        ratio = cong.get("speed_ratio")
        lines.append("Congestion: %s = speed ratio %.2f (vs free flow 80 km/h); score %.2f (0=free, 1=gridlock)." % (level, ratio or 0, score))
        if delay is not None:
            lines.append("Delay vs free flow: %.2f min extra over this route." % delay)
    return lines


def explain_risk(risk: dict, origin: str, destination: str) -> list[str]:
    """Explain risk scores from Phase 3 risk tensor."""
    lines = []
    o = risk.get("origin_risk", 0)
    d = risk.get("destination_risk", 0)
    n = risk.get("hotspot_count", 0)
    lines.append("Risk = composite of speed, density proxy, and incident rate (0–1). %s: %.2f; %s: %.2f." % (origin, o, destination, d))
    if n > 0:
        lines.append("%d road(s) in network are above 85th percentile risk (hotspots)." % n)
    return lines


def explain_routing(routes: dict) -> list[str]:
    """Explain route and eco vs fastest from actual numbers."""
    lines = []
    path = routes.get("path", [])
    n_seg = max(0, len(path) - 1) if path else 0
    total_km = routes.get("total_km", 0)
    lines.append("Path has %d segment(s), total %.2f km. Fastest = shortest by distance; eco = same path with 12%% longer distance and 15%% lower emission factor (smoother flow)." % (n_seg, total_km))
    pct = routes.get("percent_emission_reduction", 0)
    lines.append("Emission reduction eco vs fastest: %.1f%% (eco route emits less CO2 despite slightly longer distance)." % pct)
    return lines


def explain_uncertainty(forecast: dict, routes: dict) -> list[str]:
    """Explain uncertainty intervals from actual CI and emission range."""
    lines = []
    mh = forecast.get("multi_horizon_forecasts", {})
    if mh and "1h" in mh:
        lo = mh["1h"].get("ci_80_lower")
        hi = mh["1h"].get("ci_80_upper")
        if lo is not None and hi is not None:
            lines.append("Speed 80%% CI: [%.1f, %.1f] km/h (historical variance; ±15%% used if no model uncertainty)." % (lo, hi))
    pct = routes.get("percent_emission_reduction", 0)
    lines.append("Emission reduction interval: ±5%% around %.1f%% (operational uncertainty)." % (pct))
    return lines
