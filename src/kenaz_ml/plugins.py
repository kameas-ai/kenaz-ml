"""Plugin capabilities discovery — queries sigild for installed plugins and their actions."""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request

from kenaz_ml import config

logger = logging.getLogger(__name__)

_CACHE_TTL_SEC = 300
_cache: dict | None = None
_cache_ts: float = 0.0


def fetch_capabilities() -> dict:
    """Fetch plugin capabilities from sigild's HTTP API.

    Returns a dict with 'plugins' key containing a list of plugin capability dicts.
    Caches the result for 5 minutes. Returns empty on failure (non-fatal).
    """
    global _cache, _cache_ts

    now = time.time()
    if _cache is not None and (now - _cache_ts) < _CACHE_TTL_SEC:
        return _cache

    url = f"{config.sigild_plugin_url()}/api/v1/capabilities"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
            _cache = data
            _cache_ts = now
            logger.debug("plugins: fetched %d plugin capabilities", len(data.get("plugins", [])))
            return data
    except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError) as e:
        logger.debug("plugins: capabilities fetch failed (sigild may not be running): %s", e)
        return {"plugins": []}


def get_plugin_names() -> list[str]:
    """Return names of all installed plugins."""
    data = fetch_capabilities()
    return [p.get("plugin", "") for p in data.get("plugins", [])]


def get_data_sources() -> list[str]:
    """Return all data sources across installed plugins."""
    data = fetch_capabilities()
    sources: list[str] = []
    for p in data.get("plugins", []):
        sources.extend(p.get("data_sources", []))
    return sources


def get_actions() -> list[dict]:
    """Return all actions across installed plugins."""
    data = fetch_capabilities()
    actions: list[dict] = []
    for p in data.get("plugins", []):
        plugin_name = p.get("plugin", "")
        for action in p.get("actions", []):
            actions.append(
                {
                    "plugin": plugin_name,
                    "name": action.get("name", ""),
                    "description": action.get("description", ""),
                    "command": action.get("command", ""),
                }
            )
    return actions


def get_event_kinds_for_plugin(plugin_name: str) -> list[str]:
    """Return data source event kinds for a specific plugin."""
    data = fetch_capabilities()
    for p in data.get("plugins", []):
        if p.get("plugin") == plugin_name:
            return p.get("data_sources", [])
    return []


def invalidate_cache() -> None:
    """Force re-fetch on next call."""
    global _cache, _cache_ts
    _cache = None
    _cache_ts = 0.0
