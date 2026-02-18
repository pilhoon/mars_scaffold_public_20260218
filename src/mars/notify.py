from __future__ import annotations

import logging
import os
import threading
import time
import urllib.parse
import urllib.request
from typing import Any


LOGGER = logging.getLogger(__name__)
_LAST_SENT_TS: dict[str, float] = {}
_LOCK = threading.Lock()


def _get_telegram_credentials() -> tuple[str, str]:
    # Environment overrides are checked first for portability.
    token = (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
    chat_id = (os.environ.get("TELEGRAM_CHAT_ID") or "").strip()
    if token and chat_id:
        return token, chat_id
    if token or chat_id:
        raise RuntimeError(
            "both TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set (only one found)"
        )

    try:
        import redis  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "telegram credentials missing: set TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID "
            "or install redis and provide auth:telegram:* keys"
        ) from exc

    host = os.environ.get("TELEGRAM_REDIS_HOST", "localhost")
    port = int(os.environ.get("TELEGRAM_REDIS_PORT", "6379"))
    db = int(os.environ.get("TELEGRAM_REDIS_DB", "1"))
    token_key = os.environ.get("TELEGRAM_REDIS_TOKEN_KEY", "auth:telegram:token")
    chat_key = os.environ.get("TELEGRAM_REDIS_CHAT_ID_KEY", "auth:telegram:chat_id")
    client: Any = redis.Redis(
        host=host,
        port=port,
        db=db,
        decode_responses=True,
        socket_timeout=5,
    )
    token = (client.get(token_key) or "").strip()
    chat_id = (client.get(chat_key) or "").strip()
    if not token or not chat_id:
        raise RuntimeError(
            f"telegram credentials not found in redis db={db} keys={token_key},{chat_key}"
        )
    return token, chat_id


def _send_telegram_sync(text: str) -> bool:
    token, chat_id = _get_telegram_credentials()
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
    req = urllib.request.Request(url, data=payload, method="POST")
    with urllib.request.urlopen(req, timeout=10) as resp:
        if int(getattr(resp, "status", 0) or 0) != 200:
            raise RuntimeError(f"telegram send failed with status={getattr(resp, 'status', None)}")
    return True


def send_telegram(
    text: str,
    *,
    dedupe_key: str | None = None,
    min_interval_sec: float = 0.0,
) -> bool:
    if not text:
        return False

    reserved_ts: float | None = None
    if dedupe_key:
        now_ts = time.time()
        with _LOCK:
            prev = float(_LAST_SENT_TS.get(dedupe_key, 0.0))
            if now_ts - prev < max(0.0, float(min_interval_sec)):
                return False
            _LAST_SENT_TS[dedupe_key] = now_ts
            reserved_ts = now_ts

    def _worker() -> None:
        try:
            _send_telegram_sync(text)
        except Exception as exc:
            if dedupe_key and reserved_ts is not None:
                with _LOCK:
                    if _LAST_SENT_TS.get(dedupe_key) == reserved_ts:
                        del _LAST_SENT_TS[dedupe_key]
            LOGGER.warning("telegram send skipped/failed: %r", exc)

    threading.Thread(target=_worker, daemon=True).start()
    return True

