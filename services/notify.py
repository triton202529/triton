# services/notify.py
import os
import json
import smtplib
from email.message import EmailMessage
from typing import Optional, Dict, Any

import requests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_notify_config() -> Dict[str, Any]:
    cfg = _load_json(os.path.join(ROOT, "config", "notify.json"))
    # Env overrides
    cfg.setdefault("slack_webhook_url", os.getenv("SLACK_WEBHOOK_URL", ""))
    cfg.setdefault("smtp_host", os.getenv("SMTP_HOST", ""))
    cfg.setdefault("smtp_port", int(os.getenv("SMTP_PORT", "587") or 587))
    cfg.setdefault("smtp_user", os.getenv("SMTP_USER", ""))
    cfg.setdefault("smtp_password", os.getenv("SMTP_PASSWORD", ""))
    cfg.setdefault("smtp_from", os.getenv("SMTP_FROM", ""))
    cfg.setdefault("notify_to", os.getenv("NOTIFY_TO", cfg.get("notify_to", "")))
    return cfg


def _send_slack(text: str, cfg: Dict[str, Any]) -> None:
    url = cfg.get("slack_webhook_url", "")
    if not url:
        return
    try:
        requests.post(url, json={"text": text}, timeout=10)
    except Exception:
        pass


def _send_email(subject: str, body: str, cfg: Dict[str, Any]) -> None:
    host = cfg.get("smtp_host", "")
    to_list = cfg.get("notify_to", "")
    if not host or not to_list:
        return
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = cfg.get("smtp_from", cfg.get("smtp_user", ""))
        msg["To"] = to_list
        msg.set_content(body)
        with smtplib.SMTP(host, int(cfg.get("smtp_port", 587))) as s:
            s.starttls()
            user = cfg.get("smtp_user", "")
            pwd = cfg.get("smtp_password", "")
            if user and pwd:
                s.login(user, pwd)
            s.send_message(msg)
    except Exception:
        pass


def notify(event: str, text: str, extra: Optional[Dict[str, Any]] = None) -> None:
    """
    event: submit | filled | partial_fill | canceled | rejected | refresh | error
    text: human-readable one-liner
    extra: optional dictionary for future use
    """
    cfg = _load_notify_config()
    # Slack
    _send_slack(text, cfg)
    # Email
    subj = f"[TRITON] {event.upper()}"
    _send_email(subj, text, cfg)
