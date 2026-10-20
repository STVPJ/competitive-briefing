"""Delivery: always to files, optionally by SMTP.

Files are written first, so a mail failure can't lose the briefing.
"""

from __future__ import annotations

import logging
import os
import smtplib
import socket
import time
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

log = logging.getLogger(__name__)

# Retry only connection-level failures. Auth errors and refused recipients surface at
# once (smtplib.SMTPException subclasses OSError, so OSError is deliberately absent).
_TRANSIENT = (smtplib.SMTPServerDisconnected, smtplib.SMTPConnectError, ConnectionError,
              TimeoutError, socket.timeout, socket.gaierror)


def output_paths(output_dir: Path, edition_name: str, d: date, primary: bool) -> tuple[Path, Path]:
    """output/YYYY-MM-DD.html for the primary edition, output/YYYY-MM-DD.<edition>.html otherwise."""
    stem = d.isoformat() if primary else f"{d.isoformat()}.{edition_name}"
    return output_dir / f"{stem}.html", output_dir / f"{stem}.md"


def write_files(html: str, markdown: str, html_path: Path, md_path: Path) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html)
    md_path.write_text(markdown)
    log.info("wrote %s and %s", html_path, md_path)


def smtp_configured() -> bool:
    return all(os.environ.get(k) for k in ("SMTP_HOST", "SMTP_FROM"))


def send_email(subject: str, html: str, text: str, recipients: list[str], attempts: int = 3) -> int:
    """Send one individually addressed email per recipient. Returns the number sent.

    Env: SMTP_HOST, SMTP_PORT (465 = implicit TLS, otherwise STARTTLS), SMTP_USERNAME,
    SMTP_PASSWORD, SMTP_FROM.
    """
    host, port = os.environ["SMTP_HOST"], int(os.environ.get("SMTP_PORT", "465"))
    user, password = os.environ.get("SMTP_USERNAME"), os.environ.get("SMTP_PASSWORD")
    sender = os.environ["SMTP_FROM"]
    sent = 0
    for to in recipients:
        msg = MIMEMultipart("alternative")
        msg["Subject"], msg["From"], msg["To"] = subject, sender, to
        msg.attach(MIMEText(text, "plain", "utf-8"))
        msg.attach(MIMEText(html, "html", "utf-8"))
        for attempt in range(1, attempts + 1):
            try:
                if port == 465:
                    server = smtplib.SMTP_SSL(host, port, timeout=30)
                else:
                    server = smtplib.SMTP(host, port, timeout=30)
                    server.starttls()
                with server:
                    if user and password:
                        server.login(user, password)
                    server.sendmail(sender, [to], msg.as_string())
                sent += 1
                break
            except _TRANSIENT as e:
                if attempt == attempts:
                    log.error("send to %s failed: %s", to, e)
                else:
                    time.sleep(2 ** attempt)
            except smtplib.SMTPException as e:
                log.error("send to %s failed: %s", to, e)
                break
    return sent
