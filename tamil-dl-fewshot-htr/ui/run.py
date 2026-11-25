"""Launch the interactive UI.

    python ui/run.py                 # http://127.0.0.1:8000 (this computer only) and open the browser
    python ui/run.py --lan           # https://<this-PC-ip>:8443 for a phone/tablet on the same Wi-Fi
    python ui/run.py --no-browser --port 8001
"""
from __future__ import annotations

import argparse
import datetime as dt
import ipaddress
import logging
import socket
import sys
import threading
import webbrowser
from pathlib import Path

UI_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(UI_DIR))
sys.path.insert(0, str(UI_DIR.parent / "src"))

from config import force_utf8, load_config, setup_logging  # noqa: E402

log = logging.getLogger("ui.run")


def lan_ip() -> str:
    """Best-guess IPv4 address of this machine on the local network."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))           # no packet is sent; picks the outgoing interface
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def self_signed_cert(ip: str, out_dir: Path) -> tuple[Path, Path]:
    """Create (or reuse) a self-signed certificate for ``ip`` so phones allow camera access."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID
    out_dir.mkdir(parents=True, exist_ok=True)
    cert_p, key_p = out_dir / f"cert_{ip}.pem", out_dir / f"key_{ip}.pem"
    if cert_p.exists() and key_p.exists():
        return cert_p, key_p
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Tamil recogniser (local)")])
    now = dt.datetime.now(dt.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - dt.timedelta(days=1))
            .not_valid_after(now + dt.timedelta(days=365))
            .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address(ip)),
                                                        x509.DNSName("localhost")]), critical=False)
            .sign(key, hashes.SHA256()))
    cert_p.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_p.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
                                        serialization.NoEncryption()))
    return cert_p, key_p


def main() -> None:
    force_utf8()
    cfg = load_config()
    ui = cfg["ui"]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lan", action="store_true", help="serve on the local network over HTTPS (self-signed)")
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args()
    setup_logging(a.log_level)
    import uvicorn
    from server import create_app

    app = create_app()
    if a.lan:
        ip = lan_ip()
        port = a.port or int(ui["lan_port"])
        cert, key = self_signed_cert(ip, UI_DIR / "cache" / "tls")
        url = f"https://{ip}:{port}"
        log.warning("LAN mode: the app is reachable by every device on this network at %s", url)
        log.warning("Open %s on your phone. It uses a self-signed certificate: accept the browser warning "
                    "(Advanced -> Proceed). Stop the server (Ctrl+C) when you are done.", url)
        print(f"\n  Phone URL: {url}\n  (same Wi-Fi; accept the certificate warning once)\n", flush=True)
        if not a.no_browser:
            threading.Timer(1.5, lambda: webbrowser.open(f"https://127.0.0.1:{port}")).start()
        uvicorn.run(app, host="0.0.0.0", port=port, ssl_certfile=str(cert), ssl_keyfile=str(key), log_level="warning")
    else:
        host, port = str(ui["host"]), a.port or int(ui["port"])
        url = f"http://{host}:{port}"
        print(f"\n  Tamil handwriting recogniser: {url}\n", flush=True)
        if not a.no_browser:
            threading.Timer(1.2, lambda: webbrowser.open(url)).start()
        uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
