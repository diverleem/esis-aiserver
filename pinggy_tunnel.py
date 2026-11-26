"""
pinggy_tunnel.py - Modul Manajemen SSH Reverse Tunnel Pinggy untuk E.S.I.S
Memungkinkan akses publik aman (HTTPS) ke AI Server di RunPod / Cloud GPU tanpa konfigurasi port forwarding.

Penggunaan Mandiri:
    python pinggy_tunnel.py --port 8001
    python pinggy_tunnel.py --port 8001 --qr
    python pinggy_tunnel.py --port 8001 --token <PINGGY_TOKEN>

Penggunaan sebagai Modul:
    from pinggy_tunnel import PinggyTunnel
    tunnel = PinggyTunnel(port=8001)
    tunnel.start()
    print("URL Publik:", tunnel.public_url)
    ...
    tunnel.stop()
"""

import os
import sys
import time
import subprocess
import threading
import re
import atexit
import argparse
from typing import Optional

class PinggyTunnel:
    """
    Mengelola proses reverse tunnel SSH Pinggy (https://pinggy.io).
    Bekerja secara native dengan binary ssh yang sudah terpasang di Linux (RunPod) & Windows.
    """
    def __init__(self, port: int = 8001, token: Optional[str] = None, show_qr: bool = False):
        self.port = port
        self.token = token
        self.show_qr = show_qr
        self.process: Optional[subprocess.Popen] = None
        self.public_url: Optional[str] = None
        self.web_debugger_url: Optional[str] = None
        self.is_running = False
        self._thread: Optional[threading.Thread] = None

    def start(self, wait_for_url: bool = True, timeout: float = 15.0) -> Optional[str]:
        """
        Memulai tunnel SSH Pinggy.
        Jika wait_for_url=True, akan menunggu hingga URL publik didapatkan dari banner Pinggy.
        """
        if self.token:
            target_host = f"{self.token}@a.pinggy.io"
        elif self.show_qr:
            target_host = "qr@a.pinggy.io"
        else:
            target_host = "a.pinggy.io"

        ssh_cmd = [
            "ssh",
            "-p", "443",
            f"-R0:localhost:{self.port}",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null" if sys.platform != "win32" else "UserKnownHostsFile=NUL",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3",
            "-o", "ExitOnForwardFailure=yes",
            target_host
        ]

        try:
            self.process = subprocess.Popen(
                ssh_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            self.is_running = True
            atexit.register(self.stop)

            url_pattern = re.compile(r"https://[a-zA-Z0-9.-]+\.pinggy\.link[^\s]*")
            debugger_pattern = re.compile(r"http://localhost:\d+")

            def _reader():
                for line in iter(self.process.stdout.readline, ''):
                    if not line:
                        break
                    
                    if not self.public_url:
                        match = url_pattern.search(line)
                        if match:
                            self.public_url = match.group(0).rstrip("/")
                            self._print_success_banner()

                    if not self.web_debugger_url:
                        d_match = debugger_pattern.search(line)
                        if d_match:
                            self.web_debugger_url = d_match.group(0)

                self.is_running = False

            self._thread = threading.Thread(target=_reader, daemon=True)
            self._thread.start()

            if wait_for_url:
                start_wait = time.time()
                while not self.public_url and (time.time() - start_wait) < timeout and self.is_running:
                    time.sleep(0.3)

            return self.public_url

        except Exception as e:
            print(f"[PINGGY ERROR] Gagal meluncurkan perintah SSH Pinggy: {e}")
            return None

    def _print_success_banner(self):
        print("\n" + "=" * 80)
        print("   PINGGY REVERSE TUNNEL BERHASIL AKTIF! (HTTPS PUBLIC ACCESS)")
        print("=" * 80)
        print(f" * Public HTTPS URL  : {self.public_url}")
        print(f" * Target Local Port : http://localhost:{self.port}")
        print(f" * Endpoint Postman  : {self.public_url}/api/violations")
        print(f" * Cek Health/Status : {self.public_url}/health")
        print(f" * Snapshot Kamera   : {self.public_url}/snapshot")
        print(f" * Simulasi Tes API  : {self.public_url}/api/simulate")
        print("=" * 80)
        print(" [INFO] URL di atas dapat diakses dari Postman, cURL, maupun browser")
        print("        dari mana saja tanpa batasan jaringan lokal atau firewall.\n")

    def stop(self):
        """Menghentikan tunnel SSH secara aman."""
        if self.process and self.is_running:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.is_running = False
            print("[PINGGY] Tunnel SSH ditutup.")


def main():
    parser = argparse.ArgumentParser(description="E.S.I.S Pinggy SSH Tunnel Manager")
    parser.add_argument("--port", type=int, default=8001, help="Port lokal yang ingin diekspos (default: 8001)")
    parser.add_argument("--token", type=str, default=None, help="Token autentikasi Pinggy (opsional jika punya akun berbayar)")
    parser.add_argument("--qr", action="store_true", help="Tampilkan QR code untuk akses cepat dari kamera smartphone")
    args = parser.parse_args()

    print("===============================================================================")
    print("   E.S.I.S Pinggy Reverse Tunnel Launcher")
    print(f"   Menghubungkan http://localhost:{args.port} ke jaringan publik...")
    print("===============================================================================")

    tunnel = PinggyTunnel(port=args.port, token=args.token, show_qr=args.qr)
    url = tunnel.start(wait_for_url=True, timeout=15.0)

    if not url:
        print("[PERINGATAN] URL belum terdeteksi dalam 15 detik. Pastikan koneksi internet aktif.")

    try:
        while tunnel.is_running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[BERHENTI] Sinyal Ctrl+C diterima. Menutup tunnel...")
    finally:
        tunnel.stop()

if __name__ == "__main__":
    main()
