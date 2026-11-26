"""
r2_reporter.py - Modul Integrasi Cloudflare R2 & Pelaporan Otomatis Pelanggaran E.S.I.S
Digunakan oleh AI Server untuk:
1. Mengunggah foto snapshot bukti rekaman CCTV ke Cloudflare R2 (S3 API).
2. Mengirim laporan insiden via HTTP POST ke API Website E.S.I.S (/api/violations).
"""

import os
import sys
import time
import datetime
import requests
import boto3
from botocore.config import Config

# Coba import OpenCV jika tersedia
try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Direktori dasar
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(BASE_DIR, ".env")

# Muat variabel dari .env secara manual jika python-dotenv tidak terpasang
def load_env_file(filepath):
    if not os.path.exists(filepath):
        return
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

load_env_file(ENV_FILE)

# Konfigurasi R2 & API
R2_ENDPOINT_URL = os.environ.get("R2_ENDPOINT_URL", "https://ae683a81ba7db85821422f8864eee6bd.r2.cloudflarestorage.com")
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID", "2ade10a65df615bdb1e83643f6feee9c")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY", "f6fe896bbf963560a3a4ec8e6ce2151537295e623231296f49390df0bdce0def")
R2_BUCKET_NAME = os.environ.get("R2_BUCKET_NAME", "esis-violations")
R2_PUBLIC_URL = os.environ.get("R2_PUBLIC_URL", "https://violations.ucim.my.id").rstrip("/")
DEFAULT_WEB_API_URL = os.environ.get("WEB_API_URL", "http://localhost:3000/api/violations")

_s3_client = None

def get_r2_client():
    """Menginisialisasi klien S3 untuk Cloudflare R2."""
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client(
            service_name="s3",
            endpoint_url=R2_ENDPOINT_URL,
            aws_access_key_id=R2_ACCESS_KEY_ID,
            aws_secret_access_key=R2_SECRET_ACCESS_KEY,
            config=Config(signature_version="s3v4"),
            region_name="auto",
        )
    return _s3_client


def upload_evidence_to_r2(frame_or_path, filename=None) -> str:
    """
    Mengunggah bukti gambar (berupa path file lokal atau frame numpy/OpenCV) ke Cloudflare R2.
    Mengembalikan Public URL gambar (misal https://violations.ucim.my.id/pelanggaran_xxx.jpg).
    """
    s3 = get_r2_client()

    if filename is None:
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pelanggaran_{now_str}.jpg"

    # Jika input berupa numpy array (frame OpenCV)
    if HAS_CV2 and isinstance(frame_or_path, np.ndarray):
        success, buffer = cv2.imencode(".jpg", frame_or_path, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not success:
            raise ValueError("Gagal mengompresi frame OpenCV ke format JPEG.")
        body_bytes = buffer.tobytes()
    elif isinstance(frame_or_path, (bytes, bytearray)):
        body_bytes = bytes(frame_or_path)
    elif isinstance(frame_or_path, str):
        if not os.path.exists(frame_or_path):
            raise FileNotFoundError(f"File bukti gambar tidak ditemukan: {frame_or_path}")
        with open(frame_or_path, "rb") as f:
            body_bytes = f.read()
    else:
        raise TypeError("Format input frame_or_path tidak didukung (harus path str, bytes, atau numpy.ndarray).")

    # Upload ke Cloudflare R2
    s3.put_object(
        Bucket=R2_BUCKET_NAME,
        Key=filename,
        Body=body_bytes,
        ContentType="image/jpeg",
    )

    public_url = f"{R2_PUBLIC_URL}/{filename}"
    return public_url


def report_violation_to_api(
    location: str,
    evidence_image_url: str,
    location_group: str = None,
    student_id: str = None,
    student_name: str = None,
    confidence_smoking: float = None,
    face_similarity: float = None,
    notes: str = None,
    api_url: str = DEFAULT_WEB_API_URL,
    timeout: float = 8.0,
) -> dict:
    """
    Mengirim payload data insiden ke API Next.js /api/violations.
    """
    payload = {
        "location": location,
        "evidence_image_url": evidence_image_url,
    }

    if location_group:
        payload["location_group"] = location_group
    if student_id:
        payload["student_id"] = str(student_id).strip()
    if student_name:
        payload["student_name"] = str(student_name).strip()
    if confidence_smoking is not None:
        payload["confidence_smoking"] = float(confidence_smoking)
    if face_similarity is not None:
        payload["face_similarity"] = float(face_similarity)
    if notes:
        payload["notes"] = str(notes).strip()

    try:
        response = requests.post(api_url, json=payload, timeout=timeout)
        data = response.json()
        if response.ok and data.get("success", False):
            return {
                "status_code": response.status_code,
                "success": True,
                "response": data,
            }
        print(f" [API WARNING] Response Web API: {data.get('message') or response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f" [API INFO] Web API lokal ({api_url}) tidak merespons ({e}). Menggunakan direct fallback ke Supabase...")

    # Direct Supabase Fallback jika Web API lokal (localhost:3000) tidak aktif
    supabase_url = os.environ.get("SUPABASE_URL", "https://gzbiqnzwtbynnhpfsqso.supabase.co")
    supabase_key = os.environ.get("SUPABASE_KEY", "sb_publishable_eMibXfZffVVJtqfEh0hLJw_xeJ5EWKz")
    if supabase_url and supabase_key:
        try:
            sb_headers = {
                "apikey": supabase_key,
                "Authorization": f"Bearer {supabase_key}",
                "Content-Type": "application/json",
                "Prefer": "return=representation"
            }
            # Lengkapi data prodi/email jika nama mahasiswa terdeteksi
            if student_name and student_name not in ["Mahasiswa Tidak Terdaftar", "Tidak Dikenal"]:
                try:
                    m_res = requests.get(
                        f"{supabase_url.rstrip('/')}/rest/v1/mahasiswa?name=ilike.*{student_name}*&limit=1",
                        headers=sb_headers,
                        timeout=2.0
                    )
                    if m_res.ok:
                        m_list = m_res.json()
                        if isinstance(m_list, list) and len(m_list) > 0:
                            m_info = m_list[0]
                            payload["student_id"] = payload.get("student_id") or m_info.get("nim")
                            payload["student_department"] = m_info.get("prodi") or m_info.get("jurusan")
                            payload["student_email"] = m_info.get("email")
                except Exception:
                    pass

            payload["status"] = "pending"
            sb_res = requests.post(
                f"{supabase_url.rstrip('/')}/rest/v1/violations",
                json=payload,
                headers=sb_headers,
                timeout=timeout
            )
            if sb_res.status_code in [200, 201]:
                sb_data = sb_res.json()
                return {
                    "status_code": sb_res.status_code,
                    "success": True,
                    "response": {"success": True, "direct_supabase": True, "data": sb_data},
                    "message": "Pelanggaran berhasil dicatat langsung ke Supabase."
                }
            else:
                print(f" [SUPABASE ERROR] Gagal insert langsung: {sb_res.status_code} {sb_res.text}")
        except Exception as sb_err:
            print(f" [SUPABASE ERROR] Exception direct fallback: {sb_err}")

    return {
        "status_code": 0,
        "success": False,
        "message": f"Koneksi ke Web API dan Direct Supabase gagal.",
    }


def create_simulated_snapshot(location: str, student_name: str, confidence: float = 0.94):
    """Membuat frame visual simulasi rekaman CCTV bukti pelanggaran merokok."""
    if not HAS_CV2:
        return b"DUMMY_SNAPSHOT_IMAGE_BYTES"

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (35, 28, 24)

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Header CCTV OSD
    cv2.rectangle(frame, (0, 0), (640, 45), (20, 20, 20), -1)
    cv2.putText(frame, f"E.S.I.S CCTV AI PATROL | {location}", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, "REC [LIVE]", (520, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Info Kotak Kejadian
    cv2.rectangle(frame, (20, 60), (620, 420), (50, 50, 50), 1)
    cv2.putText(frame, f"WAKTU   : {now_str} WIB", (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(frame, f"LOKASI  : {location}", (40, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(frame, f"TERDUGA : {student_name}", (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.putText(frame, f"DETEKSI : MEROKOK TERKONFIRMASI (>3 DETIK)", (40, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
    cv2.putText(frame, f"KEYAKINAN: {confidence * 100:.1f}%", (40, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    cv2.putText(frame, "STATUS  : PENDING (MENUNGGU VERIFIKASI ADMIN)", (40, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 180, 255), 1)

    # Simulasi Kotak Bounding Box Rokok
    cv2.rectangle(frame, (380, 290), (480, 390), (0, 0, 255), 2)
    cv2.putText(frame, f"Rokok: {confidence * 100:.0f}%", (380, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Footer
    cv2.putText(frame, "Cloudflare R2 Storage: esis-violations | custom domain: violations.ucim.my.id", (20, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

    return frame


def capture_and_report_incident(
    frame_or_path,
    location: str,
    location_group: str = None,
    student_id: str = None,
    student_name: str = None,
    confidence_smoking: float = None,
    face_similarity: float = None,
    notes: str = None,
    filename: str = None,
    api_url: str = DEFAULT_WEB_API_URL,
) -> dict:
    """
    Fungsi all-in-one untuk AI Patrol Server:
    1. Upload foto bukti ke Cloudflare R2
    2. Kirim laporan JSON ke Web Dashboard API
    """
    try:
        # Jika input berupa keyword "simulated", buat frame snapshot sintetis
        if isinstance(frame_or_path, str) and frame_or_path == "simulated":
            frame_or_path = create_simulated_snapshot(location, student_name or "Mahasiswa Terduga", confidence_smoking or 0.94)

        # 1. Upload ke Cloudflare R2
        public_url = upload_evidence_to_r2(frame_or_path, filename=filename)
        print(f" [R2] Bukti foto berhasil diunggah ke R2: {public_url}")

        # 2. Kirim laporan ke Web API
        api_result = report_violation_to_api(
            location=location,
            evidence_image_url=public_url,
            location_group=location_group,
            student_id=student_id,
            student_name=student_name,
            confidence_smoking=confidence_smoking,
            face_similarity=face_similarity,
            notes=notes,
            api_url=api_url,
        )

        if api_result.get("success"):
            data_resp = api_result.get("response", {})
            if data_resp.get("cooldown_active"):
                print(f" [API COOLDOWN] {data_resp.get('message')}")
            else:
                print(f" [API] Laporan pelanggaran berhasil dicatat ke Dashboard Web.")
        else:
            print(f" [API WARNING] Gagal sinkronisasi ke Web API: {api_result.get('message') or api_result.get('error')}")

        return {
            "success": api_result.get("success", False),
            "evidence_url": public_url,
            "api_result": api_result,
        }
    except Exception as exc:
        print(f" [ERROR] Terjadi kesalahan pada proses capture_and_report_incident: {exc}")
        return {
            "success": False,
            "error": str(exc),
        }
