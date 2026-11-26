#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
E.S.I.S (Electronic Smoking Identification System) - Plan A: RunPod Cloud GPU
Script: main_runpod.py (Cloud GPU Inference via Pinggy Tunnel Streaming)
Folder: AI Server/Fix/Plan_A_RunPod
================================================================================
Target Kamera: Lab Riset Lantai 5 (Mainstream Cam via Pinggy MediaMTX)
Skema Alur:
1. Mengambil video feed mainstream (link Cam: cam_lab_riset_lt_5) dari Pinggy CLI.
2. Sinkronisasi otomatis URL streaming dari Supabase (tabel streaming, id=38).
3. Human Detection & Tracking stabil (YOLOv8s/n ONNX + centroid/IoU tracker).
4. Deteksi Objek Merokok (YOLO11s / YOLOv8 Grey v2/v3).
5. Aturan Verifikasi Merokok Kontinu minimal 3.0 detik.
6. Identifikasi Wajah Spasial Khusus Perokok (YuNet + FaceNet ONNX, 38 mahasiswa).
7. Jika terverifikasi -> Simpan snapshot beranotasi lengkap (kotak merah rokok,
   kotak hijau perokok, identitas mahasiswa, dan header OSD resmi).
8. Unggah bukti foto ke Cloudflare R2 Bucket & Kirim Laporan ke Web API
   (dengan direct fallback otomatis ke database Supabase jika web offline).
9. Mode Default: Terminal Headless (sangat ringan, siap deploy di RunPod Cloud).
================================================================================
"""

import os
import sys
import time
import argparse
import datetime
import threading
import json
import queue
import tempfile
from typing import Dict, List, Tuple, Optional
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

# Fix console encoding
if hasattr(sys.stdout, 'reconfigure'):
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass
if hasattr(sys.stderr, 'reconfigure'):
    try: sys.stderr.reconfigure(encoding='utf-8')
    except Exception: pass

# Konfigurasi timeout RTSP/HLS FFMPEG (3.0 detik) agar tidak freeze saat koneksi internet lambat
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;3000000|reorder_queue_size;0"
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(tempfile.gettempdir(), "ultralytics_cfg"))

# Fix Windows / Linux PermissionError pada .git commondir saat impor ultralytics
import pathlib
_orig_exists = pathlib.Path.exists
def _safe_exists(self):
    try:
        return _orig_exists(self)
    except PermissionError:
        return False
pathlib.Path.exists = _safe_exists

import numpy as np
import cv2
import onnxruntime as ort
from ultralytics import YOLO

# Import modul internal
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from r2_reporter import capture_and_report_incident, report_violation_to_api, upload_evidence_to_r2, HAS_CV2
from pinggy_tunnel import PinggyTunnel

# ------------------------------------------------------------------------------
# 1. KONFIGURASI PATH MODEL & KAMERA TARGET
# ------------------------------------------------------------------------------
MODELS_DIR = os.path.join(BASE_DIR, "models")
CAPTURES_DIR = os.path.join(BASE_DIR, "captures")
os.makedirs(CAPTURES_DIR, exist_ok=True)

def resolve_model(filename: str, fallback_rel: str = "") -> str:
    """Mencari file model di folder models/ lokal, dengan fallback jika tidak ditemukan."""
    local_p = os.path.join(MODELS_DIR, filename)
    if os.path.exists(local_p):
        return local_p
    if fallback_rel:
        fallback_p = os.path.join(BASE_DIR, fallback_rel)
        if os.path.exists(fallback_p):
            return fallback_p
    return local_p

# Default path model (Murni ONNX untuk Akselerasi GPU Cloud)
PATH_YOLO_PERSON_ONNX = os.path.join(MODELS_DIR, "yolov8s.onnx") if os.path.exists(os.path.join(MODELS_DIR, "yolov8s.onnx")) else os.path.join(MODELS_DIR, "yolov8n.onnx")
PATH_YOLO_ROKOK_ONNX = resolve_model("Model_Rokok_Esis_New_Era_Grey_v3.onnx", os.path.join("..", "Cigar Model", "Model_Rokok_Esis_New_Era_Grey_v3.onnx"))
if not os.path.exists(PATH_YOLO_ROKOK_ONNX):
    PATH_YOLO_ROKOK_ONNX = resolve_model("Model_Rokok_Esis_New_Era_Grey_v2.onnx", os.path.join("..", "Cigar Model", "Model_Rokok_Esis_New_Era_Grey_v2.onnx"))

CANDIDATE_ROKOK_MODELS = {
    "v3_onnx": resolve_model("Model_Rokok_Esis_New_Era_Grey_v3.onnx", os.path.join("..", "Cigar Model", "Model_Rokok_Esis_New_Era_Grey_v3.onnx")),
    "v2_onnx": resolve_model("Model_Rokok_Esis_New_Era_Grey_v2.onnx", os.path.join("..", "Cigar Model", "Model_Rokok_Esis_New_Era_Grey_v2.onnx")),
    "v3": resolve_model("Model_Rokok_Esis_New_Era_Grey_v3.pt"),
    "v2": resolve_model("Model_Rokok_Esis_New_Era_Grey_v2.pt", os.path.join("..", "Cigar Model", "Model_Rokok_Esis_New_Era_Grey_v2.pt")),
    "v1": resolve_model("Model Rokok YOLOv8n.pt", os.path.join("..", "AI Models", "Model Rokok YOLOv8n.pt")),
}

CANDIDATE_PERSON_MODELS = {
    "yolov8s": os.path.join(MODELS_DIR, "yolov8s.onnx"),
    "yolov8n": os.path.join(MODELS_DIR, "yolov8n.onnx"),
}

PATH_YUNET_ONNX = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
PATH_FACENET_ONNX = os.path.join(MODELS_DIR, "facenet.onnx")
PATH_DATABASE_NPZ = os.path.join(MODELS_DIR, "data_wajah_facenet_mtcnn.npz")

# Default Stream Base URL (MediaMTX HLS via Pinggy Pro port 8888)
DEFAULT_PINGGY_STREAM_BASE = os.environ.get("PINGGY_STREAM_BASE_URL", "https://fdxuqgcmfo.a.pinggy.link").rstrip("/")
DEFAULT_STREAM_URL = f"{DEFAULT_PINGGY_STREAM_BASE}/cam_lab_riset_lt_5/index.m3u8"

# Target 4 Kamera Patroli Serentak (Plan A Cloud GPU)
TARGET_LAB_RISET = {
    "id": 38,
    "key": "lab_riset_lt5",
    "name": "Lab Riset Lantai 5",
    "location_group": "Lantai 5 Gedung PLT",
    "ecam_path": "ecam_lab_riset_lt_5",
    "cam_path": "cam_lab_riset_lt_5",
    "ip": "172.20.32.44",
    "user": "admin",
    "pass": "EYQMDK",
    "channel_101": "rtsp://admin:EYQMDK@172.20.32.44:554/Streaming/Channels/101",
    "channel_102": "rtsp://admin:EYQMDK@172.20.32.44:554/Streaming/Channels/102",
    "hls_source": "https://fdxuqgcmfo.a.pinggy.link/cam_lab_riset_lt_5",
    "codec_info": "H.265 (HEVC)",
    "is_primary": True,
}

TARGET_LAB_OS = {
    "id": 36,
    "key": "lab_os",
    "name": "Lab OS",
    "location_group": "Lantai 5 Gedung PLT",
    "ecam_path": "ecam_lab_os",
    "cam_path": "cam_lab_os",
    "ip": "172.20.32.59",
    "user": "admin",
    "pass": "YEYXIJ",
    "channel_101": "rtsp://admin:YEYXIJ@172.20.32.59:554/Streaming/Channels/101",
    "channel_102": "rtsp://admin:YEYXIJ@172.20.32.59:554/Streaming/Channels/102",
    "hls_source": "https://fdxuqgcmfo.a.pinggy.link/cam_lab_os",
    "codec_info": "H.265 (HEVC)",
    "is_primary": False,
}

TARGET_LAB_PROGRAMMING = {
    "id": 37,
    "key": "lab_programming",
    "name": "Lab Programming",
    "location_group": "Lantai 5 Gedung PLT",
    "ecam_path": "ecam_lab_programming",
    "cam_path": "cam_lab_programming",
    "ip": "172.20.32.26",
    "user": "admin",
    "pass": "MDZFQM",
    "channel_101": "rtsp://admin:MDZFQM@172.20.32.26:554/Streaming/Channels/101",
    "channel_102": "rtsp://admin:MDZFQM@172.20.32.26:554/Streaming/Channels/102",
    "hls_source": "https://fdxuqgcmfo.a.pinggy.link/cam_lab_programming",
    "codec_info": "H.265 (HEVC)",
    "is_primary": False,
}

TARGET_LOBBY_LT5 = {
    "id": 42,
    "key": "lobby_lt5",
    "name": "Lobby Lantai 5",
    "location_group": "Lantai 5 Gedung PLT",
    "ecam_path": "ecam_lobby_lt_5",
    "cam_path": "cam_lobby_lt_5",
    "ip": "172.20.32.60",
    "user": "admin",
    "pass": "BVPDCJ",
    "channel_101": "rtsp://admin:BVPDCJ@172.20.32.60:554/Streaming/Channels/101",
    "channel_102": "rtsp://admin:BVPDCJ@172.20.32.60:554/Streaming/Channels/102",
    "hls_source": "https://fdxuqgcmfo.a.pinggy.link/cam_lobby_lt_5",
    "codec_info": "H.265 (HEVC)",
    "is_primary": False,
}

PATROL_LABS = [TARGET_LAB_RISET, TARGET_LAB_OS, TARGET_LAB_PROGRAMMING, TARGET_LOBBY_LT5]

# Konfigurasi Endpoint Telemetri Web
DEFAULT_TELEMETRY_URL = os.environ.get("WEB_TELEMETRY_URL", "http://localhost:3000/api/ai/live-feed")

# Worker Queue untuk pengiriman telemetri non-blocking ke Web API
telemetry_queue = queue.Queue(maxsize=1)

def start_telemetry_worker(url: str):
    """Worker background untuk mengirim telemetri koordinat bounding box ke Web API."""
    def _worker():
        import requests
        session = requests.Session()
        while True:
            try:
                payload = telemetry_queue.get()
                if payload is None:
                    break
                session.post(url, json=payload, timeout=1.0)
            except Exception:
                pass
            finally:
                time.sleep(0.12)
    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t

STUDENT_CACHE: Dict[str, Dict] = {}
STUDENT_COOLDOWNS: Dict[str, float] = {}  # nama_mahasiswa -> timestamp_terakhir_lapor

def sync_students_with_supabase():
    """Memuat data 38 mahasiswa terdaftar (Nama, NIM, Jurusan, Email) dari Supabase ke memory cache."""
    global STUDENT_CACHE
    supa_url = os.environ.get("SUPABASE_URL", "https://gzbiqnzwtbynnhpfsqso.supabase.co")
    supa_key = os.environ.get("SUPABASE_KEY", "sb_publishable_eMibXfZffVVJtqfEh0hLJw_xeJ5EWKz")
    if not supa_url or not supa_key:
        return
    try:
        endpoint = f"{supa_url.rstrip('/')}/rest/v1/mahasiswa?select=nama_lengkap,nim,jurusan,email"
        headers = {
            "apikey": supa_key,
            "Authorization": f"Bearer {supa_key}"
        }
        resp = requests.get(endpoint, headers=headers, timeout=4.0)
        if resp.status_code == 200:
            students = resp.json()
            for s in students:
                name = s.get("nama_lengkap", "").strip()
                if name:
                    STUDENT_CACHE[name.lower()] = {
                        "nama_lengkap": name,
                        "nim": s.get("nim", "").strip(),
                        "jurusan": s.get("jurusan", "").strip(),
                        "email": s.get("email", "").strip()
                    }
            print(f" * Berhasil memuat basis data {len(STUDENT_CACHE)} mahasiswa terdaftar dari Supabase.")
    except Exception as e:
        print(f" * [Info] Melewati sinkronisasi data mahasiswa Supabase ({e}).")

def sync_recent_violations_cooldown():
    """Menginisialisasi cooldown 1 jam dari riwayat pelanggaran terakhir di Supabase."""
    global STUDENT_COOLDOWNS
    supa_url = os.environ.get("SUPABASE_URL", "https://gzbiqnzwtbynnhpfsqso.supabase.co")
    supa_key = os.environ.get("SUPABASE_KEY", "sb_publishable_eMibXfZffVVJtqfEh0hLJw_xeJ5EWKz")
    if not supa_url or not supa_key:
        return
    try:
        one_hour_ago = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S+00:00")
        endpoint = f"{supa_url.rstrip('/')}/rest/v1/violations?created_at=gte.{one_hour_ago}&status=in.(pending,verified,verified_sp1,verified_sp2,verified_staff)&select=student_name,created_at"
        headers = {
            "apikey": supa_key,
            "Authorization": f"Bearer {supa_key}"
        }
        resp = requests.get(endpoint, headers=headers, timeout=4.0)
        if resp.status_code == 200:
            viols = resp.json()
            now_t = time.time()
            for v in viols:
                name = v.get("student_name")
                if name and name not in ["Mahasiswa Tidak Terdaftar", "Tidak Dikenal"]:
                    try:
                        c_t = datetime.datetime.fromisoformat(v["created_at"].replace("Z", "+00:00")).timestamp()
                        if c_t > STUDENT_COOLDOWNS.get(name, 0):
                            STUDENT_COOLDOWNS[name] = c_t
                    except Exception:
                        STUDENT_COOLDOWNS[name] = now_t
            if STUDENT_COOLDOWNS:
                print(f" * Cooldown 1 jam aktif dari Supabase untuk: {', '.join(STUDENT_COOLDOWNS.keys())}")
    except Exception as e:
        print(f" * [Info] Melewati inisialisasi cooldown Supabase ({e}).")

def sync_labs_with_supabase(labs: List[Dict]) -> List[Dict]:
    """
    Sinkronisasi konfigurasi 4 kamera (Lab Riset Lt 5, Lab OS, Lab Programming, Lobby Lt 5)
    langsung dari tabel streaming Supabase.
    """
    import requests
    supa_url = os.environ.get("NEXT_PUBLIC_SUPABASE_URL", os.environ.get("SUPABASE_URL", "https://gzbiqnzwtbynnhpfsqso.supabase.co"))
    supa_key = os.environ.get("NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY", os.environ.get("SUPABASE_KEY", "sb_publishable_eMibXfZffVVJtqfEh0hLJw_xeJ5EWKz"))
    
    if not supa_url or not supa_key:
        return labs

    try:
        endpoint = f"{supa_url.rstrip('/')}/rest/v1/streaming?select=*"
        headers = {
            "apikey": supa_key,
            "Authorization": f"Bearer {supa_key}"
        }
        resp = requests.get(endpoint, headers=headers, timeout=3.5)
        if resp.status_code == 200:
            db_cameras = resp.json()
            synced_count = 0
            for cam in labs:
                cam_key = cam.get("key", "").lower()
                target_id = cam.get("id")

                for db_cam in db_cameras:
                    db_id = db_cam.get("id")
                    db_label = (db_cam.get("label") or "").lower()

                    is_match = False
                    if target_id is not None and db_id == target_id:
                        is_match = True
                    elif "riset" in cam_key and ("riset" in db_label and ("5" in db_label or "lt 5" in db_label or "lantai 5" in db_label)):
                        is_match = True
                    elif "os" in cam_key and ("lab os" in db_label or db_label.endswith(" os") or db_label == "os"):
                        is_match = True
                    elif "programming" in cam_key and "programming" in db_label:
                        is_match = True
                    elif "lobby" in cam_key and ("lobby" in db_label and ("5" in db_label or "lt 5" in db_label or "lantai 5" in db_label)):
                        is_match = True

                    if is_match:
                        if db_cam.get("cam_path"):
                            cam["cam_path"] = db_cam["cam_path"].strip()
                        if db_cam.get("ecam_path"):
                            cam["ecam_path"] = db_cam["ecam_path"].strip()
                        if db_cam.get("cam_source"):
                            cam["channel_101"] = db_cam["cam_source"].strip()
                        if db_cam.get("ecam_source"):
                            cam["channel_102"] = db_cam["ecam_source"].strip()
                        if db_cam.get("ip_address"):
                            cam["ip"] = db_cam["ip_address"].strip()
                        if db_cam.get("username_rtsp"):
                            cam["user"] = db_cam["username_rtsp"].strip()
                        if db_cam.get("password_rtsp"):
                            cam["pass"] = db_cam["password_rtsp"].strip()
                        if db_cam.get("location_group"):
                            cam["location_group"] = db_cam["location_group"].strip()
                        if db_cam.get("hls_source"):
                            raw_hls = db_cam["hls_source"].strip()
                            cam["hls_source"] = raw_hls.replace("ecam_", "cam_") if "ecam_" in raw_hls else raw_hls
                        synced_count += 1
                        break
            print(f" * Berhasil sinkronisasi {synced_count}/{len(labs)} kamera target dari Supabase.")
    except Exception as e:
        print(f" * [Info] Melewati sinkronisasi Supabase ({e}), menggunakan konfigurasi default.")
    return labs

def resolve_camera_urls(cam: Dict, stream_mode: str = "pinggy", pinggy_base: Optional[str] = None, channel: str = "101", custom_stream_url: Optional[str] = None) -> List[str]:
    """
    Menyusun daftar prioritas URL stream untuk kamera yang bersangkutan:
    1. Custom stream URL (--stream-url jika diberikan pengguna lewat argumen CLI untuk kamera primer)
    2. HLS Stream Port 8888 via Pinggy Pro fdxuqgcmfo.a.pinggy.link
    3. HLS Source dari database Supabase / Pinggy Base URL
    4. Fallback ke direct LAN RTSP Channel 101/102
    """
    candidates = []

    # 1. Custom URL langsung dari argumen CLI (hanya jika target adalah kamera primer atau spesifik)
    if custom_stream_url and cam.get("is_primary", False):
        candidates.append(custom_stream_url.strip())

    # Path mainstream Cam (default channel 101)
    path = cam.get("cam_path") if channel == "101" else cam.get("ecam_path", cam.get("cam_path"))
    if not path:
        path = f"cam_{cam.get('key', 'camera')}"

    alt_path = path.replace("_lt_5", "_lt5") if "_lt_5" in path else path.replace("_lt5", "_lt_5")

    # 2. Prioritas Utama: HLS Stream Port 8888 via Pinggy Pro
    base_pinggy = (pinggy_base or DEFAULT_PINGGY_STREAM_BASE).rstrip("/")
    candidates.append(f"{base_pinggy}/{path}/index.m3u8")
    candidates.append(f"{base_pinggy}/{alt_path}/index.m3u8")
    candidates.append(f"{base_pinggy}/{path}")

    # 3. Kandidat dari field hls_source Supabase
    if cam.get("hls_source") and stream_mode == "pinggy":
        h_url = cam["hls_source"].strip()
        if channel == "101" and "ecam_" in h_url:
            h_url_cam = h_url.replace("ecam_", "cam_")
            candidates.append(f"{h_url_cam}/index.m3u8")
            candidates.append(f"{h_url_cam}/")
            candidates.append(h_url_cam)
        candidates.append(f"{h_url}/index.m3u8")
        candidates.append(f"{h_url}/")
        candidates.append(h_url)

    # 4. Fallback ke RTSP Lokal Kampus
    local_rtsp = cam.get("channel_101") if channel == "101" else cam.get("channel_102")
    if local_rtsp:
        candidates.append(local_rtsp)

    # Hilangkan duplikat dan jaga urutan
    seen = set()
    result = []
    for u in candidates:
        if u and u not in seen:
            seen.add(u)
            result.append(u)
    return result

# ------------------------------------------------------------------------------
# 2. MODEL PIPELINE WRAPPERS & TRACKER PERSISTENCE
# ------------------------------------------------------------------------------
class TrackedPerson:
    def __init__(self, box: Tuple[int, int, int, int], conf: float, max_lost: int = 25):
        self.box = list(box)  # [x, y, w, h]
        self.conf = conf
        self.max_lost = max_lost
        self.lost_frames = 0
        self.is_alive = True
        self.track_id = int(time.time() * 1000) % 100000

    def update(self, new_box: Tuple[int, int, int, int], new_conf: float):
        self.box = list(new_box)
        self.conf = new_conf
        self.lost_frames = 0
        self.is_alive = True

    def mark_lost(self):
        self.lost_frames += 1
        if self.lost_frames > self.max_lost:
            self.is_alive = False


def create_onnx_session(model_path: str, opts: Optional[ort.SessionOptions] = None) -> ort.InferenceSession:
    """Membuat ONNX InferenceSession dengan fallback aman: mencoba CUDA GPU dulu, jika gagal fallback ke CPU."""
    if opts is None:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    available = ort.get_available_providers()
    if 'CUDAExecutionProvider' in available:
        try:
            return ort.InferenceSession(model_path, opts, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        except Exception:
            pass
    return ort.InferenceSession(model_path, opts, providers=['CPUExecutionProvider'])

class PersonDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.20, max_lost: int = 25):
        self.conf_threshold = conf_threshold
        self.max_lost = max_lost
        self.is_onnx = model_path.endswith(".onnx")
        self.tracked_persons: List[TrackedPerson] = []

        if self.is_onnx:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = create_onnx_session(model_path, opts)
            self.input_name = self.session.get_inputs()[0].name
        else:
            self.model = YOLO(model_path)

    def reset_tracks(self):
        self.tracked_persons.clear()

    def detect(self, frame: np.ndarray) -> Tuple[bool, List[Tuple[int, int, int, int, float]]]:
        orig_h, orig_w = frame.shape[:2]
        raw_detections = []

        if self.is_onnx:
            resized = cv2.resize(frame, (640, 640))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            inp = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
            inp = np.expand_dims(inp, axis=0)
            outputs = self.session.run(None, {self.input_name: inp})[0]
            preds = np.transpose(np.squeeze(outputs))

            boxes, confidences = [], []
            for row in preds:
                cx, cy, bw, bh = row[:4]
                classes_scores = row[4:]
                person_score = float(classes_scores[0]) if len(classes_scores) > 0 else 0.0

                if person_score >= self.conf_threshold:
                    x = int((cx - bw / 2.0) * (orig_w / 640.0))
                    y = int((cy - bh / 2.0) * (orig_h / 640.0))
                    w = int(bw * (orig_w / 640.0))
                    h = int(bh * (orig_h / 640.0))
                    boxes.append([max(0, x), max(0, y), max(1, w), max(1, h)])
                    confidences.append(person_score)

            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, 0.45)
            if len(indices) > 0:
                for idx in np.array(indices).flatten():
                    bx, by, bw, bh = boxes[idx]
                    raw_detections.append(((bx, by, bw, bh), confidences[idx]))
        else:
            results = self.model.predict(frame, conf=self.conf_threshold, classes=[0], imgsz=640, verbose=False)
            for r in results:
                for box in r.boxes:
                    bx, by, bw, bh = map(int, box.xywh[0].cpu().numpy())
                    top_x = max(0, bx - bw // 2)
                    top_y = max(0, by - bh // 2)
                    conf = float(box.conf[0].cpu().numpy())
                    raw_detections.append(((top_x, top_y, bw, bh), conf))

        # Pencocokan Tracker (Centroid & IoU Matching)
        matched_tracks = set()
        matched_detections = set()

        for d_idx, (d_box, d_conf) in enumerate(raw_detections):
            dx, dy, dw, dh = d_box
            dcx, dcy = dx + dw / 2.0, dy + dh / 2.0
            best_match_idx = -1
            best_dist = float('inf')

            for t_idx, track in enumerate(self.tracked_persons):
                if t_idx in matched_tracks:
                    continue
                tx, ty, tw, th = track.box
                tcx, tcy = tx + tw / 2.0, ty + th / 2.0
                dist = ((dcx - tcx)**2 + (dcy - tcy)**2)**0.5
                max_allow_dist = max(dw, tw, dh, th) * 0.95

                if dist < max_allow_dist and dist < best_dist:
                    best_dist = dist
                    best_match_idx = t_idx

            if best_match_idx >= 0:
                self.tracked_persons[best_match_idx].update(d_box, d_conf)
                matched_tracks.add(best_match_idx)
                matched_detections.add(d_idx)

        # Tambahkan deteksi baru yang belum cocok
        for d_idx, (d_box, d_conf) in enumerate(raw_detections):
            if d_idx not in matched_detections:
                self.tracked_persons.append(TrackedPerson(d_box, d_conf, max_lost=self.max_lost))

        # Update track yang hilang pada frame ini
        for t_idx, track in enumerate(self.tracked_persons):
            if t_idx not in matched_tracks:
                track.mark_lost()

        # Bersihkan track kedaluwarsa
        self.tracked_persons = [t for t in self.tracked_persons if t.is_alive]

        person_boxes = [
            (t.box[0], t.box[1], t.box[2], t.box[3], round(float(t.conf), 2))
            for t in self.tracked_persons
        ]
        return (len(person_boxes) > 0), person_boxes


class SmokingDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.25, min_size: int = 8):
        self.conf_threshold = conf_threshold
        self.min_size = min_size
        self.is_onnx = model_path.endswith(".onnx")
        
        if self.is_onnx:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = create_onnx_session(model_path, opts)
            self.input_name = self.session.get_inputs()[0].name
        else:
            self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray, person_boxes: List[Tuple[int, int, int, int, float]]) -> List[Dict]:
        h, w = frame.shape[:2]
        smoking_events = []

        if not person_boxes:
            return smoking_events

        for p in person_boxes:
            px, py, pw, ph, pconf = p
            pad_x = int(pw * 0.12)
            pad_y = int(ph * 0.12)
            crop_x1 = max(0, px - pad_x)
            crop_y1 = max(0, py - pad_y)
            crop_x2 = min(w, px + pw + pad_x)
            crop_y2 = min(h, py + ph + pad_y)

            crop_w = crop_x2 - crop_x1
            crop_h = crop_y2 - crop_y1
            if crop_w < 30 or crop_h < 30:
                continue

            person_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

            if self.is_onnx:
                resized = cv2.resize(person_crop, (640, 640))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                inp = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
                inp = np.expand_dims(inp, axis=0)
                outputs = self.session.run(None, {self.input_name: inp})[0]
                preds = np.transpose(np.squeeze(outputs))

                boxes, confidences = [], []
                for row in preds:
                    cx, cy, bw, bh = row[:4]
                    classes_scores = row[4:]
                    score = float(np.max(classes_scores)) if len(classes_scores) > 0 else 0.0

                    if score >= self.conf_threshold:
                        x = int((cx - bw / 2.0) * (crop_w / 640.0))
                        y = int((cy - bh / 2.0) * (crop_h / 640.0))
                        box_w = int(bw * (crop_w / 640.0))
                        box_h = int(bh * (crop_h / 640.0))
                        if box_w >= self.min_size and box_h >= self.min_size:
                            boxes.append([x, y, box_w, box_h])
                            confidences.append(score)

                indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, 0.45)
                if len(indices) > 0:
                    for idx in np.array(indices).flatten():
                        bx, by, bw, bh = boxes[idx]
                        orig_sx = crop_x1 + bx
                        orig_sy = crop_y1 + by
                        smoking_events.append({
                            "is_smoking": True,
                            "box": (orig_sx, orig_sy, bw, bh),
                            "conf": confidences[idx],
                            "person_box": p
                        })
            else:
                results = self.model.predict(person_crop, conf=self.conf_threshold, imgsz=640, verbose=False)
                for r in results:
                    for box in r.boxes:
                        bx, by, bw, bh = map(int, box.xywh[0].cpu().numpy())
                        conf = float(box.conf[0].cpu().numpy())
                        if bw >= self.min_size and bh >= self.min_size:
                            orig_sx = crop_x1 + max(0, bx - bw // 2)
                            orig_sy = crop_y1 + max(0, by - bh // 2)
                            smoking_events.append({
                                "is_smoking": True,
                                "box": (orig_sx, orig_sy, bw, bh),
                                "conf": conf,
                                "person_box": p
                            })
        return smoking_events


class FaceRecognitionPipeline:
    def __init__(self, yunet_path: str, facenet_path: str, database_npz_path: str, threshold: float = 0.85):
        self.threshold = threshold
        self.yunet = cv2.FaceDetectorYN.create(
            yunet_path, "", (320, 320),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000
        )
        
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.facenet = create_onnx_session(facenet_path, opts)
        self.fn_input = self.facenet.get_inputs()[0].name
        
        self.db_names = []
        self.db_embeds = np.empty((0, 512), dtype=np.float32)
        self.load_database(database_npz_path)

    def load_database(self, db_path: str):
        if not os.path.exists(db_path):
            print(f" [PERINGATAN] Database wajah {db_path} tidak ditemukan.")
            return
        try:
            data = np.load(db_path, allow_pickle=True)
            if "x" in data and "y" in data:
                self.db_embeds = data["x"]
                self.db_names = list(data["y"])
            elif "embeddings" in data and "names" in data:
                self.db_embeds = data["embeddings"]
                self.db_names = list(data["names"])
            else:
                self.db_names = list(data.get("names", []))
                self.db_embeds = data.get("embeddings", [])
            unique_students = len(set(self.db_names)) if self.db_names else 0
            print(f" * Basis data wajah berhasil dimuat: {len(self.db_names)} profil wajah ({unique_students} mahasiswa terdaftar).")
        except Exception as e:
            print(f" [PERINGATAN] Gagal memuat database wajah {db_path}: {e}")

    def detect_faces(self, frame: np.ndarray, person_boxes: List[Tuple[int, int, int, int, float]]) -> List[Tuple]:
        h, w = frame.shape[:2]
        faces_found = []
        if not person_boxes:
            return faces_found

        for parent_person in person_boxes:
            px, py, pw, ph, *pconf = parent_person
            upper_h = int(ph * 0.65)
            x1 = max(0, px)
            y1 = max(0, py)
            x2 = min(w, px + pw)
            y2 = min(h, py + upper_h)

            crop_w = x2 - x1
            crop_h = y2 - y1
            if crop_w < 20 or crop_h < 20:
                continue

            person_crop = frame[y1:y2, x1:x2]
            self.yunet.setInputSize((crop_w, crop_h))
            _, detections = self.yunet.detect(person_crop)

            if detections is not None:
                for face in detections:
                    fx, fy, fw, fh = map(int, face[:4])
                    conf = face[-1]
                    if fw < 15 or fh < 15 or conf < 0.60:
                        continue

                    glob_fx = x1 + fx
                    glob_fy = y1 + fy
                    glob_fw = fw
                    glob_fh = fh

                    # Ekstraksi FaceNet embedding
                    pad = int(max(fw, fh) * 0.15)
                    fx1 = max(0, glob_fx - pad)
                    fy1 = max(0, glob_fy - pad)
                    fx2 = min(w, glob_fx + glob_fw + pad)
                    fy2 = min(h, glob_fy + glob_fh + pad)

                    face_crop = frame[fy1:fy2, fx1:fx2]
                    if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
                        continue

                    aligned = cv2.resize(face_crop, (160, 160))
                    aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32)
                    inp = (aligned_rgb - 127.5) / 127.5
                    inp = np.expand_dims(inp, axis=0)

                    emb = self.facenet.run(None, {self.fn_input: inp})[0][0]
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        emb = emb / norm

                    # Matching terhadap database mahasiswa
                    best_name = "Tidak Dikenal"
                    best_dist = float('inf')
                    similarity = 0.0

                    if len(self.db_embeds) > 0:
                        dists = np.linalg.norm(self.db_embeds - emb, axis=1)
                        min_idx = np.argmin(dists)
                        min_dist = float(dists[min_idx])
                        best_dist = min_dist

                        # Cosine similarity matching
                        cos_sim = np.dot(self.db_embeds[min_idx], emb)
                        similarity = round(float(cos_sim) * 100.0, 1)

                        if min_dist <= self.threshold:
                            best_name = str(self.db_names[min_idx])

                    faces_found.append((
                        (glob_fx, glob_fy, glob_fw, glob_fh),
                        best_name,
                        similarity,
                        best_dist,
                        parent_person
                    ))
        return faces_found

# ------------------------------------------------------------------------------
# 3. MULTI-CAMERA STREAM WORKERS (PINGGY MAINSTREAM CAM LOADERS)
# ------------------------------------------------------------------------------
class SingleCameraCapture:
    """Thread independen per kamera untuk mengambil video feed secara non-blocking."""
    def __init__(self, cam_info: Dict, channel: str = "101", mock: bool = False, stream_mode: str = "pinggy", pinggy_base: Optional[str] = None, custom_url: Optional[str] = None):
        self.cam = cam_info
        self.name = cam_info.get("name", "Camera")
        self.key = cam_info.get("key", "cam")
        self.channel = channel
        self.is_mock = mock
        self.stream_mode = stream_mode
        self.pinggy_base = pinggy_base
        self.custom_url = custom_url
        self.cap: Optional[cv2.VideoCapture] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.is_connected = False
        self.active_stream_url: Optional[str] = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def set_pinggy_base(self, new_base: str):
        with self.lock:
            self.pinggy_base = new_base.rstrip("/")
            if self.cap:
                try: self.cap.release()
                except Exception: pass
            self.cap = None
            self.is_connected = False
            self.latest_frame = None

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray], Dict]:
        with self.lock:
            return self.is_connected, (self.latest_frame.copy() if self.latest_frame is not None else None), self.cam

    def stop(self):
        self.running = False
        if self.cap:
            try: self.cap.release()
            except Exception: pass

    def _run(self):
        while self.running:
            if self.is_mock:
                synth = np.zeros((720, 1280, 3), dtype=np.uint8)
                synth[:] = (26, 24, 22)
                now_str = datetime.datetime.now().strftime("%H:%M:%S")
                cv2.putText(synth, f"[MOCK SIMULASI - {self.name}]", (80, 260),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 220, 255), 2)
                cv2.putText(synth, f"Waktu: {now_str} WIB | Status: Standby Simulasi", (80, 310),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
                with self.lock:
                    self.latest_frame = synth
                    self.is_connected = True
                time.sleep(0.04)
                continue

            urls_to_try = resolve_camera_urls(
                self.cam,
                stream_mode=self.stream_mode,
                pinggy_base=self.pinggy_base,
                channel=self.channel,
                custom_stream_url=self.custom_url
            )

            if self.cap is None or not self.cap.isOpened():
                opened = False
                for try_url in urls_to_try:
                    try:
                        cap = cv2.VideoCapture(try_url, cv2.CAP_FFMPEG)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        if cap.isOpened():
                            ret, f = cap.read()
                            if ret and f is not None:
                                self.cap = cap
                                self.active_stream_url = try_url
                                with self.lock:
                                    self.latest_frame = f
                                    self.is_connected = True
                                opened = True
                                print(f"\n[STREAM] {self.name} -> TERHUBUNG\n         URL: {try_url}\n", flush=True)
                                break
                            else:
                                cap.release()
                    except Exception:
                        pass

                if not opened:
                    with self.lock:
                        self.is_connected = False
                    time.sleep(1.0)
                    continue

            try:
                ret = self.cap.grab()
                if not ret:
                    with self.lock: self.is_connected = False
                    if self.cap: self.cap.release()
                    self.cap = None
                    time.sleep(0.5)
                    continue
                ret, frame = self.cap.retrieve()
                if ret and frame is not None:
                    with self.lock:
                        self.latest_frame = frame
                        self.is_connected = True
            except Exception:
                with self.lock: self.is_connected = False
                time.sleep(0.5)


class MultiCameraStreamWorker:
    """Pengelola thread video feed untuk ke-4 kamera target secara serentak."""
    def __init__(self, labs: List[Dict], channel: str = "101", mock: bool = False, stream_mode: str = "pinggy", pinggy_base: Optional[str] = None, custom_stream_url: Optional[str] = None):
        self.labs = labs
        self.workers: List[SingleCameraCapture] = []
        for idx, cam in enumerate(labs):
            c_url = custom_stream_url if (idx == 0 and custom_stream_url) else None
            self.workers.append(SingleCameraCapture(
                cam_info=cam,
                channel=channel,
                mock=mock,
                stream_mode=stream_mode,
                pinggy_base=pinggy_base,
                custom_url=c_url
            ))

    def set_pinggy_base(self, new_base: str):
        for w in self.workers:
            w.set_pinggy_base(new_base)

    def get_frame(self, idx: int) -> Tuple[bool, Optional[np.ndarray], Dict]:
        if 0 <= idx < len(self.workers):
            return self.workers[idx].get_frame()
        return False, None, {}

    def get_all_frames(self) -> List[Tuple[bool, Optional[np.ndarray], Dict]]:
        return [w.get_frame() for w in self.workers]

    def stop(self):
        for w in self.workers:
            w.stop()


# ------------------------------------------------------------------------------
# 4. PER-CAMERA AI STATE & 2x2 MOSAIC VISUALIZER
# ------------------------------------------------------------------------------
class CameraAIState:
    """Melacak status deteksi manusia, deteksi rokok, dan timer tidur per kamera."""
    def __init__(self, cam_info: Dict):
        self.cam = cam_info
        self.name = cam_info.get("name", "Camera")
        self.key = cam_info.get("key", "cam")

        # Human Gate & 1-Minute Sleep Rule
        self.has_person = False
        self.person_boxes: list = []
        self.no_person_since: Optional[float] = None
        self.sleep_until: float = 0.0  # Jika curr_t < sleep_until -> Model Rokok & Wajah dimatikan (1 menit)

        # Smoking & Face Identification State
        self.smoking_start_time: Optional[float] = None
        self.last_smoking_seen_time: Optional[float] = None
        self.last_known_smoke_event = None
        self.pending_face_ident_start: Optional[float] = None
        self.smoking_duration: float = 0.0
        self.last_incident_capture_time: float = 0.0
        self.is_smoking = False
        self.active_violator_name: Optional[str] = None
        self.active_violator_sim: float = 0.0

        # Visual & Telemetri
        self.annotated_frame: Optional[np.ndarray] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.status_badge: str = "MEMUAT"
        self.status_desc: str = "Menyiapkan stream"
        self.is_connected = False


def build_2x2_mosaic(camera_states: List[CameraAIState]) -> np.ndarray:
    """Menggabungkan feed 4 kamera menjadi 1 layar grid 2x2 (1280x720) untuk monitoring MJPEG."""
    mosaic = np.zeros((720, 1280, 3), dtype=np.uint8)
    quadrants = [
        (0, 360, 0, 640),      # Quadrant 1: Kiri Atas
        (0, 360, 640, 1280),   # Quadrant 2: Kanan Atas
        (360, 720, 0, 640),    # Quadrant 3: Kiri Bawah
        (360, 720, 640, 1280)  # Quadrant 4: Kanan Bawah
    ]

    for idx in range(min(4, len(camera_states))):
        st = camera_states[idx]
        y1, y2, x1, x2 = quadrants[idx]
        f = st.annotated_frame if st.annotated_frame is not None else st.latest_frame
        if f is not None and f.size > 0:
            thumb = cv2.resize(f, (640, 360))
        else:
            thumb = np.zeros((360, 640, 3), dtype=np.uint8)
            thumb[:] = (22, 22, 28)
            cv2.putText(thumb, f"Menghubungkan {st.name}...", (120, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 100, 200), 2)

        # Header OSD Kotak Kamera
        cv2.rectangle(thumb, (0, 0), (640, 34), (12, 16, 22), -1)
        cv2.putText(thumb, f"[{idx+1}] {st.name}", (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Badge Status Warna
        badge_color = (0, 180, 0)
        badge_txt = st.status_badge
        if "SLEEP" in badge_txt:
            badge_color = (0, 140, 255)  # Orange/Amber
        elif "ROKOK" in badge_txt:
            badge_color = (0, 0, 255)    # Merah
        elif "OFFLINE" in badge_txt:
            badge_color = (70, 70, 70)   # Abu-abu gelap
        elif "NIHIL" in badge_txt or "IDLE" in badge_txt:
            badge_color = (180, 140, 0)  # Kuning/Biru

        (bw, bh), _ = cv2.getTextSize(badge_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(thumb, (640 - bw - 20, 5), (632, 29), badge_color, -1)
        cv2.putText(thumb, badge_txt, (640 - bw - 15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        mosaic[y1:y2, x1:x2] = thumb

    # Garis Pembatas Silang 2x2
    cv2.line(mosaic, (640, 0), (640, 720), (50, 60, 75), 2)
    cv2.line(mosaic, (0, 360), (1280, 360), (50, 60, 75), 2)
    return mosaic


# ------------------------------------------------------------------------------
# 5. SHARED STATE & HTTP API SERVER (BACKGROUND DAEMON)
# ------------------------------------------------------------------------------
SHARED_RUNPOD_STATE = {
    "service": "E.S.I.S Cloud Patrol Server (Plan A - 4 Cameras RunPod)",
    "cameras": ["Lab Riset Lantai 5", "Lab OS", "Lab Programming", "Lobby Lantai 5"],
    "fps": 0.0,
    "status": "online",
    "public_url": None,
    "stream_worker": None,
    "latest_frame": None,
    "annotated_frame": None,
    "camera_states": [],
    "telemetry": {}
}

class RunPodHTTPHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Matikan log default stdout http.server

    def _send_json(self, status: int, data: dict):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]

        if path in ["/", "/status", "/api/status"]:
            st = SHARED_RUNPOD_STATE.copy()
            st.pop("latest_frame", None)
            st.pop("annotated_frame", None)
            st.pop("stream_worker", None)
            cam_summaries = []
            for cs in SHARED_RUNPOD_STATE.get("camera_states", []):
                cam_summaries.append({
                    "name": cs.name,
                    "key": cs.key,
                    "is_connected": cs.is_connected,
                    "has_person": cs.has_person,
                    "is_smoking": cs.is_smoking,
                    "status_badge": cs.status_badge,
                    "status_desc": cs.status_desc,
                    "sleep_until": cs.sleep_until
                })
            self._send_json(200, {
                "service": "E.S.I.S Cloud Patrol Server (Plan A - 4 Cameras RunPod)",
                "status": "online",
                "cameras": cam_summaries,
                "fps": SHARED_RUNPOD_STATE.get("fps", 0.0),
                "endpoints": {
                    "stream": "/stream (Grid 2x2 MJPEG)",
                    "health": "/health",
                    "snapshot": "/snapshot",
                    "telemetry": "/telemetry",
                    "simulate": "POST /api/simulate",
                    "pinggy_stream": "POST /api/patrol/pinggy-stream"
                }
            })
            return

        if path in ["/telemetry", "/api/telemetry"]:
            tel = SHARED_RUNPOD_STATE.get("telemetry", {})
            self._send_json(200, tel)
            return

        if path in ["/health", "/api/health"]:
            self._send_json(200, {
                "status": "online",
                "service": "E.S.I.S Plan A RunPod (4 Cameras)",
                "cameras_total": 4,
                "fps": SHARED_RUNPOD_STATE.get("fps", 0.0),
                "timestamp": datetime.datetime.now().isoformat()
            })
            return

        if path == "/snapshot":
            frame = SHARED_RUNPOD_STATE.get("annotated_frame")
            if frame is None:
                frame = SHARED_RUNPOD_STATE.get("latest_frame")
            if frame is not None:
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(buf.tobytes())
                    return
            self._send_json(503, {"error": "Snapshot grid 2x2 belum tersedia."})
            return

        if path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                while True:
                    frame = SHARED_RUNPOD_STATE.get("annotated_frame")
                    if frame is not None:
                        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                        if ok:
                            data = buf.tobytes()
                            header = f"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {len(data)}\r\n\r\n".encode("utf-8")
                            self.wfile.write(header + data + b"\r\n")
                    time.sleep(0.06)
            except (ConnectionResetError, BrokenPipeError):
                pass
            return

        self._send_json(404, {"error": f"Endpoint GET {path} tidak ditemukan."})

    def do_POST(self):
        path = self.path.split("?")[0]
        
        if path in ["/api/simulate", "/simulate"]:
            cam_name = "Lab Riset Lantai 5"
            now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulasi_runpod_{now_str}.jpg"
            capture_path = os.path.join(CAPTURES_DIR, filename)

            frame = SHARED_RUNPOD_STATE.get("latest_frame")
            if frame is not None:
                cv2.imwrite(capture_path, frame)
            else:
                capture_path = "simulated"

            res = capture_and_report_incident(
                frame_or_path=capture_path,
                location=cam_name,
                location_group="Lantai 5 Gedung PLT",
                student_name="Agung Susilo Adi Pratama",
                student_id="11230910000004",
                confidence_smoking=0.95,
                face_similarity=0.955,
                notes="Simulasi pelanggaran tombol API via RunPod Cloud.",
                filename=filename,
            )
            self._send_json(201 if res.get("success") else 500, res)
            return

        if path in ["/api/patrol/pinggy-stream", "/api/pinggy-stream"]:
            worker = SHARED_RUNPOD_STATE.get("stream_worker")
            if worker:
                content_len = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
                try:
                    payload = json.loads(body)
                    new_base = payload.get("base_url") or payload.get("url")
                    if new_base:
                        worker.set_pinggy_base(str(new_base))
                        self._send_json(200, {
                            "success": True,
                            "message": f"Pinggy stream base URL berhasil diperbarui ke: {new_base}",
                            "pinggy_base": new_base
                        })
                        return
                    self._send_json(400, {"error": "Parameter 'base_url' wajib disertakan."})
                except Exception as e:
                    self._send_json(500, {"error": str(e)})
            else:
                self._send_json(503, {"error": "MultiCameraStreamWorker belum siap."})
            return

        self._send_json(404, {"error": f"Endpoint POST {path} tidak ditemukan."})

# ------------------------------------------------------------------------------
# 6. MAIN PATROL LOOP (PLAN A RUNPOD CLOUD GPU - 4 KAMERA SERENTAK)
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="E.S.I.S Plan A - RunPod Cloud GPU Terminal Patrol (4 Kamera Serentak)")
    parser.add_argument("--stream-url", type=str, default="", help="Link streaming kamera primer jika ingin kustom")
    parser.add_argument("--channel", type=str, default="101", choices=["101", "102"], help="Channel stream: 101 (Mainstream Cam HD - Rekomendasi) atau 102 (Substream)")
    parser.add_argument("--mock", action="store_true", help="Gunakan mode mock simulasi")
    parser.add_argument("--stream-mode", type=str, default="pinggy", choices=["pinggy", "rtsp", "mock"], help="Sumber streaming: pinggy / publik (Default), rtsp (LAN langsung), atau mock")
    parser.add_argument("--pinggy-stream-base", type=str, default=DEFAULT_PINGGY_STREAM_BASE, help="Base URL Streaming MediaMTX (default: https://fdxuqgcmfo.a.pinggy.link)")
    parser.add_argument("--no-supabase-sync", action="store_true", help="Lewati sinkronisasi kamera dari database Supabase")
    parser.add_argument("--person-model", type=str, default="yolov8s", choices=["yolov8s", "yolov8n"], help="Model manusia: yolov8s (Akurat & Stabil) atau yolov8n (Ringan)")
    parser.add_argument("--cigar-model", type=str, default="v3_onnx", help="Model rokok: v3_onnx (YOLO11s ONNX - Default Cloud GPU), v2_onnx, v3, v2")
    parser.add_argument("--conf-smoke", type=float, default=0.25, help="Ambang batas confidence deteksi rokok (default: 0.25)")
    parser.add_argument("--conf-person", type=float, default=0.20, help="Ambang batas confidence deteksi manusia (default: 0.20)")
    parser.add_argument("--smoke-sec", type=float, default=3.0, help="Durasi rokok kontinu sebelum insiden terkonfirmasi (default: 3.0 detik)")
    parser.add_argument("--idle-sleep-sec", type=float, default=10.0, help="Durasi tanpa manusia sebelum model rokok & wajah dimatikan (default: 10.0 detik)")
    parser.add_argument("--sleep-duration-sec", type=float, default=60.0, help="Durasi tidur model rokok & wajah saat tidak ada manusia (default: 60.0 detik)")
    parser.add_argument("--headless", action="store_true", default=True, help="Jalankan di terminal tanpa GUI (Default True)")
    parser.add_argument("--gui", action="store_true", help="Buka jendela OpenCV display (jika ada monitor X11/desktop)")
    parser.add_argument("--api-port", type=int, default=8001, help="Port HTTP API latar belakang (default: 8001)")
    parser.add_argument("--pinggy", action="store_true", default=True, help="Nyalakan SSH Reverse Tunnel Pinggy untuk akses API publik (default: True)")
    parser.add_argument("--no-pinggy", action="store_true", help="Matikan Pinggy Tunnel")
    parser.add_argument("--pinggy-token", type=str, default=None, help="Token autentikasi Pinggy jika ada")
    parser.add_argument("--qr", action="store_true", help="Tampilkan QR code Pinggy di terminal")
    parser.add_argument("--web-telemetry", type=str, default=DEFAULT_TELEMETRY_URL, help="URL API Next.js untuk telemetri real-time")
    args = parser.parse_args()

    is_headless = not args.gui if args.gui else args.headless

    # Sinkronisasi ke-4 kamera dan data mahasiswa dari Supabase
    labs_list = PATROL_LABS.copy()
    if not args.no_supabase_sync:
        labs_list = sync_labs_with_supabase(labs_list)
        sync_students_with_supabase()
        sync_recent_violations_cooldown()

    # Resolusi model deteksi manusia
    person_model_path = CANDIDATE_PERSON_MODELS.get(args.person_model, PATH_YOLO_PERSON_ONNX)
    if not os.path.exists(person_model_path):
        person_model_path = os.path.join(MODELS_DIR, "yolov8n.onnx")

    # Resolusi model rokok ONNX
    if os.path.exists(args.cigar_model):
        cigar_model_path = os.path.abspath(args.cigar_model)
    else:
        cigar_model_path = CANDIDATE_ROKOK_MODELS.get(args.cigar_model, PATH_YOLO_ROKOK_ONNX)
        if not os.path.exists(cigar_model_path):
            cigar_model_path = PATH_YOLO_ROKOK_ONNX

    print("\n" + "=" * 82)
    print("   E.S.I.S (Electronic Smoking Identification System) - PLAN A (RUNPOD CLOUD GPU)")
    print("   UPGRADE: DETEKSI 4 KAMERA SERENTAK + GATED INFERENCE (HEMAT SUMBER DAYA)")
    print("=" * 82)
    print(f" * Jumlah Kamera Target : {len(labs_list)} Kamera:")
    for idx, c in enumerate(labs_list):
        print(f"     [{idx+1}] {c['name']:<20} (Path: {c.get('cam_path')})")
    print(f" * Mode Stream          : {args.stream_mode.upper()} via {args.pinggy_stream_base}")
    print(f" * Channel Stream       : Channel {args.channel} ({'Mainstream Cam HD' if args.channel == '101' else 'Substream'})")
    print(f" * Aturan Human Gate    : AI Rokok & Wajah HANYA jalan jika manusia terdeteksi")
    print(f" * Aturan Auto-Sleep    : Jika nihil manusia {args.idle_sleep_sec:.0f}s -> AI Rokok/Wajah tidur {args.sleep_duration_sec:.0f}s (1 menit)")
    print(f" * Verifikasi Merokok   : Kontinu minimal {args.smoke_sec:.1f} detik")
    print(f" * Model Manusia        : {os.path.basename(person_model_path)} (Conf >= {args.conf_person:.2f})")
    print(f" * Model Rokok          : {os.path.basename(cigar_model_path)} (Conf >= {args.conf_smoke:.2f})")
    print(f" * Model Wajah          : YuNet + FaceNet ONNX (Basis Data 38 Mahasiswa)")
    print(f" * Mode Eksekusi        : {'Terminal Headless (RunPod Mode)' if is_headless else 'Display Window GUI'}")
    print(f" * Web Telemetri API    : {args.web_telemetry}")
    print("=" * 82 + "\n", flush=True)

    # Inisialisasi Model AI (Shared instance untuk ke-4 kamera)
    print(f"[1/4] Menginisialisasi Detektor Manusia ({os.path.basename(person_model_path)})...", flush=True)
    person_detector = PersonDetector(person_model_path, conf_threshold=args.conf_person, max_lost=25)

    print(f"[2/4] Menginisialisasi Detektor Merokok ({os.path.basename(cigar_model_path)})...", flush=True)
    smoking_detector = SmokingDetector(cigar_model_path, conf_threshold=args.conf_smoke, min_size=8)

    print("[3/4] Menginisialisasi Pipeline Wajah (YuNet + FaceNet ONNX)...", flush=True)
    face_pipeline = FaceRecognitionPipeline(PATH_YUNET_ONNX, PATH_FACENET_ONNX, PATH_DATABASE_NPZ, threshold=0.85)

    # Inisialisasi State AI untuk ke-4 Kamera
    camera_states = [CameraAIState(c) for c in labs_list]
    SHARED_RUNPOD_STATE["camera_states"] = camera_states

    # Inisialisasi MultiCameraStreamWorker
    print(f"\n[4/4] Memulai pengambilan video stream serentak untuk {len(labs_list)} kamera...\n", flush=True)
    stream_worker = MultiCameraStreamWorker(
        labs=labs_list,
        channel=args.channel,
        mock=args.mock,
        stream_mode=args.stream_mode,
        pinggy_base=args.pinggy_stream_base,
        custom_stream_url=args.stream_url if args.stream_url else None
    )
    SHARED_RUNPOD_STATE["stream_worker"] = stream_worker

    # Inisialisasi HTTP Server Latar Belakang (Port 8001)
    httpd = None
    try:
        httpd = ThreadingHTTPServer(("0.0.0.0", args.api_port), RunPodHTTPHandler)
        server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        server_thread.start()
        print(f" * API Server Lokal aktif di http://0.0.0.0:{args.api_port}/")
        print(f" * Monitor Grid 2x2 MJPEG : http://localhost:{args.api_port}/stream\n", flush=True)
    except Exception as e:
        print(f" [PERINGATAN] Gagal bind HTTP server di port {args.api_port}: {e}\n", flush=True)

    # Inisialisasi Pinggy Tunnel
    pinggy_tunnel = None
    use_pinggy_api = args.pinggy and not args.no_pinggy
    if use_pinggy_api:
        print(" * Memulai Pinggy SSH Tunnel untuk akses publik API Server...", flush=True)
        pinggy_tunnel = PinggyTunnel(port=args.api_port, token=args.pinggy_token, show_qr=args.qr)
        p_url = pinggy_tunnel.start(wait_for_url=True, timeout=12.0)
        if p_url:
            SHARED_RUNPOD_STATE["public_url"] = p_url
            print(f" >>> URL Publik API Server : {p_url}")
            print(f" >>> Stream MJPEG 2x2      : {p_url}/stream\n", flush=True)

    # Jalankan background telemetry worker
    if args.web_telemetry:
        start_telemetry_worker(args.web_telemetry)

    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0.0
    last_print_time = 0.0
    last_telemetry_send_time = 0.0

    SMOKING_GRACE_PERIOD = 1.8
    SMOKING_COOLDOWN_SEC = 30.0

    window_name = "E.S.I.S Plan A - 4 Cameras Monitor (RunPod Cloud GPU)"
    if not is_headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

    print("\n" + "=" * 82)
    print("   PATROLI 4 KAMERA RUNPOD (PLAN A) BERJALAN AKTIF!")
    print("   Tekan Ctrl+C di terminal untuk menghentikan patroli.")
    print("=" * 82 + "\n")

    try:
        while True:
            curr_t = time.time()
            fps_counter += 1
            if curr_t - fps_start_time >= 1.0:
                current_fps = fps_counter / (curr_t - fps_start_time)
                fps_counter = 0
                fps_start_time = curr_t
                SHARED_RUNPOD_STATE["fps"] = round(current_fps, 1)

            # Proses inferensi ke-4 kamera secara berurutan dalam loop cepat
            for cam_idx, cam_state in enumerate(camera_states):
                cam_info = labs_list[cam_idx]
                is_conn, frame, _ = stream_worker.get_frame(cam_idx)

                cam_state.is_connected = is_conn
                cam_state.latest_frame = frame

                if not is_conn or frame is None:
                    cam_state.status_badge = "OFFLINE"
                    cam_state.status_desc = "Menghubungkan stream..."
                    cam_state.annotated_frame = None
                    continue

                # -------------------------------------------------------------
                # ATURAN GATED INFERENCE:
                # 1. Cek apakah kamera sedang dalam masa tidur (1 menit)
                # -------------------------------------------------------------
                in_sleep_mode = (curr_t < cam_state.sleep_until)

                if in_sleep_mode:
                    remaining_sleep = int(cam_state.sleep_until - curr_t)
                    cam_state.has_person = False
                    cam_state.person_boxes = []
                    cam_state.is_smoking = False
                    cam_state.smoking_duration = 0.0
                    cam_state.smoking_start_time = None
                    cam_state.last_smoking_seen_time = None
                    cam_state.pending_face_ident_start = None
                    cam_state.status_badge = f"SLEEP ({remaining_sleep}s)"
                    cam_state.status_desc = f"Model Rokok & Wajah DIMATIKAN ({remaining_sleep}s tersisa)"

                    # Model Rokok dan Wajah TIDAK dijalankan sama sekali!
                    # Tampilkan banner status tidur pada frame visual
                    disp_frame = frame.copy()
                    (fh, fw) = disp_frame.shape[:2]
                    cv2.rectangle(disp_frame, (20, 20), (min(fw - 20, 480), 80), (15, 20, 30), -1)
                    cv2.putText(disp_frame, "MODE HEMAT DAYA (1 MENIT)", (35, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
                    cv2.putText(disp_frame, f"AI Rokok & Wajah dimatikan: {remaining_sleep}s tersisa", (35, 72),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    cam_state.annotated_frame = disp_frame
                    continue

                # Jika sebelumnya dalam mode tidur dan masa tidur sudah berakhir
                if cam_state.sleep_until > 0 and curr_t >= cam_state.sleep_until:
                    print(f"\n[WAKE UP] [{cam_info['name']}] Periode hemat daya selesai. AI Rokok & Wajah kembali AKTIF.\n", flush=True)
                    cam_state.sleep_until = 0.0

                # -------------------------------------------------------------
                # 2. Jika tidak dalam mode tidur -> Jalankan Detektor Manusia
                # -------------------------------------------------------------
                has_person, person_boxes = person_detector.detect(frame)
                cam_state.has_person = has_person
                cam_state.person_boxes = person_boxes

                disp_frame = frame.copy()
                (fh, fw) = disp_frame.shape[:2]

                if has_person:
                    # Manusia terdeteksi: Reset timer idle
                    cam_state.no_person_since = None
                    cam_state.status_badge = f"MANUSIA ({len(person_boxes)})"
                    cam_state.status_desc = "AI Rokok & Wajah AKTIF"

                    # Gambar bounding box manusia (hijau tipis)
                    for p in person_boxes:
                        px, py, pw, ph, *pconf = p
                        cv2.rectangle(disp_frame, (px, py), (px + pw, py + ph), (0, 220, 0), 2)

                    # ---------------------------------------------------------
                    # 3. LANJUT KE MODEL AI ROKOK & WAJAH (KARENA ADA MANUSIA)
                    # ---------------------------------------------------------
                    faces_identified = face_pipeline.detect_faces(frame, person_boxes)
                    smoking_detections = smoking_detector.detect(frame, person_boxes)
                    smoking_events = [d for d in smoking_detections if d["is_smoking"]]

                    # Logika Verifikasi Merokok Kontinu 3 Detik
                    if smoking_events:
                        cam_state.last_smoking_seen_time = curr_t
                        cam_state.last_known_smoke_event = smoking_events[0]
                        if cam_state.smoking_start_time is None:
                            cam_state.smoking_start_time = curr_t
                        cam_state.smoking_duration = curr_t - cam_state.smoking_start_time
                        cam_state.is_smoking = True
                        cam_state.status_badge = f"ROKOK ({cam_state.smoking_duration:.1f}s)"
                    else:
                        if cam_state.last_smoking_seen_time is not None and (curr_t - cam_state.last_smoking_seen_time) < SMOKING_GRACE_PERIOD:
                            if cam_state.smoking_start_time is not None:
                                cam_state.smoking_duration = curr_t - cam_state.smoking_start_time
                            cam_state.is_smoking = True
                        else:
                            cam_state.smoking_start_time = None
                            cam_state.last_smoking_seen_time = None
                            cam_state.last_known_smoke_event = None
                            cam_state.pending_face_ident_start = None
                            cam_state.smoking_duration = 0.0
                            cam_state.is_smoking = False

                    # Anotasi visual rokok jika terdeteksi
                    if smoking_events:
                        for d in smoking_events:
                            (sx, sy, sw, sh) = d["box"]
                            conf = d["conf"]
                            b_color = (0, 0, 255) if cam_state.smoking_duration >= args.smoke_sec else (0, 140, 255)
                            cv2.rectangle(disp_frame, (sx, sy), (sx + sw, sy + sh), b_color, 3)
                            lbl = f"ROKOK: {int(conf*100)}% ({cam_state.smoking_duration:.1f}s)"
                            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                            cv2.rectangle(disp_frame, (sx, max(0, sy - th - 8)), (sx + tw + 8, max(th + 8, sy)), b_color, -1)
                            cv2.putText(disp_frame, lbl, (sx + 4, max(th + 4, sy - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

                    # Anotasi visual wajah
                    for fbox, fname, fsim, fdist, fparent in faces_identified:
                        fx, fy, fw, fh = fbox
                        f_color = (0, 255, 0) if fname != "Tidak Dikenal" else (180, 180, 180)
                        cv2.rectangle(disp_frame, (fx, fy), (fx + fw, fy + fh), f_color, 2)
                        flbl = f"{fname} ({fsim:.0f}%)" if fname != "Tidak Dikenal" else "Wajah Belum Terdaftar"
                        cv2.putText(disp_frame, flbl, (fx, max(15, fy - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, f_color, 1)

                    cam_state.annotated_frame = disp_frame

                    # ---------------------------------------------------------
                    # 4. EVALUASI VERIFIKASI PELANGGARAN JIKA >= 3 DETIK
                    # ---------------------------------------------------------
                    if cam_state.smoking_duration >= args.smoke_sec and (curr_t - cam_state.last_incident_capture_time) >= SMOKING_COOLDOWN_SEC:
                        violator_name = "Mahasiswa Tidak Terdaftar"
                        violator_sim = 0.0
                        smoker_person = None

                        primary_smoke = smoking_events[0] if smoking_events else cam_state.last_known_smoke_event
                        if primary_smoke is not None:
                            sx, sy, sw, sh = primary_smoke["box"]
                            c_rx = sx + sw / 2.0
                            c_ry = sy + sh / 2.0

                            if primary_smoke.get("person_box") is not None:
                                smoker_person = primary_smoke["person_box"]
                            elif person_boxes:
                                best_p_dist = float('inf')
                                for p in person_boxes:
                                    px, py, pw, ph, *pconf = p
                                    if (px - pw*0.15 <= c_rx <= px + pw*1.15) and (py - ph*0.15 <= c_ry <= py + ph*1.15):
                                        d = ((px + pw/2.0 - c_rx)**2 + (py + ph/2.0 - c_ry)**2)**0.5
                                        if d < best_p_dist:
                                            best_p_dist = d
                                            smoker_person = p
                        else:
                            if person_boxes:
                                smoker_person = person_boxes[0]

                        # Pencocokan Wajah Spasial
                        if smoker_person is not None and faces_identified:
                            spx, spy, spw, sph, *spconf = smoker_person
                            candidate_faces = []
                            for face_item in faces_identified:
                                fbox, fname, fsim, fdist, f_src_p = face_item
                                fx, fy, fw, fh = fbox
                                fc_x = fx + fw / 2.0
                                fc_y = fy + fh / 2.0

                                is_match = (f_src_p is not None and f_src_p == smoker_person) or \
                                           ((spx - spw*0.12 <= fc_x <= spx + spw*1.12) and (spy - 30 <= fc_y <= spy + int(sph*0.85)))
                                if is_match:
                                    d_cigar = ((fc_x - (c_rx if primary_smoke else fc_x))**2 + (fc_y - (c_ry if primary_smoke else fc_y))**2)**0.5
                                    candidate_faces.append((fbox, fname, fsim, d_cigar))

                            if candidate_faces:
                                candidate_faces.sort(key=lambda x: x[3])
                                best_face = candidate_faces[0]
                                if best_face[1] != "Tidak Dikenal":
                                    violator_name = best_face[1]
                                    violator_sim = best_face[2]

                        is_known_student = (violator_name not in ["Mahasiswa Tidak Terdaftar", "Tidak Dikenal"])

                        if cam_state.pending_face_ident_start is None:
                            cam_state.pending_face_ident_start = curr_t
                            if not is_known_student:
                                print(f"\n[BUFFER WAJAH] Merokok terdeteksi >= {args.smoke_sec:.0f}s di [{cam_info['name']}].")
                                print("               Menunggu pengenalan identitas wajah (buffer 3 detik)...\n", flush=True)

                        conf_smoke = float(primary_smoke.get("conf", 0.94)) if primary_smoke else 0.94
                        sim_face = float(violator_sim / 100.0) if violator_sim > 1.0 else float(violator_sim)

                        should_report = False
                        status_desc = ""

                        # KONDISI A: Wajah Terdaftar Teridentifikasi -> Cek Cooldown 1 Jam!
                        if is_known_student:
                            last_viol_t = STUDENT_COOLDOWNS.get(violator_name, 0.0)
                            elapsed = curr_t - last_viol_t
                            if elapsed < 3600.0:
                                sisa_menit = int((3600.0 - elapsed) // 60)
                                sisa_detik = int((3600.0 - elapsed) % 60)
                                print(f"\n[COOLDOWN 1 JAM] Mahasiswa: {violator_name} (Terakhir melanggar: {int(elapsed//60)}m lalu).")
                                print(f"                 Sisa waktu jeda cooldown: {sisa_menit}m {sisa_detik}s. Laporan baru tidak dikirim.\n", flush=True)
                                cam_state.pending_face_ident_start = None
                                cam_state.last_incident_capture_time = curr_t + 10.0
                                should_report = False
                            else:
                                should_report = True
                                status_desc = f"TERVERIFIKASI ({violator_name})"
                        # KONDISI B: Wajah Belum Dikenal setelah 3.0s -> BATALKAN! (Aturan Plan B)
                        elif (curr_t - cam_state.pending_face_ident_start) >= 3.0:
                            print(f"\n[DIABAIKAN] Pelanggaran rokok terdeteksi di [{cam_info['name']}],")
                            print("            tetapi wajah pelaku tidak terdaftar / tidak dikenal setelah buffer 3 detik.")
                            print("            -> Laporan dibatalkan sesuai aturan PLAN B (hanya mahasiswa terdaftar yang dilaporkan).\n", flush=True)
                            cam_state.pending_face_ident_start = None
                            cam_state.last_incident_capture_time = curr_t + 6.0
                            should_report = False

                        if should_report:
                            cam_state.pending_face_ident_start = None
                            cam_state.last_incident_capture_time = curr_t
                            STUDENT_COOLDOWNS[violator_name] = curr_t

                            # Resolusi NIM & Data Mahasiswa dari Cache
                            st_info = STUDENT_CACHE.get(violator_name.lower(), {})
                            resolved_student_id = st_info.get("nim")

                            now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            capture_filename = f"pelanggaran_{cam_info['key']}_{now_str}.jpg"
                            capture_path = os.path.join(CAPTURES_DIR, capture_filename)

                            evidence_frame = frame.copy()
                            eh, ew = evidence_frame.shape[:2]

                            # 1. Bounding box rokok tebal warna merah
                            if primary_smoke is not None:
                                sx, sy, sw, sh = primary_smoke["box"]
                                c_lbl = f"ROKOK: {int(conf_smoke * 100)}%"
                                cv2.rectangle(evidence_frame, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 3)
                                (tw, th), _ = cv2.getTextSize(c_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                                cv2.rectangle(evidence_frame, (sx, max(0, sy - th - 10)), (sx + tw + 10, max(th + 10, sy)), (0, 0, 255), -1)
                                cv2.putText(evidence_frame, c_lbl, (sx + 5, max(th + 5, sy - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                            # 2. Bounding box perokok warna hijau
                            if smoker_person is not None:
                                spx, spy, spw, sph, *spconf = smoker_person
                                nim_label = f" [{resolved_student_id}]" if resolved_student_id else ""
                                p_lbl = f"[PEROKOK] {violator_name}{nim_label} ({violator_sim:.1f}%)"
                                cv2.rectangle(evidence_frame, (spx, spy), (spx + spw, spy + sph), (0, 255, 0), 2)
                                (ptw, pth), _ = cv2.getTextSize(p_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                                cv2.rectangle(evidence_frame, (spx, max(0, spy - pth - 10)), (spx + ptw + 10, max(pth + 10, spy)), (0, 180, 0), -1)
                                cv2.putText(evidence_frame, p_lbl, (spx + 5, max(pth + 5, spy - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                            # 3. Header OSD Bukti CCTV
                            cv2.rectangle(evidence_frame, (0, 0), (ew, 50), (15, 20, 28), -1)
                            cv2.putText(evidence_frame, f"E.S.I.S BUKTI PELANGGARAN | {cam_info['name']}", (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                            cv2.putText(evidence_frame, f"WAKTU: {now_str} WIB | {status_desc} (>3s)", (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

                            cv2.imwrite(capture_path, evidence_frame)

                            print("\n" + "=" * 82)
                            print(f" 🚨 [PELANGGARAN TERKONFIRMASI] {cam_info['name']}")
                            print(f"    Pelaku    : {violator_name} (NIM: {resolved_student_id or '-'})")
                            print(f"    Kemiripan : {violator_sim:.1f}%")
                            print(f"    Durasi    : {cam_state.smoking_duration:.1f} detik kontinu")
                            print(f"    Snapshot  : {capture_path}")
                            print("    Mengunggah bukti ke Cloudflare R2 & Mengirim Laporan ke Website API...")
                            print("=" * 82 + "\n", flush=True)

                            threading.Thread(
                                target=capture_and_report_incident,
                                kwargs={
                                    "frame_or_path": capture_path,
                                    "location": cam_info["name"],
                                    "location_group": cam_info.get("location_group", "Lantai 5 Gedung PLT"),
                                    "student_name": violator_name,
                                    "student_id": resolved_student_id,
                                    "confidence_smoking": conf_smoke,
                                    "face_similarity": sim_face,
                                    "notes": f"Terdeteksi otomatis merokok selama {cam_state.smoking_duration:.1f} detik via Plan A di {cam_info['name']}.",
                                    "filename": capture_filename,
                                },
                                daemon=True
                            ).start()

                else:
                    # ---------------------------------------------------------
                    # TIDAK ADA MANUSIA
                    # ---------------------------------------------------------
                    cam_state.is_smoking = False
                    cam_state.smoking_duration = 0.0
                    cam_state.smoking_start_time = None
                    cam_state.last_smoking_seen_time = None
                    cam_state.pending_face_ident_start = None

                    if cam_state.no_person_since is None:
                        cam_state.no_person_since = curr_t

                    idle_duration = curr_t - cam_state.no_person_since
                    if idle_duration < args.idle_sleep_sec:
                        sisa_ke_tidur = args.idle_sleep_sec - idle_duration
                        cam_state.status_badge = f"NIHIL ({sisa_ke_tidur:.1f}s)"
                        cam_state.status_desc = f"Tidak ada manusia ({idle_duration:.1f}s / {args.idle_sleep_sec:.0f}s)"
                    else:
                        # SUDAH 10 DETIK TANPA MANUSIA -> AKTIFKAN TIDUR 1 MENIT!
                        cam_state.sleep_until = curr_t + args.sleep_duration_sec
                        cam_state.no_person_since = None
                        cam_state.status_badge = f"SLEEP ({int(args.sleep_duration_sec)}s)"
                        cam_state.status_desc = f"Tidak ada manusia >{args.idle_sleep_sec:.0f}s -> AI Rokok/Wajah dimatikan {int(args.sleep_duration_sec)}s"
                        wake_hms = datetime.datetime.fromtimestamp(cam_state.sleep_until).strftime("%H:%M:%S")
                        print(f"\n[HEMAT ENERGI] [{cam_info['name']}] Tidak ada manusia selama {args.idle_sleep_sec:.0f} detik.")
                        print(f"               Model Rokok & Wajah DIMATIKAN selama {int(args.sleep_duration_sec)} detik (hingga {wake_hms} WIB).\n", flush=True)

                    cam_state.annotated_frame = disp_frame

            # -----------------------------------------------------------------
            # 5. BUAT MOSAIC GRID 2x2 UNTUK MONITORING STREAM MJPEG & DISPLAY
            # -----------------------------------------------------------------
            mosaic_frame = build_2x2_mosaic(camera_states)
            SHARED_RUNPOD_STATE["annotated_frame"] = mosaic_frame
            SHARED_RUNPOD_STATE["latest_frame"] = mosaic_frame

            # -----------------------------------------------------------------
            # 6. PENGIRIMAN TELEMETRI REAL-TIME
            # -----------------------------------------------------------------
            if curr_t - last_telemetry_send_time >= 0.3:
                last_telemetry_send_time = curr_t
                cams_telemetry = []
                for cs in camera_states:
                    cams_telemetry.append({
                        "name": cs.name,
                        "key": cs.key,
                        "ecam_path": cs.cam.get("cam_path", f"cam_{cs.key}"),
                        "is_connected": cs.is_connected,
                        "has_person": cs.has_person,
                        "is_smoking": cs.is_smoking,
                        "smoking_duration": round(float(cs.smoking_duration), 1),
                        "status_badge": cs.status_badge,
                        "sleep_until": cs.sleep_until
                    })

                tel_payload = {
                    "cameras": cams_telemetry,
                    "fps": round(float(current_fps), 1),
                    "timestamp": curr_t
                }
                SHARED_RUNPOD_STATE["telemetry"] = tel_payload
                try:
                    telemetry_queue.put_nowait(tel_payload)
                except queue.Full:
                    pass

            # -----------------------------------------------------------------
            # 7. LOGGING TERMINAL BERKALA (STATUS 4 KAMERA)
            # -----------------------------------------------------------------
            if is_headless and (curr_t - last_print_time >= 3.0):
                last_print_time = curr_t
                now_hms = datetime.datetime.now().strftime("%H:%M:%S")
                print("\n" + "=" * 82)
                print(f" [PATROLI 4 KAMERA RUNPOD] Waktu: {now_hms} WIB | Total FPS: {current_fps:4.1f}")
                print("-" * 82)
                for c_idx, c_st in enumerate(camera_states):
                    conn_str = "ONLINE " if c_st.is_connected else "OFFLINE"
                    print(f"  [{c_idx+1}] {c_st.name:<20} | {conn_str} | Badge: {c_st.status_badge:<14} | Info: {c_st.status_desc}")
                print("=" * 82 + "\n", flush=True)

            # Display GUI jika diaktifkan
            if not is_headless:
                cv2.imshow(window_name, mosaic_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), ord('Q'), 27]:
                    print("\n[KELUAR] Tombol Q ditekan.\n", flush=True)
                    break

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\n[BERHENTI] Sinyal Ctrl+C diterima. Menghentikan worker...\n", flush=True)
    finally:
        stream_worker.stop()
        if httpd:
            try: httpd.shutdown()
            except Exception: pass
        if pinggy_tunnel:
            try: pinggy_tunnel.stop()
            except Exception: pass
        if not is_headless:
            try: cv2.destroyAllWindows()
            except Exception: pass
        print("\n * Patroli Plan A telah selesai dihentikan dengan aman.\n", flush=True)

if __name__ == "__main__":
    main()
