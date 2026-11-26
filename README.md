# E.S.I.S - PLAN A: RunPod Cloud GPU Inference (Streaming via Pinggy Tunnel)

Folder ini disiapkan khusus untuk eksekusi inferensi AI di **RunPod (Cloud GPU)** secara **Headless (Terminal Output Only)**. Sistem menerima umpan video kamera CCTV dari kampus melalui **Pinggy Tunnel (MediaMTX HLS/RTSP)** dan mengekspos endpoint API monitoring publik via HTTPS.

---

## 1. Struktur Folder Plan A
```text
Plan_A_RunPod/
├── models/
│   ├── yolov8s.onnx                           # Detektor Manusia (YOLOv8s ONNX - Bounding Box Stabil)
│   ├── yolov8n.onnx                           # Detektor Manusia Cadangan (YOLOv8n ONNX)
│   ├── Model_Rokok_Esis_New_Era_Grey_v2.pt     # Detektor Objek Rokok (Grey v2 - Cepat & Ringan)
│   ├── Model_Rokok_Esis_New_Era_Grey_v3.pt     # Detektor Objek Rokok Alternatif (Grey v3)
│   ├── face_detection_yunet_2023mar.onnx      # Detektor Wajah (YuNet ONNX)
│   ├── facenet.onnx                           # Ekstraktor Embedding Wajah (FaceNet ONNX)
│   └── data_wajah_facenet_mtcnn.npz           # Database Profil Wajah Mahasiswa
├── captures/                                  # Folder penyimpanan foto bukti snapshot teranotasi
├── main_runpod.py                             # Skrip utama patroli headless Plan A (Sync Plan B + Pinggy Stream)
├── run_runpod.sh                              # Runner Bash 1-klik untuk Linux / RunPod
├── requirements.txt                           # Seluruh dependensi Python cloud (numpy<2.0.0, torch, onnxruntime, dll.)
├── r2_reporter.py                             # Uploader Cloudflare R2 & Dispatcher Web API
├── pinggy_tunnel.py                           # Modul SSH Reverse Tunnel Pinggy
├── .env                                       # Konfigurasi kredensial R2, API URL, & Pinggy Stream Base
└── README.md                                  # Dokumentasi ini
```

---

## 2. Alur Kerja & Fitur Utama (Sinkron dengan Plan B)
1. **Streaming Input via Pinggy Tunnel**:
   - RunPod membaca video streaming CCTV lab melalui link tunnel Pinggy dari MediaMTX kampus (format HLS `index.m3u8` atau RTSP).
   - Base URL Pinggy dapat ditentukan via:
     - Argumen CLI: `--pinggy-stream-base https://xxxx.a.pinggy.link`
     - File `.env`: `PINGGY_STREAM_BASE_URL=https://xxxx.a.pinggy.link`
     - Sinkronisasi otomatis dari database Supabase (tabel `streaming` kolom `hls_source`).
     - Endpoint dinamis saat runtime: `POST http://localhost:8001/api/patrol/pinggy-stream` dengan body `{"base_url": "https://baru.a.pinggy.link"}`.
2. **Stage 1: Deteksi Manusia (YOLOv8s ONNX)**:
   - Memindai keberadaan manusia. Jika selama 10 detik berturut-turut **NIHIL MANUSIA**, sistem otomatis skip ke kamera lab berikutnya.
3. **Stage 2: Deteksi Rokok (Grey v2 .pt) & Pengenalan Wajah Spasial**:
   - Jika manusia terdeteksi, sistem menjalankan deteksi rokok dan mencocokkan wajah perokok spesifik (orang lain yang tidak merokok tidak dituduh).
4. **Buffer Wajah 3 Detik & Pembatalan Wajah Tak Dikenal**:
   - Rokok kontinu wajib terdeteksi $\ge 3$ detik.
   - Jika wajah perokok belum teridentifikasi (misal tertutup tangan saat merokok), sistem memberikan jeda toleransi 3 detik.
   - Jika dalam 3 detik wajah terbaca -> langsung dikirim.
   - Jika setelah 3 detik wajah tetap tak dikenal -> **dibatalkan** (tidak upload ke R2 & API) untuk menghemat bandwidth & storage.
5. **Anotasi Bounding Box pada Foto Bukti**:
   - Sebelum diunggah ke Cloudflare R2, frame diberi anotasi visual:
     - Kotak merah tebal pada rokok + confidence %.
     - Kotak hijau pada perokok + nama mahasiswa + similarity %.
     - Banner OSD resmi CCTV di bagian atas foto.
6. **Background API Server (Port 8001) & Pinggy Reverse Tunnel**:
   - Menyediakan endpoint `/snapshot`, `/stream`, `/health`, `/telemetry`, `POST /api/simulate`, `POST /api/patrol/switch`.

---

## 3. Cara Menjalankan di RunPod

1. **Upload folder `Plan_A_RunPod` ke RunPod** (via JupyterLab File Browser, Git, atau SCP).
2. Buka Web Terminal RunPod:
   ```bash
   cd Plan_A_RunPod
   chmod +x run_runpod.sh
   ./run_runpod.sh
   ```
   *Skrip otomatis memeriksa dependensi, mendeteksi GPU NVIDIA CUDA, dan menjalankan inferensi.*

3. **Menjalankan dengan Link Pinggy Kamera Kustom**:
   Jika URL Pinggy MediaMTX kamera Anda berubah, cukup tambahkan argumen `--pinggy-stream-base`:
   ```bash
   python3 main_runpod.py --pinggy-stream-base https://fdxuqgcmfo.a.pinggy.link
   ```

---

## 4. Pengujian Lokal Sebelum Upload ke Cloud
Di laptop Anda, Anda dapat menguji skrip Plan A tanpa koneksi cloud dengan menjalankan:
```cmd
run_runpod_mock_test.bat
```
