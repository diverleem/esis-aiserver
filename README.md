# E.S.I.S Plan A - RunPod Cloud GPU (Pure ONNX Pipeline - 4 Kamera Serentak)

Paket mandiri (*self-contained*) untuk inferensi Cloud GPU di RunPod menggunakan **100% arsitektur ONNX Runtime** dengan akselerasi NVIDIA TensorRT / CUDAExecutionProvider, mendukung pemantauan **4 kamera serentak** dengan mekanisme **Gated Inference** hemat daya.

## 1. Arsitektur Model (Full ONNX)
1. **Human Detection & Tracking**: `yolov8s.onnx` (atau `yolov8n.onnx`) dengan IoU & Centroid Tracker.
2. **Cigarette Detection**: `Model_Rokok_Esis_New_Era_Grey_v3.onnx` (YOLO11s ONNX) atau `v2.onnx`.
3. **Face Detection**: `face_detection_yunet_2023mar.onnx` (YuNet ONNX).
4. **Face Embedding**: `facenet.onnx` (512-dim embedding) dicocokkan dengan basis data 38 mahasiswa (`data_wajah_facenet_mtcnn.npz`).

## 2. Target 4 Kamera Pemantauan Serentak
1. **Lab Riset Lantai 5** (ID: 38, path: `cam_lab_riset_lt_5`)
2. **Lab OS** (ID: 36, path: `cam_lab_os`)
3. **Lab Programming** (ID: 37, path: `cam_lab_programming`)
4. **Lobby Lantai 5** (ID: 42, path: `cam_lobby_lt_5`)

URL Stream Utama: HLS Port 8888 via Pinggy Pro `https://fdxuqgcmfo.a.pinggy.link/` (sinkron otomatis dari tabel Supabase `streaming`).

## 3. Logika Pintar Gated Inference & Auto-Sleep
- **Human Gated Inference**: Model AI Rokok dan Wajah **HANYA** dijalankan jika model deteksi manusia mendeteksi kehadiran orang pada kamera tersebut.
- **Auto-Sleep 1 Menit**: Jika pada suatu kamera tidak terdeteksi manusia selama **10 detik kontinu**, model AI Rokok dan Wajah pada kamera tersebut **dimatikan selama 1 menit (60 detik)** untuk menghemat daya komputasi GPU/CPU.
- **Verifikasi Pelanggaran**:
  - Durasi rokok terverifikasi kontinu minimal 3,0 detik.
  - Wajah tidak terdaftar setelah buffer 3 detik dibatalkan secara otomatis (hanya mahasiswa terdaftar yang dilaporkan).
  - Jeda / Cooldown 1 jam per mahasiswa pelanggar.
  - NIM (`student_id`), Jurusan, dan Email mahasiswa terisi otomatis dari basis data Supabase.

## 4. Cara Menjalankan

### A. Di RunPod Linux (Cloud GPU):
```bash
bash run_runpod.sh
```

### B. Di Terminal Windows (Pengujian Lokal):
```cmd
run_runpod_terminal.bat
```
Atau dari root project:
```cmd
run_plan_a_terminal.bat
```

### C. Monitor Video Wall Grid 2x2:
Buka di browser saat server aktif:
- Stream MJPEG Grid 2x2: `http://localhost:8001/stream`
- Status API: `http://localhost:8001/status`
- Snapshot: `http://localhost:8001/snapshot`
- Telemetri Real-time: `http://localhost:8001/telemetry`
