# E.S.I.S Plan A - RunPod Cloud GPU (Pure ONNX Pipeline)

Paket mandiri (*self-contained*) untuk inferensi Cloud GPU di RunPod menggunakan **100% arsitektur ONNX Runtime** dengan akselerasi NVIDIA TensorRT / CUDAExecutionProvider.

## 1. Arsitektur Model (Full ONNX)
1. **Human Detection & Tracking**: `yolov8s.onnx` (atau `yolov8n.onnx`) dengan IoU & Centroid Tracker.
2. **Cigarette Detection**: `Model_Rokok_Esis_New_Era_Grey_v3.onnx` (YOLO11s ONNX) atau `v2.onnx`.
3. **Face Detection**: `face_detection_yunet_2023mar.onnx` (YuNet ONNX).
4. **Face Embedding**: `facenet.onnx` (512-dim embedding) dicocokkan dengan basis data 38 mahasiswa (`data_wajah_facenet_mtcnn.npz`).

## 2. Target Pemantauan & Sumber Stream
- **Lab Riset Lantai 5** (Kamera ID: 38, `cam_lab_riset_lt_5`).
- Standby 24/7 tanpa auto-skip ke lab lain.
- **URL Stream Utama**: `https://fdxuqgcmfo.a.pinggy.link/cam_lab_riset_lt_5/index.m3u8` (HLS Port 8888 Mainstream 2560x1440).

## 3. Cara Menjalankan

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

### C. Dengan Custom URL Stream:
```bash
python main_runpod.py --stream-url https://fdxuqgcmfo.a.pinggy.link/cam_lab_riset_lt_5/index.m3u8
```

### D. Mode Mock Webcam Laptop:
```bash
python main_runpod.py --mock --no-pinggy
```
