#!/usr/bin/env bash
# ===============================================================================
# E.S.I.S Plan A - RunPod Cloud GPU Launcher (Terminal Mode)
# Menjalankan Patroli Headless Lab Riset Lt 5 (Mainstream Cam) + API Server + Pinggy
# ===============================================================================

set -e

echo "==============================================================================="
echo "   E.S.I.S (Electronic Smoking Identification System)"
echo "   PLAN A: RunPod Cloud GPU Terminal Runner (Lab Riset Lantai 5)"
echo "==============================================================================="

# Pindah ke direktori skrip ini
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Cek & Install Requirements
echo ""
echo "[1/3] Memeriksa dependensi Python (requirements.txt)..."
python3 -c "import ultralytics, onnxruntime, cv2, boto3, requests" 2>/dev/null || {
    echo "      Menginstall pustaka dari requirements.txt..."
    pip install --upgrade pip
    pip install -r requirements.txt
}

which ssh >/dev/null 2>&1 || {
    echo "      Menginstall openssh-client untuk Pinggy..."
    apt-get update && apt-get install -y openssh-client
}

# 2. Cek Akselerasi GPU CUDA
echo ""
echo "[2/3] Memeriksa ketersediaan GPU NVIDIA CUDA..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'      [CUDA GPU OK] {torch.cuda.get_device_name(0)}')
else:
    print('      [INFO] CUDA GPU tidak terdeteksi, berjalan pada CPU mode.')
"

# 3. Jalankan Patroli Headless Plan A
echo ""
echo "[3/3] Memulai Patroli E.S.I.S Plan A..."
echo "      * Target Kamera : Lab Riset Lantai 5"
echo "      * Stream Source : https://stream.ucim.my.id/cam_lab_riset_lt_5/"
echo "      * Mode          : Terminal Headless"
echo "      * API Port      : 8001"
echo "      * Tekan Ctrl+C untuk keluar kapan saja."
echo "==============================================================================="
echo ""

python3 main_runpod.py --cigar-model v3_onnx --person-model yolov8s --channel 101 --api-port 8001 --headless "$@"
