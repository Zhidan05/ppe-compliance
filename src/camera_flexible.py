import csv
import json
import os
import platform
import queue
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import psutil
import requests
import torch
from dotenv import load_dotenv
from ultralytics import YOLO

# ==========================================
# PATH PROYEK & KONFIGURASI GLOBAL
# ==========================================
# Lokasi: ppe-compliance/src/camera_flexible.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

# Muat konfigurasi environment
load_dotenv(PROJECT_ROOT / ".env")

OUTPUT_CSV = DATA_DIR / "observations.csv"
# CAMERA_INDEX = 0
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE")
TRACKER_CONFIG = "botsort.yaml"

DATA_DIR.mkdir(parents=True, exist_ok=True)

PERSON_CLASS = "person"
PPE_CLASSES = ["topi", "vest"]

MIN_FRAMES = 10
MAX_HISTORY = 100
MAX_MISSING_FRAMES = 60

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_MENTIONS = ["@anggota1", "@anggota2"]
TELEGRAM_INTERVAL = 20
VIOLATION_MIN_DURATION = 3.0
JPEG_QUALITY = 85
TELEGRAM_QUEUE_SIZE = 10

# State Tracker & Notifikasi
reported_violation_ids = set()
violation_start_times = {}
active_violation_ppe = {}
tracks = {}
telegram_queue = queue.Queue(maxsize=TELEGRAM_QUEUE_SIZE)


# ==========================================
# HARDWARE PROFILING & SELEKSI MODEL
# ==========================================
def find_openvino_model(base_dir: Path) -> Path | None:
    """Mencari file model OpenVINO (.xml) yang valid."""
    ov_dir = base_dir / "openvino"
    if not ov_dir.exists():
        return None
    xml_files = list(ov_dir.rglob("*.xml"))
    return xml_files[0] if xml_files else None


def detect_hardware_profile(base_dir: Path):
    """Mendeteksi spesifikasi sistem dan memilih format model paling optimal."""
    has_cuda = torch.cuda.is_available()
    cpu_info = platform.processor().lower()
    total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    is_arm = platform.machine().lower().startswith(("arm", "aarch"))

    engine_path = base_dir / "tensorrt" / "best.engine"
    onnx_path = base_dir / "onnx" / "best.onnx"
    pt_path = base_dir / "pytorch" / "best.pt"
    tflite_path = base_dir / "tflite" / "best.tflite"
    ov_xml = find_openvino_model(base_dir)

    # 1. High Tier: GPU NVIDIA (TensorRT -> ONNX CUDA -> PyTorch CUDA)
    if has_cuda:
        if engine_path.exists():
            return {
                "tier": "NVIDIA GPU (TensorRT)",
                "model_path": engine_path,
                "imgsz": 640,
                "conf": 0.40,
                "device": 0,
            }
        elif onnx_path.exists():
            return {
                "tier": "NVIDIA GPU (ONNX CUDA)",
                "model_path": onnx_path,
                "imgsz": 640,
                "conf": 0.40,
                "device": 0,
            }
        elif pt_path.exists():
            return {
                "tier": "NVIDIA GPU (PyTorch CUDA)",
                "model_path": pt_path,
                "imgsz": 640,
                "conf": 0.40,
                "device": 0,
            }

    # 2. Mid Tier: CPU x86 / OpenVINO Optimized
    if ov_xml and ("intel" in cpu_info or "amd" in cpu_info or "ryzen" in cpu_info):
        return {
            "tier": "x86 CPU (OpenVINO)",
            "model_path": ov_xml,
            "imgsz": 640,
            "conf": 0.40,
            "device": "cpu",
        }

    # 3. Low Tier: ARM / Raspberry Pi / Low RAM
    if is_arm or total_ram_gb < 4.0:
        chosen = tflite_path if tflite_path.exists() else (onnx_path if onnx_path.exists() else pt_path)
        return {
            "tier": "Low Tier (Edge/ARM)",
            "model_path": chosen,
            "imgsz": 640,
            "conf": 0.35,
            "device": "cpu",
        }

    # 4. Standard Fallback: CPU ONNX / PyTorch
    chosen = onnx_path if onnx_path.exists() else pt_path
    return {
        "tier": "Standard CPU (ONNX/PyTorch)",
        "model_path": chosen,
        "imgsz": 640,
        "conf": 0.40,
        "device": "cpu",
    }


def prepare_onnx_cuda(model_path: Path):
    """Preload DLL CUDA sebelum runtime ONNX dimulai."""
    if torch.cuda.is_available() and str(model_path).lower().endswith(".onnx"):
        try:
            import onnxruntime as ort
            ort.preload_dlls()
        except Exception:
            pass


# ==========================================
# GEOMETRI & DETEKSI APD
# ==========================================
def center_of_box(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def point_inside_box(point, box):
    px, py = point
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def associate_ppe_to_persons(persons, ppe_detections):
    assignments = {person["track_id"]: set() for person in persons}

    for ppe in ppe_detections:
        ppe_center = center_of_box(ppe["box"])
        candidates = [p for p in persons if point_inside_box(ppe_center, p["box"])]

        if not candidates:
            continue

        best_person = min(candidates, key=lambda p: box_area(p["box"]))
        assignments[best_person["track_id"]].add(ppe["class"])

    return assignments


def majority_vote(values):
    if not values:
        return 0
    return Counter(values).most_common(1)[0][0]


# ==========================================
# NOTIFIKASI TELEGRAM (SESSION REUSED)
# ==========================================
def send_violation_to_telegram(session, raw_frame, annotated_frame, track_id, missing_ppe):
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    raw_success, raw_buffer = cv2.imencode(".jpg", raw_frame, encode_params)
    ann_success, ann_buffer = cv2.imencode(".jpg", annotated_frame, encode_params)

    if not raw_success or not ann_success:
        print("[TELEGRAM] Gagal encode frame ke JPEG.")
        return False

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    violation_text = ", ".join(missing_ppe) if missing_ppe else "APD tidak lengkap"
    mentions = " ".join(TELEGRAM_MENTIONS)

    caption = (
        "🚨 VIOLATION TERDETEKSI\n\n"
        f"👤 Person ID: {track_id}\n"
        f"❌ Pelanggaran: {violation_text}\n"
        f"🕐 Waktu: {timestamp}\n\n"
        f"{mentions}"
    )

    media = [
        {"type": "photo", "media": "attach://raw_photo", "caption": caption},
        {"type": "photo", "media": "attach://annotated_photo"},
    ]

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMediaGroup"
    files = {
        "raw_photo": ("capture.jpg", raw_buffer.tobytes(), "image/jpeg"),
        "annotated_photo": ("detection.jpg", ann_buffer.tobytes(), "image/jpeg"),
    }

    try:
        response = session.post(
            url,
            data={"chat_id": str(TELEGRAM_CHAT_ID), "media": json.dumps(media)},
            files=files,
            timeout=25,
        )
        if response.ok:
            print(f"[TELEGRAM] Sent: Track ID={track_id}")
            return True
        print(f"[TELEGRAM] Gagal kirim: {response.status_code} - {response.text}")
    except requests.RequestException as error:
        print(f"[TELEGRAM] Error jaringan: {error}")

    return False


def telegram_worker():
    # Menggunakan session HTTP tunggal untuk reuse TCP / TLS Handshake
    with requests.Session() as session:
        while True:
            item = telegram_queue.get()
            if item is None:
                telegram_queue.task_done()
                break

            raw_frame, annotated_frame, track_id, missing_ppe = item
            try:
                send_violation_to_telegram(session, raw_frame, annotated_frame, track_id, missing_ppe)
            except Exception as error:
                print(f"[TELEGRAM] Worker exception: {error}")
            finally:
                telegram_queue.task_done()


# ==========================================
# LOGGING CSV OBSERVATION
# ==========================================
def finalize_observation(track_id, track_data, csv_writer, csv_file, obs_id):
    frames = track_data["frames"]
    if frames < MIN_FRAMES:
        return obs_id

    final_ppe = {ppe: majority_vote(track_data["ppe_history"][ppe]) for ppe in PPE_CLASSES}
    violation = int(any(final_ppe[ppe] == 0 for ppe in PPE_CLASSES))

    row = [obs_id, track_data["first_seen"], track_id, frames]
    row.extend(final_ppe[ppe] for ppe in PPE_CLASSES)
    row.append(violation)

    csv_writer.writerow(row)
    csv_file.flush()

    status = "VIOLATION" if violation else "COMPLIANT"
    print(f"[OBSERVATION] ID={obs_id} Track={track_id} Frames={frames} Status={status}")
    return obs_id + 1


# ==========================================
# VISUAL HUD / DASHBOARD
# ==========================================
def draw_info_panel(frame, hw_cfg, tracked_count, violation_count, observations, fps):
    panel_x, panel_y = 16, 16
    panel_w = min(390, frame.shape[1] - 32)
    panel_h = 156

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (18, 24, 32), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    status_color = (0, 180, 255) if violation_count else (0, 200, 90)

    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (70, 80, 90), 1)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + 6, panel_y + panel_h), status_color, -1)
    cv2.putText(frame, "PPE MONITORING", (panel_x + 20, panel_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(frame, "LIVE", (panel_x + panel_w - 58, panel_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_color, 2, cv2.LINE_AA)
    cv2.line(frame, (panel_x + 18, panel_y + 42), (panel_x + panel_w - 18, panel_y + 42), (75, 85, 95), 1, cv2.LINE_AA)

    info_rows = [
        ("BACKEND", hw_cfg["tier"]),
        ("TRACKED", str(tracked_count)),
        ("VIOLATION", str(violation_count)),
        ("FPS", f"{fps:.1f}"),
    ]

    for index, (label, value) in enumerate(info_rows):
        y = panel_y + 70 + (index * 25)
        color = status_color if label == "VIOLATION" and violation_count else (215, 215, 215)
        cv2.putText(frame, label, (panel_x + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (155, 165, 175), 1, cv2.LINE_AA)
        cv2.putText(frame, value, (panel_x + 150, y), cv2.FONT_HERSHEY_SIMPLEX, 0.47, color, 1, cv2.LINE_AA)

    footer = f"OBSERVATIONS: {observations}  |  Q: QUIT"
    cv2.putText(frame, footer, (panel_x + 20, panel_y + panel_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 160, 170), 1, cv2.LINE_AA)
    return frame


# ==========================================
# INISIALISASI & VALIDASI
# ==========================================
if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    raise RuntimeError("TELEGRAM_BOT_TOKEN atau TELEGRAM_CHAT_ID belum diset di .env")

hw_cfg = detect_hardware_profile(MODELS_DIR)

if not hw_cfg["model_path"] or not Path(hw_cfg["model_path"]).exists():
    raise FileNotFoundError(f"Model tidak ditemukan di direktori: {MODELS_DIR}")

print("========================================")
print(f"Hardware Profile : {hw_cfg['tier']}")
print(f"Selected Model   : {hw_cfg['model_path']}")
print(f"Input Resolution : {hw_cfg['imgsz']}x{hw_cfg['imgsz']}")
print(f"Inference Device : {hw_cfg['device']}")
print("========================================")

prepare_onnx_cuda(hw_cfg["model_path"])
model = YOLO(str(hw_cfg["model_path"]), task="detect")

model_class_names = {str(name).lower() for name in model.names.values()}
if PERSON_CLASS not in model_class_names:
    raise ValueError(f"Class '{PERSON_CLASS}' tidak ditemukan pada model.")

missing_classes = [ppe for ppe in PPE_CLASSES if ppe not in model_class_names]
if missing_classes:
    raise ValueError(f"PPE class tidak ditemukan: {missing_classes}")

# Inisialisasi Kamera
# cap = cv2.VideoCapture(CAMERA_SOURCE, cv2.CAP_FFMPEG)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;5000000"
cap = cv2.VideoCapture(CAMERA_SOURCE, cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise RuntimeError(
        "Kamera gagal dibuka. Periksa RTSP URL, username, password, dan IP kamera."
    )

# ==========================================
# WINDOW PREVIEW
# ==========================================
# WINDOW_NORMAL + WINDOW_KEEPRATIO hanya mengubah ukuran tampilan.
# Frame asli RTSP tidak di-crop atau di-resize.
WINDOW_NAME = "PPE Compliance Monitoring"
cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
)

# Ambil resolusi asli stream RTSP.
stream_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
stream_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"RTSP Stream Resolution: {stream_width}x{stream_height}")

# Ukuran preview hanya untuk display. Seluruh field of view tetap terlihat.
if stream_width > 0 and stream_height > 0:
    preview_width = min(stream_width, 1280)
    preview_height = int(stream_height * (preview_width / stream_width))
    cv2.resizeWindow(WINDOW_NAME, preview_width, preview_height)

# Inisialisasi File CSV
file_exists = OUTPUT_CSV.exists()
csv_file = open(OUTPUT_CSV, mode="a", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)

if not file_exists:
    csv_writer.writerow(["observation_id", "timestamp", "track_id", "frames_observed", *PPE_CLASSES, "violation"])
    csv_file.flush()

observation_id = 1
if file_exists:
    try:
        with open(OUTPUT_CSV, mode="r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            if rows:
                observation_id = int(rows[-1]["observation_id"]) + 1
    except (OSError, ValueError, KeyError):
        observation_id = 1

# Start Background Thread
telegram_thread = threading.Thread(target=telegram_worker, daemon=True)
telegram_thread.start()

frame_number = 0
last_telegram_send = time.monotonic() - TELEGRAM_INTERVAL

fps_display = 0.0
fps_alpha = 0.12

print("Monitoring APD CCTV Berjalan. Tekan 'Q' untuk berhenti.")

# ==========================================
# MAIN LOOP MONITORING
# ==========================================
try:
    while True:
        frame_start_time = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame kamera.")
            break

        frame_number += 1

        # Inferensi YOLO Tracking (tanpa parameter 'half' deprecated)
        results = model.track(
            source=frame,
            imgsz=hw_cfg["imgsz"],
            device=hw_cfg["device"],
            conf=hw_cfg["conf"],
            persist=True,
            tracker=TRACKER_CONFIG,
            verbose=False,
        )

        result = results[0]
        persons = []
        ppe_detections = []

        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes
            track_ids = boxes.id.int().cpu().tolist()
            class_ids = boxes.cls.int().cpu().tolist()
            confidences = boxes.conf.cpu().tolist()
            xyxy = boxes.xyxy.cpu().tolist()

            for t_id, c_id, conf, box in zip(track_ids, class_ids, confidences, xyxy):
                c_name = str(model.names[c_id]).lower()
                if c_name == PERSON_CLASS:
                    persons.append({"track_id": t_id, "box": box, "confidence": conf})
                elif c_name in PPE_CLASSES:
                    ppe_detections.append({"class": c_name, "box": box, "confidence": conf})

        current_track_ids = {p["track_id"] for p in persons}
        ppe_assignments = associate_ppe_to_persons(persons, ppe_detections)

        # Update riwayat track personil
        for person in persons:
            track_id = person["track_id"]
            if track_id not in tracks:
                tracks[track_id] = {
                    "first_seen": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "last_seen_frame": frame_number,
                    "frames": 0,
                    "ppe_history": {ppe: [] for ppe in PPE_CLASSES},
                }

            t_data = tracks[track_id]
            t_data["last_seen_frame"] = frame_number
            t_data["frames"] += 1

            detected_ppe = ppe_assignments.get(track_id, set())
            for ppe in PPE_CLASSES:
                hist = t_data["ppe_history"][ppe]
                hist.append(int(ppe in detected_ppe))
                if len(hist) > MAX_HISTORY:
                    hist.pop(0)

        # Hapus data person yang meninggalkan frame (Memory Guard)
        tracks_to_remove = [t_id for t_id, d in tracks.items() if (frame_number - d["last_seen_frame"]) > MAX_MISSING_FRAMES]
        for t_id in tracks_to_remove:
            observation_id = finalize_observation(t_id, tracks[t_id], csv_writer, csv_file, observation_id)
            tracks.pop(t_id, None)
            violation_start_times.pop(t_id, None)
            active_violation_ppe.pop(t_id, None)
            reported_violation_ids.discard(t_id)

        # Evaluasi Pelanggaran APD
        current_time = time.monotonic()
        current_violation_tracks = set()
        current_violation_details = {}

        for person in persons:
            t_id = person["track_id"]
            missing_ppe = [ppe for ppe in PPE_CLASSES if ppe not in ppe_assignments.get(t_id, set())]

            if missing_ppe:
                current_violation_tracks.add(t_id)
                current_violation_details[t_id] = missing_ppe

                if t_id not in violation_start_times:
                    violation_start_times[t_id] = current_time
                active_violation_ppe[t_id] = missing_ppe

        # Reset timer jika person sudah patuh
        for t_id in (set(violation_start_times.keys()) - current_violation_tracks):
            violation_start_times.pop(t_id, None)
            active_violation_ppe.pop(t_id, None)
            reported_violation_ids.discard(t_id)

        # Filter validasi durasi (Minimal 3 detik)
        valid_violations = []
        for t_id in current_violation_tracks:
            start_time = violation_start_times.get(t_id)
            if start_time and (current_time - start_time >= VIOLATION_MIN_DURATION):
                missing = active_violation_ppe.get(t_id, current_violation_details.get(t_id, []))
                valid_violations.append((t_id, missing, current_time - start_time))

        # Visualisasi & Rendering HUD
        annotated_frame = result.plot()

        frame_elapsed = max(time.perf_counter() - frame_start_time, 1e-6)
        instant_fps = 1.0 / frame_elapsed
        fps_display = instant_fps if fps_display == 0.0 else (fps_alpha * instant_fps) + ((1.0 - fps_alpha) * fps_display)

        annotated_frame = draw_info_panel(
            frame=annotated_frame,
            hw_cfg=hw_cfg,
            tracked_count=len(current_track_ids),
            violation_count=len(valid_violations),
            observations=observation_id - 1,
            fps=fps_display,
        )

        # Antrean Telegram Non-Blocking
        if valid_violations and (current_time - last_telegram_send >= TELEGRAM_INTERVAL):
            v_track_id, v_missing, v_dur = valid_violations[0]
            if v_track_id not in reported_violation_ids:
                try:
                    telegram_queue.put_nowait((frame.copy(), annotated_frame.copy(), v_track_id, v_missing))
                    last_telegram_send = current_time
                    reported_violation_ids.add(v_track_id)
                    print(f"[VIOLATION] Antrean Telegram: Track={v_track_id} Missing={v_missing} Duration={v_dur:.1f}s")
                except queue.Full:
                    print("[TELEGRAM] Antrean penuh, notifikasi dilewati.")

        # Tampilkan seluruh frame RTSP tanpa crop.
        # YOLO melakukan internal letterbox untuk inferensi, tetapi result.plot()
        # dikembalikan ke koordinat/resolusi frame asli.
        cv2.imshow(WINDOW_NAME, annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("\nProgram dihentikan pengguna.")

finally:
    for t_id, t_data in list(tracks.items()):
        observation_id = finalize_observation(t_id, t_data, csv_writer, csv_file, observation_id)

    print("Menyelesaikan antrean notifikasi Telegram...")
    telegram_queue.join()
    telegram_queue.put(None)
    telegram_thread.join(timeout=5)

    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()
    print(f"Data log tersimpan di: {OUTPUT_CSV}")