import cv2
import csv
import os
from datetime import datetime
from collections import defaultdict, Counter
import time
import requests
import json

from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = r"runs\detect\train-3\weights\best.pt"

OUTPUT_CSV = "observations.csv"

CAMERA_INDEX = 0

IMAGE_SIZE = 640
CONFIDENCE = 0.70

# ============================================================
# TELEGRAM TEST CONFIGURATION
# ============================================================
# Isi dengan token dari @BotFather.
# JANGAN upload token ini ke GitHub.
TELEGRAM_BOT_TOKEN = "Redacted"

# Bisa berupa @username_channel atau chat ID angka.
TELEGRAM_CHAT_ID = -1004460895066

# Username Telegram anggota yang ingin di-ping.
# Contoh: ["@anggota1", "@anggota2"]
# Catatan: Telegram tidak punya @everyone universal seperti Discord.
TELEGRAM_MENTIONS = [
    "@anggota1",
    "@anggota2",
]

# Saat testing: kirim setiap 10 detik.
# Minimal interval antar notifikasi violation.
# Mencegah spam jika violation terdeteksi terus-menerus.
TELEGRAM_INTERVAL = 20

# Violation harus terdeteksi pada track person minimal 3 detik.
VIOLATION_MIN_DURATION = 3.0

# Track ID yang sudah pernah dilaporkan.
# Dipakai agar violation yang sama tidak terus dikirim.
reported_violation_ids = set()

# track_id -> waktu pertama kali violation terdeteksi
violation_start_times = {}

# track_id -> APD yang sedang hilang
active_violation_ppe = {}


# Nama file sementara untuk capture.
TELEGRAM_RAW_IMAGE = "telegram_capture_raw.jpg"
TELEGRAM_BOX_IMAGE = "telegram_capture_box.jpg"

# Tracker
TRACKER_CONFIG = "botsort.yaml"

# Minimal jumlah frame sebuah person harus terlihat
# sebelum dianggap sebagai observation
MIN_FRAMES = 10

# Berapa frame terakhir digunakan untuk majority voting
MAX_HISTORY = 100

# Setelah ID tidak terlihat selama beberapa frame,
# observation dianggap selesai.
MAX_MISSING_FRAMES = 30


# ============================================================
# CLASS NAME CONFIGURATION
# ============================================================
#
# Sesuaikan dengan model kamu.
#
# Contoh:
#
# person
# helmet
# vest
# mask
# gloves
# shoes
#
# Jika model kamu menggunakan nama berbeda,
# ubah dictionary ini.
# ============================================================

PERSON_CLASS = "person"

PPE_CLASSES = [
    "topi",
    "vest",
]


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

model = YOLO(MODEL_PATH)

print("\nModel classes:")
print(model.names)

print("\n")


# ============================================================
# CHECK PERSON CLASS
# ============================================================

model_class_names = [
    str(name).lower()
    for name in model.names.values()
]

if PERSON_CLASS.lower() not in model_class_names:

    print("ERROR:")
    print("Class 'person' tidak ditemukan di model.")
    print()
    print("Class yang tersedia:")
    print(model.names)
    print()
    print(
        "Model kamu harus mendeteksi 'person' "
        "agar tracking orang dapat dilakukan."
    )

    exit()


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)

if not cap.isOpened():

    print("Kamera gagal dibuka.")

    exit()


# ============================================================
# CSV FILE
# ============================================================

file_exists = os.path.exists(OUTPUT_CSV)

csv_file = open(
    OUTPUT_CSV,
    mode="a",
    newline="",
    encoding="utf-8"
)

csv_writer = csv.writer(csv_file)


# Header CSV
if not file_exists:

    header = [
        "observation_id",
        "timestamp",
        "track_id",
        "frames_observed"
    ]

    # Tambahkan class APD
    for ppe in PPE_CLASSES:
        header.append(ppe)

    header.append("violation")

    csv_writer.writerow(header)


# ============================================================
# TRACK DATA
# ============================================================

tracks = {}


# Struktur:
#
# tracks[track_id] = {
#
#     "first_seen": timestamp,
#     "last_seen_frame": frame_number,
#     "frames": 0,
#
#     "ppe_history": {
#         "helmet": [],
#         "vest": [],
#         ...
#     }
# }


observation_id = 1


# Cari observation ID terakhir dari CSV
if file_exists:

    try:

        with open(
            OUTPUT_CSV,
            mode="r",
            encoding="utf-8"
        ) as f:

            reader = csv.DictReader(f)

            rows = list(reader)

            if len(rows) > 0:

                observation_id = (
                    int(rows[-1]["observation_id"]) + 1
                )

    except Exception:

        observation_id = 1


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def center_of_box(box):
    """
    Mengambil titik tengah bounding box.
    """

    x1, y1, x2, y2 = box

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    return cx, cy


def point_inside_box(point, box):
    """
    Mengecek apakah sebuah titik berada
    di dalam bounding box.
    """

    px, py = point

    x1, y1, x2, y2 = box

    return (
        x1 <= px <= x2
        and
        y1 <= py <= y2
    )


def associate_ppe_to_person(
    person_box,
    ppe_detections
):
    """
    Menghubungkan objek APD dengan person
    berdasarkan posisi bounding box.

    Jika center bounding box APD berada
    di dalam bounding box person,
    APD dianggap milik person tersebut.
    """

    detected_ppe = set()

    for ppe in ppe_detections:

        ppe_class = ppe["class"]

        ppe_box = ppe["box"]

        ppe_center = center_of_box(
            ppe_box
        )

        if point_inside_box(
            ppe_center,
            person_box
        ):

            detected_ppe.add(
                ppe_class
            )

    return detected_ppe


def majority_vote(values):

    if len(values) == 0:

        return 0

    counter = Counter(values)

    return counter.most_common(1)[0][0]


def finalize_observation(
    track_id,
    track_data
):

    global observation_id

    frames = track_data["frames"]

    # Jangan simpan track yang terlalu singkat
    if frames < MIN_FRAMES:

        print(
            f"[SKIP] Track {track_id} "
            f"hanya {frames} frame."
        )

        return

    # ========================================================
    # MAJORITY VOTING
    # ========================================================

    final_ppe = {}

    for ppe in PPE_CLASSES:

        history = (
            track_data["ppe_history"][ppe]
        )

        final_ppe[ppe] = majority_vote(
            history
        )

    # ========================================================
    # VIOLATION
    # ========================================================

    # Jika salah satu APD tidak terdeteksi
    violation = 0

    for ppe in PPE_CLASSES:

        if final_ppe[ppe] == 0:

            violation = 1

            break

    # ========================================================
    # TIMESTAMP
    # ========================================================

    timestamp = track_data["first_seen"]

    # ========================================================
    # SAVE CSV
    # ========================================================

    row = [
        observation_id,
        timestamp,
        track_id,
        frames
    ]

    for ppe in PPE_CLASSES:

        row.append(
            final_ppe[ppe]
        )

    row.append(violation)

    csv_writer.writerow(row)

    csv_file.flush()

    # ========================================================
    # PRINT RESULT
    # ========================================================

    status = (
        "VIOLATION"
        if violation == 1
        else "COMPLIANT"
    )

    print()
    print("=" * 60)

    print(
        f"OBSERVATION #{observation_id}"
    )

    print(
        f"Track ID      : {track_id}"
    )

    print(
        f"Frames        : {frames}"
    )

    print(
        f"Status        : {status}"
    )

    print(
        "APD           :"
    )

    for ppe in PPE_CLASSES:

        status_ppe = (
            "DETECTED"
            if final_ppe[ppe] == 1
            else "NOT DETECTED"
        )

        print(
            f"  {ppe:<10}: {status_ppe}"
        )

    print("=" * 60)

    observation_id += 1


# ============================================================
# TELEGRAM FUNCTIONS
# ============================================================

def telegram_send_message(text):
    """Mengirim pesan teks ke Telegram."""
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "ISI_TOKEN_BOT_DI_SINI":
        print("[TELEGRAM] Token belum diisi.")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    try:
        response = requests.post(
            url,
            data={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": text,
            },
            timeout=15
        )

        if response.ok:
            print("[TELEGRAM] Pesan berhasil dikirim.")
            return True

        print(
            f"[TELEGRAM] Gagal mengirim pesan: "
            f"{response.status_code} {response.text}"
        )

    except requests.RequestException as e:
        print(f"[TELEGRAM] Error koneksi: {e}")

    return False


def telegram_send_photo(image_path, caption=""):
    """Mengirim foto ke Telegram."""
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "ISI_TOKEN_BOT_DI_SINI":
        print("[TELEGRAM] Token belum diisi.")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"

    try:
        with open(image_path, "rb") as photo:
            response = requests.post(
                url,
                data={
                    "chat_id": str(TELEGRAM_CHAT_ID),
                    "caption": caption,
                },
                files={
                    "photo": (
                        os.path.basename(image_path),
                        photo,
                        "image/jpeg",
                    ),
                },
                timeout=30
            )

        result = response.json()

        if result.get("ok"):
            print(f"[TELEGRAM] Foto berhasil dikirim: {image_path}")
            return True

        print(
            f"[TELEGRAM] Gagal mengirim foto: "
            f"{response.status_code} {result}"
        )


    except (requests.RequestException, OSError) as e:
        print(f"[TELEGRAM] Error mengirim foto: {e}")

    return False


def send_violation_to_telegram(
    raw_frame,
    annotated_frame,
    track_id,
    violation_ppe
):
    """
    Mengirim satu notifikasi violation ke Telegram.

    Isi notifikasi:
    1. Pesan violation.
    2. Dua gambar dikirim sebagai satu media group/bubble:
       - foto CCTV tanpa bounding box
       - foto YOLO dengan bounding box
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    raw_ok = cv2.imwrite(
        TELEGRAM_RAW_IMAGE,
        raw_frame
    )

    box_ok = cv2.imwrite(
        TELEGRAM_BOX_IMAGE,
        annotated_frame
    )

    if not raw_ok or not box_ok:
        print("[TELEGRAM] Gagal membuat capture violation.")
        return False

    # Nama APD yang tidak terdeteksi.
    if violation_ppe:
        ppe_text = ", ".join(violation_ppe)
    else:
        ppe_text = "APD tidak lengkap"

    mentions = " ".join(TELEGRAM_MENTIONS)

    message = (
        "🚨 VIOLATION TERDETEKSI\n\n"
        f"👤 Person ID: {track_id}\n"
        f"❌ Pelanggaran: {ppe_text}\n"
        f"🕐 Waktu: {timestamp}\n\n"
        f"{mentions}"
    )

    # Kirim pesan violation terlebih dahulu.
    print("[TELEGRAM] Mengirim pesan VIOLATION...")
    message_ok = telegram_send_message(message)

    if not message_ok:
        print("[TELEGRAM] Pesan violation gagal dikirim.")
        return False

    # ========================================================
    # KIRIM 2 FOTO DALAM SATU MEDIA GROUP / BUBBLE
    # ========================================================
    url = (
        f"https://api.telegram.org/"
        f"bot{TELEGRAM_BOT_TOKEN}/sendMediaGroup"
    )

    try:
        with open(TELEGRAM_RAW_IMAGE, "rb") as raw_photo, \
             open(TELEGRAM_BOX_IMAGE, "rb") as box_photo:

            files = {
                "photo1": (
                    os.path.basename(TELEGRAM_RAW_IMAGE),
                    raw_photo,
                    "image/jpeg"
                ),
                "photo2": (
                    os.path.basename(TELEGRAM_BOX_IMAGE),
                    box_photo,
                    "image/jpeg"
                ),
            }

            # Telegram media group memakai attach://filename
            # sehingga kedua foto dikirim sebagai satu album/bubble.
            media = [
                {
                    "type": "photo",
                    "media": "attach://photo1",
                    "caption": "📷 Capture CCTV — tanpa bounding box"
                },
                {
                    "type": "photo",
                    "media": "attach://photo2",
                    "caption": "🤖 Capture YOLO — dengan bounding box"
                }
            ]

            response = requests.post(
                url,
                data={
                    "chat_id": str(TELEGRAM_CHAT_ID),
                    "media": json.dumps(media)
                },
                files=files,
                timeout=30
            )

        result = response.json()

        if result.get("ok"):
            print(
                "[TELEGRAM] 2 foto violation berhasil "
                "dikirim sebagai satu media group."
            )
            return True

        print(
            f"[TELEGRAM] Gagal mengirim media group: "
            f"{response.status_code} {result}"
        )

    except (requests.RequestException, OSError, ValueError) as e:
        print(
            f"[TELEGRAM] Error media group: {e}"
        )

    return False


# ============================================================
# MAIN LOOP
# ============================================================

frame_number = 0

# Timer Telegram
last_telegram_send = time.monotonic() - TELEGRAM_INTERVAL

print()
print("========================================")
print("YOLO APD PERSON TRACKING")
print("========================================")
print(f"Telegram Chat ID : {TELEGRAM_CHAT_ID}")
print(f"Telegram Interval: {TELEGRAM_INTERVAL} detik")
print("Telegram violation notification: AKTIF")
print("Minimal durasi violation: 3 detik.")
print("Foto RAW + foto YOLO dikirim sebagai satu bubble.")
print("Press Q to exit")
print()


while True:

    ret, frame = cap.read()

    if not ret:

        print(
            "Gagal membaca frame."
        )

        break

    frame_number += 1


    # ========================================================
    # YOLO TRACKING
    # ========================================================

    results = model.track(

        source=frame,

        imgsz=IMAGE_SIZE,

        device=0,

        conf=CONFIDENCE,

        persist=True,

        tracker=TRACKER_CONFIG,

        verbose=False
    )


    result = results[0]


    # ========================================================
    # CURRENT TRACK IDs
    # ========================================================

    current_track_ids = set()


    # ========================================================
    # CHECK DETECTIONS
    # ========================================================

    # Harus diinisialisasi setiap frame.
    # Jika tidak ada detection/track pada frame, list tetap kosong.
    persons = []
    ppe_detections = []

    if (
        result.boxes is not None
        and
        result.boxes.id is not None
    ):

        boxes = result.boxes


        # ----------------------------------------------------
        # TRACK IDs
        # ----------------------------------------------------

        track_ids = (
            boxes.id
            .int()
            .cpu()
            .tolist()
        )


        # ----------------------------------------------------
        # CLASS IDS
        # ----------------------------------------------------

        class_ids = (
            boxes.cls
            .int()
            .cpu()
            .tolist()
        )


        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        confidences = (
            boxes.conf
            .cpu()
            .tolist()
        )


        # ----------------------------------------------------
        # BOUNDING BOXES
        # ----------------------------------------------------

        xyxy = (
            boxes.xyxy
            .cpu()
            .tolist()
        )


        # ====================================================
        # SEPARATE PERSON AND PPE
        # ====================================================

        persons = []

        ppe_detections = []


        for track_id, class_id, confidence, box in zip(

            track_ids,
            class_ids,
            confidences,
            xyxy

        ):

            class_name = (
                str(model.names[class_id])
                .lower()
            )


            # ------------------------------------------------
            # PERSON
            # ------------------------------------------------

            if class_name == PERSON_CLASS:

                persons.append({

                    "track_id": track_id,

                    "box": box,

                    "confidence": confidence

                })


            # ------------------------------------------------
            # PPE
            # ------------------------------------------------

            elif class_name in PPE_CLASSES:

                ppe_detections.append({

                    "class": class_name,

                    "box": box,

                    "confidence": confidence

                })


        # ====================================================
        # PROCESS EACH PERSON
        # ====================================================

        for person in persons:

            track_id = person["track_id"]

            person_box = person["box"]


            current_track_ids.add(
                track_id
            )


            # =================================================
            # CREATE TRACK RECORD
            # =================================================

            if track_id not in tracks:

                tracks[track_id] = {

                    "first_seen":
                        datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),

                    "last_seen_frame":
                        frame_number,

                    "frames":
                        0,

                    "ppe_history": {

                        ppe: []

                        for ppe in PPE_CLASSES

                    }

                }


            track_data = tracks[track_id]


            track_data[
                "last_seen_frame"
            ] = frame_number


            track_data[
                "frames"
            ] += 1


            # =================================================
            # ASSOCIATE PPE
            # =================================================

            detected_ppe = (
                associate_ppe_to_person(
                    person_box,
                    ppe_detections
                )
            )


            # =================================================
            # STORE PPE HISTORY
            # =================================================

            for ppe in PPE_CLASSES:

                if ppe in detected_ppe:

                    value = 1

                else:

                    value = 0


                history = (
                    track_data[
                        "ppe_history"
                    ][ppe]
                )


                history.append(value)


                # Jangan simpan history terlalu panjang
                if len(history) > MAX_HISTORY:

                    history.pop(0)


    # ========================================================
    # HANDLE LOST TRACKS
    # ========================================================

    tracks_to_remove = []


    for track_id, track_data in tracks.items():

        last_seen = (
            track_data[
                "last_seen_frame"
            ]
        )


        missing_frames = (
            frame_number - last_seen
        )


        # Jika track sudah lama hilang
        if missing_frames > MAX_MISSING_FRAMES:

            finalize_observation(
                track_id,
                track_data
            )


            tracks_to_remove.append(
                track_id
            )


    # ========================================================
    # DELETE FINISHED TRACKS
    # ========================================================

    for track_id in tracks_to_remove:

        del tracks[track_id]


    # ========================================================
    # DISPLAY
    # ========================================================

    annotated_frame = result.plot()


    # Informasi jumlah tracking
    cv2.putText(

        annotated_frame,

        f"Tracked: {len(tracks)}",

        (20, 40),

        cv2.FONT_HERSHEY_SIMPLEX,

        1,

        (0, 255, 0),

        2

    )


    cv2.putText(

        annotated_frame,

        f"Observations: {observation_id - 1}",

        (20, 80),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (255, 255, 255),

        2

    )


    cv2.imshow(

        "YOLO APD - Person Tracking",

        annotated_frame

    )


    # ========================================================
    # TELEGRAM - VIOLATION NOTIFICATION
    # ========================================================
    current_time = time.monotonic()

    # `persons` berisi person yang benar-benar terdeteksi pada
    # frame saat ini. Gunakan ini untuk mendapatkan bounding box
    # karena track_data tidak menyimpan "box".
    current_violation_tracks = set()
    current_violation_details = {}

    for person in persons:

        track_id = person["track_id"]
        person_box = person["box"]

        detected_ppe = associate_ppe_to_person(
            person_box,
            ppe_detections
        )

        missing_ppe = [
            ppe
            for ppe in PPE_CLASSES
            if ppe not in detected_ppe
        ]

        if missing_ppe:

            current_violation_tracks.add(
                track_id
            )

            current_violation_details[
                track_id
            ] = missing_ppe

            # Mulai timer ketika violation pertama
            # kali terlihat pada track person ini.
            if track_id not in violation_start_times:

                violation_start_times[
                    track_id
                ] = current_time

                print(
                    f"[VIOLATION] Person ID {track_id} "
                    f"mulai violation."
                )

            active_violation_ppe[
                track_id
            ] = missing_ppe

    # ========================================================
    # RESET VIOLATION YANG SUDAH BERHENTI
    # ========================================================
    ended_tracks = (
        set(violation_start_times.keys())
        - current_violation_tracks
    )

    for track_id in ended_tracks:

        duration = (
            current_time
            - violation_start_times[
                track_id
            ]
        )

        if duration < VIOLATION_MIN_DURATION:

            print(
                f"[VIOLATION] Person ID {track_id}: "
                f"{duration:.1f}s < "
                f"{VIOLATION_MIN_DURATION:.1f}s "
                f"-> tidak dikirim."
            )

        violation_start_times.pop(
            track_id,
            None
        )

        active_violation_ppe.pop(
            track_id,
            None
        )

        # Jika violation sudah selesai,
        # track ID tersebut boleh dilaporkan lagi
        # pada violation berikutnya.
        reported_violation_ids.discard(
            track_id
        )

    # ========================================================
    # CEK VIOLATION MINIMAL 3 DETIK
    # ========================================================
    valid_violations = []

    for track_id in current_violation_tracks:

        start_time = (
            violation_start_times.get(
                track_id
            )
        )

        if start_time is None:
            continue

        duration = (
            current_time
            - start_time
        )

        if duration >= VIOLATION_MIN_DURATION:

            valid_violations.append(
                (
                    track_id,
                    active_violation_ppe.get(
                        track_id,
                        current_violation_details[
                            track_id
                        ]
                    ),
                    duration
                )
            )

    # ========================================================
    # KIRIM TELEGRAM
    # ========================================================
    # Syarat:
    # - violation bertahan minimal 3 detik
    # - track ID belum dilaporkan
    # - minimal 20 detik sejak notifikasi terakhir
    if (
        valid_violations
        and
        current_time
        - last_telegram_send
        >= TELEGRAM_INTERVAL
    ):

        track_id, missing_ppe, duration = (
            valid_violations[0]
        )

        if (
            track_id
            not in reported_violation_ids
        ):

            print()
            print("========================================")
            print(
                "[TELEGRAM] VALID VIOLATION"
            )
            print(
                f"[TELEGRAM] Person ID : "
                f"{track_id}"
            )
            print(
                f"[TELEGRAM] Durasi    : "
                f"{duration:.1f} detik"
            )
            print(
                f"[TELEGRAM] APD       : "
                f"{', '.join(missing_ppe)}"
            )
            print("========================================")

            sent = send_violation_to_telegram(
                frame.copy(),
                annotated_frame.copy(),
                track_id,
                missing_ppe
            )

            if sent:

                last_telegram_send = (
                    current_time
                )

                reported_violation_ids.add(
                    track_id
                )

                print(
                    f"[TELEGRAM] Person ID "
                    f"{track_id} berhasil "
                    f"dilaporkan."
                )

    # ========================================================
    # QUIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


# ============================================================
# FINALIZE REMAINING TRACKS
# ============================================================

print()
print("Finalizing remaining observations...")


for track_id, track_data in tracks.items():

    finalize_observation(
        track_id,
        track_data
    )


# ============================================================
# CLEANUP
# ============================================================

cap.release()

csv_file.close()

cv2.destroyAllWindows()


print()
print("========================================")
print("PROGRAM SELESAI")
print("========================================")

print(
    f"Data observasi tersimpan di:"
    f" {OUTPUT_CSV}"
)