import cv2
import csv
import os
from datetime import datetime
from collections import defaultdict, Counter

from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = r"runs\detect\train-3\weights\best.pt"

OUTPUT_CSV = "observations.csv"

CAMERA_INDEX = 0

IMAGE_SIZE = 640
CONFIDENCE = 0.70

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
# MAIN LOOP
# ============================================================

frame_number = 0

print()
print("========================================")
print("YOLO APD PERSON TRACKING")
print("========================================")
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