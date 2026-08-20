import csv
import json
import os
import queue
import threading
import time
from collections import Counter
from datetime import datetime

import cv2
import requests
from dotenv import load_dotenv
from ultralytics import YOLO


load_dotenv()


MODEL_PATH = r"runs\detect\train-6\weights\best.pt"
OUTPUT_CSV = "observations.csv"

CAMERA_INDEX = 0

IMAGE_SIZE = 1280
CONFIDENCE = 0.40

TRACKER_CONFIG = "botsort.yaml"

PERSON_CLASS = "person"

PPE_CLASSES = [
    "topi",
    "vest",
]

MIN_FRAMES = 10
MAX_HISTORY = 100
MAX_MISSING_FRAMES = 60

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

TELEGRAM_MENTIONS = [
    "@anggota1",
    "@anggota2",
]

TELEGRAM_INTERVAL = 20
VIOLATION_MIN_DURATION = 3.0

JPEG_QUALITY = 85
TELEGRAM_QUEUE_SIZE = 10


reported_violation_ids = set()
violation_start_times = {}
active_violation_ppe = {}
tracks = {}

telegram_queue = queue.Queue(
    maxsize=TELEGRAM_QUEUE_SIZE
)


def center_of_box(box):
    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2,
    )


def point_inside_box(point, box):
    px, py = point
    x1, y1, x2, y2 = box

    return (
        x1 <= px <= x2
        and y1 <= py <= y2
    )


def box_area(box):
    x1, y1, x2, y2 = box

    return (
        max(0, x2 - x1)
        * max(0, y2 - y1)
    )


def associate_ppe_to_persons(
    persons,
    ppe_detections,
):
    assignments = {
        person["track_id"]: set()
        for person in persons
    }

    for ppe in ppe_detections:
        ppe_center = center_of_box(
            ppe["box"]
        )

        candidates = []

        for person in persons:
            if point_inside_box(
                ppe_center,
                person["box"],
            ):
                candidates.append(person)

        if not candidates:
            continue

        best_person = min(
            candidates,
            key=lambda person: box_area(
                person["box"]
            ),
        )

        assignments[
            best_person["track_id"]
        ].add(
            ppe["class"]
        )

    return assignments


def majority_vote(values):
    if not values:
        return 0

    return Counter(
        values
    ).most_common(1)[0][0]


def send_violation_to_telegram(
    raw_frame,
    annotated_frame,
    track_id,
    missing_ppe,
):
    raw_success, raw_buffer = cv2.imencode(
        ".jpg",
        raw_frame,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            JPEG_QUALITY,
        ],
    )

    annotated_success, annotated_buffer = (
        cv2.imencode(
            ".jpg",
            annotated_frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                JPEG_QUALITY,
            ],
        )
    )

    if (
        not raw_success
        or not annotated_success
    ):
        print(
            "[TELEGRAM] Failed to encode images."
        )
        return False

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    violation_text = (
        ", ".join(missing_ppe)
        if missing_ppe
        else "APD tidak lengkap"
    )

    mentions = " ".join(
        TELEGRAM_MENTIONS
    )

    caption = (
        "🚨 VIOLATION TERDETEKSI\n\n"
        f"👤 Person ID: {track_id}\n"
        f"❌ Pelanggaran: {violation_text}\n"
        f"🕐 Waktu: {timestamp}\n\n"
        f"{mentions}"
    )

    media = [
        {
            "type": "photo",
            "media": "attach://raw_photo",
            "caption": caption,
        },
        {
            "type": "photo",
            "media": "attach://annotated_photo",
        },
    ]

    url = (
        f"https://api.telegram.org/"
        f"bot{TELEGRAM_BOT_TOKEN}/sendMediaGroup"
    )

    files = {
        "raw_photo": (
            "capture.jpg",
            raw_buffer.tobytes(),
            "image/jpeg",
        ),
        "annotated_photo": (
            "detection.jpg",
            annotated_buffer.tobytes(),
            "image/jpeg",
        ),
    }

    try:
        response = requests.post(
            url,
            data={
                "chat_id": str(
                    TELEGRAM_CHAT_ID
                ),
                "media": json.dumps(media),
            },
            files=files,
            timeout=30,
        )

        if response.ok:
            print(
                f"[TELEGRAM] Sent "
                f"Track={track_id}"
            )
            return True

        print(
            f"[TELEGRAM] Failed: "
            f"{response.status_code} "
            f"{response.text}"
        )

    except requests.RequestException as error:
        print(
            f"[TELEGRAM] Network error: "
            f"{error}"
        )

    return False


def telegram_worker():
    while True:
        item = telegram_queue.get()

        if item is None:
            telegram_queue.task_done()
            break

        (
            raw_frame,
            annotated_frame,
            track_id,
            missing_ppe,
        ) = item

        try:
            send_violation_to_telegram(
                raw_frame,
                annotated_frame,
                track_id,
                missing_ppe,
            )

        except Exception as error:
            print(
                f"[TELEGRAM] Worker error: "
                f"{error}"
            )

        finally:
            telegram_queue.task_done()


def finalize_observation(
    track_id,
    track_data,
    csv_writer,
    csv_file,
    observation_id,
):
    frames = track_data["frames"]

    if frames < MIN_FRAMES:
        return observation_id

    final_ppe = {
        ppe: majority_vote(
            track_data["ppe_history"][ppe]
        )
        for ppe in PPE_CLASSES
    }

    violation = int(
        any(
            final_ppe[ppe] == 0
            for ppe in PPE_CLASSES
        )
    )

    row = [
        observation_id,
        track_data["first_seen"],
        track_id,
        frames,
    ]

    row.extend(
        final_ppe[ppe]
        for ppe in PPE_CLASSES
    )

    row.append(violation)

    csv_writer.writerow(row)
    csv_file.flush()

    status = (
        "VIOLATION"
        if violation
        else "COMPLIANT"
    )

    print(
        f"[OBSERVATION] "
        f"ID={observation_id} "
        f"Track={track_id} "
        f"Frames={frames} "
        f"Status={status}"
    )

    return observation_id + 1


if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError(
        "TELEGRAM_BOT_TOKEN tidak ditemukan "
        "di file .env"
    )

if not TELEGRAM_CHAT_ID:
    raise RuntimeError(
        "TELEGRAM_CHAT_ID tidak ditemukan "
        "di file .env"
    )

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model tidak ditemukan: "
        f"{MODEL_PATH}"
    )


model = YOLO(MODEL_PATH)

model_class_names = {
    str(name).lower()
    for name in model.names.values()
}

if PERSON_CLASS not in model_class_names:
    raise ValueError(
        f"Class '{PERSON_CLASS}' "
        "tidak ditemukan pada model."
    )

missing_classes = [
    ppe
    for ppe in PPE_CLASSES
    if ppe not in model_class_names
]

if missing_classes:
    raise ValueError(
        f"PPE class tidak ditemukan: "
        f"{missing_classes}"
    )


cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW,
)

if not cap.isOpened():
    raise RuntimeError(
        "Kamera gagal dibuka."
    )


file_exists = os.path.exists(
    OUTPUT_CSV
)

csv_file = open(
    OUTPUT_CSV,
    mode="a",
    newline="",
    encoding="utf-8",
)

csv_writer = csv.writer(
    csv_file
)


if not file_exists:
    header = [
        "observation_id",
        "timestamp",
        "track_id",
        "frames_observed",
        *PPE_CLASSES,
        "violation",
    ]

    csv_writer.writerow(header)
    csv_file.flush()


observation_id = 1

if file_exists:
    try:
        with open(
            OUTPUT_CSV,
            mode="r",
            encoding="utf-8",
        ) as file:

            rows = list(
                csv.DictReader(file)
            )

            if rows:
                observation_id = (
                    int(
                        rows[-1][
                            "observation_id"
                        ]
                    )
                    + 1
                )

    except (
        OSError,
        ValueError,
        KeyError,
    ):
        observation_id = 1


telegram_thread = threading.Thread(
    target=telegram_worker,
    daemon=True,
)

telegram_thread.start()


frame_number = 0

last_telegram_send = (
    time.monotonic()
    - TELEGRAM_INTERVAL
)


print(
    "YOLO PPE Monitoring started."
)

print(
    f"Model: {MODEL_PATH}"
)

print(
    f"Tracker: {TRACKER_CONFIG}"
)

print(
    f"Image size: {IMAGE_SIZE}"
)

print(
    f"Confidence: {CONFIDENCE}"
)

print(
    "Telegram worker: ACTIVE"
)

print(
    "Press Q to exit."
)


try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print(
                "Failed to read frame."
            )
            break

        frame_number += 1

        results = model.track(
            source=frame,
            imgsz=IMAGE_SIZE,
            device=0,
            conf=CONFIDENCE,
            persist=True,
            tracker=TRACKER_CONFIG,
            verbose=False,
        )

        result = results[0]

        persons = []
        ppe_detections = []

        if (
            result.boxes is not None
            and result.boxes.id is not None
        ):
            boxes = result.boxes

            track_ids = (
                boxes.id
                .int()
                .cpu()
                .tolist()
            )

            class_ids = (
                boxes.cls
                .int()
                .cpu()
                .tolist()
            )

            confidences = (
                boxes.conf
                .cpu()
                .tolist()
            )

            xyxy = (
                boxes.xyxy
                .cpu()
                .tolist()
            )

            for (
                track_id,
                class_id,
                confidence,
                box,
            ) in zip(
                track_ids,
                class_ids,
                confidences,
                xyxy,
            ):
                class_name = str(
                    model.names[class_id]
                ).lower()

                if class_name == PERSON_CLASS:
                    persons.append(
                        {
                            "track_id": track_id,
                            "box": box,
                            "confidence": confidence,
                        }
                    )

                elif class_name in PPE_CLASSES:
                    ppe_detections.append(
                        {
                            "class": class_name,
                            "box": box,
                            "confidence": confidence,
                        }
                    )

        current_track_ids = {
            person["track_id"]
            for person in persons
        }

        ppe_assignments = (
            associate_ppe_to_persons(
                persons,
                ppe_detections,
            )
        )

        for person in persons:
            track_id = person["track_id"]

            if track_id not in tracks:
                tracks[track_id] = {
                    "first_seen": (
                        datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                    ),
                    "last_seen_frame": (
                        frame_number
                    ),
                    "frames": 0,
                    "ppe_history": {
                        ppe: []
                        for ppe in PPE_CLASSES
                    },
                }

            track_data = tracks[
                track_id
            ]

            track_data[
                "last_seen_frame"
            ] = frame_number

            track_data["frames"] += 1

            detected_ppe = (
                ppe_assignments.get(
                    track_id,
                    set(),
                )
            )

            for ppe in PPE_CLASSES:
                value = int(
                    ppe in detected_ppe
                )

                history = (
                    track_data[
                        "ppe_history"
                    ][ppe]
                )

                history.append(value)

                if (
                    len(history)
                    > MAX_HISTORY
                ):
                    history.pop(0)

        tracks_to_remove = []

        for (
            track_id,
            track_data,
        ) in list(tracks.items()):

            missing_frames = (
                frame_number
                - track_data[
                    "last_seen_frame"
                ]
            )

            if (
                missing_frames
                > MAX_MISSING_FRAMES
            ):
                observation_id = (
                    finalize_observation(
                        track_id,
                        track_data,
                        csv_writer,
                        csv_file,
                        observation_id,
                    )
                )

                tracks_to_remove.append(
                    track_id
                )

        for track_id in tracks_to_remove:
            tracks.pop(
                track_id,
                None,
            )

            violation_start_times.pop(
                track_id,
                None,
            )

            active_violation_ppe.pop(
                track_id,
                None,
            )

            reported_violation_ids.discard(
                track_id
            )

        current_time = time.monotonic()

        current_violation_tracks = set()
        current_violation_details = {}

        for person in persons:
            track_id = person["track_id"]

            detected_ppe = (
                ppe_assignments.get(
                    track_id,
                    set(),
                )
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

                if (
                    track_id
                    not in violation_start_times
                ):
                    violation_start_times[
                        track_id
                    ] = current_time

                active_violation_ppe[
                    track_id
                ] = missing_ppe

        ended_tracks = (
            set(
                violation_start_times.keys()
            )
            - current_violation_tracks
        )

        for track_id in ended_tracks:
            violation_start_times.pop(
                track_id,
                None,
            )

            active_violation_ppe.pop(
                track_id,
                None,
            )

            reported_violation_ids.discard(
                track_id
            )

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

            if (
                duration
                >= VIOLATION_MIN_DURATION
            ):
                missing_ppe = (
                    active_violation_ppe.get(
                        track_id,
                        current_violation_details.get(
                            track_id,
                            [],
                        ),
                    )
                )

                valid_violations.append(
                    (
                        track_id,
                        missing_ppe,
                        duration,
                    )
                )

        annotated_frame = result.plot()

        cv2.putText(
            annotated_frame,
            f"Tracked: {len(current_track_ids)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            annotated_frame,
            (
                f"Observations: "
                f"{observation_id - 1}"
            ),
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        if (
            valid_violations
            and (
                current_time
                - last_telegram_send
                >= TELEGRAM_INTERVAL
            )
        ):
            (
                track_id,
                missing_ppe,
                duration,
            ) = valid_violations[0]

            if (
                track_id
                not in reported_violation_ids
            ):
                try:
                    telegram_queue.put_nowait(
                        (
                            frame.copy(),
                            annotated_frame.copy(),
                            track_id,
                            missing_ppe,
                        )
                    )

                    last_telegram_send = (
                        current_time
                    )

                    reported_violation_ids.add(
                        track_id
                    )

                    print(
                        f"[VIOLATION] Queued "
                        f"Track={track_id} "
                        f"Duration={duration:.1f}s "
                        f"Missing={missing_ppe}"
                    )

                except queue.Full:
                    print(
                        "[TELEGRAM] Queue full. "
                        "Notification skipped."
                    )

        cv2.imshow(
            "YOLO PPE Monitoring",
            annotated_frame,
        )

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key == ord("q"):
            break

except KeyboardInterrupt:
    print(
        "\nProgram stopped."
    )

finally:
    for (
        track_id,
        track_data,
    ) in list(tracks.items()):

        observation_id = (
            finalize_observation(
                track_id,
                track_data,
                csv_writer,
                csv_file,
                observation_id,
            )
        )

    print(
        "Waiting for Telegram queue..."
    )

    telegram_queue.join()

    telegram_queue.put(None)

    telegram_thread.join(
        timeout=5
    )

    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()

    print(
        f"Observations saved to: "
        f"{OUTPUT_CSV}"
    )