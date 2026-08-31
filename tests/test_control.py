import os
from pathlib import Path

from dotenv import load_dotenv
from pytapo import Tapo

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

CAMERA_IP = os.getenv("CAMERA_IP")
USERNAME = os.getenv("CAMERA_USERNAME")
PASSWORD = os.getenv("CAMERA_PASSWORD")

if not all([CAMERA_IP, USERNAME, PASSWORD]):
    raise RuntimeError("Konfigurasi kamera belum lengkap.")

print("Menghubungkan ke:", CAMERA_IP)
print("Username:", USERNAME)

tapo = Tapo(
    CAMERA_IP,
    USERNAME,
    PASSWORD,
    printDebugInformation=True
)

print(tapo.getBasicInfo())