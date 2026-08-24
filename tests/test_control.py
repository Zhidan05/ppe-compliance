import os
from pathlib import Path
from dotenv import load_dotenv
from pytapo import Tapo

# Root project
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env dari root project
load_dotenv(BASE_DIR / ".env")

CAMERA_IP = os.getenv("CAMERA_IP")
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")

if not all([CAMERA_IP, USERNAME, PASSWORD]):
    raise RuntimeError(
        "CAMERA_IP, USERNAME, atau PASSWORD belum diatur di .env"
    )

tapo = Tapo(
    CAMERA_IP,
    USERNAME,
    PASSWORD
)

print(tapo.getBasicInfo())