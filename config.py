import os

# === API Keys (from environment variables) ===
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = os.getenv("VOICE_ID")
SCULPTOK_KEY = os.getenv("SCULPTOK_KEY")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# === General Configuration ===
SAVE_DIR = "."
GRADIO_API_URL = "https://00e4dffac7d946c1bd.gradio.live"
WARPED_SHAPE = (500, 700)

# === File Paths ===
DEPTH_IMAGE_PATH = "assets/depth_map_image.png"
COLOR_IMAGE_PATH = "assets/colour_classified_image.png"
WARP_IMAGE_PATH = "assets/warp_reference_image.png"
INPUT_IMAGE_PATH = "assets/input_image.png"

# === Hardware Devices ===
VIDEO_INPUT_DEVICE = "/dev/video2"
ARDUINO_DEVICE = "/dev/ttyACM0"

# === Arduino Serial Connection (set at runtime) ===
arduino = None
