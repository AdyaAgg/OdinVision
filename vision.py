import cv2
import numpy as np
import os
import time
import requests
import pickle
import mediapipe as mp
import pandas as pd

from arduino import send_pwm_value
from audio import play_sound, AudioFeedbackPlayer
from config import (
    DEPTH_IMAGE_PATH,
    WARP_IMAGE_PATH,
    COLOR_IMAGE_PATH,
    VIDEO_INPUT_DEVICE,
    SCULPTOK_KEY,
)


class VisionState:
    def __init__(self):
        self.depth_map = None
        self.depth_map_image = None
        self.colour_map_image = None
        self.perspective_matrix = None
        self.inverse_matrix = None
        self.painting_detected = False
        self.latest_frame = None
        self.perspective_matrix_colour = None
        self.color_overlay = None
        self.warped_shape = (500, 700)


# Load pre-trained KNN model once at the top
with open("color_classifier.pkl", "rb") as f:
    knn = pickle.load(f)


# MediaPipe Hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.1,
    min_tracking_confidence=0.1,
)


def order_points(pts):
    """Orders 4 corner points in top-left, top-right, bottom-right, bottom-left order."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def get_rectangle_contour(img):
    """Detects the largest rectangular contour in the image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest = None
    max_area = 1000
    for c in contours:
        epsilon = 0.02 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > max_area:
                biggest = approx
                max_area = area
    return biggest


def generate_depth_map_sculptok(image_path, output_path):
    """Uploads image to SculptOK and downloads generated depth map."""
    upload_url = "https://api.sculptok.com/api-open/image/upload"
    with open(image_path, "rb") as f:
        files = {"file": f}
        headers = {"apikey": SCULPTOK_KEY}
        upload_response = requests.post(upload_url, files=files, headers=headers)

    if upload_response.status_code != 200 or upload_response.json()["code"] != 0:
        raise Exception("Upload failed: " + upload_response.text)

    image_url = upload_response.json()["data"]["src"]
    submit_url = "https://api.sculptok.com/api-open/draw/prompt"
    payload = {"imageUrl": image_url, "style": "normal"}
    headers.update({"Content-Type": "application/json"})
    submit_response = requests.post(submit_url, json=payload, headers=headers)

    if submit_response.status_code != 200 or submit_response.json()["code"] != 0:
        raise Exception("Submit failed: " + submit_response.text)

    prompt_id = submit_response.json()["data"]["promptId"]
    status_url = f"https://api.sculptok.com/api-open/draw/prompt?uuid={prompt_id}"
    while True:
        status_response = requests.get(status_url, headers=headers)
        data = status_response.json().get("data", {})
        if "imgRecords" in data and len(data["imgRecords"]) >= 1:
            depth_map_url = data["imgRecords"][0]
            break
        time.sleep(5)

    depth_map_image_resp = requests.get(depth_map_url)
    with open(output_path, "wb") as f:
        f.write(depth_map_image_resp.content)


def generate_colour_map(image_path, output_path):
    """Classifies pixel colors using trained KNN and saves the result as an image."""
    df = pd.read_csv("color_training_set.csv")
    color_map = {
        row["Label"]: [row["R"], row["G"], row["B"]] for _, row in df.iterrows()
    }

    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image.")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image_rgb.reshape(-1, 3)
    predicted_labels = knn.predict(pixels)
    classified_pixels = np.array(
        [color_map[label] for label in predicted_labels], dtype=np.uint8
    )
    classified_image = classified_pixels.reshape(image.shape)
    cv2.imwrite(output_path, cv2.cvtColor(classified_image, cv2.COLOR_RGB2BGR))
    print("Classification completed. Image saved.")


def detect_painting(state: VisionState):
    """Attempts to detect a painting in the current frame and generates overlays."""
    if state.painting_detected:
        return "Already detected. Refresh or reset to try again."

    if state.latest_frame is None:
        return "No frame available yet. Please wait."

    cnt = get_rectangle_contour(state.latest_frame)
    if cnt is None:
        return "No rectangular painting found. Try again."

    pts = cnt.reshape(4, 2)
    ordered = order_points(pts)
    dst_pts = np.array(
        [
            [0, 0],
            [state.warped_shape[0] - 1, 0],
            [state.warped_shape[0] - 1, state.warped_shape[1] - 1],
            [0, state.warped_shape[1] - 1],
        ],
        dtype="float32",
    )

    state.perspective_matrix = cv2.getPerspectiveTransform(ordered, dst_pts)
    state.inverse_matrix = cv2.getPerspectiveTransform(dst_pts, ordered)

    warped = cv2.warpPerspective(
        state.latest_frame, state.perspective_matrix, state.warped_shape
    )
    cv2.imwrite(WARP_IMAGE_PATH, warped)

    if os.path.exists(DEPTH_IMAGE_PATH):
        print("[INFO] Loading existing depth map...")
    else:
        print("[INFO] Generating new depth map via SculptOK...")
        generate_depth_map_sculptok(WARP_IMAGE_PATH, DEPTH_IMAGE_PATH)

    state.depth_map_image = cv2.imread(DEPTH_IMAGE_PATH)
    state.depth_map_image = cv2.resize(state.depth_map_image, state.warped_shape)
    state.depth_map = cv2.cvtColor(state.depth_map_image, cv2.COLOR_BGR2GRAY)

    if os.path.exists(COLOR_IMAGE_PATH):
        print("[INFO] Loading existing color map...")
    else:
        print("[INFO] Generating new color classified image...")
        generate_colour_map(WARP_IMAGE_PATH, COLOR_IMAGE_PATH)

    state.colour_map_image = cv2.imread(COLOR_IMAGE_PATH)
    state.colour_map_image = cv2.resize(state.colour_map_image, state.warped_shape)
    state.perspective_matrix_colour = cv2.getPerspectiveTransform(ordered, dst_pts)
    state.color_overlay = cv2.warpPerspective(
        state.colour_map_image, state.perspective_matrix_colour, state.warped_shape
    )

    state.painting_detected = True
    return "Painting detected. Depth and colour maps ready."


def get_classified_color(x, y, colour_map_image):
    bgr = colour_map_image[y, x]
    rgb = bgr[::-1]
    return knn.predict([rgb])[0]


def get_vibration_intensity(x, y, depth_map):
    if 0 <= x < depth_map.shape[1] and 0 <= y < depth_map.shape[0]:
        depth_value = depth_map[y, x] / 255.0
        if depth_value <= 0.01:
            return 0
        return (depth_value**1.5) * (1.0 - (1.0 - depth_value) ** 1.5)
    return 0


def stream_live(state: VisionState):
    """Streams live video, detects hand position, and overlays depth/color maps."""
    cap = cv2.VideoCapture(VIDEO_INPUT_DEVICE)
    audio_player = AudioFeedbackPlayer()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            state.latest_frame = frame.copy()
            frame_normal = frame.copy()
            display_frame = frame.copy()

            if state.painting_detected:
                overlay = frame.copy()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                # Overlay classified color
                if (
                    state.inverse_matrix is not None
                    and state.colour_map_image is not None
                ):
                    projected_colour = cv2.warpPerspective(
                        state.colour_map_image,
                        state.inverse_matrix,
                        (frame.shape[1], frame.shape[0]),
                    )
                    mask_colour = cv2.warpPerspective(
                        np.ones(state.colour_map_image.shape[:2], dtype=np.uint8) * 255,
                        state.inverse_matrix,
                        (frame.shape[1], frame.shape[0]),
                    )
                    mask_3c_colour = cv2.merge([mask_colour] * 3)
                    inv_mask_colour = cv2.bitwise_not(mask_3c_colour)
                    background_colour = cv2.bitwise_and(frame_normal, inv_mask_colour)
                    foreground_colour = cv2.bitwise_and(
                        projected_colour, mask_3c_colour
                    )
                    frame_normal = cv2.add(background_colour, foreground_colour)

                # Overlay depth image
                if (
                    state.inverse_matrix is not None
                    and state.depth_map_image is not None
                ):
                    projected = cv2.warpPerspective(
                        state.depth_map_image,
                        state.inverse_matrix,
                        (frame.shape[1], frame.shape[0]),
                    )
                    mask = cv2.warpPerspective(
                        np.ones(state.depth_map_image.shape[:2], dtype=np.uint8) * 255,
                        state.inverse_matrix,
                        (frame.shape[1], frame.shape[0]),
                    )
                    mask_3c = cv2.merge([mask] * 3)
                    inv_mask = cv2.bitwise_not(mask_3c)
                    background = cv2.bitwise_and(overlay, inv_mask)
                    foreground = cv2.bitwise_and(projected, mask_3c)
                    overlay = cv2.add(background, foreground)

                # Hand tracking and pointer feedback
                if results.multi_hand_landmarks:
                    for handLms in results.multi_hand_landmarks:
                        h, w, _ = frame.shape
                        cx = int(handLms.landmark[8].x * w)
                        cy = int(handLms.landmark[8].y * h)

                        if (
                            state.perspective_matrix_colour is not None
                            and state.color_overlay is not None
                        ):
                            src_pt = np.array([[[cx, cy]]], dtype="float32")
                            warped_pt = cv2.perspectiveTransform(
                                src_pt, state.perspective_matrix_colour
                            )
                            x_warped, y_warped = warped_pt[0][0].astype(int)
                            x_clamped = np.clip(
                                x_warped, 0, state.color_overlay.shape[1] - 1
                            )
                            y_clamped = np.clip(
                                y_warped, 0, state.color_overlay.shape[0] - 1
                            )
                            color_name = get_classified_color(
                                x_clamped, y_clamped, state.color_overlay
                            )
                            audio_player.play_sound(color_name)

                        if state.perspective_matrix is not None:
                            src_pt = np.array([[[cx, cy]]], dtype="float32")
                            warped_pt = cv2.perspectiveTransform(
                                src_pt, state.perspective_matrix
                            )
                            x_w, y_w = warped_pt[0][0].astype(int)
                            intensity = get_vibration_intensity(
                                x_w, y_w, state.depth_map
                            )
                            pwm_value = int(np.clip(intensity * 255, 0, 255))
                            send_pwm_value(pwm_value)

                        cv2.circle(overlay, (cx, cy), 10, (0, 255, 0), -1)
                        cv2.circle(frame_normal, (cx, cy), 10, (0, 255, 0), -1)

                display_frame = overlay

            else:
                cnt = get_rectangle_contour(frame_normal)
                if cnt is not None:
                    cv2.drawContours(display_frame, [cnt], -1, (0, 255, 0), 3)
                    cv2.drawContours(frame_normal, [cnt], -1, (0, 255, 0), 3)

            frame_normal_rgb = cv2.cvtColor(frame_normal, cv2.COLOR_BGR2RGB)
            display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

            yield frame_normal_rgb, display_frame_rgb

    finally:
        cap.release()
