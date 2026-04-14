import cv2
import time
import pandas as pd
import numpy as np
from ultralytics import YOLO


# ── UI Constants ──────────────────────────────────────────────────────────────
WINDOW_NAME = "Live Pose Estimation"
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL = cv2.FONT_HERSHEY_DUPLEX

# Color palette (BGR)
COL_BG       = (30, 30, 30)
COL_ACCENT   = (0, 200, 120)      # green accent
COL_ACCENT2  = (255, 160, 50)     # orange accent
COL_WHITE    = (255, 255, 255)
COL_GRAY     = (160, 160, 160)
COL_RED      = (60, 60, 255)
COL_GREEN    = (60, 220, 60)
COL_DARK     = (45, 45, 45)
COL_PANEL    = (40, 40, 40)
COL_YELLOW   = (0, 220, 255)

# ── Saved Camera Sources ──────────────────────────────────────────────────────
# Maps slot key (a-f) to {"url": str, "label": str}
# Slots let users save IP cameras and switch back to them with a single keypress.
SLOT_KEYS = ['a', 'b', 'd', 'e', 'f', 'g']  # available slot keys (skip c,h,n,q,0-9)
saved_sources = {}  # populated at runtime


def get_next_slot_key():
    """Return the next available slot key, or None if all full."""
    for k in SLOT_KEYS:
        if k not in saved_sources:
            return k
    return None


def save_source(url, label):
    """Save a camera URL to the next available slot. Returns the slot key."""
    # Check if this URL is already saved
    for k, v in saved_sources.items():
        if v["url"] == url:
            return k  # already saved
    key = get_next_slot_key()
    if key:
        saved_sources[key] = {"url": url, "label": label}
        print(f"  💾 Saved as slot [{key.upper()}] for quick switching.")
    return key


def draw_rounded_rect(img, pt1, pt2, color, radius=12, thickness=-1, alpha=0.75):
    """Draw a semi-transparent rounded rectangle."""
    overlay = img.copy()
    x1, y1 = pt1
    x2, y2 = pt2
    # Draw filled rounded rectangle on overlay
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, thickness)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_fps_badge(frame, fps):
    """Draw a sleek FPS badge in the top-left corner."""
    fps_text = f"{fps:.1f} FPS"
    (tw, th), _ = cv2.getTextSize(fps_text, FONT, 0.7, 2)
    pad_x, pad_y = 14, 10
    x, y = 16, 16
    draw_rounded_rect(
        frame,
        (x, y),
        (x + tw + pad_x * 2, y + th + pad_y * 2),
        COL_DARK, radius=10, alpha=0.8
    )
    # FPS value
    cv2.putText(frame, fps_text, (x + pad_x, y + pad_y + th),
                FONT, 0.7, COL_ACCENT, 2, cv2.LINE_AA)


def draw_info_bar(frame, model_name, cam_source, cam_connected, num_persons):
    """Draw a bottom info bar with model & camera status."""
    h, w = frame.shape[:2]
    bar_h = 44
    draw_rounded_rect(frame, (0, h - bar_h), (w, h), COL_PANEL, radius=0, alpha=0.85)

    # Model name
    cv2.putText(frame, f"Model: {model_name}", (16, h - 14),
                FONT_SMALL, 0.5, COL_WHITE, 1, cv2.LINE_AA)

    # Camera status
    status_color = COL_GREEN if cam_connected else COL_RED
    cv2.circle(frame, (w // 2 - 10, h - 22), 6, status_color, -1, cv2.LINE_AA)
    cv2.putText(frame, cam_source, (w // 2 + 4, h - 14),
                FONT_SMALL, 0.5, COL_WHITE, 1, cv2.LINE_AA)

    # Person count
    person_text = f"Persons: {num_persons}"
    (tw, _), _ = cv2.getTextSize(person_text, FONT_SMALL, 0.5, 1)
    cv2.putText(frame, person_text, (w - tw - 16, h - 14),
                FONT_SMALL, 0.5, COL_ACCENT2, 1, cv2.LINE_AA)


def draw_controls_overlay(frame):
    """Draw a small controls hint in the top-right corner, including saved sources."""
    hints = [
        ("Q", "Quit"),
        ("C", "Scan Local Cameras"),
        ("N", "Network IP Camera"),
        ("0-9", "Select Camera Index"),
        ("H", "Toggle This Panel"),
    ]
    # Add saved source slots
    saved_hints = []
    for k in SLOT_KEYS:
        if k in saved_sources:
            saved_hints.append((k.upper(), saved_sources[k]["label"]))

    h, w = frame.shape[:2]
    box_w, line_h = 270, 24
    total_lines = len(hints) + (len(saved_hints) + 1 if saved_hints else 0)
    box_h = line_h * total_lines + 28
    x = w - box_w - 16
    y = 16
    draw_rounded_rect(frame, (x, y), (x + box_w, y + box_h), COL_DARK, radius=10, alpha=0.78)

    cv2.putText(frame, "Controls", (x + 12, y + 20),
                FONT, 0.5, COL_ACCENT, 1, cv2.LINE_AA)
    row = 0
    for i, (key, desc) in enumerate(hints):
        ty = y + 42 + row * line_h
        cv2.putText(frame, f"[{key}]", (x + 12, ty),
                    FONT_SMALL, 0.40, COL_ACCENT2, 1, cv2.LINE_AA)
        cv2.putText(frame, desc, (x + 68, ty),
                    FONT_SMALL, 0.40, COL_GRAY, 1, cv2.LINE_AA)
        row += 1

    if saved_hints:
        row += 1  # spacer
        ty = y + 42 + (row - 1) * line_h
        cv2.putText(frame, "Saved Sources", (x + 12, ty + 4),
                    FONT, 0.42, COL_YELLOW, 1, cv2.LINE_AA)
        for key, desc in saved_hints:
            ty = y + 42 + row * line_h
            cv2.putText(frame, f"[{key}]", (x + 12, ty),
                        FONT_SMALL, 0.40, COL_YELLOW, 1, cv2.LINE_AA)
            cv2.putText(frame, desc, (x + 68, ty),
                        FONT_SMALL, 0.40, COL_WHITE, 1, cv2.LINE_AA)
            row += 1


def try_open_camera(index):
    """Attempt to open a camera at the given index. Returns (cap, success)."""
    print(f"  Trying camera index {index}...")
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        ret, _ = cap.read()
        if ret:
            print(f"  ✓ Camera {index} connected successfully.")
            return cap, True
        cap.release()
    print(f"  ✗ Camera {index} not available.")
    return None, False


def connect_iphone_camera(current_cap):
    """Scan camera indices 0-5 to find an iPhone / external camera."""
    print("\n🔍 Scanning for iPhone Continuity Camera...")
    # Release current capture first
    if current_cap is not None:
        current_cap.release()

    # Typically iPhone Continuity Camera appears at index 1 or 2
    for idx in [1, 2, 3, 4, 5, 0]:
        cap, ok = try_open_camera(idx)
        if ok:
            print(f"  📱 Connected to camera at index {idx}")
            return cap, idx, f"Cam {idx}"

    print("  ❌ No camera found! Please check connections.")
    return None, -1, "No Camera"


def show_error_overlay(message, sub_lines=None, duration=3.0):
    """Show a full-screen error overlay for the given duration."""
    start = time.time()
    while time.time() - start < duration:
        dialog = np.zeros((400, 600, 3), dtype=np.uint8)
        dialog[:] = (25, 25, 30)
        cv2.rectangle(dialog, (2, 2), (597, 397), COL_RED, 3)

        # Error icon
        cv2.putText(dialog, "CONNECTION FAILED", (100, 60),
                    FONT, 1.0, COL_RED, 2, cv2.LINE_AA)

        # Main message
        cv2.putText(dialog, message, (30, 120),
                    FONT_SMALL, 0.6, COL_WHITE, 1, cv2.LINE_AA)

        # Sub-lines (details)
        if sub_lines:
            for i, line in enumerate(sub_lines):
                cv2.putText(dialog, line, (30, 165 + i * 30),
                            FONT_SMALL, 0.48, COL_GRAY, 1, cv2.LINE_AA)

        # Countdown
        remaining = max(0, duration - (time.time() - start))
        cv2.putText(dialog, f"Returning in {remaining:.0f}s... (or press any key)",
                    (100, 370), FONT_SMALL, 0.45, COL_GRAY, 1, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, dialog)
        if cv2.waitKey(50) & 0xFF != 255:
            break  # any keypress dismisses


def ask_ip_address_cv2(error_msg=None):
    """Show an on-screen IP input overlay inside the OpenCV window. Returns IP or None."""
    ip_text = ""
    cursor_visible = True
    last_blink = time.time()
    error_display = error_msg  # show previous error if retrying
    error_fade_time = time.time() + 5.0 if error_msg else 0

    while True:
        # Create the input dialog frame
        dialog = np.zeros((380, 580, 3), dtype=np.uint8)
        dialog[:] = (35, 35, 35)

        # Border — red if there's an active error, green otherwise
        border_col = COL_RED if (error_display and time.time() < error_fade_time) else COL_ACCENT
        cv2.rectangle(dialog, (2, 2), (577, 377), border_col, 2)

        # Title
        cv2.putText(dialog, "Connect to Network Camera", (30, 45),
                    FONT, 0.8, COL_ACCENT, 2, cv2.LINE_AA)

        # Instructions
        cv2.putText(dialog, "Enter the iPhone IP address:", (30, 90),
                    FONT_SMALL, 0.55, COL_WHITE, 1, cv2.LINE_AA)
        cv2.putText(dialog, "(from the IP camera app, e.g. 192.168.1.105)", (30, 118),
                    FONT_SMALL, 0.45, COL_GRAY, 1, cv2.LINE_AA)

        # Input box
        box_col = COL_RED if (error_display and time.time() < error_fade_time) else COL_ACCENT
        cv2.rectangle(dialog, (28, 140), (550, 185), (60, 60, 60), -1)
        cv2.rectangle(dialog, (28, 140), (550, 185), box_col, 2)

        # Blinking cursor
        if time.time() - last_blink > 0.5:
            cursor_visible = not cursor_visible
            last_blink = time.time()
        display_text = ip_text + ("|" if cursor_visible else " ")
        cv2.putText(dialog, display_text, (38, 172),
                    FONT, 0.75, COL_WHITE, 2, cv2.LINE_AA)

        # Error message display
        if error_display and time.time() < error_fade_time:
            cv2.putText(dialog, error_display, (30, 215),
                        FONT_SMALL, 0.48, COL_RED, 1, cv2.LINE_AA)
            hint_y = 250
        else:
            error_display = None
            hint_y = 230

        # Hints
        cv2.putText(dialog, "[Enter]  Connect", (30, hint_y),
                    FONT_SMALL, 0.5, COL_ACCENT2, 1, cv2.LINE_AA)
        cv2.putText(dialog, "[Esc]    Cancel", (30, hint_y + 28),
                    FONT_SMALL, 0.5, COL_GRAY, 1, cv2.LINE_AA)
        cv2.putText(dialog, "[Bksp]   Delete character", (30, hint_y + 56),
                    FONT_SMALL, 0.5, COL_GRAY, 1, cv2.LINE_AA)

        # Show saved sources hint
        if saved_sources:
            sy = hint_y + 90
            cv2.putText(dialog, "Saved sources (press key to switch):",
                        (30, sy), FONT_SMALL, 0.42, COL_YELLOW, 1, cv2.LINE_AA)
            for k in SLOT_KEYS:
                if k in saved_sources:
                    sy += 22
                    cv2.putText(dialog, f"[{k.upper()}] {saved_sources[k]['label']}",
                                (45, sy), FONT_SMALL, 0.40, COL_GRAY, 1, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, dialog)
        key = cv2.waitKey(50) & 0xFF

        if key == 27:  # Escape
            return None
        elif key == 13 or key == 10:  # Enter
            ip = ip_text.strip()
            if ip:
                # Clean up the IP
                if ip.startswith("http://"):
                    ip = ip[7:]
                if ip.startswith("https://"):
                    ip = ip[8:]
                ip = ip.split("/")[0]
                return ip
        elif key == 8 or key == 127:  # Backspace / Delete
            ip_text = ip_text[:-1]
        elif 32 <= key <= 126:  # Printable ASCII
            if len(ip_text) < 45:  # max length guard
                ip_text += chr(key)

    return None


def show_connecting_overlay(ip, current_url, url_index, total_urls):
    """Show a 'connecting...' progress screen."""
    dialog = np.zeros((300, 580, 3), dtype=np.uint8)
    dialog[:] = (30, 30, 35)
    cv2.rectangle(dialog, (2, 2), (577, 297), COL_ACCENT2, 2)

    cv2.putText(dialog, f"Connecting to {ip}...", (30, 50),
                FONT, 0.75, COL_ACCENT2, 2, cv2.LINE_AA)

    # Progress bar
    bar_x, bar_y, bar_w, bar_h = 30, 80, 520, 20
    cv2.rectangle(dialog, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), COL_DARK, -1)
    fill_w = int(bar_w * (url_index + 1) / total_urls)
    cv2.rectangle(dialog, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), COL_ACCENT2, -1)

    cv2.putText(dialog, f"Trying: {current_url}", (30, 130),
                FONT_SMALL, 0.45, COL_GRAY, 1, cv2.LINE_AA)
    cv2.putText(dialog, f"Attempt {url_index + 1} of {total_urls}", (30, 160),
                FONT_SMALL, 0.45, COL_WHITE, 1, cv2.LINE_AA)

    cv2.imshow(WINDOW_NAME, dialog)
    cv2.waitKey(1)


def connect_network_camera(current_cap):
    """Prompt user for an IP and try common stream URL patterns."""
    ip = ask_ip_address_cv2()
    if not ip:
        print("  ⊘ Cancelled by user.")
        return current_cap, None, None

    # Common stream URL patterns used by popular IP camera apps
    stream_urls = [
        f"http://{ip}:8080/video",          # IPCam / IP Webcam (Android)
        f"http://{ip}:4747/video",           # DroidCam
        f"http://{ip}:8080/videofeed",       # IPCam (iOS)
        f"http://{ip}:8080/live",            # Generic
        f"http://{ip}:81/stream",            # ESP32-CAM / some iOS apps
        f"http://{ip}:8080/mjpegfeed",       # MJPEG variant
        f"rtsp://{ip}:8554/live",            # RTSP generic
    ]

    print(f"\n🌐 Trying to connect to camera at {ip}...")

    # Release current capture
    if current_cap is not None:
        current_cap.release()

    for i, url in enumerate(stream_urls):
        show_connecting_overlay(ip, url, i, len(stream_urls))
        print(f"  Trying {url} ...")
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)

        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"  ✓ Connected via: {url}")
                source_label = f"IP: {ip}"
                # Save to a slot for quick switching
                save_source(url, source_label)
                return cap, url, source_label
            cap.release()
        print(f"  ✗ Failed.")

    # ── Connection failed — show on-screen error ──────────────
    error_lines = [
        f"Could not connect to any stream on: {ip}",
        "",
        "Please check:",
        "  1. IP camera app is running on the phone",
        "  2. Both devices are on the same WiFi",
        f"  3. The IP address '{ip}' is correct",
        "",
        "Tried 7 common stream URL patterns.",
    ]
    show_error_overlay(
        f"No stream found at {ip}",
        sub_lines=[
            "Ensure the IP camera app is running on the phone",
            "Both devices must be on the same WiFi network",
            f"Verify the IP address: {ip}",
            "Tried 7 common URL patterns (port 8080, 4747, etc.)",
        ],
        duration=5.0,
    )
    print(f"\n  ❌ Could not connect to any stream on {ip}")
    return None, None, None


def open_saved_source(key, current_cap):
    """Open a previously saved camera source by slot key."""
    if key not in saved_sources:
        return current_cap, None, None

    src = saved_sources[key]
    url = src["url"]
    label = src["label"]
    print(f"\n🔄 Switching to saved source [{key.upper()}]: {label} ({url})")

    if current_cap is not None:
        current_cap.release()

    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)

    if cap.isOpened():
        ret, _ = cap.read()
        if ret:
            print(f"  ✓ Reconnected to {label}")
            return cap, url, label
        cap.release()

    # Failed to reconnect
    show_error_overlay(
        f"Failed to reconnect to {label}",
        sub_lines=[
            f"URL: {url}",
            "The camera may have been turned off or IP changed.",
            "Press [N] to enter a new IP address.",
        ],
        duration=3.0,
    )
    print(f"  ❌ Failed to reconnect to saved source [{key.upper()}]")
    return None, None, None


def main():
    print("=" * 60)
    print("  🏃 Live Pose Estimation — Desktop App")
    print("=" * 60)

    # ── Load Model ────────────────────────────────────────────
    model_file = "yolo26n-pose.pt"
    print(f"\n📦 Loading model: {model_file} ...")
    model = YOLO(model_file)
    model_name = "YOLOv26n-Pose"
    print("  ✓ Model loaded.\n")

    # ── Open Default Camera ───────────────────────────────────
    print("📷 Initializing camera...")
    cam_index = 1  # default: try iPhone first
    cam_source = f"Cam {cam_index}"  # display label for the info bar
    cap, cam_ok = try_open_camera(cam_index)
    if not cam_ok:
        cam_index = 0
        cap, cam_ok = try_open_camera(cam_index)
        cam_source = f"Cam {cam_index}"

    if not cam_ok:
        print("\n⚠  No local camera found — starting in disconnected mode.")
        print("   Press [C] to scan for cameras or [N] to connect via network IP.\n")
        cap = None
        cam_source = "No Camera"
    else:
        print(f"\n🎥 Streaming from camera {cam_index}. Press 'q' to quit.\n")

    # ── FPS tracking ──────────────────────────────────────────
    fps = 0.0
    frame_times = []
    show_controls = True

    # ── Main Loop ─────────────────────────────────────────────
    while True:
        t_start = time.perf_counter()

        if cap is None or not cap.isOpened():
            # Show a "no camera" placeholder
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "No Camera Connected", (100, 160),
                        FONT, 1.0, COL_RED, 2, cv2.LINE_AA)
            cv2.putText(placeholder, "Press [C] to scan for local cameras", (80, 220),
                        FONT_SMALL, 0.6, COL_GRAY, 1, cv2.LINE_AA)
            cv2.putText(placeholder, "Press [N] to connect via Network IP", (80, 252),
                        FONT_SMALL, 0.6, COL_ACCENT2, 1, cv2.LINE_AA)
            cv2.putText(placeholder, "Press [0]-[9] to select camera index", (80, 284),
                        FONT_SMALL, 0.6, COL_GRAY, 1, cv2.LINE_AA)
            # Show saved sources
            if saved_sources:
                sy = 330
                cv2.putText(placeholder, "Saved Sources:", (80, sy),
                            FONT_SMALL, 0.55, COL_YELLOW, 1, cv2.LINE_AA)
                for k in SLOT_KEYS:
                    if k in saved_sources:
                        sy += 28
                        cv2.putText(placeholder,
                                    f"[{k.upper()}]  {saved_sources[k]['label']}",
                                    (95, sy), FONT_SMALL, 0.52, COL_WHITE, 1, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, placeholder)
            key = cv2.waitKey(100) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                cap, cam_index, cam_source = connect_iphone_camera(cap)
                cam_ok = cap is not None
            elif key == ord("n"):
                cap, _, src = connect_network_camera(cap)
                if cap is not None:
                    cam_source = src
                    cam_ok = True
            elif ord("0") <= key <= ord("9"):
                idx = key - ord("0")
                if cap is not None:
                    cap.release()
                cap, cam_ok = try_open_camera(idx)
                if cam_ok:
                    cam_index = idx
                    cam_source = f"Cam {idx}"
            else:
                # Check saved source slot keys
                pressed = chr(key) if 32 <= key <= 126 else ""
                if pressed.lower() in saved_sources:
                    cap, _, src = open_saved_source(pressed.lower(), cap)
                    if cap is not None:
                        cam_source = src
                        cam_ok = True
            continue

        success, frame = cap.read()
        if not success:
            print("⚠  Frame read failed. Camera may have disconnected.")
            cap.release()
            cap = None
            cam_ok = False
            continue

        # ── Inference ─────────────────────────────────────────
        results = model(frame, device="mps", verbose=False)
        annotated_frame = results[0].plot()

        # Count detected persons
        num_persons = len(results[0].keypoints) if results[0].keypoints is not None else 0

        # ── FPS Calculation (rolling average) ─────────────────
        t_end = time.perf_counter()
        frame_times.append(t_end - t_start)
        if len(frame_times) > 30:
            frame_times.pop(0)
        fps = len(frame_times) / sum(frame_times) if frame_times else 0

        # ── Draw HUD ──────────────────────────────────────────
        draw_fps_badge(annotated_frame, fps)
        draw_info_bar(annotated_frame, model_name, cam_source, True, num_persons)
        if show_controls:
            draw_controls_overlay(annotated_frame)

        cv2.imshow(WINDOW_NAME, annotated_frame)

        # ── Keyboard Input ────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            cap, cam_index, cam_source = connect_iphone_camera(cap)
            cam_ok = cap is not None
        elif key == ord("n"):
            cap, _, src = connect_network_camera(cap)
            if cap is not None:
                cam_source = src
                cam_ok = True
            else:
                cam_ok = False
        elif key == ord("h"):
            show_controls = not show_controls
        elif ord("0") <= key <= ord("9"):
            idx = key - ord("0")
            if cap is not None:
                cap.release()
            cap, cam_ok = try_open_camera(idx)
            if cam_ok:
                cam_index = idx
                cam_source = f"Cam {idx}"
            else:
                cap = None
                cam_source = "No Camera"
        else:
            # Check saved source slot keys
            pressed = chr(key) if 32 <= key <= 126 else ""
            if pressed.lower() in saved_sources:
                cap_new, _, src = open_saved_source(pressed.lower(), cap)
                if cap_new is not None:
                    cap = cap_new
                    cam_source = src
                    cam_ok = True
                    frame_times.clear()  # reset FPS counter
                else:
                    cap = None
                    cam_ok = False
                    cam_source = "No Camera"

    # ── Save data ───────────────────────────────────────
    data = []
    for frame in results:
        keypoints = frame.keypoints.data # Tọa độ x, y, z từ YOLOv26
        data.append(keypoints.flatten())

    df = pd.DataFrame(data)
    df.to_csv("mocap_data_2026.csv")

    # ── Cleanup ───────────────────────────────────────────────
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print("\n👋 Session ended.")


if __name__ == "__main__":
    main()
