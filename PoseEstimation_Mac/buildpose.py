import cv2
import time
import threading
import numpy as np
import pandas as pd
import os
import json
from ultralytics import YOLO

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

# ── UI Constants ──────────────────────────────────────────────────────────────
WINDOW_NAME = "Live 3D Pose Estimation"
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL = cv2.FONT_HERSHEY_DUPLEX

# Color palette (BGR)
COL_BG       = (30, 30, 30)
COL_ACCENT   = (0, 200, 120)
COL_ACCENT2  = (255, 160, 50)
COL_WHITE    = (255, 255, 255)
COL_GRAY     = (160, 160, 160)
COL_RED      = (60, 60, 255)
COL_GREEN    = (60, 220, 60)
COL_DARK     = (45, 45, 45)

FRAME_W, FRAME_H = 640, 480

# ── 3D Triangulation & Camera Math ───────────────────────────────────────────

class CameraCalibration:
    """
    Camera Calibration Matrix Holder.
    Loads real intrinsic (K) and extrinsic (R, t) matrices from 'camera_calib.json' if available.
    Otherwise falls back to dummy matrices.
    """
    def __init__(self, max_cams=10):
        self.P_list = []
        
        calib_file = "camera_calib.json"
        if os.path.exists(calib_file):
            print(f"Loading real camera calibration from {calib_file}...")
            with open(calib_file, "r") as f:
                data = json.load(f)
            
            # The calibration tool saves P_list directly as lists
            p_list_data = data.get("P_list", [])
            for p in p_list_data:
                self.P_list.append(np.array(p, dtype=float))
            
            # Pad with dummy matrices if max_cams > loaded cameras
            self.K = np.array(data.get("intrinsics", [[[600, 0, 320], [0, 600, 240], [0, 0, 1]]])[0], dtype=float)
            loaded_cams = len(self.P_list)
            for i in range(loaded_cams, max_cams):
                R = np.eye(3)
                t_x = i * 1.0
                T = np.array([[t_x], [0], [0]])
                P = self.K @ np.hstack((R, T))
                self.P_list.append(P)
        else:
            print(f"Warning: {calib_file} not found. Using DUMMY calibration matrices.")
            # Dummy Intrinsic K (assuming 640x480, f=600)
            self.K = np.array([[600, 0, 320], 
                               [0, 600, 240], 
                               [0, 0, 1]], dtype=float)
            
            for i in range(max_cams):
                R = np.eye(3)
                # Default physical spread for dummy math: spread them apart
                if i == 0: t_x = 0
                elif i % 2 == 1: t_x = -((i+1)//2) * 1.0
                else: t_x = (i//2) * 1.0
                T = np.array([[t_x], [0], [0]])
                P = self.K @ np.hstack((R, T))
                self.P_list.append(P)
            
        self.P1 = self.P_list[0]
        self.P2 = self.P_list[1]


def triangulate_n_views(pts_list, P_list):
    """
    pts_list: list of N point clouds containing shape (17, 2).
    P_list: list of N 3x4 Camera Projection Matrices.
    Returns: (17, 3) 3D coordinate mapping solving AX=0.
    """
    N = len(pts_list)
    num_pts = pts_list[0].shape[0] # strictly 17
    points_3d = np.zeros((num_pts, 3))
    
    for j in range(num_pts):
        A = np.zeros((2*N, 4))
        valid_cams = 0
        for i in range(N):
            # Keypoint check against flat (0,0) failures
            if pts_list[i][j, 0] > 0 and pts_list[i][j, 1] > 0:
                P = P_list[i]
                u, v = pts_list[i][j, 0], pts_list[i][j, 1]
                A[valid_cams*2]   = u * P[2, :] - P[0, :]
                A[valid_cams*2+1] = v * P[2, :] - P[1, :]
                valid_cams += 1
                
        if valid_cams >= 2:
            A = A[:valid_cams*2]
            # Solve using Singular Value Decomposition
            U, S, Vt = np.linalg.svd(A)
            X = Vt[-1]
            if X[3] != 0:
                points_3d[j] = X[:3] / X[3]
    return points_3d

# Keypoint connections for COCO 17
SKELETON = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], 
            [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], 
            [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]

def plot_3d_pose(points_3d, width=640, height=480):
    """Render a 3D scatter plot to an OpenCV BGR image."""
    fig = plt.figure(figsize=(width/100, height/100), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    # Change view angle
    ax.view_init(elev=-80, azim=-90) 
    
    # Default range
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-1, 5]) # depth roughly
    
    # Enable axes to show 3 pole coordinate
    ax.set_xlabel('X')
    ax.set_ylabel('Depth (Y)')
    ax.set_zlabel('Height (Z)')
    
    if points_3d is not None and len(points_3d) == 17:
        # OpenCV Y is down. Matplotlib 3D: Z is UP.
        # To make it look right: X = x, Y = depth (z), Z = -y
        X = points_3d[:, 0]
        Y = points_3d[:, 2] # depth
        Z = -points_3d[:, 1]
        
        ax.scatter(X, Y, Z, c='cyan', s=20)
        
        for edge in SKELETON:
            p1, p2 = edge
            # Check if points are valid (not 0,0,0)
            if np.linalg.norm(points_3d[p1]) > 0.1 and np.linalg.norm(points_3d[p2]) > 0.1:
                ax.plot([X[p1], X[p2]], [Y[p1], Y[p2]], [Z[p1], Z[p2]], 'magenta', linewidth=2)
                    
    fig.tight_layout(pad=0)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img = buf.reshape((h, w, 4))
    plt.close(fig)
    
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    return cv2.resize(img_bgr, (width, height))

# ── Camera Stream Management ──────────────────────────────────────────────────

class CameraStream:
    """Threaded camera capture."""
    def __init__(self, src, name="Camera"):
        self.src = src
        self.name = name
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)
        
        self.grabbed, self.frame = self.cap.read()
        self.stopped = not self.grabbed
        if not self.stopped:
            self.thread = threading.Thread(target=self.update, args=())
            self.thread.daemon = True
            self.thread.start()
            
    def update(self):
        while not self.stopped:
            grabbed, frame = self.cap.read()
            if not grabbed:
                self.stopped = True
                break
            self.frame = frame
            
    def read(self):
        return self.grabbed, self.frame
        
    def stop(self):
        self.stopped = True
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        self.cap.release()

# ── Helpers & UI ─────────────────────────────────────────────────────────────

def ask_ip_address_cv2():
    """Show an on-screen IP input overlay inside a temporary OpenCV window."""
    ip_text = ""
    cursor_visible = True
    last_blink = time.time()

    # We reuse the main window if available, otherwise just use a temp canvas
    while True:
        dialog = np.zeros((480, 640, 3), dtype=np.uint8)
        dialog[:] = (35, 35, 35)
        
        cv2.rectangle(dialog, (2, 2), (637, 477), COL_ACCENT, 2)
        cv2.putText(dialog, "Add Network Camera", (30, 60), FONT, 0.8, COL_ACCENT, 2, cv2.LINE_AA)
        cv2.putText(dialog, "Enter IP Address:", (30, 120), FONT_SMALL, 0.6, COL_WHITE, 1, cv2.LINE_AA)
        
        cv2.rectangle(dialog, (30, 160), (550, 210), (60, 60, 60), -1)
        cv2.rectangle(dialog, (30, 160), (550, 210), COL_ACCENT, 2)
        
        if time.time() - last_blink > 0.5:
            cursor_visible = not cursor_visible
            last_blink = time.time()
            
        display_text = ip_text + ("|" if cursor_visible else " ")
        cv2.putText(dialog, display_text, (40, 195), FONT, 0.75, COL_WHITE, 2, cv2.LINE_AA)
        
        cv2.putText(dialog, "[Enter] Add  [Esc] Cancel", (30, 280), FONT_SMALL, 0.5, COL_GRAY, 1, cv2.LINE_AA)
        
        cv2.imshow(WINDOW_NAME, dialog)
        key = cv2.waitKey(50) & 0xFF
        if key == 27:
            return None
        elif key == 13 or key == 10:
            ip = ip_text.strip()
            if ip:
                for prefix in ["http://", "https://"]:
                    if ip.startswith(prefix):
                        ip = ip[len(prefix):]
                ip = ip.split("/")[0]
                return ip
        elif key == 8 or key == 127:
            ip_text = ip_text[:-1]
        elif 32 <= key <= 126:
            ip_text += chr(key)
    return None

def draw_info_overlay(frame, text):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), COL_DARK, -1)
    cv2.putText(frame, text, (15, 25), FONT_SMALL, 0.6, COL_WHITE, 1, cv2.LINE_AA)

def draw_controls_overlay(frame):
    hints = [
        ("Q/ESC", "Quit"),
        ("C", "Add Local Cam"),
        ("N", "Add Network Cam"),
        ("0-9", "Add Cam Index"),
        ("T", "Cycle 3D Pair"),
        ("M", "Toggle Mode (All/Pair)"),
        ("H", "Toggle Keybinds")
    ]
    h, w = frame.shape[:2]
    box_w, line_h = 240, 24
    box_h = line_h * len(hints) + 28
    x = w - box_w - 16
    y = 16
    
    cv2.rectangle(frame, (x, y), (x+box_w, y+box_h), COL_DARK, -1)
    cv2.putText(frame, "Controls", (x + 12, y + 20), FONT, 0.5, COL_ACCENT, 1, cv2.LINE_AA)
    for i, (key, desc) in enumerate(hints):
        ty = y + 42 + i * line_h
        cv2.putText(frame, f"[{key}]", (x + 12, ty), FONT_SMALL, 0.40, COL_ACCENT2, 1, cv2.LINE_AA)
        cv2.putText(frame, desc, (x + 85, ty), FONT_SMALL, 0.40, COL_WHITE, 1, cv2.LINE_AA)

def combine_grid(frames, max_cols=2):
    """Combine a list of equally sized frames into a grid."""
    if not frames:
        return np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        
    n = len(frames)
    cols = min(n, max_cols)
    rows = (n + cols - 1) // cols
    
    grid_rows = []
    for r in range(rows):
        row_frames = frames[r*cols : (r+1)*cols]
        # Pad row if needed
        while len(row_frames) < cols:
            row_frames.append(np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8))
        grid_rows.append(cv2.hconcat(row_frames))
        
    return cv2.vconcat(grid_rows)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  🏃 Multi-Cam 3D Pose Estimation — Desktop App")
    print("=" * 60)

    # ── Load Model ────────────────────────────────────────────
    model_file = "yolo26n-pose.pt"
    print(f"\n📦 Loading model: {model_file} on MPS...")
    model = YOLO(model_file)
    calib = CameraCalibration()
    
    streams = []
    
    # Try opening default cameras
    print("📷 Initializing default cameras...")
    for idx in [1, 0]: # Try continuity first, then webcam
        st = CameraStream(idx, f"Cam {idx}")
        if not st.stopped:
            streams.append(st)
            print(f"  ✓ Added Camera {idx}")
            if len(streams) >= 2: # Only auto-load up to 2 for now
                break

    recorded_3d_data = []

    print("\n🎥 Streaming started. Press 'q' to quit.")
    print("   [C] to add local camera, [N] to add IP camera.")
    
    show_controls = True
    triangulate_pair = [0, 1] # indices of streams to triangulate
    use_all_cameras = False # mode M: triangulate using all cams combined

    while True:
        # 1. Grab frames from all active streams
        frames = []
        valid_streams = []
        for st in streams:
            grabbed, frame = st.read()
            if grabbed:
                # Resize to standard size
                frame_rs = cv2.resize(frame, (FRAME_W, FRAME_H))
                frames.append(frame_rs)
                valid_streams.append(st)
            else:
                st.stop()
        
        streams = valid_streams
        frame_outputs = []
        points_2d_list = []

        # 2. Batched Inference
        if len(frames) > 0:
            # Batch inference uses single call, Metal optimized
            results = model(frames, device="mps", verbose=False)
            
            for i, res in enumerate(results):
                ann_frame = res.plot()
                draw_info_overlay(ann_frame, streams[i].name)
                frame_outputs.append(ann_frame)
                
                # Extract keypoints (Assuming 1 person for simplicity)
                kpts = res.keypoints
                if kpts is not None and len(kpts.data) > 0:
                    pts2d = kpts.data[0][:, :2].cpu().numpy() # [17, 2]
                    points_2d_list.append(pts2d)
                else:
                    points_2d_list.append(None)
                    
        # 3. 3D Triangulation & Plotting
        pts_3d = None
        info_msg = "3D View (Need 2+ cameras)"
        
        # Normalize pair bounds if a camera drops
        if triangulate_pair[1] >= len(streams):
            triangulate_pair = [0, 1]
            
        if len(streams) >= 2:
            idx1, idx2 = triangulate_pair[0], triangulate_pair[1]
            
            if use_all_cameras and len(streams) > 2:
                # Use N-camera formula
                valid_pts_list = []
                valid_p_list = []
                names_used = []
                for i in range(len(streams)):
                    if i < len(points_2d_list) and points_2d_list[i] is not None:
                        valid_pts_list.append(points_2d_list[i])
                        valid_p_list.append(calib.P_list[i])
                        names_used.append(streams[i].name)
                        
                if len(valid_pts_list) >= 2:
                    pts_3d = triangulate_n_views(valid_pts_list, valid_p_list)
                    recorded_3d_data.append(pts_3d.flatten())
                    info_msg = f"Triangulating ALL {len(valid_pts_list)} Cams (" + "|".join(names_used) + ")"
                    scale = np.max(pts_3d) - np.min(pts_3d)
                    if scale < 0.2:
                        info_msg += " - CAUTION: Distance!"
            else:
                # Use 2-camera pair
                if len(points_2d_list) > max(idx1, idx2):
                    if points_2d_list[idx1] is not None and points_2d_list[idx2] is not None:
                        pts_3d = triangulate_n_views([points_2d_list[idx1], points_2d_list[idx2]], [calib.P_list[idx1], calib.P_list[idx2]])
                        recorded_3d_data.append(pts_3d.flatten())
                        
                        # Check distance caution
                        scale = np.max(pts_3d) - np.min(pts_3d)
                        if scale < 0.2:
                            info_msg = f"Src: {streams[idx1].name} & {streams[idx2].name} - CAUTION: Move apart!"
                        else:
                            info_msg = f"Triangulating: {streams[idx1].name} & {streams[idx2].name}"
                        
        # Draw 3D Plot
        plot_img = plot_3d_pose(pts_3d, FRAME_W, FRAME_H)
        draw_info_overlay(plot_img, info_msg)
        frame_outputs.append(plot_img)
        
        # Add a placeholder if empty
        if not frames:
            empty = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
            cv2.putText(empty, "No Cameras Active", (50, 240), FONT, 1.0, COL_RED, 2)
            cv2.putText(empty, "Press [C] mapping or [N] for network", (50, 280), FONT_SMALL, 0.6, COL_WHITE, 1)
            frame_outputs = [empty]
            
        # 4. Show Grid
        final_view = combine_grid(frame_outputs, max_cols=2)
        if show_controls:
            draw_controls_overlay(final_view)
        cv2.imshow(WINDOW_NAME, final_view)
        
        # 5. Keyboard Handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord("h"):
            show_controls = not show_controls
        elif key == ord("m"):
            use_all_cameras = not use_all_cameras
            print(f"Mode toggled. Use All Cameras dynamically: {use_all_cameras}")
        elif key == ord("t") and len(streams) >= 2:
            # Cycle triangulation pair
            n = len(streams)
            idx1, idx2 = triangulate_pair
            idx2 += 1
            if idx2 >= n:
                idx1 += 1
                if idx1 >= n - 1:
                    idx1 = 0
                idx2 = idx1 + 1
            triangulate_pair = [idx1, idx2]
        elif key == ord("c"):
            # Try to add another local camera index 0-5
            added = False
            for c_idx in range(6):
                # Don't add if already streaming
                if not any(s.src == c_idx for s in streams):
                    st = CameraStream(c_idx, f"Cam {c_idx}")
                    if not st.stopped:
                        streams.append(st)
                        print(f"Added local Camera {c_idx}")
                        added = True
                        break
            if not added:
                print("No additional local cameras found.")
                
        elif key == ord("n"):
            ip = ask_ip_address_cv2()
            if ip:
                stream_urls = [
                    f"http://{ip}:8080/video",          # IPCam / IP Webcam (Android)
                    f"http://{ip}:4747/video",           # DroidCam
                    f"http://{ip}:8080/videofeed",       # IPCam (iOS)
                    f"http://{ip}:8080/live",            # Generic
                    f"http://{ip}:81/stream",            # ESP32-CAM
                    f"http://{ip}:8080/mjpegfeed",       # MJPEG variant
                    f"rtsp://{ip}:8554/live",            # RTSP generic
                ]
                print(f"\n🌐 Probing network IPs for {ip}...")
                connected = False
                for url in stream_urls:
                    print(f"  Trying {url}...")
                    st = CameraStream(url, f"IP Cam")
                    # Give it a short time to connect
                    time.sleep(0.5)
                    if not st.stopped:
                        streams.append(st)
                        print(f"  ✓ Added network camera {url}")
                        connected = True
                        break
                    else:
                        st.stop()
                if not connected:
                    print(f"  ❌ Failed to connect to any sequence on {ip}")
                    
        elif ord("0") <= key <= ord("9"):
            idx = key - ord("0")
            if not any(s.src == idx for s in streams):
                st = CameraStream(idx, f"Cam {idx}")
                if not st.stopped:
                    streams.append(st)
                    print(f"Added Camera {idx}")

    # ── Cleanup ───────────────────────────────────────────────
    for st in streams:
        st.stop()
    cv2.destroyAllWindows()
    
    if recorded_3d_data:
        print("\n💾 Saving 3D mocap data...")
        # Create column names: x0, y0, z0, x1, y1, z1 ...
        cols = []
        for i in range(17):
            cols.extend([f"x{i}", f"y{i}", f"z{i}"])
        df = pd.DataFrame(recorded_3d_data, columns=cols)
        df.to_csv("mocap_data_3d.csv", index=False)
        print(f"  ✓ Saved to mocap_data_3d.csv ({len(df)} frames)")
        
    print("👋 Session ended.")

if __name__ == "__main__":
    main()
