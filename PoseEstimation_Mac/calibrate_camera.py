import cv2
import numpy as np
import time
import os
import json

# ── Checkerboard Settings ───────────────────────────────────────────────────
CHECKERBOARD = (5, 4) # Inner corners (rows, cols) based on your 5x5 squares
SQUARE_SIZE = 0.050   # Estimated size of a square in meters (50mm on A3 paper)

# Termination criteria for corner sub-pixel accuracy
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

def main():
    print("=" * 60)
    print("  📷 Stereo Camera Calibration Tool")
    print("=" * 60)
    print(f"Checkerboard size: {CHECKERBOARD} (inner corners)")
    print(f"Square size: {SQUARE_SIZE} meters")
    
    # Ask user for camera sources
    print("\n--- Camera Configuration ---")
    print("You can use local webcams (e.g., 0 or 1) OR network cameras like DroidCam (e.g., http://192.168.1.x:4747/video)")
    
    src0 = input("Enter Camera 1 source [default: 0]: ").strip()
    if not src0: src0 = 0
    elif src0.isdigit(): src0 = int(src0)
    
    src1 = input("Enter Camera 2 source [default: 1]: ").strip()
    if not src1: src1 = 1
    elif src1.isdigit(): src1 = int(src1)

    print(f"\nConnecting to Cam 1: {src0}")
    cap0 = cv2.VideoCapture(src0)
    cap0.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    
    print(f"Connecting to Cam 2: {src1}")
    cap1 = cv2.VideoCapture(src1)
    cap1.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        
    if not cap0.isOpened() or not cap1.isOpened():
        print("Error: Could not open one or both camera streams.")
        if cap0.isOpened(): cap0.release()
        if cap1.isOpened(): cap1.release()
        return

    print("\nControls:")
    print("  [SPACE] - Capture a frame for calibration")
    print("  [ C ]   - Compute calibration (needs at least 10 pairs)")
    print("  [ Q ]   - Quit without saving")

    objpoints = [] # 3d point in real world space
    imgpoints_left = [] # 2d points in image plane.
    imgpoints_right = [] # 2d points in image plane.

    pairs_captured = 0

    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        
        if not ret0 or not ret1:
            print("Failed to grab frames.")
            break

        # Resize for consistent viewing/processing
        frame0 = cv2.resize(frame0, (640, 480))
        frame1 = cv2.resize(frame1, (640, 480))

        gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners (fast check)
        ret0_cb, corners0 = cv2.findChessboardCorners(gray0, CHECKERBOARD, None)
        ret1_cb, corners1 = cv2.findChessboardCorners(gray1, CHECKERBOARD, None)

        disp0 = frame0.copy()
        disp1 = frame1.copy()

        # Draw corners if found
        if ret0_cb:
            cv2.drawChessboardCorners(disp0, CHECKERBOARD, corners0, ret0_cb)
        if ret1_cb:
            cv2.drawChessboardCorners(disp1, CHECKERBOARD, corners1, ret1_cb)

        # Show status
        status_color = (0, 255, 0) if (ret0_cb and ret1_cb) else (0, 0, 255)
        cv2.putText(disp0, f"Pairs captured: {pairs_captured}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(disp0, "Cam 0", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(disp1, "Ready to capture!" if (ret0_cb and ret1_cb) else "Searching board...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(disp1, "Cam 1", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        combined = cv2.hconcat([disp0, disp1])
        cv2.imshow('Stereo Calibration', combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == 32: # SPACE
            if ret0_cb and ret1_cb:
                # Refine corners
                corners0_sub = cv2.cornerSubPix(gray0, corners0, (11, 11), (-1, -1), criteria)
                corners1_sub = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)

                objpoints.append(objp)
                imgpoints_left.append(corners0_sub)
                imgpoints_right.append(corners1_sub)

                pairs_captured += 1
                print(f"Captured pair {pairs_captured}")
                
                # Flash screen white
                flash = np.ones_like(combined) * 255
                cv2.imshow('Stereo Calibration', flash)
                cv2.waitKey(100)
            else:
                print("Checkerboard not fully visible in BOTH cameras.")

        elif key == ord('c'):
            if pairs_captured < 5:
                print("Need at least 5 pairs! Recommended > 15.")
                continue
                
            print("\nComputing calibration... Please wait.")
            
            # 1. Calibrate each camera individually first
            ret0, mtx0, dist0, rvecs0, tvecs0 = cv2.calibrateCamera(objpoints, imgpoints_left, gray0.shape[::-1], None, None)
            ret1, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints, imgpoints_right, gray1.shape[::-1], None, None)
            
            print(f"Cam 0 RMS Error: {ret0:.3f}")
            print(f"Cam 1 RMS Error: {ret1:.3f}")

            # 2. Stereo calibrate to get rotation and translation between cameras
            flags = cv2.CALIB_FIX_INTRINSIC
            ret_stereo, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
                objpoints, imgpoints_left, imgpoints_right, 
                mtx0, dist0, mtx1, dist1, 
                gray0.shape[::-1], criteria=criteria, flags=flags)

            print(f"Stereo RMS Error: {ret_stereo:.3f}")

            # 3. Compute Projection Matrices P1, P2
            # P1 = K1 @ [I | 0]
            P1 = mtx0 @ np.hstack((np.eye(3), np.zeros((3, 1))))
            
            # P2 = K2 @ [R | T]
            P2 = mtx1 @ np.hstack((R, T))

            calib_data = {
                "P_list": [P1.tolist(), P2.tolist()],
                "intrinsics": [mtx0.tolist(), mtx1.tolist()],
                "distortion": [dist0.tolist(), dist1.tolist()],
                "R": R.tolist(),
                "T": T.tolist(),
                "rms_error": ret_stereo
            }

            with open("camera_calib.json", "w") as f:
                json.dump(calib_data, f, indent=4)
                
            print("\n✅ Calibration saved successfully to 'camera_calib.json'")
            break

    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
