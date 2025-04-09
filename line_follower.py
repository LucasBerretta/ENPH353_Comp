#!/usr/bin/env python3

import rospy
import cv2 as cv
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

# ---------------- Global Variables ----------------
bridge = CvBridge()
latest_image = None
redcar_seen = False

# ---------------- FIRST PID (PID1) PARAMETERS ----------------
LINEAR_SPEED_NORMAL = 2.5
STEERING_GAIN = 0.27
BINARY_THRESHOLD = 150  # Grayscale threshold for the first PID
ROI_Y_PERCENTAGE = 0.65

# ---------------- SECOND PID (PID2) PARAMETERS ----------------
LINEAR_SPEED_PID2 = 2.65
STEERING_GAIN_PID2 = 0.12

PID2_ARC_CCW_DURATION = 1.0
PID2_ARC_CCW_LINEAR_SPEED = 0.8
PID2_ARC_CCW_ANGULAR_SPEED = 0.3
PID2_ROTATE_CCW_ANGULAR_SPEED = 1.0

# ---------------- RGB DEBUG (no PID, just viewing) -----------
BGR_LOWER_WHITE = np.array([122, 155, 150])   # e.g. min: fairly bright
BGR_UPPER_WHITE = np.array([255, 255, 255])   # e.g. max: pure white
ROI_Y_PERCENTAGE_2 = 0.71
CLOSE_KERNEL_SIZE = 2
OPEN_KERNEL_SIZE = 1

# ---------------- Pink-Line Behavior (initial) ----------------
PINK_DETECTION_DELAY = 0.5
PINK_FORWARD_DURATION = 0.55
PINK_FORWARD_SPEED = 2.0

#HSV PID3
PID3_H_LOWER = 15
PID3_S_LOWER = 47
PID3_V_LOWER = 40
PID3_H_UPPER = 61
PID3_S_UPPER = 255
PID3_V_UPPER = 255
PID3_HSV_BOTTOM_ROI = 45
PID3_HSV_MIN_AREA = 0




# --------------- Crosswalk / Turn Durations ----------------
LINEAR_SPEED_CROSSWALK = 0.9
CROSSING_DURATION = 2.29
ARC_DURATION = 0.7
ROTATE_DURATION = 0.8
ARC_DURATION_EXIT = 0.6
ROTATE_DURATION_EXIT = 0.8

# ---------------- Red & Pink Detection Parameters ----------------
LOWER_RED1 = np.array([0, 100, 100])
UPPER_RED1 = np.array([10, 255, 255])
LOWER_RED2 = np.array([160, 100, 100])
UPPER_RED2 = np.array([180, 255, 255])
RED_PIXEL_THRESHOLD_FRACTION = 0.05

LOWER_PINK = np.array([140, 100, 100])  
UPPER_PINK = np.array([170, 255, 255])
PINK_PIXEL_THRESHOLD_FRACTION = 0.05

# ---------------- SIFT & Reference Paths ----------------
PEDESTRIAN_FRONT_REF_PATH = "/home/fizzer/Desktop/pedestrianfront.png"
PEDESTRIAN_BACK_REF_PATH  = "/home/fizzer/Desktop/pedestrianback.png"
ROUNDABOUT_REF_PATH       = "/home/fizzer/Desktop/Roundabout.png"
SIDETRUCK_REF_PATH        = "/home/fizzer/Desktop/sidetruck.jpg"
EXIT_ROUNDABOUT_REF_PATH  = "/home/fizzer/Desktop/exitroundabout1.png"
REDCAR_REF_PATH           = "/home/fizzer/Desktop/redcar.png"  # Red car reference (new)

MIN_SIFT_MATCH_COUNT = 5
MIN_SIFT_MATCH_COUNT_ROUNDABOUT = 120
MIN_SIFT_MATCH_COUNT_SIDETRUCK = 18
MIN_SIFT_MATCH_COUNT_ROUNDABOUT_EXIT = 112
MIN_SIFT_MATCH_COUNT_REDCAR = 130  # You can adjust if needed

SHOW_DEBUG_WINDOW = False

# Initialize SIFT
sift = cv.SIFT_create()

def load_reference(path):
    ref_img = cv.imread(path, cv.IMREAD_UNCHANGED)
    if ref_img is None:
        rospy.logwarn(f"Could not load reference image at {path}.")
        return None, None, None
    # flatten if alpha channel present
    if ref_img.shape[2] == 4:
        bgr = ref_img[:, :, :3]
        alpha = ref_img[:, :, 3] / 255.0
        ref_img = np.uint8(bgr * alpha[..., None])
    gray = cv.cvtColor(ref_img, cv.COLOR_BGR2GRAY)
    kp, desc = sift.detectAndCompute(gray, None)
    return gray, kp, desc

# Load references (no changes to your existing logic)
pedestrian_front_img, pedestrian_front_kp, pedestrian_front_desc = load_reference(PEDESTRIAN_FRONT_REF_PATH)
pedestrian_back_img, pedestrian_back_kp, pedestrian_back_desc   = load_reference(PEDESTRIAN_BACK_REF_PATH)
roundabout_img, roundabout_kp, roundabout_desc                  = load_reference(ROUNDABOUT_REF_PATH)
sidetruck_img, sidetruck_kp, sidetruck_desc                     = load_reference(SIDETRUCK_REF_PATH)
exit_roundabout_img, exit_roundabout_kp, exit_roundabout_desc   = load_reference(EXIT_ROUNDABOUT_REF_PATH)
redcar_img,       redcar_kp,       redcar_desc                  = load_reference(REDCAR_REF_PATH)

bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)

# ---------------- YODA COLOR DETECTION ----------------
LOWER_YODA = np.array([115, 0, 0])
UPPER_YODA = np.array([179, 255, 255])
YODA_HISTORY_LENGTH = 120
yoda_area_history = []

# ---------------- New PID3 Parameters ----------------
# Phase 1 (HSV) constants
PID3_HSV_DURATION = 3.9       # How many seconds to stay in the HSV-based approach
PID3_HSV_LINEAR_SPEED = 0.8   # Robot speed while in PID3_HSV
PID3_HSV_STEERING_GAIN = 0.15 # Steering gain for HSV approach


# Phase 2 (RGB) constants
LINEAR_SPEED_PID3 = 1.0
STEERING_GAIN_PID3 = 0.06
NO_LANE_TURN_RATE = 2.2

# detect_yoda_color() - unchanged from your existing code
def detect_yoda_color():
    if latest_image is None:
        return 0, None
    hsv = cv.cvtColor(latest_image, cv.COLOR_BGR2HSV)
    yoda_mask = cv.inRange(hsv, LOWER_YODA, UPPER_YODA)
    # Remove Blue Regions
    blue_lower = np.array([80, 0, 0])
    blue_upper = np.array([255, 0, 0])
    blue_mask = cv.inRange(latest_image, blue_lower, blue_upper)
    non_blue_mask = cv.bitwise_not(blue_mask)
    yoda_mask = cv.bitwise_and(yoda_mask, non_blue_mask)
    # Filter out small contours
    YODA_MIN_AREA = 1300
    contours, _ = cv.findContours(yoda_mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(yoda_mask)
    for cnt in contours:
        if cv.contourArea(cnt) >= YODA_MIN_AREA:
            cv.drawContours(filtered_mask, [cnt], -1, 255, -1)
    yoda_mask = filtered_mask
    # Compute area
    area = cv.countNonZero(yoda_mask)
    if area < 1300:
        return 0, None
    # Find largest
    contours, _ = cv.findContours(yoda_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return area, None
    largest = max(contours, key=cv.contourArea)
    M = cv.moments(largest)
    if M["m00"] == 0:
        return area, None
    cx = int(M["m10"] / M["m00"])
    return area, cx

# ---------------- detect_redcar_sift() ----------------
def detect_redcar_sift():
    if latest_image is None:
        return False
    gray_frame = cv.cvtColor(latest_image, cv.COLOR_BGR2GRAY)
    frame_kp, frame_desc = sift.detectAndCompute(gray_frame, None)
    if (frame_desc is None) or (redcar_desc is None) or (len(frame_kp) < MIN_SIFT_MATCH_COUNT_REDCAR):
        return False

    matches = bf.knnMatch(redcar_desc, frame_desc, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < MIN_SIFT_MATCH_COUNT_REDCAR:
        return False

    src_pts = np.float32([redcar_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([frame_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    try:
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 10.0)
    except cv.error as e:
        rospy.logwarn(f"findHomography error (redcar): {e}")
        return False
    if mask is None:
        return False

    inliers = int(mask.sum())
    if inliers >= MIN_SIFT_MATCH_COUNT_REDCAR and SHOW_DEBUG_WINDOW and len(good) > 0:
        debug_img = cv.drawMatches(redcar_img, redcar_kp, gray_frame, frame_kp, good, None,
                                   flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv.waitKey(1)
    return inliers >= MIN_SIFT_MATCH_COUNT_REDCAR

# -------------- PID3: Phase 1 (HSV) - find largest contour on the left side -------------
def find_pid3_hsv_centroid():
  
    if latest_image is None:
        return None
    hsv = cv.cvtColor(latest_image, cv.COLOR_BGR2HSV)
    lower = np.array([PID3_H_LOWER, PID3_S_LOWER, PID3_V_LOWER])
    upper = np.array([PID3_H_UPPER, PID3_S_UPPER, PID3_V_UPPER])
    mask = cv.inRange(hsv, lower, upper)
    
    # Crop to the bottom ROI (vertical crop only)
    h, w, _ = latest_image.shape
    roi_start_y = int(h * (1.0 - PID3_HSV_BOTTOM_ROI / 100.0))
    cropped_mask = mask[roi_start_y:, :]
    
    # Find contours in the cropped mask
    contours, _ = cv.findContours(cropped_mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    # Choose the largest contour
    largest = max(contours, key=cv.contourArea)
    if cv.contourArea(largest) < PID3_HSV_MIN_AREA:
        return None
    
    # Compute centroid of the largest contour
    M = cv.moments(largest)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    # Draw the centroid on the cropped mask for debugging
    debug_crop = cv.cvtColor(cropped_mask.copy(), cv.COLOR_GRAY2BGR)
    cv.circle(debug_crop, (cx, cy), 5, (0, 0, 255), -1)
    cv.putText(debug_crop, f"Center: {cx}", (cx - 20, cy - 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    return cx

latest_image_cam3 = None

def image_callback_cam3(msg):
    global latest_image_cam3
    latest_image_cam3 = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    
VERTICAL_ROI_FRACTION = 0.65
HORIZONTAL_ROI_FRACTION = 0.28

# ---------------- Lane Detection in the Bottom-Left Quarter ----------------
def find_rgb_line_center_pid3():
    if latest_image_cam3 is None:
        return None

    debug_image = latest_image_cam3.copy()
    h, w, _ = debug_image.shape

    # 1) Crop the bottom-left quarter
    roi_start_y = int(h * VERTICAL_ROI_FRACTION)             
    roi_end_y = h
    roi_end_x = int(w * HORIZONTAL_ROI_FRACTION)           
    roi = debug_image[roi_start_y:roi_end_y, 0:roi_end_x]

    if SHOW_DEBUG_WINDOW:
        print(f"ROI shape: {roi.shape}, x=0..{roi_end_x}, y={roi_start_y}..{h}")

    # 2) Hardcoded thresholds instead of sliders
    b_lower = 100
    g_lower = 120
    r_lower = 125
    b_upper = 255
    g_upper = 255
    r_upper = 255

    # 3) Brightness correction
    gray_roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    avg_brightness = cv.mean(gray_roi)[0]
    if SHOW_DEBUG_WINDOW:
        print(f"Avg brightness in ROI: {avg_brightness:.1f}")

    target_brightness = 128.0
    factor = 0.8
    brightness_correction = int(factor * (avg_brightness - target_brightness))
    b_lower_adj = np.clip(b_lower + brightness_correction, 0, 255)
    g_lower_adj = np.clip(g_lower + brightness_correction, 0, 255)
    r_lower_adj = np.clip(r_lower + brightness_correction, 0, 255)

    lower = np.array([b_lower_adj, g_lower_adj, r_lower_adj], dtype=np.uint8)
    upper = np.array([b_upper, g_upper, r_upper], dtype=np.uint8)

    if SHOW_DEBUG_WINDOW:
        print(f"Brightness corr: {brightness_correction}, lower={lower.tolist()}, upper={upper.tolist()}")

    # 4) Threshold & find the largest contour
    mask = cv.inRange(roi, lower, upper)
    overlay = cv.bitwise_and(roi, roi, mask=mask)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv.contourArea)
    M = cv.moments(largest)
    if M["m00"] == 0:
        return None

    cx_local = int(M["m10"] / M["m00"])
    cy_local = int(M["m01"] / M["m00"])

    # 5) Replace slider offset with fixed +300 offset
    rgb_offset = 300
    cx_offset = cx_local + rgb_offset

    abs_cx = cx_offset
    abs_cy = roi_start_y + cy_local

    # Debug
    if SHOW_DEBUG_WINDOW:
        debug_vis = debug_image.copy()
        cv.circle(debug_vis, (abs_cx, abs_cy), 8, (0, 0, 255), -1)
        cv.putText(debug_vis, f"cx={abs_cx}", (abs_cx - 40, abs_cy - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv.waitKey(1)

    return (abs_cx, w)

# ---------------- ALL OTHER CODE (unchanged) ----------------
def image_callback(msg):
    global latest_image
    try:
        img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        latest_image = img
    except CvBridgeError as e:
        rospy.logerr(f"Failed to convert image: {e}")

def detect_red_line():
    if latest_image is None:
        return False
    hsv = cv.cvtColor(latest_image, cv.COLOR_BGR2HSV)
    mask1 = cv.inRange(hsv, LOWER_RED1, UPPER_RED1)
    mask2 = cv.inRange(hsv, LOWER_RED2, UPPER_RED2)
    red_mask = cv.bitwise_or(mask1, mask2)
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv.morphologyEx(red_mask, cv.MORPH_CLOSE, kernel)
    red_mask = cv.dilate(red_mask, kernel, iterations=1)
    count = cv.countNonZero(red_mask)
    total = red_mask.size
    return (count > RED_PIXEL_THRESHOLD_FRACTION * total)

def detect_pink_line():
    if latest_image is None:
        return False
    hsv = cv.cvtColor(latest_image, cv.COLOR_BGR2HSV)
    pink_mask = cv.inRange(hsv, LOWER_PINK, UPPER_PINK)
    kernel = np.ones((5, 5), np.uint8)
    pink_mask = cv.morphologyEx(pink_mask, cv.MORPH_CLOSE, kernel)
    pink_mask = cv.dilate(pink_mask, kernel, iterations=1)
    count = cv.countNonZero(pink_mask)
    total = pink_mask.size
    return (count > PINK_PIXEL_THRESHOLD_FRACTION * total)

# ---------------- SIFT-based Detectors (Roundabout / Sidetruck / Pedestrian) ----
def detect_roundabout_sift():
    if latest_image is None:
        return False
    gray_frame = cv.cvtColor(latest_image, cv.COLOR_BGR2GRAY)
    frame_kp, frame_desc = sift.detectAndCompute(gray_frame, None)
    if frame_desc is None or len(frame_kp) < MIN_SIFT_MATCH_COUNT_ROUNDABOUT:
        return False
    matches = bf.knnMatch(roundabout_desc, frame_desc, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < MIN_SIFT_MATCH_COUNT_ROUNDABOUT:
        return False
    src_pts = np.float32([roundabout_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([frame_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    try:
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 10.0)
    except cv.error as e:
        rospy.logwarn(f"findHomography error (roundabout): {e}")
        return False
    if mask is None:
        return False
    inliers = int(mask.sum())
    if inliers >= MIN_SIFT_MATCH_COUNT_ROUNDABOUT and SHOW_DEBUG_WINDOW and len(good) > 0:
        debug_img = cv.drawMatches(roundabout_img, roundabout_kp, gray_frame, frame_kp, good, None,
                                   flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return inliers >= MIN_SIFT_MATCH_COUNT_ROUNDABOUT

def detect_exit_roundabout_sift():
    if latest_image is None:
        return False
    gray_frame = cv.cvtColor(latest_image, cv.COLOR_BGR2GRAY)
    frame_kp, frame_desc = sift.detectAndCompute(gray_frame, None)
    if frame_desc is None or len(frame_kp) < MIN_SIFT_MATCH_COUNT_ROUNDABOUT_EXIT:
        return False
    matches = bf.knnMatch(exit_roundabout_desc, frame_desc, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < MIN_SIFT_MATCH_COUNT_ROUNDABOUT_EXIT:
        return False
    src_pts = np.float32([exit_roundabout_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([frame_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    try:
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 10.0)
    except cv.error as e:
        rospy.logwarn(f"findHomography error (exit roundabout): {e}")
        return False
    if mask is None:
        return False
    inliers = int(mask.sum())
    if inliers >= MIN_SIFT_MATCH_COUNT_ROUNDABOUT_EXIT and SHOW_DEBUG_WINDOW and len(good) > 0:
        debug_img = cv.drawMatches(exit_roundabout_img, exit_roundabout_kp, gray_frame, frame_kp, good, None,
                                   flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return inliers >= MIN_SIFT_MATCH_COUNT_ROUNDABOUT_EXIT

def detect_sidetruck_sift():
    if latest_image is None:
        return False
    gray_frame = cv.cvtColor(latest_image, cv.COLOR_BGR2GRAY)
    frame_kp, frame_desc = sift.detectAndCompute(gray_frame, None)
    if frame_desc is None or len(frame_kp) < MIN_SIFT_MATCH_COUNT:
        return False
    matches = bf.knnMatch(sidetruck_desc, frame_desc, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < MIN_SIFT_MATCH_COUNT_SIDETRUCK:
        return False
    src_pts = np.float32([sidetruck_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([frame_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    try:
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 10.0)
    except cv.error as e:
        rospy.logwarn(f"findHomography error (sidetruck): {e}")
        return False
    if mask is None:
        return False
    inliers = int(mask.sum())
    if inliers > 0 and SHOW_DEBUG_WINDOW and len(good) > 0:
        debug_img = cv.drawMatches(sidetruck_img, sidetruck_kp, gray_frame, frame_kp, good, None,
                                   flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return inliers > 0

def detect_pedestrian_sift():
    if latest_image is None:
        return False
    gray_frame = cv.cvtColor(latest_image, cv.COLOR_BGR2GRAY)
    frame_kp, frame_desc = sift.detectAndCompute(gray_frame, None)
    if frame_desc is None or len(frame_kp) < MIN_SIFT_MATCH_COUNT:
        return False

    def match_and_ransac(ref_img, ref_kp, ref_desc, window_name):
        matches = bf.knnMatch(ref_desc, frame_desc, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good) < MIN_SIFT_MATCH_COUNT:
            return 0
        src_pts = np.float32([ref_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([frame_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        try:
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 10.0)
        except cv.error as e:
            rospy.logwarn(f"findHomography error (pedestrian): {e}")
            return 0
        if mask is None:
            return 0
        inliers = int(mask.sum())
        if SHOW_DEBUG_WINDOW and len(good) > 0:
            debug_img = cv.drawMatches(ref_img, ref_kp, gray_frame, frame_kp, good, None,
                                       flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        return inliers

    front_inliers = match_and_ransac(pedestrian_front_img, pedestrian_front_kp, pedestrian_front_desc, "SIFT Front")
    back_inliers = match_and_ransac(pedestrian_back_img, pedestrian_back_kp, pedestrian_back_desc, "SIFT Back")
    return (front_inliers > 0 or back_inliers > 0)

# ---------------- FIRST PID (Grayscale) ----------------
def find_line_center():
    if latest_image is None:
        return None
    debug_image = latest_image.copy()
    gray = cv.cvtColor(latest_image, cv.COLOR_BGR2GRAY)
    h, w = gray.shape
    roi_start_y = int(h * ROI_Y_PERCENTAGE)
    roi = gray[roi_start_y:, :]
    _, mask = cv.threshold(roi, BINARY_THRESHOLD, 255, cv.THRESH_BINARY_INV)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv.contourArea)
    M = cv.moments(largest)
    if M['m00'] == 0:
        return None

    center_x = int(M['m10'] / M['m00'])
    abs_center_y = roi_start_y + int(np.mean(largest[:, 0, 1]))

    cv.circle(debug_image, (center_x, abs_center_y), 8, (0, 255, 0), -1)
    cv.putText(debug_image, f'Lane(PID1): {center_x}',
               (center_x - 80, abs_center_y - 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


    return (center_x, w)

# ---------------- RGB PID2 ----------------
def find_rgb_line_center():
    if latest_image is None:
        return None

    debug_image = latest_image.copy()
    h, w, _ = debug_image.shape
    roi_start_y = int(h * ROI_Y_PERCENTAGE_2)
    roi = debug_image[roi_start_y:, :]

    # Keep your original logic / do not remove your constants
    roi = debug_image[roi_start_y:, :int(w // 2.5)]

    if SHOW_DEBUG_WINDOW:
        print(f"ROI starts at y={roi_start_y}, image height={h}")

    mask = cv.inRange(roi, BGR_LOWER_WHITE, BGR_UPPER_WHITE)
    white_pixel_count = cv.countNonZero(mask)
    if SHOW_DEBUG_WINDOW:
        print("White pixels:", white_pixel_count)

    overlay = cv.bitwise_and(roi, roi, mask=mask)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv.contourArea)
    M = cv.moments(largest)
    if M['m00'] == 0:
        return None

    cx = int(M['m10'] / M['m00']) + 280
    abs_y = roi_start_y + int(np.mean(largest[:, 0, 1]))


    return (cx, w)

# Pink sub-state durations
PID2_PINK_FORWARD_DURATION = 0.5
PID2_PINK_FORWARD_SPEED = 1.5
PID2_PINK_ROTATE_DURATION = 1.5
PID2_PINK_ROTATE_SPEED = 1.5

# RedCar approach
REDCAR_FORWARD_DURATION = 1.925
REDCAR_FORWARD_SPEED = 1.05
REDCAR_ROTATE_DURATION = 1.2
REDCAR_ROTATE_SPEED = 1.4

def main():
    global redcar_seen
    rospy.init_node('line_follower_with_pedestrian', anonymous=True)


    cmd_vel_pub = rospy.Publisher('/B1/cmd_vel', Twist, queue_size=1)
    rospy.Subscriber('/B1/rrbot/camera1/image_raw', Image, image_callback)
    #rospy.Subscriber('/B1/rrbot/camera3/image_raw', Image, image_callback_cam3)

    rospy.sleep(1.0)

    rate = rospy.Rate(120)
    twist = Twist()

    # States
    state = "NORMAL"
    roundabout_triggered = False
    exit_roundabout_triggered = False
    crossing_start_time = None
    turn_left_start_time = None

    pink_delay_start_time = None
    pink_forward_start_time = None
    exit_roundabout_start_time = None

    # Pink sub-states
    pid2_pink_forward_start = None
    pid2_pink_rotate_start = None

    # Yoda & redcar times
    yoda_follow_start_time = None
    redcar_forward_start_time = None
    redcar_rotate_start_time = None

    # [NEW] PID3 times
    pid3_hsv_start_time = None
    pid3_rgb_start_time = None
    pid3_hsv_done = False  # to track if we've finished phase 1

    while not rospy.is_shutdown():
        if state == "NORMAL":
            # ...
            # (unchanged logic, leaving it intact)
            if (not exit_roundabout_triggered) and detect_exit_roundabout_sift():
                exit_roundabout_triggered = True
                state = "EXIT_ROUNDABOUT"
                exit_roundabout_start_time = rospy.Time.now()
                twist.linear.x = 0.0
                twist.angular.z = 0.0

            elif (not roundabout_triggered) and detect_roundabout_sift():
                roundabout_triggered = True
                state = "WAITING_SIDETRUCK"
                twist.linear.x = 0.0
                twist.angular.z = 0.0

            elif not redcar_seen and detect_pink_line():
                state = "PINK_DELAY"
                pink_delay_start_time = rospy.Time.now()
                twist.linear.x = 0.0
                twist.angular.z = 0.0

            elif detect_red_line():
                state = "WAITING"
                twist.linear.x = 0.0
                twist.angular.z = 0.0

            else:
                lane_result = find_line_center()
                if lane_result is None:
                    rospy.logwarn("No lane found (PID1). Stopping.")
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                else:
                    cx, w = lane_result
                    deviation = cx - (w // 2)
                    twist.linear.x = LINEAR_SPEED_NORMAL
                    twist.angular.z = -STEERING_GAIN * deviation

        elif state == "PINK_DELAY":
            # ...
            elapsed = (rospy.Time.now() - pink_delay_start_time).to_sec()
            if elapsed < PINK_DETECTION_DELAY:
                twist.linear.x = 0.0
                twist.angular.z = 0.0
            else:
                pink_forward_start_time = rospy.Time.now()
                state = "PINK_FORWARD"

        elif state == "PINK_FORWARD":
            # ...
            elapsed_fwd = (rospy.Time.now() - pink_forward_start_time).to_sec()
            if elapsed_fwd < PINK_FORWARD_DURATION:
                twist.linear.x = PINK_FORWARD_SPEED
                twist.angular.z = 0.0
            else:
                state = "RGB_PID2"
                twist.linear.x = 0.0
                twist.angular.z = 0.0

        elif state == "RGB_PID2":
            # ...
            if not redcar_seen and detect_pink_line():
                state = "PID2_PINK_FORWARD"
                pid2_pink_forward_start = rospy.Time.now()
                twist.linear.x = 0.0
                twist.angular.z = 0.0
            else:
                lane_result = find_rgb_line_center()
                if lane_result is None:
                    rospy.logwarn("No RGB lane found (PID2). Stopping.")
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                else:
                    cx, w = lane_result
                    deviation = cx - (w // 2)
                    twist.linear.x = LINEAR_SPEED_PID2
                    twist.angular.z = -STEERING_GAIN_PID2 * deviation

        elif state == "PID2_PINK_FORWARD":
          
            elapsed_pf = (rospy.Time.now() - pid2_pink_forward_start).to_sec()
            if elapsed_pf < PID2_PINK_FORWARD_DURATION:
                twist.linear.x = PID2_PINK_FORWARD_SPEED
                twist.angular.z = -3.0
            else:
                state = "PID2_PINK_ROTATE"
                pid2_pink_rotate_start = rospy.Time.now()
                twist.linear.x = 0.0
                twist.angular.z = 0.0

        elif state == "PID2_PINK_ROTATE":
        
            elapsed_pr = (rospy.Time.now() - pid2_pink_rotate_start).to_sec()
            if elapsed_pr < PID2_PINK_ROTATE_DURATION:
                twist.linear.x = 0.0
                twist.angular.z = PID2_PINK_ROTATE_SPEED
            else:
                state = "YODA_WAITING"
                twist.linear.x = 0.0
                twist.angular.z = 0.0

        elif state == "YODA_WAITING":
     
            twist.linear.x = 0.0
            twist.angular.z = 0.0

            if latest_image is not None:
                hsv = cv.cvtColor(latest_image, cv.COLOR_BGR2HSV)
                yoda_mask = cv.inRange(hsv, LOWER_YODA, UPPER_YODA)
                yoda_debug = cv.bitwise_and(latest_image, latest_image, mask=yoda_mask)
            
                cv.waitKey(1)

            yoda_area, _ = detect_yoda_color()
            if latest_image is not None:
                debug_img = latest_image.copy()
                cv.putText(debug_img, f"Yoda Area: {yoda_area}", (10, 30),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv.waitKey(1)

            yoda_area_history.append(yoda_area)
            if len(yoda_area_history) > YODA_HISTORY_LENGTH:
                yoda_area_history.pop(0)

            twist.linear.x = 0.0
            twist.angular.z = 0.0

            if len(yoda_area_history) < 120:
                pass  # still collecting data, do nothing
            else:
                old_avg = sum(yoda_area_history[:-1]) / len(yoda_area_history[:-1])
                new_area = yoda_area_history[-1]
                if new_area < old_avg:
                    state = "FOLLOW_YODA"
                else:
                    pass  # remain in YODA_WAITING


        elif state == "FOLLOW_YODA":
            # ...
            if detect_redcar_sift():
                redcar_seen = True
                state = "REDCAR_FORWARD"
                redcar_forward_start_time = rospy.Time.now()
                twist.linear.x = 0.0
                twist.angular.z = 0.0
            else:
                yoda_area, cx = detect_yoda_color()
                if yoda_area == 0 or cx is None:
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                else:
                    TARGET_YODA_MIN = 1000
                    TARGET_YODA_MAX = 4000
                    base_speed = 2.2
                    if yoda_area < TARGET_YODA_MIN:
                        speed_factor = 1.0 + (TARGET_YODA_MIN - yoda_area) / float(TARGET_YODA_MIN)
                    elif yoda_area > TARGET_YODA_MAX:
                        speed_factor = max(0.0, 1.0 - (yoda_area - TARGET_YODA_MAX) / float(TARGET_YODA_MAX))
                    else:
                        speed_factor = 1.0
                    adjusted_speed = base_speed * speed_factor
                    w = latest_image.shape[1]
                    deviation = cx - (w // 2)
                    twist.linear.x = adjusted_speed
                    twist.angular.z = -0.02 * deviation

        elif state == "REDCAR_FORWARD":
            elapsed_rf = (rospy.Time.now() - redcar_forward_start_time).to_sec()
            if elapsed_rf < REDCAR_FORWARD_DURATION:
                twist.linear.x = REDCAR_FORWARD_SPEED
                twist.angular.z = 0.84
            else:
                state = "REDCAR_ROTATE"
                redcar_rotate_start_time = rospy.Time.now()
                twist.linear.x = 0.0
                twist.angular.z = 0.0

        elif state == "REDCAR_ROTATE":
            elapsed_rr = (rospy.Time.now() - redcar_rotate_start_time).to_sec()
            if elapsed_rr < REDCAR_ROTATE_DURATION:
                twist.linear.x = 0.3
                twist.angular.z = REDCAR_ROTATE_SPEED
            else:
                state = "PID3_HSV"
                pid3_hsv_start_time = rospy.Time.now()
                pid3_hsv_done = False
                twist.linear.x = 0.0
                twist.angular.z = 0.0

        elif state == "WAITING":
            # ...
            if detect_pedestrian_sift():
                state = "CROSSING"
                crossing_start_time = rospy.Time.now()
                twist.linear.x = LINEAR_SPEED_CROSSWALK
                twist.angular.z = 0.0
            else:
                twist.linear.x = 0.0
                twist.angular.z = 0.0

        elif state == "WAITING_SIDETRUCK":
            # ...
            if detect_sidetruck_sift():
                state = "TURNING_LEFT"
                turn_left_start_time = rospy.Time.now()
            else:
                twist.linear.x = 0.0
                twist.angular.z = 0.0

        elif state == "CROSSING":
            # ...
            twist.linear.x = LINEAR_SPEED_CROSSWALK
            twist.angular.z = 0.0
            if (rospy.Time.now() - crossing_start_time) > rospy.Duration(CROSSING_DURATION):
                state = "NORMAL"
                twist.linear.x = 0.0
                twist.angular.z = 0.0

        elif state == "TURNING_LEFT":
            # ...
            elapsed_tl = (rospy.Time.now() - turn_left_start_time).to_sec()
            if elapsed_tl < ARC_DURATION:
                twist.linear.x = 1.2
                twist.angular.z = -0.45
            elif elapsed_tl < (ARC_DURATION + ROTATE_DURATION):
                twist.linear.x = 0.0
                twist.angular.z = -2.0
            else:
                state = "NORMAL"
                twist.linear.x = 0.0
                twist.angular.z = 0.0

        elif state == "EXIT_ROUNDABOUT":
            # ...
            elapsed_exit = (rospy.Time.now() - exit_roundabout_start_time).to_sec()
            if elapsed_exit < ARC_DURATION_EXIT:
                twist.linear.x = 2.05
                twist.angular.z = -0.246
            elif elapsed_exit < (ARC_DURATION_EXIT + ROTATE_DURATION_EXIT):
                twist.linear.x = 0.0
                twist.angular.z = -3.95
            else:
                state = "NORMAL"
                twist.linear.x = 0.0
                twist.angular.z = 0.0

        # ---------------- PID3 STATES ----------------
        elif state == "PID3_HSV":
            elapsed_pid3hsv = (rospy.Time.now() - pid3_hsv_start_time).to_sec()
            if elapsed_pid3hsv < PID3_HSV_DURATION:
                cx = find_pid3_hsv_centroid()
                if cx is None:
                    rospy.logwarn("PID3_HSV: No valid contour found in the bottom ROI -> Stopping.")
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                else:
                    w = latest_image.shape[1]
                    deviation = cx - (w // 2)
                    twist.linear.x = PID3_HSV_LINEAR_SPEED
                    twist.angular.z = -PID3_HSV_STEERING_GAIN * deviation
            else:
                rospy.loginfo("PID3_HSV phase done. Stopping robot and launching yoda_hsv_debug.py")

                # 1. Stop the robot immediately
                stop_twist = Twist()
                cmd_vel_pub.publish(stop_twist)
                rospy.sleep(0.2)  # Let it settle

                # 2. Launch next script
                import os
                import subprocess
                subprocess.Popen(["python3", "/home/fizzer/ros_ws/src/my_controller/scripts/yoda_hsv_debug.py"])

                # 3. Exit current script
                rospy.signal_shutdown("Handing off to yoda_hsv_debug.py")
                return



        cmd_vel_pub.publish(twist)
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
