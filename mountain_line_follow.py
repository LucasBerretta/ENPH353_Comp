#!/usr/bin/env python3
import rospy
import cv2 as cv
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

sift_drive_start_time = None


# ----- SIFT SETUP -----
SIFT_REF_PATH = "/home/fizzer/Desktop/finalsift.png"
sift = cv.SIFT_create()
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)

ref_img = cv.imread(SIFT_REF_PATH, cv.IMREAD_GRAYSCALE)
ref_kp, ref_desc = sift.detectAndCompute(ref_img, None)
MIN_MATCH_COUNT = 100  # Tune this threshold as needed

sift_drive_start_time = None  # timer for forward drive
SIFT_DRIVE_DURATION = 3.0  # seconds


# ---------------- Globals ----------------
bridge = CvBridge()
latest_image = None

# ----- USER TUNABLE PARAMETERS -----
LINEAR_SPEED_PID3 = 1.5
STEERING_GAIN_PID3 = 0.06
NO_LANE_TURN_RATE = 2.2
RGB_OFFSET = 300  # This is your hardcoded offset added to cx_local

B_LOWER = 100
G_LOWER = 120
R_LOWER = 125
B_UPPER = 255
G_UPPER = 255
R_UPPER = 255

SHOW_DEBUG_WINDOW = True

# ROI cropping parameters
VERTICAL_ROI_FRACTION = 0.65
HORIZONTAL_ROI_FRACTION = 0.28

# ---------------- Image Callback ----------------
def image_callback(msg):
    global latest_image
    latest_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

def detect_final_sift():
    if latest_image is None:
        return False
    gray_frame = cv.cvtColor(latest_image, cv.COLOR_BGR2GRAY)
    kp_frame, desc_frame = sift.detectAndCompute(gray_frame, None)
    if desc_frame is None or ref_desc is None:
        return False
    matches = bf.knnMatch(ref_desc, desc_frame, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    return len(good) >= MIN_MATCH_COUNT


# ---------------- Lane Detection ----------------
def find_rgb_line_center():
    if latest_image is None:
        return None

    debug_image = latest_image.copy()
    h, w, _ = debug_image.shape

    roi_start_y = int(h * VERTICAL_ROI_FRACTION)
    roi_end_y = h
    roi_end_x = int(w * HORIZONTAL_ROI_FRACTION)
    roi = debug_image[roi_start_y:roi_end_y, 0:roi_end_x]

    gray_roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    avg_brightness = cv.mean(gray_roi)[0]
    target_brightness = 128.0
    factor = 0.8
    brightness_correction = int(factor * (avg_brightness - target_brightness))

    b_lower_adj = np.clip(B_LOWER + brightness_correction, 0, 255)
    g_lower_adj = np.clip(G_LOWER + brightness_correction, 0, 255)
    r_lower_adj = np.clip(R_LOWER + brightness_correction, 0, 255)

    lower = np.array([b_lower_adj, g_lower_adj, r_lower_adj], dtype=np.uint8)
    upper = np.array([B_UPPER, G_UPPER, R_UPPER], dtype=np.uint8)

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

    cx_offset = cx_local + RGB_OFFSET
    abs_cx = cx_offset
    abs_cy = roi_start_y + cy_local


    return (abs_cx, w)

# ---------------- Main Loop ----------------
def main():
    rospy.init_node("pid3_rgb_node")
    rospy.Subscriber('/B1/rrbot/camera3/image_raw', Image, image_callback)
    cmd_vel_pub = rospy.Publisher('/B1/cmd_vel', Twist, queue_size=1)
    global sift_drive_start_time
    rate = rospy.Rate(120)
    twist = Twist()

    while not rospy.is_shutdown():
        if latest_image is None:
            rate.sleep()
            continue

        now = rospy.Time.now()

        # Check for SIFT match
        if sift_drive_start_time is None and detect_final_sift():
            sift_drive_start_time = now
            rospy.loginfo("SIFT match found — driving forward!")

        # If within 1 second of match, drive forward
        if sift_drive_start_time is not None:
            elapsed = (now - sift_drive_start_time).to_sec()
            if elapsed < SIFT_DRIVE_DURATION:
                twist.linear.x = 1.0
                twist.angular.z = 0.125
            else:
                sift_drive_start_time = None  # reset after done
                twist.linear.x = 0.0
                twist.angular.z = 0.0
        else:
            lane_result = find_rgb_line_center()
            if lane_result is None:
                twist.linear.x = LINEAR_SPEED_PID3
                twist.angular.z = NO_LANE_TURN_RATE
            else:
                cx, width = lane_result
                deviation = cx - (width // 2)
                raw_turn = -STEERING_GAIN_PID3 * deviation
                twist.linear.x = LINEAR_SPEED_PID3
                twist.angular.z = raw_turn


        cmd_vel_pub.publish(twist)
        cv.waitKey(1)
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
