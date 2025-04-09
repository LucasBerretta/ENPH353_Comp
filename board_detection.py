#!/usr/bin/env python3

import rospy
import rospkg
import os
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from collections import deque

# Two states for our finite state machine
STATE_BOARD_SEARCH = 0
STATE_CLUE_DETECTION = 1

class BoardDetector:
    def __init__(self):
        rospy.init_node('board_detector')

        self.bridge = CvBridge()

        # Camera subscriber
        self.image_sub = rospy.Subscriber("/B1/rrbot/camera2/image_raw",
                                          Image, self.image_callback)

        # Publishers
        self.result_pub = rospy.Publisher('/homography_result', Image, queue_size=10)
        self.annotated_pub = rospy.Publisher("/camera/annotated_image", Image, queue_size=1)

        # Listen for "clue found 3 times" signal from image_processing
        self.clue_done_sub = rospy.Subscriber('/clue_detection_done', Image, self.on_clue_done)

        # Keep a queue of images for CLUE_DETECTION mode
        self.image_queue = deque(maxlen=100)  # buffer up to 100 frames, adjust as needed

        # State machine variable
        self.state = STATE_BOARD_SEARCH

        # SIFT setup
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('my_controller')
        template_path = os.path.join(package_path, "clue_banner.png")

        self.board_template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    

        self.sift = cv2.SIFT_create()
        self.kp_template, self.desc_template = self.sift.detectAndCompute(self.board_template, None)
        self.flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=3), dict(checks=20))

        self.min_ransac_inliers = 30
        

    def on_clue_done(self, msg):
        
        # rospy.loginfo("Received /clue_detection_done → Resetting to BOARD_SEARCH.")
        self.state = STATE_BOARD_SEARCH
        self.image_queue.clear()  # discard leftover frames
        # Optionally do any other cleanup

    def image_callback(self, msg):
     
        if self.state == STATE_BOARD_SEARCH:
            self.handle_board_search(msg)
        else:  # STATE_CLUE_DETECTION
            self.handle_clue_detection(msg)

    def handle_board_search(self, msg):
        
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"handle_board_search: CVBridge error: {e}")
            return

        annotated = self.try_detect_board(frame)
        if annotated is not None:
            # Publish the annotated image (with bounding box, etc.)
            try:
                annotated_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
                self.annotated_pub.publish(annotated_msg)
            except CvBridgeError as e:
                rospy.logerr(f"Failed to publish annotated image: {e}")

    def handle_clue_detection(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.image_queue.append(frame)
        except CvBridgeError as e:
            rospy.logerr(f"handle_clue_detection: CVBridge error: {e}")

    def try_detect_board(self, frame):
    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, desc_frame = self.sift.detectAndCompute(gray, None)
        if desc_frame is None:
            return None

        matches = self.flann.knnMatch(self.desc_template, desc_frame, k=2)
        good_points = [m for m, n in matches if m.distance < 0.6*n.distance]

        if len(good_points) < 5:
            return None

        query_pts = np.float32([self.kp_template[m.queryIdx].pt for m in good_points]).reshape(-1,1,2)
        train_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_points]).reshape(-1,1,2)

        matrix, mask = cv2.findHomography(train_pts, query_pts, cv2.RANSAC, 5.0)
        if matrix is None or mask is None:
            return None

        inlier_count = np.sum(mask)
        if inlier_count < self.min_ransac_inliers:
            return None

        # Board found -> warp
        h, w = self.board_template.shape
        warped = cv2.warpPerspective(frame, matrix, (w, h))
        warped = cv2.resize(warped, (600, 400))

        # Publish to image_processing
        try:
            warped_msg = self.bridge.cv2_to_imgmsg(warped, encoding="bgr8")
            self.result_pub.publish(warped_msg)
        except CvBridgeError as e:
            rospy.logerr(f"Failed to publish warped image: {e}")

        # Draw bounding polygon on 'frame'
        matrix1, _ = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
        if matrix1 is not None:
            corners = np.float32([[0,0],[0,h],[w,h],[w,0]]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(corners, matrix1)
            cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
            cv2.putText(frame, "Board FOUND", (80,120),
                        cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 0), 2)

        #rospy.loginfo("Board found → Switching to CLUE_DETECTION state.")
        self.state = STATE_CLUE_DETECTION
        self.image_queue.clear()  # start fresh collecting frames
        return frame

    def run(self):
    
        rate = rospy.Rate(30)  # process up to 30 frames per second
        while not rospy.is_shutdown():
            if self.state == STATE_CLUE_DETECTION and self.image_queue:
                # Pop the oldest image
                frame = self.image_queue.popleft()
                # Run SIFT again
                annotated = self.try_detect_board(frame)
                # Optionally publish the annotated frame if you'd like
                if annotated is not None:
                    try:
                        ann_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
                        self.annotated_pub.publish(ann_msg)
                    except CvBridgeError as e:
                        rospy.logerr(f"Failed to publish annotated: {e}")

            # If state == BOARD_SEARCH, we do SIFT in the callback, 
            # so we do nothing else here.
            rate.sleep()

if __name__ == "__main__":
    try:
        node = BoardDetector()
        node.run()
    except rospy.ROSInterruptException:
        pass
