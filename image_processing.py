#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from tensorflow.keras.models import load_model

model_path = "/home/fizzer/ros_ws/src/my_controller/models/test_model.h5"
model = load_model(model_path)

bridge = CvBridge()
score_pub = rospy.Publisher('/score_tracker', String, queue_size=1)
clue_done_pub = rospy.Publisher('/clue_detection_done', Image, queue_size=1)

classes = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '
index_to_class = {i: c for i, c in enumerate(classes)}

clue_map = {
    'SIZE': '1',
    'VICTIM': '2',
    'CRIME': '3',
    'TIME': '4',
    'PLACE': '5',
    'MOTIVE': '6',
    'WEAPON': '7',
    'BANDIT': '8',
}

def preprocess_image(img):
    img_expanded = np.expand_dims(img, axis=0)
    return img_expanded

def predict_letter(roi):
    roi_preprocessed = preprocess_image(roi)
    prediction = model.predict(roi_preprocessed, verbose = 0)
    return np.argmax(prediction)

def slice_image(image):
    first_character = image[250:350, 30:75]
    second_character = image[250:350, 75:120]
    third_character = image[250:350, 120:165]
    fourth_character = image[250:350, 165:210]
    fifth_character = image[250:350, 210:255]
    sixth_character = image[250:350, 255:300]
    seventh_character = image[250:350, 300:345]
    eigth_character = image[250:350, 345:390]
    ninth_character = image[250:350, 390:435]
    tenth_character = image[250:350, 435:480]
    eleventh_character = image[250:350, 480:525]
    twelvth_character = image[250:350, 525:570]
    return [
        first_character, second_character, third_character, fourth_character,
        fifth_character, sixth_character, seventh_character, eigth_character,
        ninth_character, tenth_character, eleventh_character, twelvth_character
    ]

def slice_clue(image):
    first_character = image[30:130, 250:295]
    second_character = image[30:130, 295:340]
    third_character = image[30:130, 340:385]
    fourth_character = image[30:130, 385:430]
    fifth_character = image[30:130, 430:475]
    sixth_character = image[30:130, 475:520]
    return [
        first_character, second_character, third_character,
        fourth_character, fifth_character, sixth_character
    ]

def find_clue(message):
    return clue_map.get(message, '100')

def is_between_0_and_8(s):
    if s.isdigit():
        return 0 <= int(s) <= 8
    return False

class ClueDetector:
    def __init__(self):
        rospy.init_node('image_processing', anonymous=True)

        # Subscribe to warped board images
        rospy.Subscriber('/homography_result', Image, self.on_warped_image)

        self.publish_count = np.zeros(8)
        self.consecutive_clue_count = 0
        self.last_clue = None

        # Send an initial scoreboard message
        team_name = "Lucas and Heejae"
        password = "12345678"
        start_message = f"{team_name},{password},0, Starting Simulation"
        score_pub.publish(start_message)

    def on_warped_image(self, msg):
        try:
            cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"Failed to convert warped board: {e}")
            return

        # 1) Slice for "clue" portion (top)
        clue_slices = slice_clue(cv_image)
        if not clue_slices:
            return

        # 2) Build the clue string
        clue_str = ""
        for roi in clue_slices:
            pred = predict_letter(roi)
            clue_str += index_to_class[pred]
        clue_str = clue_str.strip()
        # rospy.loginfo(f"[ClueDetector] Detected clue string: {clue_str}")

        # 3) Determine clue_id
        clue_id = find_clue(clue_str)
        if is_between_0_and_8(clue_id):
            # Check if repeated
            if self.last_clue == clue_id:
                self.consecutive_clue_count += 1
            else:
                self.last_clue = clue_id
                self.consecutive_clue_count = 1

            # rospy.loginfo(f"[ClueDetector] Clue {clue_id}, consecutive={self.consecutive_clue_count}")

            # 3 times in a row => publish final message
            if self.consecutive_clue_count >= 1:
                # rospy.loginfo(f"[ClueDetector] Clue {clue_str} (ID: {clue_id}) repeated 3 times!!!")

                # Optionally slice the bottom word too
                second_word_str = ""
                word_slices = slice_image(cv_image)
                for roi2 in word_slices:
                    pred2 = predict_letter(roi2)
                    second_word_str += index_to_class[pred2]
                second_word_str = second_word_str.strip()

                # 4) Publish the final clue to scoreboard
                team_name = "Lucas and Heejae"
                password = "12345678"


                if self.publish_count[int(clue_id)-1] < 2.0:
                    final_msg = f"{team_name},{password},{clue_id},{second_word_str}"
                # rospy.loginfo(f"[ClueDetector] Publishing final_msg: {final_msg}")
                    score_pub.publish(final_msg)
                    self.publish_count[int(clue_id)-1] += 1

                # 5) If the last clue is "BANDIT" (clue_id == '8'), also send -1,NA
                if clue_id == '8':
                    # final_msg = f"{team_name},{password},{clue_id},{second_word_str}"
                    #rospy.loginfo("[ClueDetector] 'BANDIT' is the final clue => Stopping timer.")
                    stop_msg = f"{team_name},{password},-1,NA"
                    score_pub.publish(stop_msg)

                # 6) Also let board_detection node revert to searching
                clue_done_pub.publish(msg)

                # Reset
                self.last_clue = None
                self.consecutive_clue_count = 0
        else:
            # Not recognized
            self.last_clue = None
            self.consecutive_clue_count = 0

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = ClueDetector()
        node.run()
    except rospy.ROSInterruptException:
        pass