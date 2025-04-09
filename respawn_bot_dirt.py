#!/usr/bin/env python3

import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Twist

def stop_robot():
    """Stop the robot's motion before teleporting it."""
    pub = rospy.Publisher('/B1/cmd_vel', Twist, queue_size=1)
    rospy.sleep(0.5)  # Give time for publisher registration
    stop_msg = Twist()
    pub.publish(stop_msg)
    rospy.loginfo("🛑 Published zero Twist to stop robot.")

def respawn_car(position):
    rospy.wait_for_service('/gazebo/set_model_state')
    try:
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        msg = ModelState()
        msg.model_name = 'B1'
        msg.pose.position.x = position[0]
        msg.pose.position.y = position[1]
        msg.pose.position.z = position[2]
        msg.pose.orientation.x = position[3]
        msg.pose.orientation.y = position[4]
        msg.pose.orientation.z = position[5]
        msg.pose.orientation.w = position[6]

        # Make sure robot is not moving after respawn
        msg.twist.linear.x = 0.0
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = 0.0
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.0

        msg.reference_frame = 'world'

        resp = set_state(msg)
        rospy.loginfo("✅ Robot respawned.")
    except rospy.ServiceException as e:
        rospy.logerr(f"❌ Service call failed: {e}")

if __name__ == '__main__':
    rospy.init_node('respawn_node')

    # Finalized spawn pose from robots.launch
    spawn_pose = [-2.45, -2.2, 0.5, 0.0, 0.0, 0.0, 0.25]

    stop_robot()
    rospy.sleep(0.2)
    respawn_car(spawn_pose)
