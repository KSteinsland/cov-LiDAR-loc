import rospy
import tf
import tf2_msgs
import tf2_ros


if __name__ == '__main__':
    #rospy.set_param('use_sim_time', True)
    rospy.init_node('tf_listener')
    listener = tf.TransformListener()
   

    rate = rospy.Rate(10.0)

    while not rospy.is_shutdown():
        try:
            (trans,rot) = listener.lookupTransform('/velodyne', '/world', rospy.Time(0))
            print(trans, rot)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue
        try:
            rate.sleep()
        except (rospy.exceptions.ROSTimeMovedBackwardsException):
            continue