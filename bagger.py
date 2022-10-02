from cmath import inf
from genericpath import isfile
from typing import List
import rospy
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
import open3d 
import numpy as np
import ros_numpy
import tf2_msgs
import tf2_ros
import pickle
import os
import tf
import time

class pc_sampler:

    num_samples = 3
    sample_interval = 3 #sec
    
    samples = []

    gt_odometry_latest = None
    time_last_sample = -inf

    tf_listener = tf.TransformListener

    def callback_pcl(self, pc2_msg):
        if len(self.samples) < self.num_samples:
            if time.time() - self.time_last_sample > self.sample_interval:
                self.time_last_sample = time.time()

                try:
                    (trans,rot) = self.tf_listener.lookupTransform('/velodyne', '/world', rospy.Time(0))
                    print(trans, rot)
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    pass
                
                xyz_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(pc2_msg)
                pcd = open3d.geometry.PointCloud()
                pcd.points = open3d.utility.Vector3dVector(xyz_array)

                self.samples.append((pcd, np.array(trans), np.array(rot)))
            
        else:
            rospy.signal_shutdown("exit")
            return
    
    def callback_odo(self, odo_msg):
        T = self.transform_from_odo_msg(odo_msg)
        print(self.inverse_transform(T))
        self.gt_odometry_latest = odo_msg
    
    def transform_from_odo_msg(self, odo):
        q = odo.pose.pose.orientation
        q_a = np.array([q.x, q.y, q.z, q.w])
        R = open3d.geometry.get_rotation_matrix_from_quaternion(q_a)
        T = np.eye(4)
        T[:3, :3] = R
        p = odo.pose.pose.position
        t = np.array([p.x, p.y, p.z])
        T[:3, 3] = t
        return T

    def inverse_transform(self, T):
        R = T[:3, :3]
        t = T[:3, 3]
        T_inv = np.eye(4)
        T_inv[:3, :3] = R.T
        T_inv[:3, 3] = -R.T@t
        return T_inv

    def run(self):
        #rospy.set_param('use_sim_time', True)
        
        rospy.init_node('listener', anonymous=True)
        rospy.Subscriber("/velodyne_points", PointCloud2, self.callback_pcl)
        #rospy.Subscriber("/ground_truth/state", Odometry, self.callback_odo)

        self.tf_listener = tf.TransformListener()

        rospy.spin()

    def get_clouds(self):
        return [ptc for ptc, _, _ in self.samples]
        
    def get_transformed_clouds(self):
        ptcs = []
        for ptc, trans, rot in self.samples:
            R = open3d.geometry.get_rotation_matrix_from_quaternion(rot)
            ptc = ptc.translate(trans)
            ptc = ptc.rotate(R)
            ptcs.append(ptc)
        return ptcs

    def save_samples(self):
        tf = []
        for i, (ptc, trans, rot) in enumerate(self.samples):
            open3d.io.write_point_cloud(f"clouds/cloud{i}.pcd", ptc)
            tf.append((trans, rot))
        with open("ground_truth.p", "wb") as f:
            pickle.dump(tf, f)

    def load_samples(self):
        tf = None
        with open("ground_truth.p", "rb") as f:
            tf = pickle.load(f)

        dir = "clouds"
        pcds = []
        for filename in os.listdir(dir):
            f = os.path.join(dir, filename)
            if os.path.isfile(f):
                pcds.append(open3d.io.read_point_cloud(f))

        sample_pairs = []
        for i in range(len(tf)):
            sample_pairs.append((pcds[i], tf[i][0], tf[i][1]))

        self.samples = sample_pairs

if __name__ == "__main__":
    h = pc_sampler()
    
    h.run()
    h.save_samples()
    
    #h.load_samples()

    #for (ptc, trans, rot) in h.samples:
    #    print(trans, rot)

    open3d.visualization.draw_geometries(h.get_transformed_clouds())



