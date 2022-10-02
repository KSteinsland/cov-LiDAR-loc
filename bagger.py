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
from ros_numpy import numpify

class pc_sampler:

    num_samples = 100
    sample_interval = 3 #sec
    
    samples = []

    gt_odometry_latest = None
    time_last_sample = -inf

    """
    base_link velodyne transform
    - Translation: [0.230, 0.000, 0.694]
    - Rotation: in Quaternion [0.000, 0.000, 0.000, 1.000]
    """
    vld_T : np.array

    def __init__(self):
        self.vld_T = np.eye(4)
        self.vld_T[0, 3] = 0.230
        self.vld_T[1, 3] = 0.000
        self.vld_T[2, 3] = 0.694
      

    def callback_pcl(self, pc2_msg):
        if len(self.samples) < self.num_samples:
            if rospy.get_time() - self.time_last_sample > self.sample_interval:
                self.time_last_sample = rospy.get_time()

                xyz_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(pc2_msg)
                pcd = open3d.geometry.PointCloud()
                pcd.points = open3d.utility.Vector3dVector(xyz_array)
                self.samples.append((pcd, self.gt_odometry_latest))
            
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
        rospy.set_param('use_sim_time', True)
        
        rospy.init_node('listener', anonymous=True)
        rospy.Subscriber("/velodyne_points", PointCloud2, self.callback_pcl)
        rospy.Subscriber("/ground_truth/state", Odometry, self.callback_odo)

        rospy.spin()

    def get_clouds(self):
        return [ptc for ptc, _ in self.samples]
        
    def get_transformed_clouds(self):
        ptcs = []
        for ptc, odo in self.samples:
            """
            # create inverse rigid transform
            q = odo.pose.pose.orientation
            q_a = np.array([q.x, q.y, q.z, q.w])
            #print(q_a)
            R = open3d.geometry.get_rotation_matrix_from_quaternion(q_a)
            #print(R)

            T = np.eye(4)
            T[:3, :3] = R.T
            p = odo.pose.pose.position
            t = np.array([p.x, p.y, p.z])
            T[:3, 3] = -R.T@t
            print(T)
            """
            T = self.transform_from_odo_msg(odo)
            T_inv = self.inverse_transform(T)

            ptc = ptc.transform(T_inv)
            ptcs.append(ptc)
        return ptcs

    def save_samples(self):
        odom = []
        for i, (ptc, odo) in enumerate(self.samples):
            open3d.io.write_point_cloud(f"clouds/cloud{i}.pcd", ptc)
            odom.append(odo)
        with open("ground_truth.p", "wb") as f:
            pickle.dump(odom, f)

    def load_samples(self):
        odom = None
        with open("ground_truth.p", "rb") as f:
            odom = pickle.load(f)

        dir = "clouds"
        pcds = []
        for filename in os.listdir(dir):
            f = os.path.join(dir, filename)
            if os.path.isfile(f):
                pcds.append(open3d.io.read_point_cloud(f))

        sample_pairs = []
        for i in range(len(odom)):
            sample_pairs.append((pcds[i], odom[i]))

        self.samples = sample_pairs

if __name__ == "__main__":
    h = pc_sampler()
    h.run()
    
    #h.save_samples()
    #h.load_samples()

    #open3d.visualization.draw_geometries(h.get_transformed_clouds())



