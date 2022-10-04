from cmath import inf
from genericpath import isfile
from typing import List
import rospy
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
import rosgraph_msgs 
import open3d
import numpy as np
import ros_numpy
import tf2_msgs
import tf2_ros
import pickle
import os
import tf2_ros
import time
import pcl_ros
pcl_ros.transform_point_cloud()

class pc_sampler:

    num_samples = 3
    sample_interval = 3 #sec
    
    samples = []

    gt_odometry_latest = None
    time_last_sample = -inf

    #tf_listener = tf.TransformListener
    #tf_listener.transformPointCloud
    

    def callback_pcl(self, pc2_msg):
        """
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
        """
        
    
    def callback_odo(self, odo_msg):
        T = self.transform_from_odo_msg(odo_msg)
        print(self.inverse_transform(T))
        self.gt_odometry_latest = odo_msg
    
    def transform_from_msg(self, msg):
        q = msg.transform.rotation
        q_a = np.array([q.x, q.y, q.z, q.w])
        R = open3d.geometry.get_rotation_matrix_from_quaternion(q_a)
        T = np.eye(4)
        T[:3, :3] = R
        p = msg.transform.translation
        t = np.array([p.x, p.y, p.z])
        T[:3, 3] = t
        #print("---------------------")
        #print(T)
        #print(msg.transform, "/n")
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
        #rospy.Subscriber("/velodyne_points", PointCloud2, self.callback_pcl, queue_size=1) #might try somethin with queue size = 1
        #rospy.Subscriber("/ground_truth/state", Odometry, self.callback_odo)
        tf_buffer = tf2_ros.Buffer(rospy.Duration(50))
        tf_listener = tf2_ros.TransformListener(tf_buffer)
        time.sleep(1)
        
        points = np.array([[0, 0, 0]])
        while(len(self.samples) < self.num_samples):
            print("\n---------------")
            pc2_msg = rospy.wait_for_message("/velodyne_points", PointCloud2, timeout=10)
            #print(pc2_msg.header.frame_id)

            #print(pc2_msg.header)
            trans = tf_buffer.lookup_transform('world', 'velodyne', pc2_msg.header.stamp, rospy.Duration(100.0))
            #print(trans)
            t = trans.transform.translation
            t = np.array([t.x, t.y, t.z])
            q = trans.transform.rotation
            q = np.array([q.x, q.y, q.z, q.w])
            R = open3d.geometry.get_rotation_matrix_from_quaternion(q)
            
            print(t)
            print(q)

            Q = np.array([
                [-1, 0, 0],
                [0, -1, 0],
                [0, 0, 1]
            ])
            P = np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]
            ])


            xyz_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(pc2_msg)
            
            #xyz_array_world = xyz_array - np.array([t.x, t.y, t.z])
            
            xyz_array = xyz_array + Q@t
            #xyz_array = xyz_array.dot(Q)
            xyz_array = xyz_array.dot(R)
            xyz_array = xyz_array.dot(P)
            

            points = np.concatenate((points, xyz_array), axis=0)

            self.samples.append((xyz_array, t, q))
            #print(rospy.get_time())
            time.sleep(self.sample_interval)

        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(points)
        frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
        open3d.visualization.draw_geometries([pcd, frame])

    def get_clouds(self):
        return [ptc for ptc, _, _ in self.samples]
        
    def get_transformed_clouds(self):
        ptcs = []
        for ptc, t, q in self.samples:
            Q = np.array([
                [0, 0, 1],
                [0, -1, 0],
                [1, 0, 0]
            ])
            R = open3d.geometry.get_rotation_matrix_from_quaternion(q)
            
            """
            T = np.eye(4)
            T[:3, :3] = R@Q
            T[:3, 3] = -t
            """
            ptc.translate(t, relative=False)
            ptc.rotate(R, center=t)
            
            
           
            
            
           
            ptcs.append(ptc)
        return ptcs

    def save_samples(self):
        tf = []
        for i, (ptc, T) in enumerate(self.samples):
            open3d.io.write_point_cloud(f"clouds/cloud{i}.pcd", ptc)
            tf.append(T)
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
            sample_pairs.append((pcds[i], tf[i]))

        self.samples = sample_pairs

if __name__ == "__main__":
    h = pc_sampler()
    
    h.run()
    #h.save_samples()
    
    #h.load_samples()

    #for (ptc, trans, rot) in h.samples:
    #    print(trans, rot)

    frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])

    #open3d.visualization.draw_geometries(h.get_clouds() + [frame])
    #print("numb samples: ", len(h.samples))



