import rospy
from sensor_msgs.msg import PointCloud2
import tf
import numpy as np
import ros_numpy
import pickle
import time
import os
import matplotlib.pyplot as plt
from manifpy import SE3, SO3, SE3Tangent

class PCSampler:

    samples = [] # holds tuples [(xyz_array_points, mat44_transformation), ...]

    def sample(self, num_samples, sample_interval):
        """
        Listens for point clouds and transforms and adds them as pairs in self.samples.
        Returns all sampled point clouds concatendated, in world frame. 
        """

        rospy.init_node('listener', anonymous=True)
        tf_listener = tf.TransformListener()
        time.sleep(1)
        
        full_cloud = np.array([[0, 0, 0]])
        while(len(self.samples) < num_samples):
            print("\nGetting cloud and transform.")
            pc2_msg = rospy.wait_for_message("/velodyne_points", PointCloud2, timeout=10)
            xyz_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(pc2_msg)

            tf_listener.waitForTransform("/velodyne", "/world", pc2_msg.header.stamp, rospy.Duration(100.0))
            mat44 = tf_listener.asMatrix("/world", pc2_msg.header)

            self.samples.append((xyz_array, mat44))
            
            xyz_array_world = transform_cloud(xyz_array, mat44)
            full_cloud = np.concatenate((full_cloud, xyz_array_world), axis=0)

            time.sleep(sample_interval)

        return full_cloud

    def get_clouds(self):
        return [xyz_array for xyz_array, _ in self.samples]

    def get_transforms(self, use_quat=False):
        if use_quat:
            samples = []
            for ptc, T in self.samples:
                q = tf.transformations.quaternion_from_matrix(T)
                t = T[:3, 3]
                samples.append(np.concatenate((t, q)))
            return samples

        return [T for _, T in self.samples]
        
    def get_transformed_clouds(self):
        ptcs = []
        for xyz_array, mat44 in self.samples:
            ptcs.append(transform_cloud(xyz_array, mat44))
        return ptcs

    def save_samples(self):
        import datetime 
        dt = datetime.datetime.now()
        timestamp = dt.strftime("%Y%m%d-%H%M%S")
        with open(f"clouds/{timestamp}.p", "wb") as f:
            pickle.dump(self.samples, f)

    def save_samples_csv(self, clouds_filename):
        foldername = clouds_filename.split(".")[0]
        os.makedirs(f'clouds_csv/{foldername}', exist_ok=True)
        samples = self.load_samples(f'./clouds/{clouds_filename}')
        for i, (xyz_array, _) in enumerate(samples):
            np.savetxt(f'clouds_csv/{foldername}/cloud{i:03d}.csv', xyz_array, delimiter=",")

    def load_samples(self, pickle_filename="newest"):
        if pickle_filename == "newest":
            import glob, os
            files = glob.glob("./clouds/*.p")
            pickle_filename = max(files, key=os.path.getctime)
        with open(pickle_filename, "rb") as f:
            self.samples = pickle.load(f)        
        return self.samples

def add_noise_to_cloud(cloud, std_noise):
    noise = std_noise * np.random.normal(size=cloud.shape)
    return cloud + noise

def transform_cloud(xyz_array, mat44):
        def xf(p):
            xyz = tuple(np.dot(mat44, np.array([p[0], p[1], p[2], 1.0])))[:3]
            return xyz
        return np.array([xf(p) for p in xyz_array])

def vis_pc(xyz_array_list):
    import open3d
    
    if isinstance(xyz_array_list[0], open3d.geometry.PointCloud):
        frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
        open3d.visualization.draw_geometries(xyz_array_list + [frame])

    elif isinstance(xyz_array_list[0], np.ndarray):
        pcds = []
        for xyz_array in xyz_array_list:
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(xyz_array)
            pcds.append(pcd)
        frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
        open3d.visualization.draw_geometries(pcds + [frame])
    

def plot_pc(xyz_array_list):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for xyz_array in xyz_array_list:
        xs = xyz_array[:,0]
        ys = xyz_array[:,1]
        zs = xyz_array[:,2]
        ax.scatter(xs, ys, zs)
    plt.show()

def plot_proj_ax(ax, xyz_array, color, label=None):
    xs, ys = [], []
    for i in range(len(xyz_array)):
        d = np.sqrt(xyz_array[i,0]**2 + xyz_array[i,1]**2)
        if xyz_array[i,2] > 0.1 and d < 150 and xyz_array[i,1] < 25 and np.random.random() < 0.1:#0.005:
            xs.append(xyz_array[i,0])
            ys.append(xyz_array[i,1])
    ax.plot(xs, ys, color + "o", markersize="0.1", label=label)

def TtoSE3(T):
    #fix
    q = tf.transformations.quaternion_from_matrix(T)
    t = T[:3, 3]
    return SE3(np.concatenate((t, q)))


if __name__ == "__main__":
    num_samples = 30
    sample_interval = 6 #sec
    
    S = PCSampler()
    dataset = "20221217-153418"#"20221107-185525"
    #S.load_samples(pickle_filename="./clouds/20221217-153418.p")
    S.load_samples(pickle_filename=f"./clouds/{dataset}.p")

    clouds_world = S.get_clouds()
    ts = S.get_transforms()
    c0 = 0
    c1 = 299#29
    fig, ax = plt.subplots()
    
    for c in range(c0,c1):
        #c0_xyz = clouds_world[c0]
        #c1_xyz = clouds_world[c1]
        #T0 = ts[c0]
        #T1 = ts[c1]
        #T_rel = (TtoSE3(T0).inverse()*TtoSE3(T1))

        eul = tf.transformations.euler_from_matrix(ts[c][:3,:3])
        #print(c, " ", eul[2]*180/3.14)
        ax.plot(ts[c][0,3], ts[c][1,3], "go", markersize=1)
        if c % 30 == 0: ax.annotate(f"{c}", ts[c][:2,3])
        plot_proj_ax(ax, transform_cloud(clouds_world[c], ts[c]), "r")
    #plot_proj_ax(ax, transform_cloud(c1_xyz, T_rel.transform()), "b", label="Cloud 2")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_xlim((10,110))#ax.set_xlim((25,75))#
    ax.set_ylim((-15,20))#ax.set_ylim((-30,5))#
    #plt.legend(loc="best")
    #dummy labels
    ax.plot([-100], [-100], "go", label= "Robot $xy$-pos")
    ax.plot([-100], [-100], "ro", label= "Point clouds $xy$-projection")
    lgnd = plt.legend(loc="lower left", prop={'size': 7})
   

    plt.savefig("./imgsEnv/fullPath300HQ.png", format="png", dpi=300)


    #S.load_samples()
    #S.save_samples_csv('20221107-185525.p')

    #run and save point clouds
    #S.sample(num_samples, sample_interval)
    #S.save_samples()

    #show point clouds
    #S.load_samples()
    #clouds_world = S.get_transformed_clouds()
    #vis_pc([add_noise_to_cloud(clouds_world[0], 0.1)])
    #print(samples[0][1][:2,3])
    #print(len(samples))



    



