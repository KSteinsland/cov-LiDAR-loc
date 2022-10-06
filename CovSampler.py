import imp
from random import sample
from traceback import print_tb
from PCSampler import PCSampler, vis_pc, transform_cloud
from manifpy import SE3, SO3, SE3Tangent
import open3d as o3d
import numpy as np
import tf

np.random.seed(0)

S = PCSampler()
S.load_samples()
clouds = S.get_clouds()
transforms = S.get_transforms(use_quat=True)

def calc_mean_cov(tf_list):
    n = len(tf_list)
    mean = np.zeros(6)
    cov = np.zeros((6,6))
    for T, T_p in tf_list: 
        q = tf.transformations.quaternion_from_matrix(T)
        t = T[:3, 3]
        tf_b = SE3(np.concatenate((t, q)))
        tf_l = (T_p.inverse()*tf_b).log().coeffs()
        mean = np.add(mean, tf_l)
        cov = np.add(cov, np.outer(tf_l, tf_l))
    mean = mean / n
    cov = cov / (n - 1)
    return mean, cov

samples = [] 

for i in range(len(clouds) - 1):
    #get the two clouds and their tranformations to world
    c0, c1 = clouds[i], clouds[i+1]
    T0, T1 = SE3(transforms[i]), SE3(transforms[i+1])

    #get ground truth relative transformation
    T = T0.inverse()*T1
    c1_0 = transform_cloud(c1, T.transform())
    #vis_pc([c0, c1_0])

    num_samples = 30
    icp_transformations = []

    for i in range(num_samples):
        #give T some random perturbations to simulate noisy odometry
        cov_pos = 0.5
        cov_yaw = 0.01
        p = np.zeros(6)
        p[:2] = np.sqrt(cov_pos)*np.random.normal(size=2) # we only perturb x,y
        p[5] = np.sqrt(cov_yaw)*np.random.normal(size=1) # we only perturb yaw
        T_p = SE3Tangent(p) + T #I guess we want to add p on the left side? ask nikhil

        c1_0p = transform_cloud(c1, T_p.transform())
        #vis_pc([c0, c1_0p])

        #convert clouds to open3d clouds
        pcd0 = o3d.geometry.PointCloud()
        pcd0.points = o3d.utility.Vector3dVector(c0)
        pcd0.estimate_normals()

        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(c1)
        pcd0.estimate_normals()

        #do registration using open3ds point to plane icp for now
        treshold = 1
        initial_T = T_p.transform()
        reg = o3d.pipelines.registration.registration_icp(
            pcd1, pcd0, treshold, initial_T, 
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        icp_transform = reg.transformation
        c1_registered = transform_cloud(c1, icp_transform)

        icp_transformations.append((icp_transform, T_p))
    
    mean, cov = calc_mean_cov(icp_transformations)

    Ts = [icp_transform for icp_transform, _ in icp_transformations]
    samples.append((mean, cov, T1.transform(), Ts))

def save_samples():
    import datetime 
    import pickle
    dt = datetime.datetime.now()
    timestamp = dt.strftime("%Y%m%d-%H%M%S")
    with open(f"covariances/cov_samples_{timestamp}.p", "wb") as f:
        pickle.dump(samples, f)

#project samples for to visualize covergence clusters
#sampled_pose_projected = np.array([[icp_transform[0,3], icp_transform[1,3], -3]])
#projected_samples = np.array([[samples[i][3][0,3], samples[i][3][1,3], -3] for i in range(20)])
#vis_pc([c0, c1_0, projected_samples])


