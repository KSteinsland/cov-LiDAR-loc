from random import sample
from traceback import print_tb
from turtle import color
from PCSampler import PCSampler, vis_pc, transform_cloud
from manifpy import SE3, SO3, SE3Tangent
import open3d as o3d
import numpy as np
import tf
import matplotlib.pyplot as plt

def calc_mean_cov(tf_list, T_gt):
    n = len(tf_list)
    mean = np.zeros(6)
    cov = np.zeros((6,6))
    for T in tf_list: 
        q = tf.transformations.quaternion_from_matrix(T)
        t = T[:3, 3]
        tf_m = SE3(np.concatenate((t, q)))
        tf_t = (T_gt.inverse()*tf_m).log().coeffs()
        mean += tf_t
        cov += np.outer(tf_t, tf_t)
    mean = mean / n
    cov = cov / (n - 1)
    return mean, cov

def open3d_icp(source, target, initial_T):
    #convert clouds to open3d clouds
    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(target)
    pcd0.estimate_normals()

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(source)
    pcd0.estimate_normals()

    #do registration using open3ds point to plane icp for now
    treshold = 1.0
    reg = o3d.pipelines.registration.registration_icp(
        pcd1, pcd0, treshold, initial_T, 
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    return reg.transformation

def sample_registrations(cloud_source, cloud_target, rel_T, num_samples):
    """
    Will sample an initial tranform from distribution with mean at rel_t and covariance cov6x6.
    Registers clouds_source to cloud_target with this intital transform as prior. 
    Does this num_samples times and returns list of transforms. 
    """

    samples = [] 
    for i in range(num_samples):
        #give T some random perturbations to simulate noisy odometry
        #TODO take in cov6x6 instead of doing this  
        cov_pos = 0.5
        cov_yaw = 0.01
        p = np.zeros(6)
        p[:2] = np.sqrt(cov_pos)*np.random.normal(size=2) # we only perturb x,y
        p[5] = np.sqrt(cov_yaw)*np.random.normal(size=1) # we only perturb yaw
        T_p = SE3Tangent(p) + rel_T #I guess we want to add p on the left side? ask nikhil
        initial_T = T_p.transform()

        icp_transform = open3d_icp(cloud_source, cloud_target, initial_T)

        samples.append(icp_transform)
    return samples

def save_samples(samples):
    import datetime 
    import pickle
    dt = datetime.datetime.now()
    timestamp = dt.strftime("%Y%m%d-%H%M%S")
    with open(f"covariances/{timestamp}.p", "wb") as f:
        pickle.dump(samples, f)

def plot_ellipse(ax, mean, cov, n=50, chi2_val=9.21, fill_alpha=0., fill_color='lightsteelblue'):
    u, s, _ = np.linalg.svd(cov)
    scale = np.sqrt(chi2_val * s)

    theta = np.linspace(0, 2*np.pi, n+1)
    x = np.cos(theta)
    y = np.sin(theta)

    R = u
    t = np.reshape(mean, [2, 1])
    circle_points = (R * scale) @ np.vstack((x.flatten(), y.flatten())) + t

    ax.fill(circle_points[0, :], circle_points[1, :], alpha=fill_alpha, facecolor=fill_color)
    ax.plot(circle_points[0, :], circle_points[1, :], color=fill_color)

def plot_results_2d(ax, c0_w, sample_points2_w, mean2_w, cov2x2_w, z_lim=1):
    ax.plot(c0_w[c0_w[:,2] > z_lim][:,0], c0_w[c0_w[:,2] > z_lim][:,1], 'bo', markersize=0.1)
    ax.plot(sample_points2_w[:,0], sample_points2_w[:,1], 'go', markersize=2)
    plot_ellipse(ax, mean2_w, cov2x2_w, chi2_val=5)

if __name__ == "__main__":
    np.random.seed(0)

    S = PCSampler()
    S.load_samples()
    clouds = S.get_clouds()
    transforms = S.get_transforms(use_quat=True)

    fig, ax = plt.subplots()

    cov_samples = []
    for i in range(len(clouds) - 1):
        #get the two clouds and their tranformations to world
        c0, c1 = clouds[i], clouds[i+1]
        T0, T1 = SE3(transforms[i]), SE3(transforms[i+1])
        c0_w = transform_cloud(c0, T0.transform())
        c1_w = transform_cloud(c1, T1.transform())

        #get ground truth relative transformation
        T = T0.inverse()*T1
        c1_0 = transform_cloud(c1, T.transform())

        num_samples = 30
        samples = sample_registrations(c1, c0, T, num_samples)

        mean, cov = calc_mean_cov(samples, T)
        
        A = T1.adj()
        cov_w = T1.transform()[:3,:3]@cov[:3,:3]@(T1.transform()[:3,:3].T) #use this?
        cov_w_A = A@cov@A.T # or this?

        # the mean is T in T0 and T1 in world
        cov_samples.append((T1.transform(), cov_w_A, )) 

        #get sample points 2d to visualize covergence clusters
        sample_points2 = np.array([[samples[i][0,3], samples[i][1,3], 0] for i in range(num_samples)])
        sample_points2_w = transform_cloud(sample_points2, T0.transform())

        plot_results_2d(ax, c0_w, sample_points2_w, T1.transform()[:2,3], cov_w_A[:2,:2])

        
    #save_samples(cov_samples)

    #plot
    ax.relim()
    ax.autoscale_view()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()