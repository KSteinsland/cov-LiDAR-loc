from random import sample
from traceback import print_tb
from turtle import color
from PCSampler import PCSampler, vis_pc, transform_cloud
from manifpy import SE3, SO3, SE3Tangent
import open3d as o3d
import numpy as np
import tf
import matplotlib.pyplot as plt
from pathlib import Path
import os
import glob
import subprocess

#params
base_path = Path('/home/ksteins/covest/')
config_path = base_path / Path('./libpointmatcher/martin/config/base_config.yaml')
results_path = base_path / Path("./results")
os.makedirs(results_path, exist_ok=True)

cloud_dir = "20221005-204202"
clouds_path = base_path / Path("./clouds_csv/") / cloud_dir
clouds = glob.glob(str(clouds_path) + "/*")

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

def str_T(T):
    if T.ndim == 1:
        output = '[' + str(T[0])
        for i in range(1, T.shape[0]):
            output += ',' + str(T[i])
    elif T.ndim == 2:
        output = '[' + str(T[0, 0])
        for i in range(T.shape[0]):
            for j in range(T.shape[1]):
                if j == 0 and i == 0:
                    continue
                output += ',' + str(T[i, j])
    else:
        print('error')
    output += ']'
    return output

def icp_without_cov(pc_ref, pc_in, T_init):
    initTranslation = str_T(T_init[:3, 3])
    initRotation = str_T(T_init[:3, :3])

    base_path = Path('/home/ksteins/covest/')
    lpm_path = base_path / './libpointmatcher/'
    config_yaml = lpm_path / 'martin' / 'config' / "base_config.yaml"
    pose_path = base_path / 'results_icp_without_cov'
    os.makedirs(pose_path, exist_ok=True)

    pose_path_str = str(pose_path) + "/T.txt"

    command = "cd " + str(lpm_path) + "/build/ \n" + " " + "examples/icp_without_cov"
    command += " " + "--config" + " " + str(config_yaml)
    command += " " + "--output" + " " + pose_path_str
    command += " " + "--initTranslation" + " " + initTranslation
    command += " " + "--initRotation" + " " + initRotation
    command += " " + pc_ref
    command += " " + pc_in
    subprocess.run(command, shell=True)

    data = np.genfromtxt(pose_path_str)
    T = data[:4]
    init_T = data[4:]
    return T


def icp_with_cov(pc_ref, pc_in, T_init):
    initTranslation = str_T(T_init[:3, 3])
    initRotation = str_T(T_init[:3, :3])

    base_path = Path('/home/ksteins/covest/')
    lpm_path = base_path / './libpointmatcher/'
    config_path = lpm_path / 'martin' / 'config' / "base_config.yaml"
    pose_path = base_path / 'results_icp_with_cov'
    os.makedirs(pose_path, exist_ok=True)

    pose_path_str = str(pose_path) + "/T.txt"
    cov_path_str = str(pose_path) + "/cov.txt"

    command = "cd " + str(lpm_path) + "/build/ \n" + " " + "examples/icp_with_cov"
    command += " " + "--config" + " " + str(config_path)
    command += " " + "--output" + " " + pose_path_str
    command += " " + "--output_cov" + " " + cov_path_str
    command += " " + "--initTranslation" + " " + initTranslation
    command += " " + "--initRotation" + " " + initRotation
    command += " " + pc_ref
    command += " " + pc_in
    subprocess.run(command, shell=True)

    data = np.genfromtxt(pose_path_str)
    T = data[:4]
    init_T = data[4:]

    data_cov = np.genfromtxt(pose_path_str)
    censi = data_cov[:6]
    bonnabel = data_cov[:6]

    return T, censi, bonnabel

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
        cov_pos = 5e-1
        cov_yaw = 2e-3
        p = np.zeros(6)
        p[:2] = np.sqrt(cov_pos)*np.random.normal(size=2) # we only perturb x,y
        p[5] = np.sqrt(cov_yaw)*np.random.normal(size=1) # we only perturb yaw
        T_p = SE3Tangent(p) + rel_T #I guess we want to add p on the left side? ask nikhil
        initial_T = T_p.transform()

        #replace this with the new icp, read and write direct
        #icp_transform = open3d_icp(cloud_source, cloud_target, initial_T)
        icp_transform= icp_without_cov(cloud_source, cloud_target, initial_T)


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
    clouds = S.get_clouds()[:2]
    transforms = S.get_transforms(use_quat=True)

    cloud_dir = "20221005-204202"
    clouds_path = base_path / Path("./clouds_csv/") / cloud_dir
    clouds_path_list = sorted(glob.glob(str(clouds_path) + "/*"))

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

        num_samples = 20
        #clouds_path_list[1]
        samples = sample_registrations(clouds_path_list[i], clouds_path_list[i+1], T, num_samples)
        #samples = sample_registrations(c1, c0, T, num_samples)
        
        c1_0 = transform_cloud(c1, samples[0])
        #vis_pc([c0, c1_0])

        mean, cov = calc_mean_cov(samples, T)
        
        A = T1.adj()
        cov_w = T1.transform()[:3,:3]@cov[:3,:3]@(T1.transform()[:3,:3].T) #use this?
        cov_w_A = A@cov@A.T # or this?

        # the mean is T in T0 and T1 in world
        cov_samples.append((T1.transform(), cov_w_A)) 

        #get sample points 2d to visualize covergence clusters
        sample_points2 = np.array([[samples[i][0,3], samples[i][1,3], 0] for i in range(num_samples)])
        sample_points2_w = transform_cloud(sample_points2, T0.transform())

        plot_results_2d(ax, c0_w, sample_points2_w, T1.transform()[:2,3], cov_w_A[:2,:2])

        
    #save_samples(cov_samples)

    #plot
    ax.relim()
    ax.autoscale_view()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show(block=True)