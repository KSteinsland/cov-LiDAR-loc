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

    data_cov = np.genfromtxt(cov_path_str)
    censi = data_cov[:6]
    bonnabel = data_cov[6:]

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

def sample_registrations(cloud_source, cloud_target, rel_T, num_samples, clouds_path, clouds_noise_path, std_sensor_noise):
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
        #p[:2] = np.sqrt(cov_pos)*np.random.normal(size=2) # we only perturb x,y
        #p[5] = np.sqrt(cov_yaw)*np.random.normal(size=1) # we only perturb yaw
        T_p = SE3Tangent(p) + rel_T #I guess we want to add p on the left side? ask nikhil
        initial_T = T_p.transform()

        add_noise_to_clouds(clouds_path, clouds_noise_path, std_sensor_noise)

        #replace this with the new icp, read and write direct
        #icp_transform = open3d_icp(cloud_source, cloud_target, initial_T)
        icp_transform = icp_without_cov(cloud_source, cloud_target, initial_T)

        samples.append(icp_transform)

    return samples

def save_samples(samples):
    import datetime 
    import pickle
    dt = datetime.datetime.now()
    timestamp = dt.strftime("%Y%m%d-%H%M%S")
    with open(f"covariances/{timestamp}.p", "wb") as f:
        pickle.dump(samples, f)

def plot_ellipse(ax, mean, cov, n=50, chi2_val=9.21, fill_alpha=0., fill_color='lightsteelblue', label=None):
    u, s, _ = np.linalg.svd(cov)
    scale = np.sqrt(chi2_val * s)

    theta = np.linspace(0, 2*np.pi, n+1)
    x = np.cos(theta)
    y = np.sin(theta)

    R = u
    t = np.reshape(mean, [2, 1])
    circle_points = (R * scale) @ np.vstack((x.flatten(), y.flatten())) + t

    ax.fill(circle_points[0, :], circle_points[1, :], alpha=fill_alpha, facecolor=fill_color)
    ax.plot(circle_points[0, :], circle_points[1, :], color=fill_color, label=label + " - 3sigma")

def plot_results_2d(ax, c0_w, sample_points2_w, mean2_w, cov2x2_w_mc, cov2x2_w_censi, cov2x2_w_bonna, z_lim=1):
    ax.plot(c0_w[c0_w[:,2] > z_lim][:,0], c0_w[c0_w[:,2] > z_lim][:,1], 'bo', markersize=0.1)
    ax.plot(sample_points2_w[:,0], sample_points2_w[:,1], 'go', markersize=2)
    plot_ellipse(ax, mean2_w, cov2x2_w_mc)
    plot_ellipse(ax, mean2_w, cov2x2_w_censi, fill_color="red")
    plot_ellipse(ax, mean2_w, cov2x2_w_bonna, fill_color="green")


def add_noise_to_cloud(cloud, std_noise):
    noise = std_noise * np.random.normal(size=cloud.shape)
    return cloud + noise

def add_noise_to_clouds(clouds_path, result_save_path, std_noise):
    clouds_path_list = sorted(glob.glob(str(clouds_path) + "/*"))
    for cloud_path in clouds_path_list:
        cloud = np.genfromtxt(cloud_path, delimiter=',')
        cloud_noisy = add_noise_to_cloud(cloud, std_noise)
        save_path = str(result_save_path) + "/" + cloud_path.split('/')[-1]
        np.savetxt(save_path, cloud_noisy, delimiter=",")

if __name__ == "__main__":
    np.random.seed(0)

    S = PCSampler()
    S.load_samples()
    clouds = S.get_clouds()
    transforms = S.get_transforms(use_quat=True)

    cloud_dir = "20221107-185525"
    clouds_path = base_path / Path("./clouds_csv/") / cloud_dir
    

    std_sensor_noise = 0.01
    clouds_noise_path = base_path / Path("./clouds_csv_noise/") / cloud_dir
    os.makedirs(clouds_noise_path, exist_ok=True)
    add_noise_to_clouds(clouds_path, clouds_noise_path, std_sensor_noise)
    
    clouds_path_list = sorted(glob.glob(str(clouds_noise_path) + "/*"))

    cov_samples = []
    for i in range(len(clouds) - 1):
        """ if i < 10:#18:
            continue """
        print("cloud number ", i, " and ", i+1)
        #get the two clouds and their tranformations to world
        c0, c1 = clouds[i], clouds[i+1]
        T0, T1 = SE3(transforms[i]), SE3(transforms[i+1])
        c0_w = transform_cloud(c0, T0.transform())
        c1_w = transform_cloud(c1, T1.transform())

        #get ground truth relative transformation
        T = T0.inverse()*T1
        #c1_0 = transform_cloud(c1, T.transform())

        num_samples = 100
        samples = sample_registrations(clouds_path_list[i], clouds_path_list[i+1], T, num_samples, clouds_path, clouds_noise_path, std_sensor_noise)
        
        mean, sampeled_cov = calc_mean_cov(samples, T)
        
        #censi
        icp_transform, censi, bonnabel = icp_with_cov(clouds_path_list[i], clouds_path_list[i+1], T.transform())
        A = T.transform()
        censi_cov = std_sensor_noise * censi[:3,:3]

        cov_samples.append((T1.transform(), sampeled_cov, censi_cov))
        
        #plot 
        sample_points2d = np.array([[samples[i][0,3], samples[i][1,3], 0] for i in range(num_samples)])
        sample_points2d = transform_cloud(sample_points2d, T.inverse().transform())

        fig, ax = plt.subplots()

        #ax.set_xlim(-1,1)
        #ax.set_ylim(-1,1)

        ax.plot(sample_points2d[:,0], sample_points2d[:,1], 'ro', label="samples")
        plot_ellipse(ax, [0, 0], sampeled_cov[:2], fill_color='red', label="sample cov")
        plot_ellipse(ax, [0, 0], censi_cov[:2], fill_color='blue', label="censi",)
        plt.legend(loc="best")
        plt.savefig(f'./imgs_cov/clouds{i}and{i+1}.png')
        #plt.show()






















        #A = T1.adj()
        #cov_w = T1.transform()[:3,:3]@cov[:3,:3]@(T1.transform()[:3,:3].T) #use this?
        #cov_w_A = A@cov@A.T # or this?

        #std_sensor = 1 

        #censi_W_A = std_sensor**2 * T1.transform()[:3,:3]@censi[3][:3,:3]@(T1.transform()[:3,:3].T)
        #bonna_W_A = std_sensor**2 * T1.transform()[:3,:3]@bonna[3][:3,:3]@(T1.transform()[:3,:3].T)

        # the mean is T in T0 and T1 in world
        

        #get sample points 2d to visualize covergence clusters
        #sample_points2 = np.array([[samples[i][0,3], samples[i][1,3], 0] for i in range(num_samples)])
        #sample_points2_w = transform_cloud(sample_points2, T0.transform())

        

        #plot_results_2d(ax, c0_w, sample_points2_w, T1.transform()[:2,3], cov_w_A[:2,:2], censi_W_A[:2,:2], bonna_W_A[:2,:2])

        
    #save_samples(cov_samples)

    #plot
    #ax.relim()
    #ax.autoscale_view()
    #plt.gca().set_aspect('equal', adjustable='box')
    #ax.legend(["clouds", "sample points", "sample cov", "censi", "bonnabel"])
    #plt.show(block=True)