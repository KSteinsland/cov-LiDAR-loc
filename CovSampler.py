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
import pickle

""" def calc_mean_cov(tf_list, T_gt):
    n = len(tf_list)
    mean = np.zeros(6)
    cov = np.zeros((6,6))
    for T in tf_list: 
        q = tf.transformations.quaternion_from_matrix(T)
        t = T[:3, 3]
        tf_m = SE3(np.concatenate((t, q)))
        tf_t = (T_gt.inverse()*tf_m).log().coeffs()
        #tf_t = (tf_m*T_gt.inverse()).log().coeffs()
        mean += tf_t
        cov += np.outer(tf_t, tf_t)
    mean = mean / n
    cov = cov / (n - 1)
    t = np.trace(cov)
    return mean, cov """

def TtoSE3(T):
    #fix
    q = tf.transformations.quaternion_from_matrix(T)
    t = T[:3, 3]
    return SE3(np.concatenate((t, q)))



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
    command += " " + str(pc_ref)
    command += " " + str(pc_in)
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
    command += " " + str(pc_ref)
    command += " " + str(pc_in)
    subprocess.run(command, shell=True)

    data = np.genfromtxt(pose_path_str)
    T = data[:4]
    init_T = data[4:]

    data_cov = np.genfromtxt(cov_path_str)
    censi = data_cov[:6]
    bonnabel = data_cov[6:]

    return T, censi, bonnabel

def get_cloud(path):
    return np.genfromtxt(path, delimiter=',')

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

def create_noisy_clouds(cloud_ref, cloud_in, save_path, std_noise):
    cloud = np.genfromtxt(cloud_ref, delimiter=',')
    cloud_noisy = add_noise_to_cloud(cloud, std_noise)
    ref_path = save_path / Path('./ref.csv')
    np.savetxt(ref_path, cloud_noisy, delimiter=",")

    cloud = np.genfromtxt(cloud_in, delimiter=',')
    cloud_noisy = add_noise_to_cloud(cloud, std_noise)
    in_path = save_path / Path('./in.csv')
    np.savetxt(in_path, cloud_noisy, delimiter=",")
    
    return ref_path, in_path

def inv(T):
    T_inv = np.eye(4)
    T_inv[:3, :3] = T[:3, :3].T
    T_inv[:3, 3] = -T_inv[:3, :3].dot(T[:3, 3])
    return T_inv

def cov_registration_sensor_noise(cloud_dataset_path, T_gt, result_dataset_path, num_samples, std_sensor_noise_levels, overwrite=False):
    """
    should:
    - get path to a folder of clouds
    - for every pair 
        - calculate mc_cov, censi_cov
        - store this in folder as following 


    - base
        - sensor_noise_results
            - dataset_20221107..  iput: cloud_dataset_path 
                - noise_lvl_0001
                    - cloud_pair1_covs.p
                    - cloud_pair2_covs.p
                - noise_lvl_0005 

        - odom_noise_results
            - same structure 


        - dataset 
            - working set 


    """

    #make dirs
    os.makedirs(result_dataset_path, exist_ok=True)

    for noise_level in std_sensor_noise_levels:
        s = str(format(noise_level, '.4f')).replace('.','') 
        noise_lvl_result_path = result_dataset_path / Path(f'./noise_lvl_{s}')
        if os.path.exists(noise_lvl_result_path) and not overwrite:
            print(noise_lvl_result_path, " already exist")
            continue
        os.makedirs(noise_lvl_result_path, exist_ok=True)

        clouds_path_list = sorted(glob.glob(str(cloud_dataset_path) + "/*"))

        for cld_indx in range(len(clouds_path_list)-1):
            #if cld_indx != 18: continue
            cloud_ref, cloud_in = clouds_path_list[cld_indx], clouds_path_list[cld_indx+1]
            T_gt_ref, T_gt_in = T_gt[cld_indx], T_gt[cld_indx+1]
            T_rel = T_gt_ref.inverse()*T_gt_in         

            # a place to temorarily store cloud with added noise
            working_cloud_path = result_dataset_path / Path('working_noisy_cloud')
            os.makedirs(working_cloud_path, exist_ok=True)

            #sample cov
            samples = [] 
            for i in range(num_samples):
                noisy_cloud_ref, noisy_cloud_in = create_noisy_clouds(cloud_ref, cloud_in, working_cloud_path, noise_level)
                icp_transform = icp_without_cov(noisy_cloud_ref, noisy_cloud_in, T_rel.transform())
                samples.append(icp_transform)
            
            #censi cov
            noisy_cloud_ref, noisy_cloud_in = create_noisy_clouds(cloud_ref, cloud_in, working_cloud_path, noise_level)
            icp_transform, censi, bonnabel = icp_with_cov(noisy_cloud_ref, noisy_cloud_in, T_rel.transform())
            censi_cov = noise_level **2 * censi

            #save
            cov_result_save_path = noise_lvl_result_path / Path(f'cloud_pair_{cld_indx:03d}.p')
            with open(cov_result_save_path, 'wb') as f:
                pickle.dump((censi_cov, samples, T_rel.transform()), f)

def calc_mean_cov(tf_list, T_gt):
    n = len(tf_list)
    T_bar = TtoSE3(T_gt)
    for t in range(30):
        #compute mean tangent vector at T_bar
        psi_bar = 1/n * sum((T_bar.inverse()*TtoSE3(T)).log().coeffs() 
                    for T in tf_list) 
        #update
        T_bar_last = T_bar
        T_bar = T_bar*(SE3Tangent(psi_bar).exp())

        eps = np.linalg.norm((T_bar.inverse()*T_bar_last).log().coeffs())
        if eps < 1e-16:
            break

    mean = T_bar.log().coeffs()
   
    cov = 1/(n-1) * sum(
         np.outer(
            (T_bar.inverse()*TtoSE3(T)).log().coeffs(),
            (T_bar.inverse()*TtoSE3(T)).log().coeffs())
         for T in tf_list)

    return mean, cov

def results_reg_sens_noise(results_path):
    noise_level_result = sorted(glob.glob(str(results_path) + "/noise_*"))
    noise_levels = [int(n.split('_')[-1])/10000 for n in noise_level_result]
    for i, lvl in enumerate(noise_levels):
        if lvl == 0.0: continue
        results = sorted(glob.glob(str(noise_level_result[i]) + "/*"))
        for r in results:
            #if int(r.split('_')[-1].split('.')[0]) != 18: continue
            with open(r, 'rb') as f:
                censi_cov, samples, T_rel = pickle.load(f)
                sample_mean, sample_cov = calc_mean_cov(samples, T_rel)
                T_bar = SE3Tangent(sample_mean).exp()
                
                #sample_points2d = transform_cloud(sample_points2d, SE3Tangent(sample_mean).exp().inverse().transform())
                samples_rel_Tbar = [ (T_bar.inverse()*TtoSE3(T)).transform() for T in samples]
                sample_points2d = np.array([[s[0,3], s[1,3], 0] for s in samples_rel_Tbar])

                #A = T_bar.inverse().adj() 
                #censi_cov_Tbar = A@censi_cov@A.T
                
                #plot 2d 
                fig, ax = plt.subplots()
                ax.plot(sample_points2d[:,0], sample_points2d[:,1], 'ro', label="samples")
                plot_ellipse(ax, [0,0], sample_cov[:2,:2], fill_color='red', label="sample cov")
                plot_ellipse(ax, [0,0], censi_cov[:2,:2], fill_color='blue', label="censi")
                plt.legend(loc="best")
                plt.title(f"Sensor noise standard deviation: {lvl}")
                #plt.savefig(f'./sample_and_censi_v_noise/clouds{i}and{i+1}_noise_{std_sensor_noise}.png')
                plt.show()

if __name__ == "__main__":
    np.random.seed(0)
    
    #params
    base_path = Path('/home/ksteins/covest/')
    config_path = base_path / Path('./libpointmatcher/martin/config/base_config.yaml')

    cloud_dir = "20221107-185525"
    dataset_clouds_path = base_path / Path("./clouds_csv/") / cloud_dir
    results_path = base_path / Path("./results_new") / cloud_dir
    os.makedirs(results_path, exist_ok=True)

    #get gt transforms
    S = PCSampler()
    S.load_samples()
    clouds = S.get_clouds()
    transforms = S.get_transforms(use_quat=True)
    transforms_se3 = [SE3(T) for T in transforms]

    numb_mc_samples = 100
    std_sensor_noise_levels = [n/1000 for n in range(51)]
    
    #cov_registration_sensor_noise(dataset_clouds_path, transforms_se3, results_path, numb_mc_samples, std_sensor_noise_levels, overwrite=True)
    results_reg_sens_noise(results_path)



            

    


def cov_registration_odom_noise():
    pass



























""" cov_samples.append((T1.transform(), sampeled_cov, censi_cov))
            cov_traces_censi.append(np.trace(censi_cov))
            cov_traces_sample.append(np.trace(sampeled_cov)) """

""" #plot 
    sample_points2d = np.array([[samples[i][0,3], samples[i][1,3], 0] for i in range(num_samples)])
    sample_points2d = transform_cloud(sample_points2d, T.inverse().transform())

    fig, ax = plt.subplots() """


""" #get the two clouds and their tranformations to world
        c0, c1 = clouds[i], clouds[i+1]
        T0, T1 = SE3(transforms[i]), SE3(transforms[i+1])
        c0_w = transform_cloud(c0, T0.transform())
        c1_w = transform_cloud(c1, T1.transform())

        #get ground truth relative transformation
        T = T0.inverse()*T1
        #c1_0 = transform_cloud(c1, T.transform()) """

