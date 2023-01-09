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
from scipy.linalg import block_diag
from ut import *
import time

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['axes.titlesize'] = 'x-large'
matplotlib.rcParams["axes.grid"] = True


def TtoSE3(T):
    q = tf.transformations.quaternion_from_matrix(T)
    t = T[:3, 3]
    return SE3(np.concatenate((t, q)))

def str_T(T):
    #Source: https://github.com/CAOR-MINES-ParisTech/libpointmatcher
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

times_without_cov = [] # for timing
def icp_without_cov(pc_ref, pc_in, T_init):
    #Source: https://github.com/CAOR-MINES-ParisTech/libpointmatcher
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
    start = time.time()
    subprocess.run(command, shell=True)
    timer = time.time() - start
    times_without_cov.append(timer)

    data = np.genfromtxt(pose_path_str)
    T = data[:4]
    init_T = data[4:]
    return T

times_with_cov = [] #for timing
def icp_with_cov(pc_ref, pc_in, T_init):
    #Source: https://github.com/CAOR-MINES-ParisTech/libpointmatcher
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
    start = time.time()
    subprocess.run(command, shell=True)
    timer = time.time() - start
    times_with_cov.append(timer)

    data = np.genfromtxt(pose_path_str)
    T = data[:4]
    init_T = data[4:]

    data_cov = np.genfromtxt(cov_path_str)
    censi = data_cov[:6]
    bonnabel = data_cov[6:]

    return T, censi, bonnabel

def get_cloud(path):
    return np.genfromtxt(path, delimiter=',')

def plot_ellipse(ax, mean, cov, n=50, chi2_val=9.21, fill_alpha=0., fill_color='lightsteelblue', label=None):
    #source: https://github.com/tussedrotten/simple-factorgraph-example
    u, s, v = np.linalg.svd(cov)
    scale = np.sqrt(chi2_val * s)

    theta = np.linspace(0, 2*np.pi, n+1)
    x = np.cos(theta)
    y = np.sin(theta)

    R = u
    t = np.reshape(mean, [2, 1])
    circle_points = (R * scale) @ np.vstack((x.flatten(), y.flatten())) + t

    ax.fill(circle_points[0, :], circle_points[1, :], alpha=fill_alpha, facecolor=fill_color)
    ax.plot(circle_points[0, :], circle_points[1, :], color=fill_color, label=label)

def plot_results_2d(ax, c0_w, sample_points2_w, mean2_w, cov2x2_w_mc, cov2x2_w_censi, cov2x2_w_bonna, z_lim=1):
    ax.plot(c0_w[c0_w[:,2] > z_lim][:,0], c0_w[c0_w[:,2] > z_lim][:,1], 'bo', markersize=0.1)
    ax.plot(sample_points2_w[:,0], sample_points2_w[:,1], 'go', markersize=2)
    plot_ellipse(ax, mean2_w, cov2x2_w_mc)
    plot_ellipse(ax, mean2_w, cov2x2_w_censi, fill_color="red")
    plot_ellipse(ax, mean2_w, cov2x2_w_bonna, fill_color="green")

def add_noise_to_cloud(cloud, std_noise):
    noise = std_noise * np.random.normal(size=cloud.shape)
    return cloud + noise

def add_range_noise_to_cloud(cloud, std_noise):
    dirs = cloud / np.linalg.norm(cloud, axis=1, keepdims=True)
    noise = std_noise * dirs * np.random.normal(size=cloud.shape)
    return cloud + noise

def add_noise_to_clouds(clouds_path, result_save_path, std_noise):
    clouds_path_list = sorted(glob.glob(str(clouds_path) + "/*"))
    for cloud_path in clouds_path_list:
        cloud = np.genfromtxt(cloud_path, delimiter=',')
        cloud_noisy = add_range_noise_to_cloud(cloud, std_noise)
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


def cov_registration_sensor_noise(cloud_dataset_path, T_gt, result_dataset_path, num_samples, std_sensor_noise_levels, ut, odom_noise=None, clouds_mask=None, overwrite=False):
    """
    directories:
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
            - dataset_20221107..
                - std_pos_0001-std_rot_0.01
                        -cloud_pair1_covs.p 


        - dataset 
            - working set 


    """

    #make dirs
    os.makedirs(result_dataset_path, exist_ok=True)

    for noise_level in std_sensor_noise_levels:
        s = str(format(noise_level, '.4f')).replace('.','') 
        noise_lvl_result_path = result_dataset_path / Path(f'./noise_lvl_{s}')
        os.makedirs(noise_lvl_result_path, exist_ok=True)

        clouds_path_list = sorted(glob.glob(str(cloud_dataset_path) + "/*"))

        for cld_indx in range(len(clouds_path_list)-1):
            if clouds_mask != None and cld_indx not in clouds_mask: continue

            cov_result_save_path = noise_lvl_result_path / Path(f'cloud_pair_{cld_indx:03d}.p')
            if os.path.exists(cov_result_save_path) and not overwrite:
                print(cov_result_save_path, " already exist")
                continue

            cloud_ref, cloud_in = clouds_path_list[cld_indx], clouds_path_list[cld_indx+1]
            T_gt_ref, T_gt_in = T_gt[cld_indx], T_gt[cld_indx+1]
            T_rel = T_gt_ref.inverse()*T_gt_in         

            # a place to temorarily store cloud with added noise
            working_cloud_path = result_dataset_path / Path('working_noisy_cloud')
            os.makedirs(working_cloud_path, exist_ok=True)

            #simulate noisy odometry
            T_init = T_rel.transform()
            if odom_noise is not None:
                std_pos, std_rot = odom_noise
                xi = np.hstack((np.random.normal(0, std_pos, 3),
                                np.random.normal(0, std_rot, 3)))
                T_init = (SE3Tangent(xi).exp()*T_rel).transform()

                
            #censi cov
            noisy_cloud_ref, noisy_cloud_in = create_noisy_clouds(cloud_ref, cloud_in, working_cloud_path, noise_level)
            icp_transform, censi, bonnabel = icp_with_cov(noisy_cloud_ref, noisy_cloud_in, T_init)
            censi_cov = (noise_level **2) * censi 


            #Brossard cov
            sps = ut.sp.sigma_points(ut.Q_prior)
            # unscented transform
            T_ut = []
            for n in range(13):
                T_init_n = (SE3Tangent(sps[n]).exp()*T_rel).transform() #SE3.normalize(SE3.mul(SE3.exp(sps[n]), T_init))  #  T_sp = exp(xi) T_hat
                icp_transform = icp_without_cov(noisy_cloud_ref, noisy_cloud_in, T_init_n)
                T_ut.append(TtoSE3(icp_transform))
            _, _, cov_ut, _ = ut.unscented_transform_se3(T_ut)
            cov_brossard = (noise_level ** 2) * bonnabel + cov_ut            

            #sample cov
            samples = [] 
            for i in range(num_samples):
                
                T_init = T_rel.transform()
                if odom_noise is not None:
                    std_pos, std_rot = odom_noise
                    xi = np.hstack((np.random.normal(0, std_pos, 3),
                                    np.random.normal(0, std_rot, 3)))
                    T_init = (SE3Tangent(xi).exp()*T_rel).transform()

                noisy_cloud_ref, noisy_cloud_in = create_noisy_clouds(cloud_ref, cloud_in, working_cloud_path, noise_level)
                icp_transform = icp_without_cov(noisy_cloud_ref, noisy_cloud_in, T_init)
                samples.append(icp_transform)
            
            #save
            with open(cov_result_save_path, 'wb') as f:
                pickle.dump((censi_cov, cov_brossard, samples, T_rel.transform()), f)

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

        epsilon = np.linalg.norm((T_bar.inverse()*T_bar_last).log().coeffs())
        if epsilon < 1e-16:
            break

    mean = T_bar.log().coeffs()
   
    cov = 1/(n-1) * sum(
         np.outer(
            (T_bar.inverse()*TtoSE3(T)).log().coeffs(),
            (T_bar.inverse()*TtoSE3(T)).log().coeffs())
         for T in tf_list)

    return mean, cov


def results_reg_sens_noise(results_path, results_figures_path, save=False, clouds_mask=None, noise_levels_mask=None, plot_cov=False, plot_trace=True):
    noise_level_result = sorted(glob.glob(str(results_path) + "/noise_*"))
    noise_levels = ([int(n.split('_')[-1])/10000 for n in noise_level_result])
    #filter noise levels after mask
    if noise_levels_mask != None: 
        noise_levels = list(set(noise_levels).intersection(noise_levels_mask))
        noise_level_result = list(filter(lambda n: int(n.split('_')[-1])/10000 in noise_levels_mask, noise_level_result))
    noise_levels = np.sort(noise_levels)

    traces_sample_cov_avg_pos = []
    traces_censi_cov_avg_pos = []
    traces_brossard_cov_avg_pos = []
    traces_sample_cov_avg_rot = []
    traces_censi_cov_avg_rot = []
    traces_brossard_cov_avg_rot = []
    angle_errors_bross_avg = []
    angle_errors_censi_avg = []

    for i, lvl in enumerate(noise_levels):
        results = sorted(glob.glob(str(noise_level_result[i]) + "/*"))

        
        traces_sample_cov_pos = []
        traces_censi_cov_pos = []
        traces_brossard_cov_pos = []
        traces_sample_cov_rot = []
        traces_censi_cov_rot = []
        traces_brossard_cov_rot = []
        angle_errors_censi = []
        angle_errors_bross = []

        for r in results:
            #for every registraion at this noise level
            scan_number = int(r.split('_')[-1].split('.')[0])
            # only do clouds in mask
            if clouds_mask != None and scan_number not in clouds_mask: continue
            with open(r, 'rb') as f:
                censi_cov, brossard_cov, samples, T_rel = pickle.load(f)
                #censi_cov, brossard_cov = censi_cov*1e1, brossard_cov*1e1

                sample_mean, sample_cov = calc_mean_cov(samples, T_rel)

                #transform mean and covariance
                T_bar = TtoSE3(T_rel)
                A = T_bar.adj()
                sample_cov = A@sample_cov@A.T
                sample_mean = (SE3Tangent(sample_mean).exp()*T_bar.inverse()).transform()
                origin = sample_mean[:2,3]

                #transfrom sample points and get 2d
                sample_points2d = np.zeros((len(samples), 2))
                T_rel_inv = T_bar.inverse()
                for n in range(len(samples)):
                    mc = (TtoSE3(samples[n])*T_rel_inv)
                    sample_points2d[n] = mc.transform()[:2,3]    
            
                #trace
                t_censi_pos, t_censi_rot = np.trace(censi_cov[:3,:3]), np.trace(censi_cov[3:,3:])
                t_sample_pos, t_sample_rot = np.trace(sample_cov[:3,:3]), np.trace(sample_cov[3:,3:])
                t_brossard_pos, t_brossard_rot = np.trace(brossard_cov[:3,:3]), np.trace(brossard_cov[3:,3:])
                traces_censi_cov_pos.append(t_censi_pos)
                traces_sample_cov_pos.append(t_sample_pos)
                traces_brossard_cov_pos.append(t_brossard_pos)
                traces_censi_cov_rot.append(t_censi_rot)
                traces_sample_cov_rot.append(t_sample_rot)
                traces_brossard_cov_rot.append(t_brossard_rot)
                
                #eigenvalues
                wc, ec = np.linalg.eig(censi_cov[:2,:2])
                ws, es = np.linalg.eig(sample_cov[:2,:2])
                wb, eb = np.linalg.eig(brossard_cov[:2,:2])

                max_c, min_c = wc.argmax(), wc.argmin()
                max_s, min_s = ws.argmax(), ws.argmin()
                max_b, min_b = wb.argmax(), wb.argmin()

                d_angle_c = np.arccos(ec[:,max_c].dot(es[:,max_s]))
                d_angle_c = np.round(d_angle_c * 180 / np.pi, 0)
                if d_angle_c > 90: d_angle_c = 180 - d_angle_c
                angle_errors_censi.append(d_angle_c)

                d_angle_b = np.arccos(eb[:,max_b].dot(es[:,max_s]))
                d_angle_b = np.round(d_angle_b * 180 / np.pi, 1)
                if d_angle_b > 90: d_angle_b = 180 - d_angle_b
                angle_errors_bross.append(d_angle_b)
                
                print(f"""
                    clouds {scan_number} and {scan_number+1}, Sensor noise: {lvl}.
                    Trace sample cov pos: {t_sample_pos:.3e}, rot: {t_sample_rot:.3e},
                    Censi cov: angle error {d_angle_c}, trace pos {t_censi_pos:.3e}, trace rot {t_censi_rot:.3e},
                    Brossard cov: angle error {d_angle_b}, trace pos {t_brossard_pos:.3e}, trace rot {t_brossard_rot:.3e},
                    sample mean: {np.linalg.norm(sample_mean[:2,3])}\n""")
                
                with np.printoptions(precision=1):
                    print("sample cov: ", (sample_cov))
                    print("censis cov: ", (censi_cov))
                    print("brossa cov: ", (brossard_cov))

                
                if plot_cov: 
                    #plot cov 2d 
                    fig, ax = plt.subplots()
                    ax.set_xlabel("x [m]")
                    ax.set_ylabel("y [m]")

                    plot_ellipse(ax, origin, censi_cov[:2,:2], fill_color='blue', label="Censi's cov $3\sigma$.")
                    plot_ellipse(ax, origin, brossard_cov[:2,:2], fill_color='green', label="Brossard's cov $3\sigma$.")

                    ax.plot(sample_points2d[:,0], sample_points2d[:,1], 'r.', label="Samples.")
                    plot_ellipse(ax, origin, sample_cov[:2,:2], fill_color='red', label="Sample cov $3\sigma$.")
                    
                    """ a_scale, a_width = 2, 1e-2
                    ax.arrow(*[0,0], *ec[:,max_c]*np.sqrt(wc[max_c])*a_scale, width=np.sqrt(wc[max_c])*a_width, color="darkblue")
                    ax.arrow(*[0,0], *ec[:,min_c]*np.sqrt(wc[min_c])*a_scale, width=np.sqrt(wc[min_c])*a_width, color="b")

                    ax.arrow(*[0,0], *es[:,max_s]*np.sqrt(ws[max_s])*a_scale, width=np.sqrt(ws[max_s])*a_width, color="darkred")
                    ax.arrow(*[0,0], *es[:,min_s]*np.sqrt(ws[min_s])*a_scale, width=np.sqrt(ws[min_s])*a_width, color="r") """
                    
                    plt.legend(loc="best")
                    #plt.title("Covariances")
                    
                
                    if save:
                        lvl_s = str(format(lvl, '.4f')).replace('.','') 
                        save_path = results_figures_path / Path(f'./clouds{scan_number}and{scan_number+1}_noise_{lvl_s}.svg') 
                        plt.savefig(save_path, format="svg")
                    else: plt.show()
        
        """ traces_sample_cov_avg_pos.append(np.average(traces_sample_cov_pos))
        traces_censi_cov_avg_pos.append(np.average(traces_censi_cov_pos))
        traces_brossard_cov_avg_pos.append(np.average(traces_brossard_cov_pos)) 
        traces_sample_cov_avg_rot.append(np.average(traces_sample_cov_rot))
        traces_censi_cov_avg_rot.append(np.average(traces_censi_cov_rot))
        traces_brossard_cov_avg_rot.append(np.average(traces_brossard_cov_rot))  """
        angle_errors_censi_avg.append(np.average(angle_errors_censi))
        angle_errors_bross_avg.append(np.average(angle_errors_bross))

    
    trace_MSE_censi_pos = np.square(np.subtract(np.log(np.array(traces_censi_cov_pos) ), np.log(np.array(traces_sample_cov_pos)))).mean()
    trace_MSE_bross_pos = np.square(np.subtract(np.log(np.array(traces_brossard_cov_pos)), np.log(np.array(traces_sample_cov_pos)))).mean()
    trace_MSE_censi_rot = np.square(np.subtract(np.log(np.array(traces_censi_cov_rot)), np.log(np.array(traces_sample_cov_rot)))).mean()
    trace_MSE_bross_rot = np.square(np.subtract(np.log(np.array(traces_brossard_cov_rot)), np.log(np.array(traces_sample_cov_rot)))).mean()

    #plot traces
    print("trace MSE censi pos: ", trace_MSE_censi_pos)
    print("trace MSE censi rot: ", trace_MSE_censi_rot)
    print()
    print("trace MSE bross pos: ", trace_MSE_bross_pos)
    print("trace MSE bross rot: ", trace_MSE_bross_rot)
   
    print("average angle error censi: ", np.average(angle_errors_censi_avg))
    print("average angle error bross: ", np.average(angle_errors_bross_avg))
    if plot_trace:
        #censi
        fig, ax = plt.subplots()
        ax.plot(noise_levels, traces_sample_cov_avg_pos, label="Trace sample cov.")
        ax.plot(noise_levels, traces_censi_cov_avg_pos, label="Trace censi's cov.")
        plt.title("Sampeled variance vs Censi's estimate.")
        plt.xlabel("Sensor noise standard deviation $\sigma_{sens}$ [m].")
        plt.legend(loc="best")
        if save: plt.savefig(results_figures_path / Path("./noiseCovCensi.svg", format="svg"))
        else: plt.show()

        fig, ax = plt.subplots()
        ax.plot(noise_levels, angle_errors_censi_avg, label="")
        plt.xlabel("Sensor noise standard deviation $\sigma_{sens}$ [m].")
        plt.title("Angle error degrees.")
        if save: plt.savefig(results_figures_path / Path("./angleErrorCensi.svg", format="svg"))
        else: plt.show()

        #brossard
        fig, ax = plt.subplots()
        ax.plot(noise_levels, traces_sample_cov_avg_pos, label="Trace sample cov.")
        ax.plot(noise_levels, traces_brossard_cov_avg_pos, label="Trace Brossard's cov.")
        plt.title("Sampeled variance vs Brossard's estimate.")
        plt.xlabel("Sensor noise standard deviation $\sigma_{sens}$ [m].")
        plt.legend(loc="best")
        if save: plt.savefig(results_figures_path / Path("./noiseCovBross.svg", format="svg"))
        else: plt.show()

        fig, ax = plt.subplots()
        ax.plot(noise_levels, angle_errors_bross_avg, label="")
        plt.xlabel("Sensor noise standard deviation $\sigma_{sens}$ [m].")
        plt.title("Angle error degrees.")
        if save: plt.savefig(results_figures_path / Path("./angleErrorBross.svg", format="svg"))
        else: plt.show()

if __name__ == "__main__":
    np.random.seed(0)
    
    #params
    base_path = Path('/home/ksteins/covest/')
    config_path = base_path / Path('./libpointmatcher/martin/config/base_config.yaml')

    cloud_dir = "20221107-185525"
    dataset_clouds_path = base_path / Path("./clouds_csv/") / cloud_dir
    results_path_sensor_noise = base_path / Path("./results/result_brossard") / cloud_dir
    results_figures_path = base_path / Path("./imgs/compSVG/")
    os.makedirs(results_figures_path, exist_ok=True)
    os.makedirs(results_path_sensor_noise, exist_ok=True)
    
    #get gt transforms
    S = PCSampler()
    S.load_samples()
    clouds = S.get_clouds()
    transforms = S.get_transforms(use_quat=True)
    transforms_se3 = [SE3(T) for T in transforms]

    numb_mc_samples = 30
    clouds_mask = [17,18,19,20] #list of clouds to use in dataset
    std_sensor_noise_levels = [n/1000 for n in range(51)]#[0.04]##[0.02] # sigma sensor #0.014
    odom_noise = (0.08, 0.04) #pos, rot

    #ut
    cov_Q_odo = block_diag(odom_noise[0]**2 * np.eye(3), odom_noise[1]**2 * np.eye(3))
    cov_ut = UTSE3(6, 1, 2, 0, cov_Q_odo)  # parameter of unscented transform

    #cov_registration_sensor_noise(dataset_clouds_path, transforms_se3, results_path_sensor_noise, 
    #    numb_mc_samples, std_sensor_noise_levels, cov_ut, clouds_mask=clouds_mask, odom_noise=odom_noise, overwrite=False)
    results_reg_sens_noise(results_path_sensor_noise, results_figures_path, noise_levels_mask=std_sensor_noise_levels, 
        clouds_mask=clouds_mask, plot_cov=False, plot_trace=True, save=True)
 
    #print(times_with_cov)
    print(len(times_with_cov))
    print(np.average(np.array(times_with_cov)), "\n")

    #print(times_without_cov)
    print(len(times_without_cov))
    print(np.average(np.array(times_without_cov)), "\n")
        

