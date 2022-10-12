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

def sample_registrations(cloud_source, cloud_target, rel_T, num_samples):
    samples = [] 
    for i in range(num_samples):
        #give T some random perturbations to simulate noisy odometry
        cov_pos = 0.5
        cov_yaw = 0.01
        p = np.zeros(6)
        p[:2] = np.sqrt(cov_pos)*np.random.normal(size=2) # we only perturb x,y
        p[5] = np.sqrt(cov_yaw)*np.random.normal(size=1) # we only perturb yaw
        T_p = SE3Tangent(p) + rel_T #I guess we want to add p on the left side? ask nikhil

        #convert clouds to open3d clouds
        pcd0 = o3d.geometry.PointCloud()
        pcd0.points = o3d.utility.Vector3dVector(cloud_target)
        pcd0.estimate_normals()

        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(cloud_source)
        pcd0.estimate_normals()

        #do registration using open3ds point to plane icp for now
        treshold = 1.5
        initial_T = T_p.transform()
        reg = o3d.pipelines.registration.registration_icp(
            pcd1, pcd0, treshold, initial_T, 
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        icp_transform = reg.transformation
        samples.append(icp_transform)
    return samples

def ellipse_params(covmat2x2):
    a = covmat2x2[0,0]
    b = covmat2x2[0,1]
    c = covmat2x2[1,1]
    width  = (a+c)/2 + np.sqrt(((a-c)/2)**2 + b**2)
    length = (a+c)/2 - np.sqrt(((a-c)/2)**2 + b**2)
    angle = 0
    if b == 0 and a >= c:
        angle = 0
    elif b == 0 and a < c:
        angle = np.pi / 2
    else:
        angle = np.arctan2(width - a, b)
    return np.sqrt(width), np.sqrt(length), angle

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


if __name__ == "__main__":
    np.random.seed(1)

    S = PCSampler()
    S.load_samples()
    clouds = S.get_clouds()
    transforms = S.get_transforms(use_quat=True)

    """ cov_samples = []
    for i in range(len(clouds) - 1):
        #get the two clouds and their tranformations to world
        c0, c1 = clouds[i], clouds[i+1]
        T0, T1 = SE3(transforms[i]), SE3(transforms[i+1])

        #get ground truth relative transformation
        T = T0.inverse()*T1
        c1_0 = transform_cloud(c1, T.transform())

        num_samples = 30
        samples = sample_registrations(c1, c0, T, num_samples)

        mean, cov = calc_mean_cov(samples, T)
        print(mean)
        with np.printoptions(precision=3, suppress=True):
            print(cov)
        cov_samples.append((mean, cov, T0.transform()))
    
    save_samples(cov_samples) """

    nc1, nc2 = 12, 20
    c0, c1 = clouds[nc1], clouds[nc2]
    T0, T1 = SE3(transforms[nc1]), SE3(transforms[nc2])
    c0_w = transform_cloud(c0, T0.transform())
    c1_w = transform_cloud(c1, T1.transform())

    #get ground truth relative transformation
    T = T0.inverse()*T1
    c1_0 = transform_cloud(c1, T.transform())

    num_samples = 10
    samples = sample_registrations(c1, c0, T, num_samples)

    mean, cov = calc_mean_cov(samples, T)

    A = T1.adj()
    cov_w = T1.transform()[:3,:3]@cov[:3,:3]@(T1.transform()[:3,:3].T)
    cov_w_A = A@cov@A.T

    with np.printoptions(precision=4, suppress=True):
        print(cov_w, '\n')
        print(cov_w_A[:3,:3])

        # the mean and samples are taken in the tangent space of c1
        # so we need to transform to world 
        #mean_w = T1.adj()@mean
        #print(mean_w)   

   
    #project samples to visualize covergence clusters
    projected_samples = np.array([[samples[i][0,3], samples[i][1,3], 0] for i in range(num_samples)])
    projected_samples_w = transform_cloud(projected_samples, T0.transform())
    #vis_pc([c0, c1_0, projected_samples, np.array([[mean[0], mean[1], 5]])])

    fig, ax = plt.subplots()
    ax.plot(c0_w[:,0], c0_w[:,1], 'bo', markersize=0.1)
    ax.plot(c1_w[:,0], c1_w[:,1], 'ro', markersize=0.1)
    ax.plot(projected_samples_w[:,0], projected_samples_w[:,1], 'go', markersize=2)
    #ax.add_artist(e)
    plot_ellipse(ax, T1.transform()[:2,3], cov_w[:2,:2], chi2_val=5)
    plot_ellipse(ax, T1.transform()[:2,3], cov_w_A[:2,:2], chi2_val=5, fill_color='red')
    ax.relim()
    ax.autoscale_view()
    plt.gca().set_aspect('equal', adjustable='box')
    #ax.set_xlim(20, 25)
    #ax.set_ylim(-2.5, 2.5)
    #ax.set_xlim(min(ax.get_xlim()[0], ax.get_ylim()[0], max(ax.get_xlim()[1], ax.get_ylim()[1])))
    #ax.set_ylim(ax.get_xlim()[0], ax.get_xlim()[1])
    
    plt.show()




""" 
    from matplotlib.patches import Ellipse

    cov_w = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0 , 0]])

    w, l, a = ellipse_params(cov_w[:2,:2])
    print(w,l,a)
    e = Ellipse(T1.transform()[:2,3], w, l, angle=np.rad2deg(a))
    e.set_fill(False) """