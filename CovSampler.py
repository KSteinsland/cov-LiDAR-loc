from PCSampler import PCSampler, vis_pc, transform_cloud
from manifpy import SE3, SO3, SE3Tangent
import open3d as o3d
import numpy as np

S = PCSampler()
S.load_samples()
clouds = S.get_clouds()
transforms = S.get_transforms(use_quat=True)

#get the two clouds and their tranformations to world
c0, c1 = clouds[0], clouds[8]
T0, T1 = SE3(transforms[0]), SE3(transforms[8])

#get ground truth relative transformation
T = T0.inverse()*T1
#c1_0 = transform_cloud(c1, T.transform())
#vis_pc([c0, c1_0])

#give T some random perturbations to simulate noisy odometry
cov_pos = 10
cov_yaw = 0.01
p = np.zeros(6)
p[:2] = np.sqrt(cov_pos)*np.random.normal(size=2) # we only perturb x,y
p[5] = np.sqrt(cov_yaw)*np.random.normal(size=1) # we only perturb yaw
T_p = SE3Tangent(p) + T #I guess we want to add p on the left side?

c1_0p = transform_cloud(c1, T_p.transform())
vis_pc([c0, c1_0p])

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
print(icp_transform)
c1_registered = transform_cloud(c1, icp_transform)

vis_pc([c0, c1_registered])


