"""
convert pointclouds to correct csv

run libpointmatcher
"""

import numpy as np
from PCSampler import PCSampler, vis_pc, transform_cloud
from pathlib import Path
import subprocess
import os
import glob

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

def icp_with_cov(pc_ref, pc_in, T_init, config_path, pose_path, cov_path):
    initTranslation = str_T(T_init[:3, 3])
    initRotation = str_T(T_init[:3, :3])

    lpm_path = '/home/ksteins/covest/libpointmatcher/'

    command = "cd " + lpm_path + "build/ \n" + " " + "examples/icp_with_cov"
    command += " " + "--config" + " " + config_path
    command += " " + "--output" + " " + pose_path
    command += " " + "--output_cov" + " " + cov_path
    command += " " + "--initTranslation" + " " + initTranslation
    command += " " + "--initRotation" + " " + initRotation
    command += " " + pc_ref
    command += " " + pc_in
    subprocess.run(command, shell=True)


if __name__ == '__main__':
    base_path = Path('/home/ksteins/covest/')
    config_path = base_path / Path('./libpointmatcher/martin/config/base_config.yaml')
    results_path = base_path / Path("./results")
    os.makedirs(results_path, exist_ok=True)

    cloud_dir = "20221005-204202"
    clouds_path = base_path / Path("./clouds_csv/") / cloud_dir
    clouds = glob.glob(str(clouds_path) + "/*")

    for i in range(len(clouds) - 1):
        c0 = clouds[i]
        c1 = clouds[i + 1]
        
        results_cloud_path = results_path / cloud_dir / f'cloud{i}'
        os.makedirs(results_cloud_path, exist_ok=True)
        pose_path = results_cloud_path / "T_censi.txt"
        cov_path = results_cloud_path / "cov_censi.txt"
       

        icp_with_cov(str(c0), str(c1), np.eye(4), str(config_path), str(pose_path), str(cov_path))