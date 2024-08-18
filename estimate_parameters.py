import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def depth_derivatives(old_id, new_id, mask_np=np.array([]), real_dist=1, plot=True):
    """ A method for calculating the roughness of the beam of a ship.
    
    Args:
        old_id (str): The original image name
        new_id (str): The name after saving in store
        mask_np (np.array): The mask of the background
        real_dist (float): The expected mean height

    Returns:
        floats
    """

    ### Not in use?
    # distance_map = img_np[0,0,:]
    # z = distance_map
    # h, w = distance_map.shape
    # aspect_ratio = h/w  
    # x, y = np.meshgrid(np.arange(w), np.arange(h))  # Not in use?
    # point_cloud = np.dstack((x, y, z)).reshape(-1, 3)  # Not in use?

    img = torch.load(os.getcwd() + "/Data/Depth_Estim/" + new_id.rsplit(".")[0]+".pt")
    img_np = img.cpu().numpy()[:]

    if real_dist != 1:
        ratio = real_dist/np.max(img_np[0,0,::])
        points = img_np[0,0,::] * ratio
    else:
        points = img_np[0,0,::]

    if len(mask_np)==0:
        mask_np = np.ones_like(points)
        mask_np = (mask_np==1.)
    
    points[~mask_np] = np.NaN 

    # mean_grad_z = np.mean(np.gradient(points[:,2]))  # Not in use?
    mean_act = np.nanmean(points[:,2])
    ES = np.nanmean(np.abs(np.gradient(points[:,2])))
    ka = np.nanmean(points[:,2] - mean_act)
    
    k_rms = (np.nanmean(np.power(points[:,2] - mean_act,2)))**0.5
    SK = 1/(k_rms**3) * np.nanmean(np.power(points[:,2] - mean_act, 3))
    
    num_bins = int(len(points)**0.5)
    # counts, bin_edges = np.histogram(np.abs(np.gradient(points[:,2])), bins=num_bins)

    # k = 5
    # sorted_indices = np.argsort(counts)[::-1] 
    # top_k_indices = sorted_indices[:k]  

    # bin_means = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in top_k_indices]
    # kz = np.mean(bin_means)

    with open("parameters_results.txt", "a") as f:
        f.write("\n")
        f.write(f"Image {old_id} \n")
        # f.write("Bin means:", bin_means)
        f.write(f"mean: {mean_act} \n")
        f.write(f"std: {np.nanstd(points)} \n")    
        f.write(f"ES: {ES}, ka: {ka}, k_rms: {k_rms}, Sk: {SK} \n")  # , k_z: {kz}
        f.write("\n")

    # if plot:
    #     plot_results(points)
        

def plot_results(points, path):

    f, ax = plt.subplots(2, 2, figsize=(10, 8))

    plt.pcolormesh(points)
    plt.colorbar()
    plt.show()

    ct = ax[0, 0].contourf(points-np.mean(points,axis=0)[None,:],levels=[0.0,0.1,0.2,0.3],alpha=0.3)
    f.colorbar(ct, ax=ax[0, 0])
    
    ct = ax[0, 1].contourf(points-np.mean(points,axis=1)[:,None],levels=[0.0,0.1,0.2,0.3],alpha=0.3)
    f.colorbar(ct, ax=ax[0, 1])
    
    ax[1, 0].plot(np.mean(points,axis=0)); ax[1, 1].plot(np.mean(points,axis=1))

    ax[0, 0].set_title("Axis=0", fontsize='large', pad=20); ax[0, 1].set_title("Axis=1", fontsize='large', pad=20)

    f.text(0.04, 0.5, "Dist from mean", va='center', rotation='vertical', fontsize='large', fondweight='bold')
    f.text(0.04, 0.5, "Mean", va='center', rotation='vertical', fontsize='large', fondweight='bold')

    plt.tight_layout()
    plt.savefig(path)
