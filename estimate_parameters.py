import torch
import os
import numpy as np
import matplotlib.pyplot as plt



def depth_derivatives(img_pt, real_dist, plot=True):
    """ A method for calculating the roughness of the beam of a ship.
    
    Args:
        img_pt (torch.tensor): A tensor with heights (distance of each pixel to the camera)
        real_dist (float): The expected mean height

    Returns:
        floats
    """

    pt_path = os.path.getcwd() + "/Data/" + img_pt
    tsr = torch.load(pt_path).cpu()

    test_array = tsr.numpy()[:]
    distance_map = test_array[0,0,:]

    z, (h, w) = distance_map, distance_map.shape
    aspect_ratio = h/w
    
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    point_cloud = np.dstack((x, y, z)).reshape(-1, 3)

    ratio = real_dist/np.max(test_array[0,0,::])
    points = test_array[0,0,::] * ratio

    # mean_grad_z = np.mean(np.gradient(points[:,2]))
    mean_act = np.mean(points[:,2])
    ES = np.mean(np.abs(np.gradient(points[:,2])))
    ka = np.mean(points[:,2] - mean_act)
    
    k_rms = (np.mean(np.power(points[:,2] - mean_act,2)))**0.5
    SK = 1/(k_rms**3) * np.mean(np.power(points[:,2] - mean_act,3))
    
    num_bins = int(len(points)**0.5)
    counts, bin_edges = np.histogram(np.abs(np.gradient(points[:,2])), bins=num_bins)

    k = 5
    sorted_indices = np.argsort(counts)[::-1] 
    top_k_indices = sorted_indices[:k]  

    bin_means = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in top_k_indices]
    print(bin_means)
    kz = np.mean(bin_means)
    print("std: ",np.std(points))
    print(f"ES: {ES}, ka: {ka}, k_rms: {k_rms}, Sk: {SK}, k_z: {kz}")

    if plot:
        plot_results(points)
        

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