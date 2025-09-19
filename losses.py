import torch
from pytorch3d.ops.knn import knn_points, knn_gather

# define losses
def voxel_loss(voxel_src, voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	# loss = 
	# implement some loss for binary voxel grids
	# Can use predefined loss from pytorch
	# Maximize log likelihod L06 P49
	# -wn (y log x + (1-y) log(1-x))
	loss_fn = torch.nn.BCELoss(reduction='mean')
	loss = loss_fn(voxel_src, voxel_tgt)
	return loss

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  
	# loss_chamfer = 
	# implement chamfer loss from scratch
 
	dists, idx, nn = knn_points(p1=point_cloud_src, p2=point_cloud_tgt, lengths1=None, lengths2=None, norm=2, K=1, return_nn=False)
	# nn[n, i, k] gives kth nn for p1[n, i]
	L2_src = torch.sum(dists)
 
	dists, idx, nn = knn_points(p1=point_cloud_tgt, p2=point_cloud_src, lengths1=None, lengths2=None, norm=2, K=1, return_nn=False)
	L2_dst = torch.sum(dists)
 
	loss_chamfer = L2_src + L2_dst
	# L2_src = torch.linalg.norm(point_cloud_src - nn)

	# sum over points in S1 of L2 dist to closest point in S2, then other way around

	#Can use pytorch3d.ops.knn.knn_gather or knn_points
	return loss_chamfer

def smoothness_loss(mesh_src):
	# loss_laplacian = 
	# implement laplacian smoothening loss

	# Can use predefined loss from pytorch
	return loss_laplacian