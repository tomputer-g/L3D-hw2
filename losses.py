import torch

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

	#Can use pytorch3d.ops.knn.knn_gather or knn_points
	return loss_chamfer

def smoothness_loss(mesh_src):
	# loss_laplacian = 
	# implement laplacian smoothening loss

	# Can use predefined loss from pytorch
	return loss_laplacian