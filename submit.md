# 16-825 Assigment 2: Single View to 3D

# 1. Exploring loss functions 

## 1.1 Fitting a Voxel Grid (Optimized voxel grid | Ground Truth)

<image src="submit_images/q1/hw2q1p1_vox_src.gif" width=256>
<image src="submit_images/q1/hw2q1p1_vox_tgt.gif" width=256>

## 1.2 Fitting a point cloud (Optimized point cloud | Ground Truth)

<image src="submit_images/q1/hw2q1p2_pc_src.gif" width=256>
<image src="submit_images/q1/hw2q1p2_pc_tgt.gif" width=256>

## 1.3 Fitting a mesh (Optimized mesh | Ground Truth)

<image src="submit_images/q1/hw2q1p3_mesh_src.gif" width=256>
<image src="submit_images/q1/hw2q1p3_mesh_tgt.gif" width=256>


# 2. Reconstructing 3D from single view

## 2.1 Image to Voxel Grid

TODO Visualize three examples (input RGB, predicted voxel, ground truth mesh)

## 2.2. Image to Point Cloud

TODO Visualize three examples (input RGB, predicted point cloud, ground truth mesh)

## 2.3 Image to Mesh

TODO Visualize three examples (input RGB, predicted mesh, ground truth mesh)

## 2.4 Quantitative comparisons

Here are the F1 score curves of the three different modalities:

<image src="submit_images/q2/eval_vox.png" width=512>

<image src="submit_images/q2/eval_point_10k.png" width=512>

<image src="submit_images/q2/eval_mesh.png" width=512>

Quantitatively, we see that the point cloud method yields the best F1 score out of the three modalities, followed by voxels and finally mesh. TODO talk more abt this

TODO load checkpoints and talk about actual score

## 2.5 Analyze effects of Hyperparam variations

I chose to experiment with the `n_points` parameter for the point cloud model, between 1k, 5k, and 10k points. Here are the F1 score curves:

1K:
<image src="submit_images/q2/eval_point_1k.png" width=512>

5K:
<image src="submit_images/q2/eval_point_5k.png" width=512>

10K:
<image src="submit_images/q2/eval_point_10k.png" width=512>

The loss increases proportionally with the number of points, which is expected due to the chamfer loss being calculated per each point. 

We can clearly see a significant increase in F1 score as the number of points increases. The increased representation capacity allows for a denser reconstruction of the scene that includes additional detail from the scene. The increased number of points also allows the model to cover more volume and have additional points get closer to the evaluation points, which leads to an increased F1 score.

## 2.6 Interpret your model 

TODO see which visualizations highlight what your model does

# 3. Exploring Architectures/Datasets

## 3.3 Extended Dataset for Training

TODO Compare quantitative and qualitative results of training on one class vs multiple classes. How does F1 score change, how does 3D consistency/diversity change
