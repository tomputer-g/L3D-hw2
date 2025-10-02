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

F1@0.05 score: 73.4

| Input Image | Ground Truth Mesh | Predicted Voxel Grid |
|-------------|------------------|----------------------|
| <image src="submit_images/q2/vox/0_rgb.png" width=256> | <image src="submit_images/q2/vox/0_actual.gif" width=256> | <image src="submit_images/q2/vox/pred_0_vox.gif" width=256> |
| <image src="submit_images/q2/vox/200_rgb.png" width=256> | <image src="submit_images/q2/vox/200_actual.gif" width=256> | <image src="submit_images/q2/vox/pred_200_vox.gif" width=256> |
| <image src="submit_images/q2/vox/400_rgb.png" width=256> | <image src="submit_images/q2/vox/400_actual.gif" width=256> | <image src="submit_images/q2/vox/pred_400_vox.gif" width=256> |

## 2.2. Image to Point Cloud

F1@0.05 score is 87.9 (with n_points = 10k).

| Input Image | Ground Truth Mesh | Predicted Point Cloud |
|-------------|------------------|----------------------|
| <image src="submit_images/q2/point/0_rgb.png" width=256> | <image src="submit_images/q2/point/0_actual.gif" width=256> | <image src="submit_images/q2/point/pred_0_point.gif" width=256> |
| <image src="submit_images/q2/point/100_rgb.png" width=256> | <image src="submit_images/q2/point/100_actual.gif" width=256> | <image src="submit_images/q2/point/pred_100_point.gif" width=256> |
| <image src="submit_images/q2/point/200_rgb.png" width=256> | <image src="submit_images/q2/point/200_actual.gif" width=256> | <image src="submit_images/q2/point/pred_200_point.gif" width=256> |

## 2.3 Image to Mesh

F1@0.05: 73.6.

| Input Image | Ground Truth Mesh | Predicted Voxel Grid |
|-------------|------------------|----------------------|
| <image src="submit_images/q2/mesh/0_rgb.png" width=256> | <image src="submit_images/q2/mesh/0_actual.gif" width=256> | <image src="submit_images/q2/mesh/pred_0_mesh.gif" width=256> |
| <image src="submit_images/q2/mesh/200_rgb.png" width=256> | <image src="submit_images/q2/mesh/200_actual.gif" width=256> | <image src="submit_images/q2/mesh/pred_200_mesh.gif" width=256> |
| <image src="submit_images/q2/mesh/400_rgb.png" width=256> | <image src="submit_images/q2/mesh/400_actual.gif" width=256> | <image src="submit_images/q2/mesh/pred_400_mesh.gif" width=256> |


## 2.4 Quantitative comparisons

| Voxel Grid (F1 = 73.4)| Point Cloud (F1 = 87.9) | Mesh (F1 = 73.6) |
|-------------|------------------|----------------------|
| <image src="submit_images/q2/F1/eval_vox.png" width=512> | <image src="submit_images/q2/F1/eval_point.png" width=512> | <image src="submit_images/q2/F1/eval_mesh.png" width=512> |


Quantitatively, we see that the point cloud method yields the best F1 score out of the three modalities, followed by mesh and finally voxel grid. The point cloud predictions are the most free-form and able to capture subtle detail. Compared to this, meshes are restricted by the requirement of connectivity between vertices as well as smoothness losses, and voxels are constrained on resolution by the 32x32x32 grid of predictions. Visually, we can also see that some additional loss or architecture definitions may be required to further optimize the predicted meshes and constrain the connectivity between vertices, as currently it produces a jagged prediction for chairs.

## 2.5 Analyze effects of Hyperparam variations

I chose to experiment with the `n_points` parameter for the point cloud model, between 1k, 5k, and 10k points. Here are the F1 score curves:
| 1K Points | 5K Points | 10K Points |
|-----------|-----------|------------|
| <image src="submit_images/q2/F1/eval_point_1k.png" width=512> | <image src="submit_images/q2/F1/eval_point_5k.png" width=512> | <image src="submit_images/q2/F1/eval_point.png" width=512> |

The loss increases proportionally with the number of points, which is expected due to the chamfer loss being calculated per each point. 

We can clearly see a significant increase in F1 score as the number of points increases. The increased representation capacity allows for a denser reconstruction of the scene that includes additional detail from the scene. The increased number of points also allows the model to cover more volume and predict additional points that are nearer to the evaluation points, which leads to an increased F1 score.

## 2.6 Interpret your model 

I chose to visualize the effect of the marching cubes isovalue on the rendered meshes from the voxel prediction outputs. This was interesting as a concept because the network outputs probabilities of a voxel being occupied or not, and this is essentially a confidence value from 0 to 1 of how likely the network thinks the volume is occupied by part of the chair. 

Changing the isovalue parameter allows us to visualize how confident the voxel sv3d network is, by thresholding the confidence threshold for mesh prediction. Thus, a smaller isovalue threshold should allow more volume to be occupied, creating an overly large resulting mesh, while a larger threshold should shrink the volume so that only the most 'confident' volumes are occupied.

| Image | 0.1 Threshold (largest) | 0.3 Threshold | 0.5 Threshold (normal) | 0.7 Threshold | 0.9 Threshold (smallest) |
|---------------|---------------|---------------|-----------------------|---------------|---------------|

| <image src="submit_images/q2/interpret/0_rgb.png" width=256> | <image src="submit_images/q2/interpret/pred_0_vox_iso_10.gif" width=256> | <image src="submit_images/q2/interpret/pred_0_vox_iso_30.gif" width=256> | <image src="submit_images/q2/interpret/pred_0_vox_iso_50.gif" width=256> | <image src="submit_images/q2/interpret/pred_0_vox_iso_70.gif" width=256> | <image src="submit_images/q2/interpret/pred_0_vox_iso_90.gif" width=256> |
| <image src="submit_images/q2/interpret/200_rgb.png" width=256> | <image src="submit_images/q2/interpret/pred_200_vox_iso_10.gif" width=256> | <image src="submit_images/q2/interpret/pred_200_vox_iso_30.gif" width=256> | <image src="submit_images/q2/interpret/pred_200_vox_iso_50.gif" width=256> | <image src="submit_images/q2/interpret/pred_200_vox_iso_70.gif" width=256> | <image src="submit_images/q2/interpret/pred_200_vox_iso_90.gif" width=256> |
| <image src="submit_images/q2/interpret/400_rgb.png" width=256> | <image src="submit_images/q2/interpret/pred_400_vox_iso_10.gif" width=256> | <image src="submit_images/q2/interpret/pred_400_vox_iso_30.gif" width=256> | <image src="submit_images/q2/interpret/pred_400_vox_iso_50.gif" width=256> | <image src="submit_images/q2/interpret/pred_400_vox_iso_70.gif" width=256> | <image src="submit_images/q2/interpret/pred_400_vox_iso_90.gif" width=256> |

We can see that for the large sofa (top row), the network is very confident about the seatback, armrests, and seat of the sofa, whereas for the bottom two instances the network is much less confident about the resulting prediction of the chair structure. The middle row folding chair was challenging for the network to predict, which may suggest that more training weight or dataset balancing is required for this type of chair. We can also see that for the bottom image (where the input image is the back of the chair), the network is very unsure of the structure with regards to the front and the seat of the chair, which is valid given the invisibility of these features, but a good first guess at the location of armrests and seats are visualized with a lower threshold for isovalue.

# 3. Exploring Architectures/Datasets

## 3.3 Extended Dataset for Training

Trained on chair: F1@0.05 is 83.0 on average. The F1 score for chair are typically >95, but on the chair and car classes, the network struggles and the f1 score dips as low as 5 on specific instances. As shown below, the network (trained solely on chairs) could only reconstruct chair-like shapes even when the input image is a car or a plane. The chair back is predicted as if the plane's tail or the car's spoiler are backs of chairs, and similarly, chair legs are predicted even though the car image and plane image do not imply existence of legs.


| Input Image | Ground Truth Mesh | Predicted Point Cloud |
|-------------|------------------|----------------------|
| <image src="submit_images/q3/chair_only_ckpt/0_rgb.png" width=256> | <image src="submit_images/q3/chair_only_ckpt/0_actual.gif" width=256> | <image src="submit_images/q3/chair_only_ckpt/pred_0_point.gif" width=256> |
| <image src="submit_images/q3/chair_only_ckpt/600_rgb.png" width=256> | <image src="submit_images/q3/chair_only_ckpt/600_actual.gif" width=256> | <image src="submit_images/q3/chair_only_ckpt/pred_600_point.gif" width=256> |
| <image src="submit_images/q3/chair_only_ckpt/1200_rgb.png" width=256> | <image src="submit_images/q3/chair_only_ckpt/1200_actual.gif" width=256> | <image src="submit_images/q3/chair_only_ckpt/pred_1200_point.gif" width=256> |

The network trained on all three classes performs significantly better, with an average F1@0.05 score of 91.7. It is clear that the same model picked up on the different shape modalities of planes and cars, where the predicted point clouds have no chair-like features for the car and plane classes. We can conclude that providing the full dataset with additional classes trained the network to predict other modalities of objects successfully. However, given that the same network architecture was used for both instances, we can see a slight degradation in the predicted point cloud for the chair (looking closely, the points are spread out slightly further than the above case, and there are more points under the seat of the chair). This suggests that the predicted point cloud modalities are not completely separated, and the priors of the 'average plane' or 'average car' influence the output of the chair prediction (since both other classes tend to have a flat base instead of a raised seat like a chair).


| Input Image | Ground Truth Mesh | Predicted Point Cloud |
|-------------|------------------|----------------------|
| <image src="submit_images/q3/full_ckpt/0_rgb.png" width=256> | <image src="submit_images/q3/full_ckpt/0_actual.gif" width=256> | <image src="submit_images/q3/full_ckpt/pred_0_point.gif" width=256> |
| <image src="submit_images/q3/full_ckpt/600_rgb.png" width=256> | <image src="submit_images/q3/full_ckpt/600_actual.gif" width=256> | <image src="submit_images/q3/full_ckpt/pred_600_point.gif" width=256> |
| <image src="submit_images/q3/full_ckpt/1200_rgb.png" width=256> | <image src="submit_images/q3/full_ckpt/1200_actual.gif" width=256> | <image src="submit_images/q3/full_ckpt/pred_1200_point.gif" width=256> |