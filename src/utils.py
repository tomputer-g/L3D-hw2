import imageio
import numpy as np
import torch
from pathlib import Path
import mcubes
import pytorch3d
from tqdm import tqdm

from pytorch3d.renderer import (
    AlphaCompositor,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    HardPhongShader,
    TexturesVertex,
)

def create_gif_from_image_list(images_list: list[np.ndarray], gif_path: Path, FPS=15):
    # images_list is a list of (H,W,3) images
    assert images_list[0].shape[2] == 3
    
    frame_duration_ms = 1000 // FPS
    imageio.mimsave(gif_path, images_list, duration=frame_duration_ms, loop=0)


def add_texture_to_mesh(mesh: pytorch3d.structures.Meshes):
    """
    Adds a vertex texture to the given mesh with the specified color.

    Args:
        mesh (pytorch3d.structures.Meshes): The mesh to add texture to.
        color (tuple): RGB color to assign to all vertices.

    Returns:
        pytorch3d.structures.Meshes: Mesh with vertex textures.
    """
    verts = mesh.verts_list()[0]
    textures = torch.ones_like(verts) * torch.tensor([0.7, 0.7, 1], device=verts.device)
    mesh.textures = TexturesVertex([textures]).to(verts.device)
    return mesh

def get_color_pointcloud(pointcloud: torch.Tensor):
    return torch.ones_like(pointcloud).to(pointcloud.device) * torch.tensor([0.7,0.7,1], device=pointcloud.device)


def render_vox_to_mesh(vox: torch.Tensor): # -> pytorch3d.structures.Meshes:
    # print(vox.shape) #1, 32, 32, 32
    # H,W,D = vox.shape[1:]
    # voxel_size=32
    # min_value=-16
    # max_value=16
    vertices_src, faces_src = mcubes.marching_cubes(vox.detach().cpu().squeeze().numpy(), isovalue=0.5) #0.5
    vertices_src = torch.tensor(vertices_src).float()
    # vertices_src = (vertices_src / voxel_size) * (max_value - min_value) + min_value

    faces_src = torch.tensor(faces_src.astype(int))
    textures = torch.ones_like(vertices_src)  # (1, N_v, 3)
    textures = textures * torch.tensor([0.7, 0.7, 1])  # (1, N_v, 3)
    mesh_src = pytorch3d.structures.Meshes([vertices_src], [faces_src], TexturesVertex([textures])).to(vox.device)
    return mesh_src

def get_mesh_renderer(image_size=512, lights=None, device='cuda'):
    """
    Returns a Pytorch3D Mesh Renderer.

    Args:
        image_size (int): The rendered image size.
        lights: A default Pytorch3D lights object.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
    """
    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device, lights=lights),
    )
    return renderer


def get_points_renderer(
    image_size=512, device="cuda", radius=0.01, background_color=(1, 1, 1)
):
    """
    Returns a Pytorch3D renderer for point clouds.

    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.
    
    Returns:
        PointsRenderer.
    """
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer

def render_mesh_to_gif(
        mesh: pytorch3d.structures.Meshes,
        gif_path: Path,
        cam_dist: float = 60,
        cam_elev: float = 10,
        device: str="cuda",
        image_size: int = 512,
        azimuth_step: int = 10):
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device)
    renderer = get_mesh_renderer(image_size=image_size, device=device)

    image_list = []
    for azimuth in tqdm(range(0, 360, azimuth_step), desc="Rendering mesh..."): 
        R, T = pytorch3d.renderer.look_at_view_transform(dist=cam_dist, elev=cam_elev, azim=azimuth)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        rend = renderer(mesh, cameras=cameras, lights=lights)
        img = rend.detach().cpu().numpy()[0, ..., :3].clip(0,1)  # (B, H, W, 4) -> (H, W, 3)
        img *= 255
        img = img.astype('uint8')
        image_list.append(img)
        
    create_gif_from_image_list(image_list, gif_path)


def render_pointcloud_to_gif(
    V: torch.Tensor,
    rgb: torch.Tensor,
    gif_path: Path,
    cam_dist: float = 60,
    cam_elev: float = 10,
    device:str = "cuda",
    background_color=(1, 1, 1),
    downsample_factor=1,
    image_size=512,
):
    """
    Renders a point cloud.
    """
    renderer = get_points_renderer(
        radius=0.007,image_size=image_size, background_color=background_color
    )
    
    verts = V[::downsample_factor].to(device)
    rgb = rgb[::downsample_factor].to(device)
    point_cloud = pytorch3d.structures.Pointclouds(points=verts, features=rgb)
    
    image_list = []
    for azimuth in tqdm(range(0, 360, 10), desc="Rendering pointcloud..."): 
        R, T = pytorch3d.renderer.look_at_view_transform(cam_dist, cam_elev, azimuth)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        rend = renderer(point_cloud, cameras=cameras)
        img = rend.detach().cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
        img *= 255
        img = img.astype('uint8')
        image_list.append(img)
        
    create_gif_from_image_list(image_list, gif_path)