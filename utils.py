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

def render_vox_to_mesh(vox: torch.Tensor): # -> pytorch3d.structures.Meshes:
    # print(vox.shape) #1, 32, 32, 32
    # H,W,D = vox.shape[1:]
    vertices_src, faces_src = mcubes.marching_cubes(vox.detach().cpu().squeeze().numpy(), isovalue=0.5)
    vertices_src = torch.tensor(vertices_src).float()
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

def render_mesh_to_gif(
        mesh: pytorch3d.structures.Meshes,
        gif_path: Path,
        cam_dist: float = 60,
        cam_elev: float = 10,
        device: str="cuda",
        image_size: int = 512):
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device)
    renderer = get_mesh_renderer(image_size=image_size, device=device)

    image_list = []
    for azimuth in tqdm(range(0, 360, 10), desc="Rendering mesh..."): 
        R, T = pytorch3d.renderer.look_at_view_transform(dist=cam_dist, elev=cam_elev, azim=azimuth)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        rend = renderer(mesh, cameras=cameras, lights=lights)
        img = rend.cpu().numpy()[0, ..., :3].clip(0,1)  # (B, H, W, 4) -> (H, W, 3)
        img *= 255
        img = img.astype('uint8')
        image_list.append(img)
        
    create_gif_from_image_list(image_list, gif_path)