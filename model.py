from torchvision import models as torchvision_models
from torchvision import transforms
import time
import torch.nn as nn
import torch
from pytorch3d.utils import ico_sphere
import pytorch3d

class SingleViewto3D(nn.Module):
    def __init__(self, args):
        super(SingleViewto3D, self).__init__()
        self.device = args.device
        if not args.load_feat:
            vision_model = torchvision_models.__dict__[args.arch](pretrained=True)
            self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

        # define decoder
        if args.type == "vox":
            # Input: b x 512
            # Output: b x 32 x 32 x 32

            # Batch size: 16
            layers = []
            layers.append(torch.nn.Linear(512, 2048))
            layers.append(torch.nn.GELU())
            layers.append(torch.nn.Linear(2048, 4096))
            layers.append(torch.nn.GELU())
            layers.append(torch.nn.Linear(4096, 8192))
            layers.append(torch.nn.GELU())
            layers.append(torch.nn.Linear(8192, 32768))
            layers.append(torch.nn.Unflatten(1, (1,32,32,32)))
            self.decoder = torch.nn.Sequential(*layers)

            # pass
            # TODO:
            # self.decoder =             
        elif args.type == "point":
            # Input: b x 512
            # Output: b x args.n_points x 3  

            # Batch size 64, n_points 10k
            self.n_point = args.n_points
            layers = []
            layers.append(torch.nn.Linear(512, 2048))
            layers.append(torch.nn.GELU())
            layers.append(torch.nn.Linear(2048, 4096))
            layers.append(torch.nn.GELU())
            layers.append(torch.nn.Linear(4096, 8192))
            layers.append(torch.nn.GELU())
            layers.append(torch.nn.Linear(8192, self.n_point*3))
            layers.append(torch.nn.Unflatten(1, (self.n_point,3)))
            self.decoder = torch.nn.Sequential(*layers)
            # TODO:
            # self.decoder =             
        elif args.type == "mesh":
            # Input: b x 512
            # Output: b x mesh_pred.verts_packed().shape[0] x 3  
            # try different mesh initializations
            mesh_pred = ico_sphere(4, self.device)
            self.mesh_pred = pytorch3d.structures.Meshes(mesh_pred.verts_list()*args.batch_size, mesh_pred.faces_list()*args.batch_size)
            # TODO:
            # self.decoder =             

    def forward(self, images, args):
        results = dict()

        total_loss = 0.0
        start_time = time.time()

        B = images.shape[0]

        if not args.load_feat:
            images_normalize = self.normalize(images.permute(0,3,1,2))
            encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1) # b x 512
        else:
            encoded_feat = images # in case of args.load_feat input images are pretrained resnet18 features of b x 512 size

        # call decoder
        if args.type == "vox":
            # TODO:
            # voxels_pred = 
            if not args.load_feat:
                voxels_pred = self.decoder(encoded_feat)  
                return voxels_pred
            return None

        elif args.type == "point":
            # TODO:
            pointclouds_pred = self.decoder(encoded_feat)
            return pointclouds_pred

        elif args.type == "mesh":
            # TODO:
            # deform_vertices_pred =             
            mesh_pred = self.mesh_pred.offset_verts(deform_vertices_pred.reshape([-1,3]))
            return  mesh_pred          

