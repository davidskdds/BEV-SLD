import torch.nn as nn
import torch
from .blocks import ResidualBlock, DownStepResBlock


class bev_sld_net(nn.Module):
    def __init__(
        self,
        coords_init: torch.Tensor,        # Tensor of shape (n_coords, 2)
    ):
        super().__init__()

        #  validate
        assert coords_init.ndim == 2 and coords_init.shape[1] == 2, \
            f"coords_init must be (n_coords, 2), got {tuple(coords_init.shape)}"

        n_coords = coords_init.shape[0]

        #  compute bounds from coords_init 
        xs = coords_init[:, 0]
        ys = coords_init[:, 1]
        min_x, max_x = float(xs.min()), float(xs.max())
        min_y, max_y = float(ys.min()), float(ys.max())

        #  normalization constants 
        self.mean_x  = (min_x + max_x) / 2
        self.mean_y  = (min_y + max_y) / 2
        self.scale_x = (max_x - min_x) / 2
        self.scale_y = (max_y - min_y) / 2

        # descale
        init_pos_scaled = coords_init

        init_pos_scaled[:,0] = init_pos_scaled[:,0] - self.mean_x
        init_pos_scaled[:,1] = init_pos_scaled[:,1] - self.mean_y

        init_pos_scaled[:,0] =  init_pos_scaled[:,0] / self.scale_x
        init_pos_scaled[:,1] = init_pos_scaled[:,1] / self.scale_y

        # heat map
        self.conv_in = nn.Sequential(ResidualBlock(1,32))

        # downsampling blocks
        self.down1 = DownStepResBlock(32,64) # 256x256
        self.down2 = DownStepResBlock(64,64) # 128x128
        self.down3 = DownStepResBlock(64,128) # 64x64
        self.down4 = DownStepResBlock(128,128) # 32x32
        self.down5 = DownStepResBlock(128,128) # 16x16
        self.down6 = DownStepResBlock(128,128) # 8x8
        self.down7 = DownStepResBlock(128,128) # 4x4
        self.down8 = DownStepResBlock(128,128) # 2x2
        
        self.corresp_5 = nn.Sequential(ResidualBlock(128,128), ResidualBlock(128,128))
        self.corresp_6 = nn.Sequential(ResidualBlock(128,128), ResidualBlock(128,128))
        self.corresp_8 = nn.Sequential(ResidualBlock(128,128), ResidualBlock(128,128))
        
        # GroupNorm with 1 group is LayerNorm
        self.corresp_output_layers = nn.Sequential(nn.Conv2d(384, 384, kernel_size=1),nn.GroupNorm(1,384),nn.LeakyReLU(),\
            nn.Conv2d(384, 384, kernel_size=1),nn.GroupNorm(1,384),nn.LeakyReLU(),\
            nn.Conv2d(384, 384, kernel_size=1),nn.GroupNorm(1,384),nn.LeakyReLU(),\
            nn.Conv2d(384, coords_init.shape[0], kernel_size=1))

        self.heatmap_1 = nn.Sequential(ResidualBlock(64,32), ResidualBlock(32,16))
        self.heatmap_3 = nn.Sequential(ResidualBlock(128,64),ResidualBlock(64,64),ResidualBlock(64,16))
        self.heatmap_5 = nn.Sequential(ResidualBlock(128,64),ResidualBlock(64,64),ResidualBlock(64,16))
        self.heatmap_6 = nn.Sequential(ResidualBlock(128,64),ResidualBlock(64,64),ResidualBlock(64,16))

        # GroupNorm with 1 group is LayerNorm
        self.heatmap_output_layers = nn.Sequential(nn.Conv2d(64, 32, kernel_size=1),nn.GroupNorm(1,32),nn.LeakyReLU(),\
            nn.Conv2d(32, 16, kernel_size=1),nn.GroupNorm(1,16),nn.LeakyReLU(),\
            nn.Conv2d(16, 8, kernel_size=1),nn.GroupNorm(1,8),nn.LeakyReLU(),\
            nn.Conv2d(8, 8, kernel_size=1),nn.GroupNorm(1,8),nn.LeakyReLU(),\
            nn.Conv2d(8, 1, kernel_size=1))#,\

        # upsampling
        self.restore16 = nn.Upsample((16,16),mode='bilinear',align_corners=True)
        self.restore256 = nn.Upsample((256,256),mode='bilinear',align_corners=True)
        self.restore512 = nn.Upsample((512,512),mode='bicubic',align_corners=True)
        
        # landmark coordinate list as embedding
        self.embd_coords = nn.Embedding(n_coords, 2)

        with torch.no_grad():
            self.embd_coords.weight.copy_(init_pos_scaled)



    def freeze_embd(self):
        self.embd_coords.weight.requires_grad = False

    def unfreeze_embd(self):
        self.embd_coords.weight.requires_grad = True


    def forward(self, x):
        # Input x should have dimensions [N, 1, 512, 512] where N is batch size
        
        # feature extraction
        out = self.conv_in(x)
        out = self.down1(out)
        f1 = self.heatmap_1(out)
        out = self.down2(out)
        out = self.down3(out)
        f3 = self.heatmap_3(out)
        out = self.down4(out)
        
        out = self.down5(out)
        f5 = self.heatmap_5(out)
        c5 = self.corresp_5(out)

        out = self.down6(out)
        f6 = self.heatmap_6(out)
        c6 = self.corresp_6(out)
        out = self.down7(out)
        out = self.down8(out)
        c8 = self.corresp_8(out)
        
        # correspondence prediction
        corresp_features = torch.cat((c5,self.restore16(c6),self.restore16(c8)),dim=1) # 384x32x32
        corresp_pred = self.corresp_output_layers(corresp_features)

        # heatmap prediction
        heatmap_features = torch.cat((f1,self.restore256(f3),self.restore256(f5),self.restore256(f6)),dim=1) # 24x512x512
        heatmap = self.heatmap_output_layers(heatmap_features)

        # to 512x512
        heatmap = self.restore512(heatmap)

        # input independent landmark coordinates
        coords = self.embd_coords.weight

        coords_scaled = torch.zeros(coords.shape[0], 2,
                            device=coords.device,
                            dtype=coords.dtype)

        coords_scaled[:,0] = coords[:,0] * self.scale_x + self.mean_x
        coords_scaled[:,1] = coords[:,1] * self.scale_y + self.mean_y

        # heatmap: local landmark positions; corresp_pred: correspondence prediction; coords_scaled: list of global landmarks
        return heatmap,corresp_pred, coords_scaled
