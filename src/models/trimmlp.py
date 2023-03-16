# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import PatchEmbed , _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_,  DropPath

from .lr_layers import LRLinear

from utils import load_pretrained

__all__ = [
    'trimmlp_12'
]

# class Affine(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.alpha = nn.Parameter(torch.ones(dim))
#         self.beta = nn.Parameter(torch.zeros(dim))

#     def forward(self, x):
#         return self.alpha * x + self.beta 
def Affine(dim):
    # Use linear layer to instaed of Affine, so it can be easiler when doing quantize
    return nn.Linear(dim, dim)

class Mlp(nn.Module):
    def __init__(self, ratios, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = LRLinear(ratios[0], in_features, hidden_features)
        self.fc2 = LRLinear(ratios[1], hidden_features, out_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1.VT(x)
        x = self.fc1.U(x)
        x = self.act(x)
        # x = self.drop(x)
        x = self.fc2.VT(x)
        x = self.fc2.U(x)
        # x = self.drop(x)
        return x
    
class layers_scale_mlp_blocks(nn.Module):

    def __init__(self, dim, drop=0., drop_path=0., act_layer=nn.GELU,init_values=1e-4,num_patches = 196, svd_config = None):
        super().__init__()
        self.norm1 = Affine(dim)
        self.attn = nn.Linear(num_patches, num_patches)
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.skip_add = nn.quantized.FloatFunctional()
        self.norm2 = Affine(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(4.0 * dim), act_layer=act_layer, drop=drop, ratios = svd_config)
        # self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        # self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)

        # Use linear layer to instaed of parameter, so it can be easiler when doing quantize
        self.gamma_1 = nn.Linear(dim, dim, bias=False)
        self.gamma_2 = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        # x = x + self.drop_path(self.gamma_1(self.attn(self.norm1(x).transpose(1,2)).transpose(1,2)))
        # x = x + self.drop_path(self.gamma_2(self.mlp(self.norm2(x))))
        residual = x
        #x = self.skip_add.add(residual, self.drop_path(self.gamma_1 * self.attn(self.norm1(x).transpose(1,2)).transpose(1,2)))
        x = self.skip_add.add(residual, self.gamma_1(self.attn(self.norm1(x).transpose(1,2)).transpose(1,2)))
        
        residual = x
        #x = self.skip_add.add(residual, self.drop_path(self.gamma_2 * self.mlp(self.norm2(x))))
        x = self.skip_add.add(residual, self.gamma_2(self.mlp(self.norm2(x))))
        
        return x 


class LRResMLPSubnet(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,drop_rate=0.,
                 Patch_layer=PatchEmbed,act_layer=nn.GELU,
                 drop_path_rate=0.0,init_scale=1e-4,adaptive_column=False, svd_config = None):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  
        self.adaptive_column = adaptive_column

        self.patch_embed = Patch_layer(
                img_size=img_size, patch_size=patch_size, in_chans=int(in_chans), embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        dpr = [drop_path_rate for i in range(depth)]

        self.blocks = nn.ModuleList([
            layers_scale_mlp_blocks(
                dim=embed_dim,drop=drop_rate,drop_path=dpr[i],
                act_layer=act_layer,init_values=init_scale,
                num_patches=num_patches,
                svd_config=svd_config[i],
                )
            for i in range(depth)])


        self.norm = Affine(embed_dim)

        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)



    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]

        x = self.patch_embed(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x)
        
        x = self.norm(x)
        x = x.mean(dim=1).reshape(B,1,-1)

        return x[:, 0]

    def forward(self, x):
        x  = self.forward_features(x)
        x = self.head(x)
        return x 

@register_model
def trimmlp_12(pretrained=False, **kwargs):
    """
    MODEL:
    TYPE = 'lr_resmlp_subnet'
    NAME = 'lr_resmlp_s12_subnet_05'
    """
    IMG_SIZE = 224
    PATCH_SIZE = 16
    DROP_PATH_RATE = 0.0
    DROP_RATE = 0.0
    EMBED_DIM = 384
    DEPTH = 12
    SVD_CONFIG = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]
    
    model = LRResMLPSubnet(
        svd_config = SVD_CONFIG,
        img_size = IMG_SIZE,
        patch_size = PATCH_SIZE,
        embed_dim = EMBED_DIM,
        depth = DEPTH,
    )

    if pretrained:
        load_pretrained(model, model_path = 'lr_resmlp_s12_subnet_05.pth')

    return model