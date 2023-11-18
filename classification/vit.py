import torch

from timm import create_model
from timm.models.registry import register_model
from tome.patch.timm import apply_patch
from tome import merge

@register_model
def tome_vit_base_patch16_224_augreg_in21k_ft_in1k(pretrained=True, r=4, **kwargs):
    model = create_model("vit_base_patch16_224.augreg_in21k_ft_in1k", pretrained=pretrained)
    
    apply_patch(model)
    
    model.r = r
    
    return model