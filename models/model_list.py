import argparse

# 2D models
from models.model_2d.ae import Network2D as AE
from models.model_2d.ae_2d_to_2d import Network2D as AE_2D_TO_2D
from models.model_2d.ae_6_2d_to_6_2d import Network2D as AE_6_2D_TO_6_2D
from models.model_2d.gap_cnn import Network2D as GAP_CNN
from models.model_2d.vgg_ae_demo import Network2D as VGG_AE_DEMO

# 3D models
from models.model_3d.ae_6_2d_to_3d import Network3D as AE_6_2D_TO_3D
from models.model_3d.ae_3d_to_3d import Network3D as AE_3D_TO_3D


def init_model(args: argparse.Namespace):
    model_map = {
        # 2D models
        "ae": AE,
        "ae_2d_to_2d": AE_2D_TO_2D,
        "ae_6_2d_to_6_2d": AE_6_2D_TO_6_2D,
        "gap_cnn": GAP_CNN,
        "vgg_ae_demo": VGG_AE_DEMO,

        # 3D models
        "ae_6_2d_to_3d": AE_6_2D_TO_3D,
        "ae_3d_to_3d": AE_3D_TO_3D
    }
    if args.model in list(model_map.keys()):
        return model_map[args.model](args=args)
    else:
        raise Exception("Model not supported")
