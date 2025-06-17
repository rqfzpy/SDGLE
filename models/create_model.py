from .vit import ViT
from .swin import SwinTransformer
from .resnet import SupCEResNet


def create_model(img_size, n_classes, args):
    if args.model == 'vit':
        patch_size = 4 if img_size == 32 else 8
        model = ViT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=192, 
                    mlp_dim_ratio=2, depth=9, heads=12, dim_head=192//12,
                    stochastic_depth=args.sd, is_SPT=args.is_SPT, is_LSA=args.is_LSA,is_MABS=args.is_MABS,is_PAM=args.is_PAM,DT=args.detach)
    elif args.model == 'vitti': 
        patch_size = 4 if img_size == 32 else 8
        model = ViT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=256, 
                    mlp_dim_ratio=2, depth=12, heads=4, dim_head=256//4,
                    stochastic_depth=args.sd, is_SPT=args.is_SPT, is_LSA=args.is_LSA,is_MABS=args.is_MABS,is_PAM=args.is_PAM,DT=args.detach)
    elif args.model == 'vits': 
        patch_size = 4 if img_size == 32 else 8
        model = ViT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=320, 
                    mlp_dim_ratio=2, depth=12, heads=5, dim_head=320//5,
                    stochastic_depth=args.sd, is_SPT=args.is_SPT, is_LSA=args.is_LSA,is_MABS=args.is_MABS,is_PAM=args.is_PAM,DT=args.detach)
    elif args.model == 'vitb': 
        patch_size = 4 if img_size == 32 else 8
        model = ViT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=512, 
                    mlp_dim_ratio=2, depth=12, heads=8, dim_head=512//8,
                    stochastic_depth=args.sd, is_SPT=args.is_SPT, is_LSA=args.is_LSA,is_MABS=args.is_MABS,is_PAM=args.is_PAM,DT=args.detach)
    elif args.model == 'vitl': 
        patch_size = 4 if img_size == 32 else 8
        model = ViT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=640, 
                    mlp_dim_ratio=2, depth=12, heads=10, dim_head=640//10,
                    stochastic_depth=args.sd, is_SPT=args.is_SPT, is_LSA=args.is_LSA,is_MABS=args.is_MABS,is_PAM=args.is_PAM,DT=args.detach)
        
    elif args.model =='swin':
        depths = [2, 2,6,2]
        num_heads = [3,6,12,24]
        mlp_ratio = 2
        window_size = 7

        if img_size == 32:
            patch_size = 2
        elif img_size == 64:
            patch_size = 4
        else:
            patch_size = 8
        window_size = patch_size*2

            
        model = SwinTransformer(img_size=img_size, window_size=window_size, 
                                patch_size=patch_size, mlp_ratio=mlp_ratio, depths=depths, num_heads=num_heads, num_classes=n_classes)
    elif 'resnet18' in args.model:
        model = SupCEResNet(name=args.model,num_classes=n_classes)

    return model