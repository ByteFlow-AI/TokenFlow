from dataset.imagenet import build_imagenet
from dataset.coco import build_coco
from dataset.openimage import build_openimage

def build_dataset(args, **kwargs):
    # images
    if args.dataset == 'imagenet':
        return build_imagenet(args, **kwargs)
    if args.dataset == 'coco':
        return build_coco(args, **kwargs)
    if args.dataset == 'openimage':
        return build_openimage(args, **kwargs)
    
    raise ValueError(f'dataset {args.dataset} is not supported')