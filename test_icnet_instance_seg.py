import os
import torch
import argparse
import numpy as np
import scipy.misc as misc


from ptsemseg.models import get_model
from ptsemseg.loader import get_loader
from ptsemseg.utils import convert_state_dict

from train_icnet_instance_seg import FullModel, DataParallel_withLoss

from sklearn.decomposition import PCA as sklearnPCA
from sklearn.cluster import MeanShift, estimate_bandwidth
import imageio
from PIL import Image
import matplotlib.pyplot as plt
try:
    import pydensecrf.densecrf as dcrf
except:
    print(
        "Failed to import pydensecrf,\
           CRF post-processing will not work"
    )


def test(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_file_name = os.path.split(args.model_path)[1]
    model_name = model_file_name[: model_file_name.find("_")]
    model_name = 'icnet_is'
    # import ipdb
    # ipdb.set_trace()
    # Setup image
    print("Read Input Image from : {}".format(args.img_path))
    img = imageio.imread(args.img_path)
    original_img = Image.fromarray(img).convert('RGBA')

    data_loader = get_loader(args.dataset)
    loader = data_loader(root=None, is_transform=True, img_norm=args.img_norm, test_mode=True)
    n_classes = loader.n_classes

    resized_img = misc.imresize(img, (loader.img_size[0], loader.img_size[1]), interp="bicubic")

    orig_size = img.shape[:-1]
    if model_name in ["pspnet", "icnet", "icnetBN", "icnet_is"]:
        # uint8 with RGB mode, resize width and height which are odd numbers
        img = misc.imresize(img, (orig_size[0] // 2 * 2 + 1, orig_size[1] // 2 * 2 + 1))
    else:
        img = misc.imresize(img, (loader.img_size[0], loader.img_size[1]))

    img = img[:, :, ::-1]
    img = img.astype(np.float64)
    img -= loader.mean
    # if args.img_norm:
    #     img = img.astype(float) / 255.0

    # NHWC -> NCHW
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()
    images = img.to(device)
    
    if args.origianl_icnet_semantic_pred:
        model_dict = {"arch": "icnet"}
        model = get_model(model_dict, n_classes, version=args.dataset)
        state = convert_state_dict(torch.load("pretrained_models/icnetBN_cityscapes_trainval_90k.pth")["model_state"])
        # state = torch.load(args.model_path)["model_state"]
        model.load_state_dict(state)
        model.eval()
        model.to(device)
        outputs = model(images)

    # Setup Model
    model_dict = {"arch": model_name}
    model = get_model(model_dict, n_classes, version=args.dataset)
    model = FullModel(model,None)

    state = convert_state_dict(torch.load(args.model_path)["model_state"])
    # state = torch.load(args.model_path)["model_state"]
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    if args.origianl_icnet_semantic_pred:
        _, outputs_inst = model.model(images)
    else:
        outputs, outputs_inst = model.model(images)

    if args.dcrf:
        unary = outputs.data.cpu().numpy()
        unary = np.squeeze(unary, 0)
        unary = -np.log(unary)
        unary = unary.transpose(2, 1, 0)
        w, h, c = unary.shape
        unary = unary.transpose(2, 0, 1).reshape(loader.n_classes, -1)
        unary = np.ascontiguousarray(unary)

        resized_img = np.ascontiguousarray(resized_img)

        d = dcrf.DenseCRF2D(w, h, loader.n_classes)
        d.setUnaryEnergy(unary)
        d.addPairwiseBilateral(sxy=5, srgb=3, rgbim=resized_img, compat=1)

        q = d.inference(50)
        mask = np.argmax(q, axis=0).reshape(w, h).transpose(1, 0)
        decoded_crf = loader.decode_segmap(np.array(mask, dtype=np.uint8))
        dcrf_path = args.out_path[:-4] + "_drf.png"
        misc.imsave(dcrf_path, decoded_crf)
        print("Dense CRF Processed Mask Saved at: {}".format(dcrf_path))

    pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
    pred_original = np.copy(pred)
    if model_name in ["pspnet", "icnet", "icnetBN", "icnet_is"]:
        pred = pred.astype(np.float32)
        # float32 with F mode, resize back to orig_size
        pred = misc.imresize(pred, orig_size, "nearest", mode="F")

    interested_semantic_class_train_id = 17


    outputs_inst = outputs_inst.cpu().detach().numpy()

    outputs_inst = outputs_inst[0, ...]
    outputs_inst = outputs_inst.transpose((1, 2, 0))
    h, w, c = outputs_inst.shape
    outputs_inst_transformed = np.copy(outputs_inst.reshape((h * w, c)))
    pca = sklearnPCA(n_components=3)

    pca.fit(outputs_inst_transformed)
    outputs_inst_transformed = pca.transform(outputs_inst_transformed)
    outputs_inst_transformed -= outputs_inst_transformed.min(axis = 0)
    outputs_inst_transformed /= outputs_inst_transformed.max(axis = 0)
    outputs_inst_img = outputs_inst_transformed.reshape((h, w, 3))
    outputs_inst_img = (outputs_inst_img * 255).astype(int)



    decoded = loader.decode_segmap(pred)
    print("Classes found: ", np.unique(pred))
    imageio.imsave(args.out_path, decoded)
    imageio.imsave("inst_"+args.out_path, outputs_inst_img)
    print("Segmentation Mask Saved at: {}".format(args.out_path))


    outputs_inst_transformed_single = np.copy(outputs_inst.reshape((h * w, c)))
    pred_transformed = pred_original.reshape((h * w))
    pca.fit(outputs_inst_transformed_single[pred_transformed == interested_semantic_class_train_id, :])
    outputs_inst_transformed_single = pca.transform(outputs_inst_transformed_single)
    outputs_inst_transformed_single -= outputs_inst_transformed_single.min(axis = 0)
    outputs_inst_transformed_single /= outputs_inst_transformed_single.max(axis = 0)
    outputs_inst_transformed_single[pred_transformed != interested_semantic_class_train_id, :] = 0
    outputs_inst_single_img = outputs_inst_transformed_single.reshape((h, w, 3))
    outputs_inst_single_img = Image.fromarray((outputs_inst_single_img * 255).astype(np.uint8))
    outputs_inst_single_img.save("inst_single_"+args.out_path)


    outputs_inst_transformed_single = np.copy(outputs_inst.reshape((h * w, c)))
    bandwidth = estimate_bandwidth(outputs_inst_transformed_single[pred_transformed == interested_semantic_class_train_id, :], quantile=0.1, n_samples=1000, n_jobs = 12)
    print(bandwidth)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs = 12)
    ms.fit(outputs_inst_transformed_single[pred_transformed == interested_semantic_class_train_id, :])
    clustering_label = ms.labels_
    inst_segment_map_single = np.zeros((h * w))
    inst_segment_map_single[pred_transformed == interested_semantic_class_train_id] = clustering_label + 1
    inst_segment_map_single = inst_segment_map_single.reshape(h, w)
    cmap = plt.cm.jet
    norm = plt.Normalize(vmin=inst_segment_map_single.min(), vmax=inst_segment_map_single.max())
    # import ipdb
    # ipdb.set_trace()
    # map the normalized data to colors
    # image is now RGBA (512x512x4) 
    inst_segment_map_single_image = cmap(norm(inst_segment_map_single))
    inst_segment_map_single_image[inst_segment_map_single == 0] = [0, 0, 0, 1]
    inst_segment_map_single_image = Image.fromarray((inst_segment_map_single_image * 255).astype(np.uint8))

    # save the image
    inst_segment_map_single_image.save('inst_seg_map_' + args.out_path)
    # import ipdb
    # ipdb.set_trace()
    original_img = original_img.resize(inst_segment_map_single_image.size)
    inst_segment_map_single_image.putalpha(128)

    overlayed_image = Image.alpha_composite(original_img, inst_segment_map_single_image)
    overlayed_image.save('inst_seg_map_overlayed_' + args.out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Params")
    parser.add_argument(
        "--model_path",
        nargs="?",
        type=str,
        default="fcn8s_pascal_1_26.pkl",
        help="Path to the saved model",
    )
    parser.add_argument(
        "--dataset",
        nargs="?",
        type=str,
        default="pascal",
        help="Dataset to use ['pascal, camvid, ade20k etc']",
    )

    parser.add_argument(
        "--img_norm",
        dest="img_norm",
        action="store_true",
        help="Enable input image scales normalization [0, 1] \
                              | True by default",
    )
    parser.add_argument(
        "--no-img_norm",
        dest="img_norm",
        action="store_false",
        help="Disable input image scales normalization [0, 1] |\
                              True by default",
    )
    parser.set_defaults(img_norm=True)

    parser.add_argument(
        "--dcrf",
        dest="dcrf",
        action="store_true",
        help="Enable DenseCRF based post-processing | \
                              False by default",
    )
    parser.add_argument(
        "--no-dcrf",
        dest="dcrf",
        action="store_false",
        help="Disable DenseCRF based post-processing | \
                              False by default",
    )
    parser.add_argument(
        "--use_original_icnet",
        dest="origianl_icnet_semantic_pred",
        action="store_true",
        help="Use original icnet semantic prediction | \
                              False by default",
    )
    parser.set_defaults(dcrf=False)
    parser.set_defaults(use_original_icnet=False)

    parser.add_argument(
        "--img_path", nargs="?", type=str, default=None, help="Path of the input image"
    )
    parser.add_argument(
        "--out_path", nargs="?", type=str, default=None, help="Path of the output segmap"
    )
    args = parser.parse_args()
    test(args)
