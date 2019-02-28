import os
import torch
import argparse
import numpy as np
import scipy.misc as misc
import glob

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

from utils import pad_one_more, remove_pad_one_more

has_inst_class = [
    {'trainID': 11, 'id': 24, 'bandwidth': 0.6}, #person
    {'trainID': 12, 'id': 25, 'bandwidth': 0.8}, #rider
    {'trainID': 13, 'id': 26, 'bandwidth': 0.6}, #car
    {'trainID': 14, 'id': 27, 'bandwidth': 0.8}, #truck
    {'trainID': 15, 'id': 28, 'bandwidth': 0.8}, #bus
    {'trainID': 16, 'id': 31, 'bandwidth': 1.0}, #train
    {'trainID': 17, 'id': 32, 'bandwidth': 0.6}, #motorcycle
    {'trainID': 18, 'id': 33, 'bandwidth': 0.6}, #bicycle
]



def test(args):
    imgList = glob.glob('datasets/cityscapes/leftImg8bit/val/*/*_leftImg8bit.png')
    outputDir = 'datasets/cityscapes/results'
    overlayedDir = 'datasets/cityscapes/overlayed_results'
    gtSemDir = 'datasets/cityscapes/gtFine/val'

    data_loader = get_loader(args.dataset)
    loader = data_loader(root=None, is_transform=True, img_norm=args.img_norm, test_mode=True)
    n_classes = loader.n_classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    model_file_name = os.path.split(args.model_path)[1]
    model_name = 'icnet_is'


    if args.origianl_icnet_semantic_pred:
        model_dict = {"arch": "icnet"}
        model_icnet = get_model(model_dict, n_classes, version=args.dataset)
        state = convert_state_dict(torch.load("pretrained_models/icnetBN_cityscapes_trainval_90k.pth")["model_state"])
        # state = torch.load(args.model_path)["model_state"]
        model_icnet.load_state_dict(state)
        model_icnet.eval()
        model_icnet.to(device)
   
    # Setup Model
    model_dict = {"arch": model_name}
    model = get_model(model_dict, n_classes, version=args.dataset)
    model = FullModel(model,None)

    state = convert_state_dict(torch.load(args.model_path)["model_state"])
    # state = torch.load(args.model_path)["model_state"]
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    img_processed = 0
    for imgPath in imgList:
        img_processed += 1
        imgId = os.path.split(imgPath)[1].split('.')[0]
        output_txt = open(os.path.join(outputDir, imgId + '.txt'), 'w')
        # import ipdb
        # ipdb.set_trace()
        # Setup image
        print("Read Input Image from : {} ({}/{})".format(imgPath, img_processed, len(imgList)))
        # if img_processed > 10: break
        img = imageio.imread(imgPath)
        original_img = Image.fromarray(img).convert('RGBA')

        # resized_img = misc.imresize(img, (1025, 2049), interp="bicubic")

        orig_size = img.shape[:-1]
        # if model_name in ["pspnet", "icnet", "icnetBN", "icnet_is"]:
        #     # uint8 with RGB mode, resize width and height which are odd numbers
        #     img = misc.imresize(img, (orig_size[0] // 2 * 2 + 1, orig_size[1] // 2 * 2 + 1), 'bilinear')
        # else:
        #     img = misc.imresize(img, (loader.img_size[0], loader.img_size[1]))
        img = pad_one_more(img)

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
            outputs = model_icnet(images)
            _, outputs_inst = model.model(images)
        else:
            outputs, outputs_inst = model.model(images)

        # if args.dcrf:
        #     unary = outputs.data.cpu().numpy()
        #     unary = np.squeeze(unary, 0)
        #     unary = -np.log(unary)
        #     unary = unary.transpose(2, 1, 0)
        #     w, h, c = unary.shape
        #     unary = unary.transpose(2, 0, 1).reshape(loader.n_classes, -1)
        #     unary = np.ascontiguousarray(unary)

        #     resized_img = np.ascontiguousarray(resized_img)

        #     d = dcrf.DenseCRF2D(w, h, loader.n_classes)
        #     d.setUnaryEnergy(unary)
        #     d.addPairwiseBilateral(sxy=5, srgb=3, rgbim=resized_img, compat=1)

        #     q = d.inference(50)
        #     mask = np.argmax(q, axis=0).reshape(w, h).transpose(1, 0)
        #     decoded_crf = loader.decode_segmap(np.array(mask, dtype=np.uint8))
        #     dcrf_path = args.out_path[:-4] + "_drf.png"
        #     misc.imsave(dcrf_path, decoded_crf)
        #     print("Dense CRF Processed Mask Saved at: {}".format(dcrf_path))
        
        pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
        pred = remove_pad_one_more(np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0))

        outputs_inst = outputs_inst.cpu().detach().numpy()
        outputs_sem = outputs.cpu().detach().numpy()
        # if model_name in ["pspnet", "icnet", "icnetBN", "icnet_is"]:
        #     pred = pred.astype(np.float32)
        #     # float32 with F mode, resize back to orig_size
        #     pred = misc.imresize(pred, orig_size, "nearest", mode="F")
        #     outputs_inst = misc.imresize(outputs_inst, orig_size, "nearest", mode="F")
        #     outputs_sem = misc.imresize(outputs_sem, orig_size, "nearest", mode="F")


        outputs_inst = outputs_inst[0, ...]
        outputs_inst = outputs_inst.transpose((1, 2, 0))
        outputs_inst = remove_pad_one_more(outputs_inst)

        outputs_sem = outputs_sem[0, ...]
        outputs_sem = outputs_sem.transpose((1, 2, 0))
        outputs_sem = remove_pad_one_more(outputs_sem)

        h, w, c = outputs_inst.shape

        pred_flattened = pred.reshape((h * w))
        outputs_inst_flattened = np.copy(outputs_inst.reshape((h * w, c)))
        inst_num = 0
        min_inst_size = 500
        single_obj_dist = 1.5
        bd_decay_rate = 0.9

        if args.use_gt_sem_map:
            imgId_np = ('_').join(imgId.split('_')[:-1])
            gtImgDir = os.path.join(gtSemDir, imgId.split('_')[0], imgId_np + '_gtFine_labelTrainIds.png')
            pred = imageio.imread(gtImgDir)
            # pred_flattened = pred.reshape(h * w)
            pred_flattened = misc.imresize(pred, (outputs_sem.shape[0], outputs_sem.shape[1])).reshape((h * w))

        for inst_class in has_inst_class:
            
            interested_semantic_class_train_id = inst_class['trainID']
            predID = inst_class['id']

            # if interested_semantic_class_train_id != 13: continue

            if np.sum(pred_flattened == interested_semantic_class_train_id) == 0: continue
            
            inst_segment_map = np.zeros((h * w), dtype = np.uint16)

            avg_dist = estimate_bandwidth(outputs_inst_flattened[pred_flattened == interested_semantic_class_train_id, :], quantile=1.0, n_samples=1000, n_jobs = 12)
            if avg_dist > single_obj_dist:
                bandwidth = inst_class['bandwidth']
                while True:
                    # ms = MeanShift(bandwidth=inst_class['bandwidth'], bin_seeding=True, n_jobs = 12)
                    try:
                        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs = 12)
                        ms.fit(outputs_inst_flattened[pred_flattened == interested_semantic_class_train_id, :])
                        clustering_label = ms.labels_
                        break
                    except:
                        bandwidth *= bd_decay_rate
                        print(bandwidth)
                inst_segment_map[pred_flattened == interested_semantic_class_train_id] = clustering_label + 1
            else:
                inst_segment_map[pred_flattened == interested_semantic_class_train_id] = 1

            for lbl in range(inst_segment_map.max()):
                if np.sum(inst_segment_map == lbl + 1) < min_inst_size: continue
                inst_num += 1
                mask_file_name = imgId + '_inst_{:03d}.png'.format(inst_num)
                mask_dir = os.path.join(outputDir, mask_file_name)
                mask_img = np.zeros((h * w), dtype = np.uint8)
                mask_img[inst_segment_map == lbl + 1] = 255
                mask_img.resize((h, w))
                # mask_img_orig_size = misc.imresize(mask_img, orig_size)
                imageio.imsave(mask_dir, mask_img)
                sem_lbl_pred = predID
                conf = np.mean(outputs_sem[..., interested_semantic_class_train_id][mask_img > 0]) - outputs_sem.min()
                output_txt.write(mask_file_name + ' ' + str(sem_lbl_pred) + ' {:.4f}\n'.format(conf))


            if inst_num > 0:
                # import ipdb
                # ipdb.set_trace()
                                
    
                inst_segment_map = inst_segment_map.reshape(h, w)
                cmap = plt.cm.jet
                norm = plt.Normalize(vmin=inst_segment_map.min(), vmax=inst_segment_map.max())
                # import ipdb
                # ipdb.set_trace()
                # map the normalized data to colors
                # image is now RGBA (512x512x4) 
                inst_segment_map_single_image = cmap(norm(inst_segment_map))
                inst_segment_map_single_image[inst_segment_map == 0] = [0, 0, 0, 1]
                inst_segment_map_single_image = Image.fromarray((inst_segment_map_single_image * 255).astype(np.uint8))

                # save the image
                # inst_segment_map_single_image.save('inst_seg_map_' + args.out_path)
                # import ipdb
                # ipdb.set_trace()
                original_img = original_img.resize(inst_segment_map_single_image.size)
                inst_segment_map_single_image.putalpha(128)

                overlayed_image = Image.alpha_composite(original_img, inst_segment_map_single_image)
                overlayed_image_path = os.path.join(overlayedDir, str(interested_semantic_class_train_id), imgId + '.png')
                print(overlayed_image_path)
                if not os.path.exists(os.path.dirname(overlayed_image_path)):
                    try:
                        os.makedirs(os.path.dirname(overlayed_image_path))
                    except:
                        pass
                overlayed_image.save(overlayed_image_path)



        output_txt.close()
        # import ipdb
        # ipdb.set_trace()



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

    parser.add_argument(
        "--use_gt_sem_map",
        dest="use_gt_sem_map",
        action="store_true",
        help="Use ground truth semantic segmenattion | \
                              False by default",
    )

    parser.set_defaults(dcrf=False)
    parser.set_defaults(use_original_icnet=False)
    parser.set_defaults(use_gt_sem_map=False)
    parser.add_argument(
        "--img_path", nargs="?", type=str, default=None, help="Path of the input image"
    )
    parser.add_argument(
        "--out_path", nargs="?", type=str, default=None, help="Path of the output segmap"
    )
    args = parser.parse_args()
    test(args)
