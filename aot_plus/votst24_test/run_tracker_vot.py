import torch
import torch.nn.functional as F
import os
import sys
import cv2
import importlib
import collections
import pdb

Rectangle = collections.namedtuple('Rectangle', ['x', 'y', 'width', 'height'])
DIR_PATH = os.path.join(os.path.dirname(__file__), '../')
sys.path.append(DIR_PATH)
import vot_utils
import copy
from tools.transfer_predicted_mask2vottype import transfer_mask

import dataloaders.video_transforms as tr
from torchvision import transforms
from networks.engines import build_engine
from utils.checkpoint import load_network
from utils.metric import pytorch_iou
from networks.models import build_vos_model
from utils.image import flip_tensor

random_seed = 1
import os
os.environ['CUDNN_DETERMINISTIC'] = '1'
os.environ['PYTHONHASHSEED'] = str(random_seed)
import random
random.seed(random_seed + 1)
import numpy as np
np.random.seed(random_seed + 2)
torch.manual_seed(random_seed + 3)
torch.cuda.manual_seed(random_seed + 4)
torch.cuda.manual_seed_all(random_seed + 5)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


seed_torch(1000000007)
torch.set_num_threads(4)
torch.autograd.set_grad_enabled(False)


class Tracker(object):
    def __init__(self, cfg, gpu_id):
        self.gpu_id = gpu_id

        self.transform = transforms.Compose([
            tr.MultiRestrictSize(cfg.TEST_MIN_SIZE,
                                 cfg.TEST_MAX_SIZE, cfg.TEST_FLIP,
                                 cfg.TEST_MULTISCALE, cfg.MODEL_ALIGN_CORNERS),
            tr.MultiToTensor()
        ])

        self.aug_nums = len(cfg.TEST_MULTISCALE)
        if cfg.TEST_FLIP:
            self.aug_nums = len(cfg.TEST_MULTISCALE) * 2

        vos_model = build_vos_model(cfg.MODEL_VOS, cfg)
        print('cfg.TEST_CKPT_PATH = ', cfg.TEST_CKPT_PATH)

        self.model, _ = load_network(vos_model, cfg.TEST_CKPT_PATH, 0)

        self.all_engines = []

        self.model.eval()

        for aug_idx in range(self.aug_nums):
            if len(self.all_engines) <= aug_idx:
                self.all_engines.append(
                    build_engine(cfg.MODEL_ENGINE,
                                 phase='eval',
                                 aot_model=self.model if aug_idx == 0 else copy.deepcopy(self.model),
                                 gpu_id=gpu_id,
                                 long_term_mem_gap=cfg.TEST_LONG_TERM_MEM_GAP))
                self.all_engines[-1].eval()

        self.engine_A = build_engine(cfg.MODEL_ENGINE, phase='eval',
                                     aot_model=copy.deepcopy(self.model),
                                     gpu_id=gpu_id, long_term_mem_gap=cfg.TEST_LONG_TERM_MEM_GAP)
        self.engine_A.eval()

        self.engine_B = build_engine(cfg.MODEL_ENGINE, phase='eval',
                                     aot_model=copy.deepcopy(self.model),
                                     gpu_id=gpu_id, long_term_mem_gap=cfg.TEST_LONG_TERM_MEM_GAP)
        self.engine_B.eval()

    # add the first frame and label
    def add_first_frame(self, frame, mask, object_num):
        sample = {
            'current_img': frame,
            'current_label': mask,
        }
        sample['meta'] = {
            'obj_num': object_num,
            'height': frame.shape[0],
            'width': frame.shape[1],
            'flip': False,
        }
        sample = self.transform(sample)

        if self.aug_nums > 1:
            torch.cuda.empty_cache()
        for aug_idx in range(self.aug_nums):
            frame = sample[aug_idx]['current_img'].unsqueeze(0).float().cuda(self.gpu_id, non_blocking=True)
            mask = sample[aug_idx]['current_label'].unsqueeze(0).float().cuda(self.gpu_id, non_blocking=True)
            mask = F.interpolate(mask, size=frame.size()[2:], mode="nearest")
            # add reference frame
            self.all_engines[aug_idx].add_reference_frame(frame, mask, frame_step=0, obj_nums=object_num)

    def add_first_frame_A(self, frame, mask, object_num):
        sample = {
            'current_img': frame,
            'current_label': mask,
        }
        sample['meta'] = {
            'obj_num': object_num,
            'height': frame.shape[0],
            'width': frame.shape[1],
            'flip': False,
        }
        sample = self.transform(sample)

        if self.aug_nums > 1:
            torch.cuda.empty_cache()
        for aug_idx in range(self.aug_nums):
            frame = sample[aug_idx]['current_img'].unsqueeze(0).float().cuda(self.gpu_id, non_blocking=True)
            mask = sample[aug_idx]['current_label'].unsqueeze(0).float().cuda(self.gpu_id, non_blocking=True)
            mask = F.interpolate(mask, size=frame.size()[2:], mode="nearest")
            # add reference frame
            self.engine_A.add_reference_frame(frame, mask, frame_step=0, obj_nums=object_num)

    def add_first_frame_B(self, frame, mask, object_num):
        sample = {
            'current_img': frame,
            'current_label': mask,
        }
        sample['meta'] = {
            'obj_num': object_num,
            'height': frame.shape[0],
            'width': frame.shape[1],
            'flip': False,
        }
        sample = self.transform(sample)

        if self.aug_nums > 1:
            torch.cuda.empty_cache()
        for aug_idx in range(self.aug_nums):
            frame = sample[aug_idx]['current_img'].unsqueeze(0).float().cuda(self.gpu_id, non_blocking=True)
            mask = sample[aug_idx]['current_label'].unsqueeze(0).float().cuda(self.gpu_id, non_blocking=True)
            mask = F.interpolate(mask, size=frame.size()[2:], mode="nearest")
            # add reference frame
            self.engine_B.add_reference_frame(frame, mask, frame_step=0, obj_nums=object_num)

    def track(self, image):
        height = image.shape[0]
        width = image.shape[1]

        sample = {'current_img': image}
        sample['meta'] = {
            'height': height,
            'width': width,
            'flip': False
        }
        sample = self.transform(sample)

        all_preds = []
        for aug_idx in range(self.aug_nums):
            output_height = sample[aug_idx]['meta']['height']
            output_width = sample[aug_idx]['meta']['width']
            image = sample[aug_idx]['current_img'].unsqueeze(0).float().cuda(self.gpu_id, non_blocking=True)
            pred_logit = self.all_engines[aug_idx].match_propogate_one_frame(image,
                                                                             output_size=(output_height, output_width))
            is_flipped = sample[aug_idx]['meta']['flip']

            pred_label = torch.argmax(torch.softmax(pred_logit, dim=1), dim=1, keepdim=True).float()
            _pred_label = F.interpolate(pred_label, size=self.all_engines[aug_idx].input_size_2d, mode="nearest")
            self.all_engines[aug_idx].update_memory(_pred_label)

            if is_flipped:
                pred_logit = flip_tensor(pred_logit, 3)
            pred_prob = torch.softmax(pred_logit, dim=1)

            all_preds.append(pred_prob)

        cat_all_preds = torch.cat(all_preds, dim=0)
        pred_prob = torch.mean(cat_all_preds, dim=0, keepdim=True)

        pred_label = torch.argmax(pred_prob, dim=1, keepdim=True).float()

        _pred_label = F.interpolate(pred_label,
                                    size=self.all_engines[aug_idx].input_size_2d,
                                    mode="nearest")
        mask = pred_label.detach().cpu().numpy()[0][0].astype(np.uint8)

        return mask

    def track_A(self, image):
        height = image.shape[0]
        width = image.shape[1]
        sample = {'current_img': image}
        sample['meta'] = {
            'height': height,
            'width': width,
            'flip': False
        }
        sample = self.transform(sample)

        all_preds = []
        for aug_idx in range(self.aug_nums):
            output_height = sample[aug_idx]['meta']['height']
            output_width = sample[aug_idx]['meta']['width']
            image = sample[aug_idx]['current_img'].unsqueeze(0).float().cuda(self.gpu_id, non_blocking=True)
            pred_logit = self.engine_A.match_propogate_one_frame(image, output_size=(output_height, output_width))
            pred_label = torch.argmax(torch.softmax(pred_logit, dim=1), dim=1, keepdim=True).float()
            _pred_label = F.interpolate(pred_label, size=self.engine_A.input_size_2d, mode="nearest")
            self.engine_A.update_memory(_pred_label)
            pred_prob = torch.softmax(pred_logit, dim=1)
            all_preds.append(pred_prob)

        cat_all_preds = torch.cat(all_preds, dim=0)
        pred_prob = torch.mean(cat_all_preds, dim=0, keepdim=True)
        pred_label = torch.argmax(pred_prob, dim=1, keepdim=True).float()
        _pred_label = F.interpolate(pred_label, size=self.engine_A.input_size_2d, mode="nearest")
        mask = pred_label.detach().cpu().numpy()[0][0].astype(np.uint8)

        return mask

    def track_B(self, image):
        height = image.shape[0]
        width = image.shape[1]
        sample = {'current_img': image}
        sample['meta'] = {
            'height': height,
            'width': width,
            'flip': False
        }
        sample = self.transform(sample)

        all_preds = []
        for aug_idx in range(self.aug_nums):
            output_height = sample[aug_idx]['meta']['height']
            output_width = sample[aug_idx]['meta']['width']
            image = sample[aug_idx]['current_img'].unsqueeze(0).float().cuda(self.gpu_id, non_blocking=True)
            pred_logit = self.engine_B.match_propogate_one_frame(image, output_size=(output_height, output_width))
            pred_label = torch.argmax(torch.softmax(pred_logit, dim=1), dim=1, keepdim=True).float()
            _pred_label = F.interpolate(pred_label, size=self.engine_B.input_size_2d, mode="nearest")
            self.engine_B.update_memory(_pred_label)
            pred_prob = torch.softmax(pred_logit, dim=1)
            all_preds.append(pred_prob)

        cat_all_preds = torch.cat(all_preds, dim=0)
        pred_prob = torch.mean(cat_all_preds, dim=0, keepdim=True)
        pred_label = torch.argmax(pred_prob, dim=1, keepdim=True).float()
        _pred_label = F.interpolate(pred_label, size=self.engine_B.input_size_2d, mode="nearest")
        mask = pred_label.detach().cpu().numpy()[0][0].astype(np.uint8)

        return mask


def make_full_size(x, output_sz):
    '''
    zero-pad input x (right and down) to match output_sz
    x: numpy array e.g., binary mask
    output_sz: size of the output [width, height]
    '''
    if x.shape[0] == output_sz[1] and x.shape[1] == output_sz[0]:
        return x
    pad_x = output_sz[0] - x.shape[1]
    if pad_x < 0:
        x = x[:, :x.shape[1] + pad_x]
        # padding has to be set to zero, otherwise pad function fails
        pad_x = 0
    pad_y = output_sz[1] - x.shape[0]
    if pad_y < 0:
        x = x[:x.shape[0] + pad_y, :]
        # padding has to be set to zero, otherwise pad function fails
        pad_y = 0
    return np.pad(x, ((0, pad_y), (0, pad_x)), 'constant', constant_values=0)


def read_img(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


#####################
# config
#####################

cur_colors = [(0, 255, 255),  # yellow b g r
              (255, 0, 0),  # blue
              (0, 255, 0),  # green
              (0, 0, 255),  # red
              (255, 255, 255),  # white
              (0, 0, 0),  # black
              (255, 255, 0),  # Cyan
              (225, 228, 255),  # MistyRose
              (180, 105, 255),  # HotPink
              (255, 0, 255),  # Magenta
              ] * 100

config = {
    'exp_name': 'default',
    'model': 'r50_deaotl',
    # 'pretrain_model_path': 'pretrain_models/aotplus_R50_AOTL_Temp_pe_Slot_4_ema_20000.pth',
    'pretrain_model_path': 'pretrain_models/aotplus_R50_DeOTL_Temp_pe_Slot_4_ema_20000.pth',
    'config': 'pre_vost',
    'long_max': 10,
    'long_gap': 5,
    'short_gap': 2,
    'patch_wised_drop_memories': False,
    'patch_max': 999999,
    'gpu_id': 0,
    'flip': False,
    'ms': [1.0],
    'max_resolution': 600
}
vis_results = False
save_mask = False
save_dir = '/media/File_JiaWenZ2/VOTS24/codes/RMem-main/aot_plus/votst24_val_multi/Mask_assemble'
from PIL import Image

palette_template = Image.open(
    '/media/titan3/File_JiaWenZ2/VOTS24/codes/RMemAOT/aot_plus/tools/mask_palette.png').getpalette()

# get first frame and mask
handle = vot_utils.VOT("mask", multiobject=True)

objects = handle.objects()
imagefile = handle.frame()
if not imagefile:
    sys.exit(0)
seq_name = imagefile.split('/')[-3]

# get first
image_init = read_img(imagefile)

# Get merged-mask
merged_mask = np.zeros((image_init.shape[0], image_init.shape[1]))
object_num = len(objects)
object_id = 1
mask_objects_init = []
for object in objects:
    mask = make_full_size(object, (image_init.shape[1], image_init.shape[0]))
    mask_objects_init.append(mask)
    mask = np.where(mask > 0, object_id, 0)
    merged_mask += mask
    object_id += 1
    # print("Save")

# set cfg
engine_config = importlib.import_module('configs.' + f'{config["config"]}')
cfg = engine_config.EngineConfig(config['exp_name'], config['model'])
cfg.TEST_CKPT_PATH = os.path.join(DIR_PATH, config['pretrain_model_path'])
cfg.TEST_LONG_TERM_MEM_GAP = config['long_gap']
cfg.TEST_FLIP = config['flip']
cfg.TEST_MULTISCALE = config['ms']
cfg.TEST_MIN_SIZE = None
cfg.TEST_MAX_SIZE = config['max_resolution'] * 800. / 480.
# Rmem
cfg.TEST_EMA = True
cfg.FORMER_MEM_LEN = 1
cfg.LATTER_MEM_LEN = 7

if vis_results:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_dir_main = save_dir.replace('Mask_assemble', 'Mask_assemble_main')
    save_dir_A = save_dir.replace('Mask_assemble', 'Mask_assemble_A')
    save_dir_B = save_dir.replace('Mask_assemble', 'Mask_assemble_B')
    if not os.path.exists(save_dir_main):
        os.makedirs(save_dir_main)
    if not os.path.exists(save_dir_A):
        os.makedirs(save_dir_A)
    if not os.path.exists(save_dir_B):
        os.makedirs(save_dir_B)

    save_dir = os.path.join(save_dir, seq_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_dir_main = os.path.join(save_dir_main, seq_name)
    if not os.path.exists(save_dir_main):
        os.makedirs(save_dir_main)
    save_dir_A = os.path.join(save_dir_A, seq_name)
    if not os.path.exists(save_dir_A):
        os.makedirs(save_dir_A)
    save_dir_B = os.path.join(save_dir_B, seq_name)
    if not os.path.exists(save_dir_B):
        os.makedirs(save_dir_B)

### init trackers
tracker = Tracker(cfg, config["gpu_id"])

# initialize tracker
tracker.add_first_frame(image_init, merged_mask, object_num)
mask_size = merged_mask.shape

init_flag = True
frame_count = 0
while True:
    frame_count += 1
    imagefile = handle.frame()
    if not imagefile:
        break
    image = read_img(imagefile)
    m = tracker.track(image)
    m = F.interpolate(torch.tensor(m)[None, None, :, :], size=mask_size, mode="nearest").numpy().astype(np.uint8)[0][0]
    m_main = m.copy()

    if frame_count == 1:
        tracker.add_first_frame_A(image, m, object_num)
    if frame_count == 2:
        tracker.add_first_frame_B(image, m, object_num)

    if frame_count > 1:
        m_A = tracker.track_A(image)
        m_A = \
        F.interpolate(torch.tensor(m_A)[None, None, :, :], size=mask_size, mode="nearest").numpy().astype(np.uint8)[0][
            0]
    if frame_count > 2:
        m_B = tracker.track_B(image)
        m_B = \
        F.interpolate(torch.tensor(m_B)[None, None, :, :], size=mask_size, mode="nearest").numpy().astype(np.uint8)[0][
            0]


    if frame_count > 2:
        obj_list = np.unique(m)
        mask_ = np.zeros_like(m)
        mask_2 = np.zeros_like(m)
        masks_ls = []
        for i in obj_list:
            mask = (m == i).astype(np.uint8)
            mask_A = (m_A == i).astype(np.uint8)
            mask_B = (m_B == i).astype(np.uint8)
            if i == 0:
                masks_ls.append(mask_)
                continue
            if mask.sum() == 0:
                if mask_A.sum() == 0:
                    if mask_B.sum() == 0:
                        masks_ls.append(mask_)
                    else:
                        masks_ls.append(mask_B)
                else:
                    masks_ls.append(mask_A)
                continue

            iou_A = pytorch_iou(torch.from_numpy(mask).cuda().unsqueeze(0),
                                torch.from_numpy(mask_A).cuda().unsqueeze(0), [1]).cpu().numpy()
            iou_B = pytorch_iou(torch.from_numpy(mask).cuda().unsqueeze(0),
                                torch.from_numpy(mask_B).cuda().unsqueeze(0), [1]).cpu().numpy()
            iou_AB = pytorch_iou(torch.from_numpy(mask_A).cuda().unsqueeze(0),
                                 torch.from_numpy(mask_B).cuda().unsqueeze(0), [1]).cpu().numpy()

            iou_thred = 0.25
            if ((iou_A < iou_thred) or (iou_B < iou_thred)) and (mask_A.sum()>3 and mask_B.sum()>3):
                if (iou_AB > max(iou_A, iou_B)) and (iou_AB>0.5):
                    output = mask_B if iou_A<iou_B else mask_A
                else:
                    output = mask
            else:
                output = mask
            masks_ls.append(output)
            mask_2 = mask_2 + output * i

        masks_ls = np.stack(masks_ls)
        masks_ls_argmax = np.argmax(masks_ls, axis=0)
        masks_ls_ = masks_ls.sum(0)

        rs = np.where(masks_ls_ > 1, masks_ls_argmax, mask_2)
        m = np.array(rs).astype(np.uint8)

    masks = transfer_mask(m, object_num)
    handle.report(masks)

    if vis_results:
        VIS_box = False
        VIS_img_mask = True

        # original image + mask
        if VIS_img_mask:
            image_ori = image[:, :, ::-1].copy()  # RGB --> BGR
            image_name = imagefile.split('/')[-1]
            image_m = image_ori.copy().astype(np.float32)

            for idx, m in enumerate(masks):
                image_m[:, :, 0]+=image_m[:, :, 0] * cur_colors[idx][0] * m
                image_m[:, :, 1]+=image_m[:, :, 1] * cur_colors[idx][1] * m
                image_m[:, :, 2]+=image_m[:, :, 2] * cur_colors[idx][2] * m

                contours, _ = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                image_m = cv2.drawContours(image_m, contours, -1, cur_colors[idx], 4)

            image_m = image_m.clip(0, 255).astype(np.uint8)
            image_mask_name_m = image_name.replace('.jpg', '_mask.jpg')
            save_path = os.path.join(save_dir, image_mask_name_m)
            cv2.imwrite(save_path, image_m)


            if init_flag:
                image_init = image_init[:, :, ::-1].copy()
                image_name_init = '00000001.jpg'
                image_m_init = image_init.copy().astype(np.float32)

                for idx, m in enumerate(mask_objects_init):
                    image_m_init[:, :, 1] += 127.0 * m
                    image_m_init[:, :, 2] += 127.0 * m
                    contours, _ = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    image_m_init = cv2.drawContours(image_m_init, contours, -1, cur_colors[idx], 2)

                image_m_init = image_m_init.clip(0, 255).astype(np.uint8)
                image_mask_name_m = image_name_init.replace('.jpg', '_mask.jpg')
                save_path = os.path.join(save_dir, image_mask_name_m)
                cv2.imwrite(save_path, image_m_init)
                init_flag = False
