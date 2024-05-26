import pdb
import os
os.environ['CUDA_VISIBLE_DEVICES']='3'

import numpy as np


import cv2
# 读取txt文件中的顶点坐标
import torch
import os.path as osp
import pickle
import glob
from third_parties.lpips import LPIPS
from third_parties.lpips import LPIPS
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import mean_squared_error as compare_mse


def load_points():
    points = np.loadtxt("/mnt/data2/lcj/SelfReconCode2/nerfcaptemplates hapeT.txt")
    points=torch.Tensor(points)
    # 创建mesh对象
    triangles = o3d.utility.Vector3iVector(np.zeros((0, 3), dtype=int))
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(points), triangles)

    # 为mesh添加颜色
    mesh.paint_uniform_color([0.7, 0.7, 0.7])

    # 显示mesh
    o3d.visualization.draw_geometries([mesh])

    # 输出为.ply文件
    o3d.io.write_triangle_mesh("output.ply", mesh)

# vedio2img
def vedio2img():
    video_path = '/mnt/data2/lcj/dataset/MonoPerfCapDataset/Franzi_studio/'
    folder_name = video_path + "masks"   # imgs & masks
    os.makedirs(folder_name, exist_ok=True)
    vc = cv2.VideoCapture(video_path + "mask.mp4")  # 读入视频文件
    c = 0
    rval = vc.isOpened()

    while rval:  # 循环读取视频帧
        rval, frame = vc.read()
        pic_path = folder_name + '/'
        if rval:
            cv2.imwrite(pic_path + '%06d.png'%c, frame)  # 存储为图像,保存名为 文件夹名_数字（第几个文件）.jpg
            cv2.waitKey(1)
        else:
            break
        c = c + 1
    vc.release()
    print('save_success')


# img2vedio
def img2vedio():
    image_folder = '/mnt/data2/lcj/humannerf-main/experiments/human_nerf/wild/red/single_gpu/latest/movement'
    output_path = '/mnt/data2/lcj/humannerf-main/experiments/human_nerf/wild/red/single_gpu/latest/color.mp4'
    fps = 30
    images = sorted(os.listdir(image_folder))   # 根据首位sort,所以命名不是很好的图像无法生成比较好的视频
    num=len(images)
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或者使用其他的视频编码器
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i in range(1000):
        image_path = os.path.join(image_folder, '%06d.png' % (i))
        frame = cv2.imread(image_path)
        video_writer.write(frame)

    # 命名规范的时候可以用
    # for image_name in images:
    #     image_path = os.path.join(image_folder, image_name)
    #     frame = cv2.imread(image_path)
    #     video_writer.write(frame)

    video_writer.release()




# camera.pkl-->camera.npz
def pkl2npz():
    # 新增加extrinsic，外参矩阵
    ps = np.array([981.4922,526.2165])
    fs = np.array([1515.6488,1516.3548])
    trans = np.zeros(3)  # 啥也不是 写进pkl的时候就是0
    rt = np.zeros(3)
    # extrinsic = cam_data['camera_e']
    assert (np.linalg.norm(
        rt) < 0.0001)  # The cameras of snapshot dataset seems no rotation and translation 相机固定的意思
    H = 1080
    W = 1920

    quat = np.array([np.cos(np.pi / 2.), 0., 0., np.sin(np.pi / 2.)])
    T = trans
    fx = fs[0]
    fy = fs[1]
    cx = ps[0]
    cy = ps[1]

    np.savez(osp.join('/mnt/data2/lcj/dataset/MonoPerfCapDataset/Franzi_studio/', 'camera.npz'), fx=fx, fy=fy, cx=cx, cy=cy, quat=quat,
             T=T)
    print('pkl2npz is ok!')

# 可视化mask之间的差距 #
def vis_mask_delta():
    device = torch.device(0)
    folder_path = "/mnt/data2/lcj/dataset/deepcap/Antonia1"
    gt_mask_folder_path = os.path.join(folder_path, "masks")
    result_folder_path=os.path.join(folder_path, "result_origin200")
    pred_mask_folder_path = os.path.join(result_folder_path, "masks1")
    gt_image_files = glob.glob(os.path.join(gt_mask_folder_path, "*.png"))
    num_images = len(gt_image_files)
    print("Number of images in folder:", num_images)
    mask_delta_save_folder_path=os.path.join(pred_mask_folder_path, "masks_delta")
    os.makedirs(mask_delta_save_folder_path, exist_ok = True)
    writer_mask_delta = cv2.VideoWriter(osp.join(mask_delta_save_folder_path, 'mask_delta.mp4'),
                                        cv2.VideoWriter.fourcc('m', 'p', '4', 'v'), 30., (1920, 1080))

    for i in range(num_images):
        print(i)
        gt_mask_path = osp.join(gt_mask_folder_path, '%06d.png' % i)
        gt_mask = cv2.imread(gt_mask_path)/255.
        pred_mask_path = osp.join(pred_mask_folder_path, '%d.png' % i)
        pred_mask = cv2.imread(pred_mask_path)/255.
        mask_delta = (np.abs(pred_mask - gt_mask) * 255.).astype(np.uint8)
        writer_mask_delta.write(mask_delta)
        cv2.imwrite(osp.join(mask_delta_save_folder_path, '%06d.png' %i), mask_delta)
        torch.cuda.empty_cache()
    writer_mask_delta.release()
    print("Done!")


# 原论文对mask的操作 #
# mask>0 and dilation #
def original_mask_operation():
    device = torch.device(0)
    folder_path="/mnt/data2/lcj/dataset/deepcap/Magdalena0"
    gt_mask_original_process_save_path = os.path.join(folder_path, "masks_origin_process")
    gt_mask_0_process_save_path=os.path.join(folder_path, "masks>0_process")
    gt_mask_folder_path = os.path.join(folder_path, "masks")
    gt_image_files = glob.glob(os.path.join(gt_mask_folder_path, "*.png"))
    num_images = len(gt_image_files)
    print("Number of images in folder:", num_images)
    for i in range(num_images):
        print(i)
        gt_mask_path = osp.join(gt_mask_folder_path, '%06d.png' % i)
        # 读取mask
        gt_mask=(torch.from_numpy(cv2.imread(gt_mask_path))>0).view(1080,1920,-1).any(-1).float().to(device)
        gt_mask0=gt_mask
        gtMasks0=(gt_mask0 * 255.).detach().cpu().numpy().astype(np.uint8)
        cv2.imwrite(osp.join(gt_mask_0_process_save_path, '%06d.png' % i), gtMasks0)
        radius=int(np.round(0.0041/ 2. * float(min(1080, 1920)) / 1.2))
        mgtMs = torch.nn.functional.max_pool2d(gt_mask.reshape(1,1080,1920), kernel_size=2 * radius + 1, stride=1, padding=radius)
        gtMasks = (mgtMs * 255.).detach().cpu().numpy().astype(np.uint8)
        cv2.imwrite(osp.join(gt_mask_original_process_save_path, '%06d.png' %i), gtMasks.reshape(1080,1920))
        torch.cuda.empty_cache()
    print("mask operation is ok!")

###### metircs ######
### color ###
## 只在mask中计算 ##
# def psnr(pred_img, gt_img, peak=255.):
#     pdb.set_trace()
#     return 10 * torch.log10(peak ** 2 / torch.mean((1. * pred_img - 1. * gt_img) ** 2))
#
# def dssim(pred_img, gt_img, range=255.):
#     from skimage.measure import compare_ssim
#     return (1 - compare_ssim(pred_img, gt_img, data_range=range, multichannel=True)) / 2.

# [0,1]->[-1,1]
def scale_for_lpips(image_tensor):
    return image_tensor * 2. - 1.

def compute_lpips(lpips,pred_img, gt_img):
    lpips_metric = lpips(scale_for_lpips(pred_img.permute(2, 0, 1)),
                          scale_for_lpips(gt_img.permute(2, 0, 1))) # (1080,1920,3)-->(3, 1080, 1920)
    return lpips_metric

def compute_iou(pred_mask, gt_mask):
    # 要换掉，现在的方法好像只适用于二进制
    """
    Calculate intersection over union (IoU) of two binary masks.
    Args:
        mask1: numpy array of shape (H, W)
        mask2: numpy array of shape (H, W)
    Returns:
        IoU: float, the IoU of the two masks
    """
    # intersection = np.logical_and(pred_mask, gt_mask)
    # union = np.logical_or(pred_mask, gt_mask)
    # iou_score = np.sum(intersection) / np.sum(union)
    iou_score = (pred_mask * gt_mask).reshape(-1).sum(0) / (np.abs((pred_mask + gt_mask - pred_mask * gt_mask)).reshape(-1).sum(0))

    return iou_score


### color ###
def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def compute_metrics():
    device = torch.device(0)
    folder_path="/mnt/data2/lcj/dataset/DynaCap/red/"
    gt_img_folder_path =os.path.join(folder_path, "imgs")
    gt_mask_folder_path = os.path.join(folder_path, "masks")
    gt_normal_folder_path = os.path.join(folder_path, "normals")
    gt_image_files = glob.glob(os.path.join(gt_img_folder_path, "*.png"))
    num_images = len(gt_image_files)
    print("Number of images in folder:", num_images)
    pred_folder_path=os.path.join(folder_path, "result_spin_dia1")
    pred_img_folder_path = os.path.join(pred_folder_path, "colors1")
    pred_mask_folder_path = os.path.join(pred_folder_path, "masks1")
    pred_normal_folder_path = os.path.join(pred_folder_path, "normals1")
    print("Compute metrics:")
    metrics = {}
    metrics['mse'] = -1. * np.ones(num_images)
    metrics['psnr'] = -1. * np.ones(num_images)
    metrics['ssim'] = -1. * np.ones(num_images)
    metrics['lpips'] = -1. * np.ones(num_images)
    metrics['mae'] = -1. * np.ones(num_images)
    metrics['iou'] = -1. * np.ones(num_images)
    metrics['mask_mse'] = -1. * np.ones(num_images)
    mse_total=0.
    psnr_total = 0.
    ssim_total = 0.
    lpips_total = 0.
    mae_total=0.
    iou_total = 0.
    mask_mse_total = 0.
    lpips = LPIPS(net='vgg').to(device)
    set_requires_grad(lpips, requires_grad=False)
    W = 1284
    H = 940
    for i in range(num_images):
        gt_img_path = osp.join(gt_img_folder_path, '%06d.png' % i)
        pred_img_path = osp.join(pred_img_folder_path, '%d.png' % i)
        gt_mask_path = osp.join(gt_mask_folder_path, '%06d.png' % i)
        pred_mask_path = osp.join(pred_mask_folder_path, '%d.png' % i)
        gt_normal_path = osp.join(gt_normal_folder_path, '%06d.png' % i)
        pred_normal_path = osp.join(pred_normal_folder_path, '%d.png' % i)
        # 读取img 转换为浮点数 归一化到[0,1]
        gt_img_origin = cv2.imread(gt_img_path) / 255.
        pred_img_origin = cv2.imread(pred_img_path) / 255.
        # 读取mask
        gt_mask = cv2.imread(gt_mask_path)/ 255.
        pred_mask = cv2.imread(pred_mask_path) / 255.
        # 读取normal
        gt_normal_origin = cv2.imread(gt_normal_path)[:, :, ::-1]
        gt_normal_origin = 2. * gt_normal_origin.astype(np.float32) / 255. - 1.
        pred_normal_origin = cv2.imread(pred_normal_path)[:, :, ::-1]
        pred_normal_origin = 2. * pred_normal_origin.astype(np.float32) / 255. - 1.
        # 只取mask内的(如果是概率mask的话，这样相乘，边缘的颜色会有问题)
        # 所以一定要使用二值mask
        #gt_img = gt_img_origin*gt_mask
        gt_img = gt_img_origin
        gt_img[gt_mask == 0] = [1.]
        pred_img=pred_img_origin
        gt_normal=gt_normal_origin*gt_mask
        pred_normal = pred_normal_origin * gt_mask
        ## 计算metric
        # color
        mse_single = 10000*compare_mse(pred_img, gt_img)
        psnr_single = compare_psnr(pred_img, gt_img)
        ssim_single = compare_ssim(pred_img, gt_img,data_range=1.0,channel_axis=-1)
        lpips_single = 1000*compute_lpips(lpips,torch.from_numpy(pred_img).view(H, W,3).float().to(device), torch.from_numpy(gt_img).view(H, W,3).float().to(device))
        # normal #
        mae_single = np.mean(np.abs(pred_normal - gt_normal))
        # mask #
        # pred_mask由pytorch的渲染器渲染mesh生成，因此mash不是0就是1,3个通道的值也是一样的
        # gtmask由他人算法得到，范围在[0~1]，大小为[1080,1920,3],但是3通道的数值都一样，因为可以直接取某一通道作为该像素点mask的值
        # iou
        iou_single = compute_iou(pred_mask[:, :, 0].reshape(H, W), gt_mask[:, :, 0].reshape(H, W))
        # mse #
        mask_mse_single = 10000 * compare_mse(pred_mask[:, :, 0].reshape(H, W), gt_mask[:, :, 0].reshape(H, W))

        # 记录指标 #
        metrics['mse'][i] = mse_single
        metrics['psnr'][i] = psnr_single
        metrics['ssim'][i] = ssim_single
        metrics['lpips'][i] = lpips_single
        metrics['mae'][i] = mae_single
        metrics['iou'][i] = iou_single
        metrics['mask_mse'][i] = mask_mse_single
        print("image", i, " color_mse:", mse_single," color_psnr:", psnr_single, " color_ssim:", ssim_single, " color_lpips:", lpips_single,
              " normal_mae:", mae_single," mask_iou:", iou_single," mask_mse:", mask_mse_single)
        mse_total+=mse_single
        psnr_total += psnr_single
        ssim_total += ssim_single
        lpips_total += lpips_single
        mae_total += mae_single
        iou_total += iou_single
        mask_mse_total += mask_mse_single
        torch.cuda.empty_cache()

    # 分别写进一个txt文件 #
    with open(osp.join(pred_folder_path, 'color_mse.txt'), 'w') as ff:
        ff.write('      mse\n')
        mse = metrics['mse']
        for ind, e in enumerate(mse.tolist()):
            if e >= 0.:
                ff.write('%6d: %.9f\n' % (ind, e))
        ff.write('mse mean: %.6f, max: %.6f, min: %.6f, maxinds:' % (mse.mean(), mse.max(), mse.min()))  # mse 越小越好，找出最大mse，也就是找有问题的几帧
        # 在默认情况下，argsort() 返回的是按升序排列的数组的索引，即最小值的索引排在最前面。
        # 因此，如果我们要找到最大值的索引，我们可以将待排序数组取负数，然后进行排序，这样就会将最大值变成最小值，再取前几个元素，就可以得到最大值的索引。
        for ind in (-mse).argsort()[:10]:
            ff.write('%d ' % ind)
    print('color mse done')
    with open(osp.join(pred_folder_path, 'color_psnr.txt'), 'w') as ff:
        ff.write('      psnr\n')
        psnr = metrics['psnr']
        for ind, e in enumerate(psnr.tolist()):
            if e >= 0.:
                ff.write('%6d: %.9f\n' % (ind, e))
        ff.write('psnr mean: %.6f, max: %.6f, min: %.6f, mininds:' % (psnr.mean(), psnr.max(), psnr.min()))  # psnr 越高越好，找出最小psnr，也就是找有问题的几帧
        for ind in (psnr).argsort()[:10]:
            ff.write('%d ' % ind)
    print('color psnr done')
    with open(osp.join(pred_folder_path, 'color_ssim.txt'), 'w') as ff:
        ff.write('      ssim\n')
        ssim = metrics['ssim']
        for ind, e in enumerate(ssim.tolist()):
            if e >= 0.:
                ff.write('%6d: %.9f\n' % (ind, e))
        ff.write('ssim mean: %.6f, max: %.6f, min: %.6f, mininds:' % (ssim.mean(), ssim.max(), ssim.min()))  # ssim 越高越好
        for ind in (ssim).argsort()[:10]:
            ff.write('%d ' % ind)
    print('color ssim done')
    with open(osp.join(pred_folder_path, 'color_lpips.txt'), 'w') as ff:
        ff.write('      lpips\n')
        lpips = metrics['lpips']
        for ind, e in enumerate(lpips.tolist()):
            if e >= 0.:
                ff.write('%6d: %.9f\n' % (ind, e))
        ff.write('lpips mean: %.6f, max: %.6f, min: %.6f, maxinds:' % (lpips.mean(), lpips.max(), lpips.min())) # lpips 越小越好
        for ind in (-lpips).argsort()[:10]:
            ff.write('%d ' % ind)
    print('color lpips done')

    with open(osp.join(pred_folder_path, 'normal_mae.txt'), 'w') as ff:
        ff.write('      mae\n')
        mae = metrics['mae']
        for ind, e in enumerate(mae.tolist()):
            if e >= 0.:
                ff.write('%6d: %.9f\n' % (ind, e))
        ff.write('mae mean: %.6f, max: %.6f, min: %.6f, maxinds:' % (mae.mean(), mae.max(), mae.min()))  # mae 越小越好，找出最大mae，也就是找有问题的几帧
        for ind in (-mae).argsort()[:10]:
            ff.write('%d ' % ind)
    print('normal mae done')

    with open(osp.join(pred_folder_path, 'mask_iou.txt'), 'w') as ff:
        ff.write('      iou\n')
        iou = metrics['iou']
        for ind, e in enumerate(iou.tolist()):
            if e >= 0.:
                ff.write('%6d: %.9f\n' % (ind, e))
        ff.write('iou mean: %.6f, max: %.6f, min: %.6f, mininds:' % (iou.mean(), iou.max(), iou.min()))  # iou 越高越好，找出最小iou，也就是找有问题的几帧
        for ind in (iou).argsort()[:10]:
            ff.write('%d ' % ind)
    print('mask iou done')

    with open(osp.join(pred_folder_path, 'mask_mse.txt'), 'w') as ff:
        ff.write('      mse\n')
        mse = metrics['mse']
        for ind, e in enumerate(mse.tolist()):
            if e >= 0.:
                ff.write('%6d: %.9f\n' % (ind, e))
        ff.write('mse mean: %.6f, max: %.6f, min: %.6f, maxinds:' % (mse.mean(), mse.max(), mse.min()))  # mse 越小越好
        for ind in (-mse).argsort()[:10]:
            ff.write('%d ' % ind)
    print('mask mse done')

    # num_temp=end-start
    mse_average=mse_total/num_images
    psnr_average = psnr_total / num_images
    ssim_average = ssim_total / num_images
    lpips_average = lpips_total / num_images
    mae_average = mae_total / num_images
    iou_average = iou_total / num_images
    mask_mse_average = mask_mse_total / num_images
    print("mse_average:", mse_average,"psnr_average:", psnr_average, " ssim_average:", ssim_average, " lpips_average:", lpips_average,
          " mae_average:", mae_average," iou_average:", iou_average," mask_mse_average:", mask_mse_average)
    print("End!")

### 测试mask 指标 ###
# iou MAD MSE Grad Conn dtSSD
def compute_mask_metrics():
    device = torch.device(0)
    folder_path="/mnt/data2/lcj/dataset/deepcap/Antonia1/"
    gt_img_folder_path =os.path.join(folder_path, "imgs")
    gt_mask_folder_path = os.path.join(folder_path, "masks")
    gt_image_files = glob.glob(os.path.join(gt_img_folder_path, "*.png"))
    num_images = len(gt_image_files)
    print("Number of images in folder:", num_images)
    pred_folder_path=os.path.join(folder_path, "finetune_mlp_lbs_p_lpips_1.0")
    pred_mask_folder_path = os.path.join(pred_folder_path, "masks1")
    print("Compute metrics:")
    metrics = {}
    metrics['iou'] = -1. * np.ones(num_images)
    metrics['mse'] = -1. * np.ones(num_images)
    iou_total = 0.
    mse_total = 0.
    for i in range(num_images):
        gt_mask_path = osp.join(gt_mask_folder_path, '%06d.png' % i)
        pred_mask_path = osp.join(pred_mask_folder_path, '%d.png' % i)
        # 读取mask
        gt_mask = cv2.imread(gt_mask_path)/ 255.
        pred_mask = cv2.imread(pred_mask_path) / 255.
        ## 计算metric
        # iou
        iou_single=compute_iou(pred_mask[:,:,0].reshape(1080,1920),gt_mask[:,:,0].reshape(1080,1920))
        # mse #
        mse_single = 10000*compare_mse(pred_mask[:, :, 0].reshape(1080, 1920), gt_mask[:, :, 0].reshape(1080, 1920))

        # 记录指标 #
        metrics['iou'][i] = iou_single
        metrics['mse'][i] = mse_single
        print("image", i," mask_iou:", iou_single," mask_mse:", mse_single)
        iou_total += iou_single
        mse_total += mse_single
        torch.cuda.empty_cache()

    # 分别写进一个txt文件 #
    with open(osp.join(pred_folder_path, 'test_mask_iou.txt'), 'w') as ff:
        ff.write('      iou\n')
        iou = metrics['iou']
        for ind, e in enumerate(iou.tolist()):
            if e >= 0.:
                ff.write('%6d: %.9f\n' % (ind, e))
        ff.write('iou mean: %.6f, max: %.6f, min: %.6f, mininds:' % (iou.mean(), iou.max(), iou.min()))  # iou 越高越好，找出最小iou，也就是找有问题的几帧
        for ind in (iou).argsort()[:10]:
            ff.write('%d ' % ind)
    print('mask iou done')
    with open(osp.join(pred_folder_path, 'test_mask_mse.txt'), 'w') as ff:
        ff.write('      mse\n')
        mse = metrics['mse']
        for ind, e in enumerate(mse.tolist()):
            if e >= 0.:
                ff.write('%6d: %.9f\n' % (ind, e))
        ff.write('mse mean: %.6f, max: %.6f, min: %.6f, maxinds:' % (mse.mean(), mse.max(), mse.min()))  # mse 越小越好
        for ind in (-mse).argsort()[:10]:
            ff.write('%d ' % ind)
    print('mask mse done')

    iou_average = iou_total / num_images
    mse_average = mse_total / num_images
    print(" iou_average:", iou_average," mse_average:", mse_average)
    print("End!")

#compute_metrics()

# humannerf metric
def compute_metrics_humannerf():
    device = torch.device(0)
    folder_path="/mnt/data2/lcj/humannerf-main/dataset/wild/red/"
    gt_img_folder_path =os.path.join(folder_path, "images")
    gt_mask_folder_path = os.path.join(folder_path, "masks")
    gt_image_files = glob.glob(os.path.join(gt_img_folder_path, "*.png"))
    num_images = len(gt_image_files)
    print("Number of images in folder:", num_images)
    pred_folder_path="/mnt/data2/lcj/humannerf-main/experiments/human_nerf/wild/red/single_gpu/latest/"
    pred_img_folder_path = os.path.join(pred_folder_path, "movement")
    print("Compute metrics:")
    metrics = {}
    metrics['mse'] = -1. * np.ones(num_images)
    metrics['psnr'] = -1. * np.ones(num_images)
    metrics['ssim'] = -1. * np.ones(num_images)
    metrics['lpips'] = -1. * np.ones(num_images)
    mse_total=0.
    psnr_total = 0.
    ssim_total = 0.
    lpips_total = 0.
    lpips = LPIPS(net='vgg').to(device)
    set_requires_grad(lpips, requires_grad=False)
    W=1284
    H=940
    for i in range(num_images):
        gt_img_path = osp.join(gt_img_folder_path, '%06d.png' % i)
        pred_img_path = osp.join(pred_img_folder_path, '%06d.png' % i)
        gt_mask_path = osp.join(gt_mask_folder_path, '%06d.png' % i)
        # 读取img 转换为浮点数 归一化到[0,1]
        gt_img_origin = cv2.imread(gt_img_path) / 255.
        pred_img_origin = cv2.imread(pred_img_path) / 255.
        # 读取mask
        gt_mask = cv2.imread(gt_mask_path)/ 255.
        # 只取mask内的(如果是概率mask的话，这样相乘，边缘的颜色会有问题)
        # 所以一定要使用二值mask
        gt_img=gt_img_origin
        gt_img[gt_mask==0] = [1.]
        pred_img=pred_img_origin
        ## 计算metric
        # color
        mse_single = 10000*compare_mse(pred_img, gt_img)
        psnr_single = compare_psnr(pred_img, gt_img)
        ssim_single = compare_ssim(pred_img, gt_img,data_range=1.0,channel_axis=-1)  # (1080,1920,3)
        lpips_single = 1000*compute_lpips(lpips,torch.from_numpy(pred_img).view(H,W,3).float().to(device), torch.from_numpy(gt_img).view(H,W,3).float().to(device))

        # 记录指标 #
        metrics['mse'][i] = mse_single
        metrics['psnr'][i] = psnr_single
        metrics['ssim'][i] = ssim_single
        metrics['lpips'][i] = lpips_single

        print("image", i, " color_mse:", mse_single," color_psnr:", psnr_single, " color_ssim:", ssim_single, " color_lpips:", lpips_single)
        mse_total+=mse_single
        psnr_total += psnr_single
        ssim_total += ssim_single
        lpips_total += lpips_single
        torch.cuda.empty_cache()

    # 分别写进一个txt文件 #
    with open(osp.join(pred_folder_path, 'color_mse.txt'), 'w') as ff:
        ff.write('      mse\n')
        mse = metrics['mse']
        for ind, e in enumerate(mse.tolist()):
            if e >= 0.:
                ff.write('%6d: %.9f\n' % (ind, e))
        ff.write('mse mean: %.6f, max: %.6f, min: %.6f, maxinds:' % (mse.mean(), mse.max(), mse.min()))  # mse 越小越好，找出最大mse，也就是找有问题的几帧
        # 在默认情况下，argsort() 返回的是按升序排列的数组的索引，即最小值的索引排在最前面。
        # 因此，如果我们要找到最大值的索引，我们可以将待排序数组取负数，然后进行排序，这样就会将最大值变成最小值，再取前几个元素，就可以得到最大值的索引。
        for ind in (-mse).argsort()[:10]:
            ff.write('%d ' % ind)
    print('color mse done')
    with open(osp.join(pred_folder_path, 'color_psnr.txt'), 'w') as ff:
        ff.write('      psnr\n')
        psnr = metrics['psnr']
        for ind, e in enumerate(psnr.tolist()):
            if e >= 0.:
                ff.write('%6d: %.9f\n' % (ind, e))
        ff.write('psnr mean: %.6f, max: %.6f, min: %.6f, mininds:' % (psnr.mean(), psnr.max(), psnr.min()))  # psnr 越高越好，找出最小psnr，也就是找有问题的几帧
        for ind in (psnr).argsort()[:10]:
            ff.write('%d ' % ind)
    print('color psnr done')
    with open(osp.join(pred_folder_path, 'color_ssim.txt'), 'w') as ff:
        ff.write('      ssim\n')
        ssim = metrics['ssim']
        for ind, e in enumerate(ssim.tolist()):
            if e >= 0.:
                ff.write('%6d: %.9f\n' % (ind, e))
        ff.write('ssim mean: %.6f, max: %.6f, min: %.6f, mininds:' % (ssim.mean(), ssim.max(), ssim.min()))  # ssim 越高越好
        for ind in (ssim).argsort()[:10]:
            ff.write('%d ' % ind)
    print('color ssim done')
    with open(osp.join(pred_folder_path, 'color_lpips.txt'), 'w') as ff:
        ff.write('      lpips\n')
        lpips = metrics['lpips']
        for ind, e in enumerate(lpips.tolist()):
            if e >= 0.:
                ff.write('%6d: %.9f\n' % (ind, e))
        ff.write('lpips mean: %.6f, max: %.6f, min: %.6f, maxinds:' % (lpips.mean(), lpips.max(), lpips.min())) # lpips 越小越好
        for ind in (-lpips).argsort()[:10]:
            ff.write('%d ' % ind)
    print('color lpips done')


    mse_average=mse_total/num_images
    psnr_average = psnr_total / num_images
    ssim_average = ssim_total / num_images
    lpips_average = lpips_total / num_images

    print("mse_average:", mse_average,"psnr_average:", psnr_average, " ssim_average:", ssim_average, " lpips_average:", lpips_average)
    print("End!")


### econ直接可视化比较吧，不比指标了
# 前512*512是gt 中间512*512是pred正面 最后512*512是pred背面
# 分割出gt和pred，normal和gtnormal比较；根据pred得到mask，和gtmask比较
def compute_metrics_econ():
    device = torch.device(0)
    folder_path = "/mnt/data2/lcj/dataset/deepcap/Antonia1/"
    gt_mask_folder_path = os.path.join(folder_path, "masks")
    gt_normal_folder_path = os.path.join(folder_path, "normals")
    gt_mask_files = glob.glob(os.path.join(gt_mask_folder_path, "*.png"))
    num_images = len(gt_mask_files)
    print("Number of images in folder:", num_images)

    pred_folder_path = os.path.join(folder_path, "econ_results")
    pred_econ_folder_path = os.path.join(pred_folder_path, "econ")
    pred_econ_img_folder_path = os.path.join(pred_econ_folder_path, "png")
    culled_pred_econ_img_folder_path=os.path.join(pred_econ_folder_path, "culled_png")

    os.makedirs(culled_pred_econ_img_folder_path,exist_ok=True)

    for i in range(num_images):
        gt_mask_path = osp.join(gt_mask_folder_path, '%06d.png' % i)
        # 读入原始图像
        image = cv2.imread(osp.join(pred_econ_img_folder_path, '%06d_crop.png' % i))

        # 获取原始图像的尺寸
        height, width, _ = image.shape

        # 计算要裁剪的中间部分的起始和结束位置
        start_x = 512
        end_x = start_x + 512


        # 裁剪图像
        cropped_image = image[:, start_x:end_x]

        # 保存裁剪后的图像
        cv2.imwrite(osp.join(culled_pred_econ_img_folder_path, '%06d.png' % i),cropped_image)

    print("Compute metrics:")
    metrics = {}
    metrics['mse'] = -1. * np.ones(num_images)
    metrics['psnr'] = -1. * np.ones(num_images)
    metrics['ssim'] = -1. * np.ones(num_images)
    metrics['lpips'] = -1. * np.ones(num_images)
    metrics['mae'] = -1. * np.ones(num_images)
    metrics['iou'] = -1. * np.ones(num_images)
    metrics['mask_mse'] = -1. * np.ones(num_images)
    mse_total = 0.
    psnr_total = 0.
    ssim_total = 0.
    lpips_total = 0.
    mae_total = 0.
    iou_total = 0.
    mask_mse_total = 0.
    lpips = LPIPS(net='vgg').to(device)
    set_requires_grad(lpips, requires_grad=False)
    for i in range(num_images):
        gt_mask_path = osp.join(gt_mask_folder_path, '%06d.png' % i)
        pred_mask_path = osp.join(pred_mask_folder_path, '%d.png' % i)
        gt_normal_path = osp.join(gt_normal_folder_path, '%06d.png' % i)
        pred_normal_path = osp.join(pred_normal_folder_path, '%d.png' % i)
        # 读取mask
        gt_mask = cv2.imread(gt_mask_path)/ 255.
        pred_mask = cv2.imread(pred_mask_path) / 255.
        # 读取normal
        gt_normal_origin = cv2.imread(gt_normal_path)[:, :, ::-1]
        gt_normal_origin = 2. * gt_normal_origin.astype(np.float32) / 255. - 1.
        pred_normal_origin = cv2.imread(pred_normal_path)[:, :, ::-1]
        pred_normal_origin = 2. * pred_normal_origin.astype(np.float32) / 255. - 1.
        # 只取mask内的(如果是概率mask的话，这样相乘，边缘的颜色会有问题)
        # 所以一定要使用二值mask
        gt_img=gt_img_origin*pred_mask
        pred_img=pred_img_origin*pred_mask
        gt_normal=gt_normal_origin*pred_mask
        pred_normal = pred_normal_origin * pred_mask
        ## 计算metric
        # color
        mse_single = 10000*compare_mse(pred_img, gt_img)
        psnr_single = compare_psnr(pred_img, gt_img)
        ssim_single = compare_ssim(pred_img, gt_img,multichannel=True)
        lpips_single = 1000*compute_lpips(lpips,torch.from_numpy(pred_img).view(1080,1920,3).float().to(device), torch.from_numpy(gt_img).view(1080,1920,3).float().to(device))
        # normal #
        mae_single = np.mean(np.abs(pred_normal - gt_normal))
        # mask #
        # pred_mask由pytorch的渲染器渲染mesh生成，因此mash不是0就是1,3个通道的值也是一样的
        # gtmask由他人算法得到，范围在[0~1]，大小为[1080,1920,3],但是3通道的数值都一样，因为可以直接取某一通道作为该像素点mask的值
        # iou
        iou_single = compute_iou(pred_mask[:, :, 0].reshape(1080, 1920), gt_mask[:, :, 0].reshape(1080, 1920))
        # mse #
        mask_mse_single = 10000 * compare_mse(pred_mask[:, :, 0].reshape(1080, 1920), gt_mask[:, :, 0].reshape(1080, 1920))

        # 记录指标 #
        metrics['mae'][i] = mae_single
        metrics['iou'][i] = iou_single
        metrics['mask_mse'][i] = mask_mse_single
        print("image", i, " normal_mae:", mae_single," mask_iou:", iou_single," mask_mse:", mask_mse_single)
        mae_total += mae_single
        iou_total += iou_single
        mask_mse_total += mask_mse_single
        torch.cuda.empty_cache()

    with open(osp.join(pred_folder_path, 'normal_mae.txt'), 'w') as ff:
        ff.write('      mae\n')
        mae = metrics['mae']
        for ind, e in enumerate(mae.tolist()):
            if e >= 0.:
                ff.write('%6d: %.9f\n' % (ind, e))
        ff.write('mae mean: %.6f, max: %.6f, min: %.6f, maxinds:' % (mae.mean(), mae.max(), mae.min()))  # mae 越小越好，找出最大mae，也就是找有问题的几帧
        for ind in (-mae).argsort()[:10]:
            ff.write('%d ' % ind)
    print('normal mae done')

    with open(osp.join(pred_folder_path, 'mask_iou.txt'), 'w') as ff:
        ff.write('      iou\n')
        iou = metrics['iou']
        for ind, e in enumerate(iou.tolist()):
            if e >= 0.:
                ff.write('%6d: %.9f\n' % (ind, e))
        ff.write('iou mean: %.6f, max: %.6f, min: %.6f, mininds:' % (iou.mean(), iou.max(), iou.min()))  # iou 越高越好，找出最小iou，也就是找有问题的几帧
        for ind in (iou).argsort()[:10]:
            ff.write('%d ' % ind)
    print('mask iou done')

    with open(osp.join(pred_folder_path, 'mask_mse.txt'), 'w') as ff:
        ff.write('      mse\n')
        mse = metrics['mse']
        for ind, e in enumerate(mse.tolist()):
            if e >= 0.:
                ff.write('%6d: %.9f\n' % (ind, e))
        ff.write('mse mean: %.6f, max: %.6f, min: %.6f, maxinds:' % (mse.mean(), mse.max(), mse.min()))  # mse 越小越好
        for ind in (-mse).argsort()[:10]:
            ff.write('%d ' % ind)
    print('mask mse done')


    mae_average = mae_total / num_images
    iou_average = iou_total / num_images
    mask_mse_average = mask_mse_total / num_images
    print(" mae_average:", mae_average," iou_average:", iou_average," mask_mse_average:", mask_mse_average)
    print("End!")

# 前512*512是gt 中间512*512是pred 最后512*512是smpl+衣服
def compute_metrics_scarf():
    folder_path = "/mnt/data/lcj/SCARF/exps/mine/male-3-casual/hybrid/visualization/"
    pred_folder_path=os.path.join(folder_path, "capture")
    pred_files = glob.glob(os.path.join(pred_folder_path, "*.jpg"))
    num_images = len(pred_files)//2
    print("Number of images in folder:", num_images)

    device = torch.device(0)
    print("Compute metrics:")
    metrics = {}
    metrics['mse'] = -1. * np.ones(num_images)
    metrics['psnr'] = -1. * np.ones(num_images)
    metrics['ssim'] = -1. * np.ones(num_images)
    metrics['lpips'] = -1. * np.ones(num_images)
    metrics['mae'] = -1. * np.ones(num_images)
    metrics['iou'] = -1. * np.ones(num_images)
    metrics['mask_mse'] = -1. * np.ones(num_images)
    mse_total = 0.
    psnr_total = 0.
    ssim_total = 0.
    lpips_total = 0.
    mae_total = 0.
    iou_total = 0.
    mask_mse_total = 0.
    lpips = LPIPS(net='vgg').to(device)
    set_requires_grad(lpips, requires_grad=False)

    # start=295
    # end=345
    for i in range(num_images):
        # 读入
        image = cv2.imread(osp.join(pred_folder_path, '_f%06d.jpg' % (2*i)))
        image = 2. * image / 255. - 1.
        # mask = cv2.imread(osp.join('/mnt/data/lcj/SCARF/exps/Antonia1/Antonia1/matting/', 'Antonia1_f000000.png'),
        #                   cv2.IMREAD_UNCHANGED)
        #
        # mask = mask[:, :, 3]
        # mask = mask / 255 > 0.5
        # import torchvision
        # torchvision.utils.save_image(torch.from_numpy(mask).float(), '/mnt/data/lcj/SCARF/a_' +str(i)  +'.png')

        # 计算要裁剪的中间部分的起始和结束位置
        start_x = 512
        end_x = start_x + 512


        # 裁剪图像
        gt_img = image[:, 0:start_x]
        pred_img = image[:, start_x:end_x]
        cv2.imwrite(osp.join("/mnt/data/lcj/SCARF/", 'gt_img.png'), (gt_img+1)*0.5 * 255)
        cv2.imwrite(osp.join("/mnt/data/lcj/SCARF/", 'pred_img.png'), (pred_img+1)*0.5 * 255)

        ## 计算metric
        # color
        mse_single = 10000 * compare_mse(pred_img, gt_img)
        psnr_single = compare_psnr(pred_img, gt_img)
        ssim_single = compare_ssim(pred_img, gt_img,data_range=1.0,channel_axis=-1)
        lpips_single = 1000*compute_lpips(lpips,torch.from_numpy(pred_img).view(512,512,3).float().to(device), torch.from_numpy(gt_img).view(512,512,3).float().to(device))

        torch.cuda.empty_cache()

        # 记录指标 #
        metrics['mse'][i] = mse_single
        metrics['psnr'][i] = psnr_single
        metrics['ssim'][i] = ssim_single
        metrics['lpips'][i] = lpips_single

        print("image", i*2, " color_mse:", mse_single, " color_psnr:", psnr_single, " color_ssim:", ssim_single,
              " color_lpips:", lpips_single)
        mse_total += mse_single
        psnr_total += psnr_single
        ssim_total += ssim_single
        lpips_total += lpips_single
        torch.cuda.empty_cache()


    # 分别写进一个txt文件 #
    with open(osp.join(folder_path, 'color_mse.txt'), 'w') as ff:
        ff.write('      mse\n')
        mse = metrics['mse']
        for ind, e in enumerate(mse.tolist()):
            if e >= 0.:
                ff.write('%6d: %.9f\n' % (ind, e))
        ff.write('mse mean: %.6f, max: %.6f, min: %.6f, maxinds:' % (mse.mean(), mse.max(), mse.min()))  # mse 越小越好，找出最大mse，也就是找有问题的几帧
        # 在默认情况下，argsort() 返回的是按升序排列的数组的索引，即最小值的索引排在最前面。
        # 因此，如果我们要找到最大值的索引，我们可以将待排序数组取负数，然后进行排序，这样就会将最大值变成最小值，再取前几个元素，就可以得到最大值的索引。
        for ind in (-mse).argsort()[:10]:
            ff.write('%d ' % ind)
    print('color mse done')
    with open(osp.join(folder_path, 'color_psnr.txt'), 'w') as ff:
        ff.write('      psnr\n')
        psnr = metrics['psnr']
        for ind, e in enumerate(psnr.tolist()):
            if e >= 0.:
                ff.write('%6d: %.9f\n' % (ind, e))
        ff.write('psnr mean: %.6f, max: %.6f, min: %.6f, mininds:' % (psnr.mean(), psnr.max(), psnr.min()))  # psnr 越高越好，找出最小psnr，也就是找有问题的几帧
        for ind in (psnr).argsort()[:10]:
            ff.write('%d ' % ind)
    print('color psnr done')
    with open(osp.join(folder_path, 'color_ssim.txt'), 'w') as ff:
        ff.write('      ssim\n')
        ssim = metrics['ssim']
        for ind, e in enumerate(ssim.tolist()):
            if e >= 0.:
                ff.write('%6d: %.9f\n' % (ind, e))
        ff.write('ssim mean: %.6f, max: %.6f, min: %.6f, mininds:' % (ssim.mean(), ssim.max(), ssim.min()))  # ssim 越高越好
        for ind in (ssim).argsort()[:10]:
            ff.write('%d ' % ind)
    print('color ssim done')
    with open(osp.join(folder_path, 'color_lpips.txt'), 'w') as ff:
        ff.write('      lpips\n')
        lpips = metrics['lpips']
        for ind, e in enumerate(lpips.tolist()):
            if e >= 0.:
                ff.write('%6d: %.9f\n' % (ind, e))
        ff.write('lpips mean: %.6f, max: %.6f, min: %.6f, maxinds:' % (lpips.mean(), lpips.max(), lpips.min())) # lpips 越小越好
        for ind in (-lpips).argsort()[:10]:
            ff.write('%d ' % ind)
    print('color lpips done')

    #num_temp=end-start

    mse_average=mse_total/num_images
    psnr_average = psnr_total / num_images
    ssim_average = ssim_total / num_images
    lpips_average = lpips_total / num_images

    print("mse_average:", mse_average,"psnr_average:", psnr_average, " ssim_average:", ssim_average, " lpips_average:", lpips_average)
    print("End!")

def compute_metrics_v2a():
    device = torch.device(0)
    folder_path="/mnt/data2/lcj/videoavatars-master/data/red/"
    gt_img_folder_path =os.path.join(folder_path, "imgs")
    gt_mask_folder_path = os.path.join(folder_path, "masks")   # mask:v2a的mask不太行 masks:通过rvm获得的gtmask
    gt_normal_folder_path = os.path.join(folder_path, "normals")    # pifu的normal感觉甚至没有v2a的好
    gt_image_files = glob.glob(os.path.join(gt_img_folder_path, "*.png"))
    num_images = len(gt_image_files)
    num_images=140
    print("Number of images in folder:", num_images)
    pred_folder_path="/mnt/data2/lcj/videoavatars-master/outputs/Video/red/"
    pred_img_folder_path = os.path.join(pred_folder_path, "test_fg_rendering")
    pred_mask_folder_path = os.path.join(pred_folder_path, "test_mask")
    pred_normal_folder_path = os.path.join(pred_folder_path, "test_normal")
    print("Compute metrics:")
    metrics = {}
    metrics['mse'] = -1. * np.ones(num_images)
    metrics['psnr'] = -1. * np.ones(num_images)
    metrics['ssim'] = -1. * np.ones(num_images)
    metrics['lpips'] = -1. * np.ones(num_images)
    metrics['mae'] = -1. * np.ones(num_images)
    metrics['iou'] = -1. * np.ones(num_images)
    metrics['mask_mse'] = -1. * np.ones(num_images)
    mse_total=0.
    psnr_total = 0.
    ssim_total = 0.
    lpips_total = 0.
    mae_total=0.
    iou_total = 0.
    mask_mse_total = 0.
    lpips = LPIPS(net='vgg').to(device)
    set_requires_grad(lpips, requires_grad=False)
    W=1284
    H=940
    for i in range(num_images):
        gt_img_path = osp.join(gt_img_folder_path, '%06d.png' % i)
        pred_img_path = osp.join(pred_img_folder_path, '%04d.png' % i)
        gt_mask_path = osp.join(gt_mask_folder_path, '%06d.png' % i)
        pred_mask_path = osp.join(pred_mask_folder_path, '%04d.png' % i)
        gt_normal_path = osp.join(gt_normal_folder_path, '%06d.png' % i)
        pred_normal_path = osp.join(pred_normal_folder_path, '%04d.png' % i)
        # 读取img 转换为浮点数 归一化到[0,1]
        gt_img_origin = cv2.imread(gt_img_path) / 255.
        pred_img_origin = cv2.imread(pred_img_path) / 255.
        # 读取mask
        gt_mask = cv2.imread(gt_mask_path)/ 255.
        pred_mask = cv2.imread(pred_mask_path) / 255.
        # 读取normal
        gt_normal_origin = cv2.imread(gt_normal_path)[:, :, ::-1]
        gt_normal_origin = 2. * gt_normal_origin.astype(np.float32) / 255. - 1.
        pred_normal_origin = cv2.imread(pred_normal_path)[:, :, ::-1]
        pred_normal_origin = 2. * pred_normal_origin.astype(np.float32) / 255. - 1.
        # 只取mask内的(如果是概率mask的话，这样相乘，边缘的颜色会有问题)
        # 所以一定要使用二值mask
        gt_img=gt_img_origin*gt_mask   # 和pred_mask相乘还是gt_mask相乘 v2a方法和gt_mask相乘
        pred_img=pred_img_origin*pred_mask
        gt_normal=gt_normal_origin*gt_mask
        pred_normal = pred_normal_origin * pred_mask
        ## 计算metric
        # color
        mse_single = 10000*compare_mse(pred_img, gt_img)
        psnr_single = compare_psnr(pred_img, gt_img)
        ssim_single = compare_ssim(pred_img, gt_img,data_range=1.0,channel_axis=-1)
        lpips_single = 1000*compute_lpips(lpips,torch.from_numpy(pred_img).view(H,W,3).float().to(device), torch.from_numpy(gt_img).view(H,W,3).float().to(device))
        # normal #
        mae_single = np.mean(np.abs(pred_normal - gt_normal))
        # mask #
        # pred_mask由pytorch的渲染器渲染mesh生成，因此mash不是0就是1,3个通道的值也是一样的
        # gtmask由他人算法得到，范围在[0~1]，大小为[1080,1920,3],但是3通道的数值都一样，因为可以直接取某一通道作为该像素点mask的值
        # iou
        iou_single = compute_iou(pred_mask[:, :, 0].reshape(H, W), gt_mask[:, :, 0].reshape(H, W))
        # mse #
        mask_mse_single = 10000 * compare_mse(pred_mask[:, :, 0].reshape(H, W), gt_mask[:, :, 0].reshape(H, W))

        # 记录指标 #
        metrics['mse'][i] = mse_single
        metrics['psnr'][i] = psnr_single
        metrics['ssim'][i] = ssim_single
        metrics['lpips'][i] = lpips_single
        metrics['mae'][i] = mae_single
        metrics['iou'][i] = iou_single
        metrics['mask_mse'][i] = mask_mse_single
        print("image", i, " color_mse:", mse_single," color_psnr:", psnr_single, " color_ssim:", ssim_single, " color_lpips:", lpips_single,
              " normal_mae:", mae_single," mask_iou:", iou_single," mask_mse:", mask_mse_single)
        mse_total+=mse_single
        psnr_total += psnr_single
        ssim_total += ssim_single
        lpips_total += lpips_single
        mae_total += mae_single
        iou_total += iou_single
        mask_mse_total += mask_mse_single
        torch.cuda.empty_cache()

    # 分别写进一个txt文件 #
    with open(osp.join(pred_folder_path, 'color_mse.txt'), 'w') as ff:
        ff.write('      mse\n')
        mse = metrics['mse']
        for ind, e in enumerate(mse.tolist()):
            if e >= 0.:
                ff.write('%6d: %.9f\n' % (ind, e))
        ff.write('mse mean: %.6f, max: %.6f, min: %.6f, maxinds:' % (mse.mean(), mse.max(), mse.min()))  # mse 越小越好，找出最大mse，也就是找有问题的几帧
        # 在默认情况下，argsort() 返回的是按升序排列的数组的索引，即最小值的索引排在最前面。
        # 因此，如果我们要找到最大值的索引，我们可以将待排序数组取负数，然后进行排序，这样就会将最大值变成最小值，再取前几个元素，就可以得到最大值的索引。
        for ind in (-mse).argsort()[:10]:
            ff.write('%d ' % ind)
    print('color mse done')
    with open(osp.join(pred_folder_path, 'color_psnr.txt'), 'w') as ff:
        ff.write('      psnr\n')
        psnr = metrics['psnr']
        for ind, e in enumerate(psnr.tolist()):
            if e >= 0.:
                ff.write('%6d: %.9f\n' % (ind, e))
        ff.write('psnr mean: %.6f, max: %.6f, min: %.6f, mininds:' % (psnr.mean(), psnr.max(), psnr.min()))  # psnr 越高越好，找出最小psnr，也就是找有问题的几帧
        for ind in (psnr).argsort()[:10]:
            ff.write('%d ' % ind)
    print('color psnr done')
    with open(osp.join(pred_folder_path, 'color_ssim.txt'), 'w') as ff:
        ff.write('      ssim\n')
        ssim = metrics['ssim']
        for ind, e in enumerate(ssim.tolist()):
            if e >= 0.:
                ff.write('%6d: %.9f\n' % (ind, e))
        ff.write('ssim mean: %.6f, max: %.6f, min: %.6f, mininds:' % (ssim.mean(), ssim.max(), ssim.min()))  # ssim 越高越好
        for ind in (ssim).argsort()[:10]:
            ff.write('%d ' % ind)
    print('color ssim done')
    with open(osp.join(pred_folder_path, 'color_lpips.txt'), 'w') as ff:
        ff.write('      lpips\n')
        lpips = metrics['lpips']
        for ind, e in enumerate(lpips.tolist()):
            if e >= 0.:
                ff.write('%6d: %.9f\n' % (ind, e))
        ff.write('lpips mean: %.6f, max: %.6f, min: %.6f, maxinds:' % (lpips.mean(), lpips.max(), lpips.min())) # lpips 越小越好
        for ind in (-lpips).argsort()[:10]:
            ff.write('%d ' % ind)
    print('color lpips done')

    with open(osp.join(pred_folder_path, 'normal_mae.txt'), 'w') as ff:
        ff.write('      mae\n')
        mae = metrics['mae']
        for ind, e in enumerate(mae.tolist()):
            if e >= 0.:
                ff.write('%6d: %.9f\n' % (ind, e))
        ff.write('mae mean: %.6f, max: %.6f, min: %.6f, maxinds:' % (mae.mean(), mae.max(), mae.min()))  # mae 越小越好，找出最大mae，也就是找有问题的几帧
        for ind in (-mae).argsort()[:10]:
            ff.write('%d ' % ind)
    print('normal mae done')

    with open(osp.join(pred_folder_path, 'mask_iou.txt'), 'w') as ff:
        ff.write('      iou\n')
        iou = metrics['iou']
        for ind, e in enumerate(iou.tolist()):
            if e >= 0.:
                ff.write('%6d: %.9f\n' % (ind, e))
        ff.write('iou mean: %.6f, max: %.6f, min: %.6f, mininds:' % (iou.mean(), iou.max(), iou.min()))  # iou 越高越好，找出最小iou，也就是找有问题的几帧
        for ind in (iou).argsort()[:10]:
            ff.write('%d ' % ind)
    print('mask iou done')

    with open(osp.join(pred_folder_path, 'mask_mse.txt'), 'w') as ff:
        ff.write('      mse\n')
        mse = metrics['mse']
        for ind, e in enumerate(mse.tolist()):
            if e >= 0.:
                ff.write('%6d: %.9f\n' % (ind, e))
        ff.write('mse mean: %.6f, max: %.6f, min: %.6f, maxinds:' % (mse.mean(), mse.max(), mse.min()))  # mse 越小越好
        for ind in (-mse).argsort()[:10]:
            ff.write('%d ' % ind)
    print('mask mse done')

    mse_average=mse_total/num_images
    psnr_average = psnr_total / num_images
    ssim_average = ssim_total / num_images
    lpips_average = lpips_total / num_images
    mae_average = mae_total / num_images
    iou_average = iou_total / num_images
    mask_mse_average = mask_mse_total / num_images
    print("mse_average:", mse_average,"psnr_average:", psnr_average, " ssim_average:", ssim_average, " lpips_average:", lpips_average,
          " mae_average:", mae_average," iou_average:", iou_average," mask_mse_average:", mask_mse_average)
    print("End!")

compute_metrics()