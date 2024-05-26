import faulthandler
import pdb

faulthandler.enable()

import torch
import numpy as np
from dataset.dataset import getDatasetAndLoader
from model import getOptNet
import model.texture_generation as TexNet
from model.tex_network import neural_rendering_network
from pyhocon import ConfigFactory,HOCONConverter
import argparse
import openmesh as om
import os
import os.path as osp

import utils.obj_io as obj_io
from utils.general import Get_sharpen


from MCAcc import Seg3dLossless
import utils
import cv2
from tqdm import tqdm
from pytorch3d.renderer import (
    RasterizationSettings, 
    HardPhongShader,
    PointsRasterizationSettings,
	PointsRenderer,
	PointsRasterizer,
	AlphaCompositor
)

parser = argparse.ArgumentParser(description='neu video body infer')
parser.add_argument('--gpu-ids',nargs='+',type=int,metavar='IDs',
					help='gpu ids')
parser.add_argument('--batch-size',default=1,type=int,metavar='IDs',
					help='batch size')
parser.add_argument('--rec-root',default=None,metavar='M',
					help='data root')
parser.add_argument('--frames',default=-1,type=int,metavar='frames',
					help='render frame nums')
parser.add_argument('--nV',action='store_true',help='not save video')
parser.add_argument('--nI',action='store_true',help='not save image')
parser.add_argument('--C',action='store_true',help='overlay on gtimg')
parser.add_argument('--nColor',action='store_true',help='not render images')
args = parser.parse_args()

assert(not(args.nV and args.nI))

# resolutions = [
# 	(32+1, 32+1, 32+1),
# 	(64+1, 64+1, 64+1),
# 	(128+1, 128+1, 128+1),
# 	(256+1, 256+1, 256+1),
# 	(512+1, 512+1, 512+1),
# ]
resolutions = [
	(14+1, 20+1, 8+1),
	(28+1, 40+1, 16+1),
	(56+1, 80+1, 32+1),
	(112+1, 160+1, 64+1),
	(224+1, 320+1, 128+1),
]
# resolutions = [
# 	(18+1, 24+1, 12+1),
# 	(36+1, 48+1, 24+1),
# 	(72+1, 96+1, 48+1),
# 	(144+1, 192+1, 96+1),
# 	(288+1, 384+1, 192+1),
# ]

config=ConfigFactory.parse_file(osp.join(args.rec_root,'config.conf'))
device=args.gpu_ids[0]
deformer_condlen=config.get_int('mlp_deformer.condlen')
renderer_condlen=config.get_int('render_net.condlen')
#deformer_finetune_condlen=config.get_int('mlp_deformer_finetune.condlen')
# batch_size=config.get_int('train.coarse.batch_size')
batch_size=args.batch_size
shuffle=False
# 初始化dataset dataloader optNet sdf_initialized
dataset,dataloader=getDatasetAndLoader(osp.normpath(osp.join(args.rec_root,osp.pardir)),
									   {'deformer':deformer_condlen,'renderer':renderer_condlen},batch_size,
						shuffle,config.get_int('train.num_workers'),
						False,False,False)   # osp.pardir 获得上一级/父目录 将

optNet,sdf_initialized=getOptNet(dataset,batch_size,None,None,resolutions,device,config)

# 然后通过load latest model赋值进去
print('load model: '+osp.join(args.rec_root,'latest.pth'))
optNet,dataset,endepoch=utils.load_model(osp.join(args.rec_root,'latest.pth'),optNet,dataset,device)
optNet.dataset=dataset
optNet.eval()
# A renderer in PyTorch3D is composed of a rasterizer and a shader. Create a renderer in a few simple steps:
raster_settings = RasterizationSettings(
	image_size=(dataset.H,dataset.W),   # 要栅格化的输出图像的像素大小
	blur_radius=0,    # 设置模糊半径会使形状周围的边缘变得模糊，而不是硬边界。0代表不模糊
	faces_per_pixel=1,  # 每个像素要保存的面的数量
	perspective_correct=True,   # 如果使用的是透视相机，则应将此设置为True
	clip_barycentric_coords=False,  # 是否将面外的位置（即负的arycentric坐标）"校正 "到面的边缘位置。
	cull_backfaces=False   # 是否只对相机可见的网格面进行光栅化处理
	)
optNet.maskRender.rasterizer.raster_settings=raster_settings
optNet.maskRender.shader=HardPhongShader(device,optNet.maskRender.rasterizer.cameras)
optNet.pcRender=None
H=dataset.H   # 1080
W=dataset.W   # 1080
if 'train.fine.point_render' in config:  # into
	raster_settings_silhouette = PointsRasterizationSettings(
		image_size=(H,W),
		radius=config.get_float('train.fine.point_render.radius'),
		# radius=0.002,
		bin_size=64,   # 用于从粗到细的栅格化的栅格大小
		points_per_pixel=50,
		)   
	optNet.pcRender=PointsRenderer(
		rasterizer=PointsRasterizer(
			cameras=optNet.maskRender.rasterizer.cameras, 
			raster_settings=raster_settings_silhouette
		),
			compositor=AlphaCompositor(background_color=(1,1,1,1))
		).to(device)



ratio={'sdfRatio':1.,'deformerRatio':1.,'renderRatio':1.}
TmpVs,Tmpfs=optNet.discretizeSDF(ratio,None)   # (76164, 3)  (152324, 3)


mesh = om.TriMesh(TmpVs.detach().cpu().numpy(), Tmpfs.cpu().numpy())
om.write_mesh(osp.join(args.rec_root,'tmp.ply'),mesh)





errors={}
errors['maskE']=-1.*np.ones((len(dataset)))
gts={}


avatar_normal=None
avatar_color=None
not_color = None


### model: mesh color

neural_texture_model =TexNet.getTexNet(device, config.get_config('tex_net'))



learnable_ws=dataset.learnable_weights()   # 这是dataset中ewai的参数，e.g. pose、trans
optimizer = torch.optim.Adam([{'params':learnable_ws},{'params':[p for p in neural_texture_model.parameters() if p.requires_grad]}], lr=config.get_float('train.learning_rate'))
# for parameter in neural_texture_model.parameters():#打印出参数矩阵及值
# 	if parameter.requires_grad!=True:
# 		print(parameter)

tex_model=osp.join(args.rec_root,"neural_texture_model.pth")
curepoch=0
if tex_model is not None and osp.isfile(tex_model):   # not into  none
	print('load model: '+tex_model,end='')
	neural_texture_model, optimizer,curepoch,loss = utils.load_model_tex(tex_model,neural_texture_model,optimizer,device)

neural_texture_model.train()


mesh_infer_root=osp.join(args.rec_root, "mesh_infer")
mesh_finetune_root=osp.join(args.rec_root, "mesh_finetune")
mesh_optimize_root=osp.join(args.rec_root, "mesh_optimize_root")
os.makedirs(mesh_infer_root, exist_ok=True)
os.makedirs(mesh_finetune_root, exist_ok=True)
os.makedirs(mesh_optimize_root, exist_ok=True)
correct_pose = True
lbs_weight_finetune= True
allow_nonrigid_mlp= True
vis_ws= False
lbs_weight_root = None


###### 感觉接下来的3步效果都一样啊，可以整合到一个代码里，用不同开关开启 ######
###### 1. get mesh color ######
###### (x,n)->color ######
for infer_epoch in range(curepoch,70):
	not_color = None
	print("===>>>infer mesh color epoch:",infer_epoch)
	for data_index, (frame_ids, outs) in enumerate(dataloader):
		if data_index * batch_size > args.frames if args.frames >= 0 else False:
			break
		imgs = outs['img']
		masks = outs['mask']
		if args.nColor:
			print(data_index * batch_size)
		else:  # false
			print(data_index * batch_size, end='	')
		frame_ids = frame_ids.long().to(device)
		gts['mask'] = masks.to(device)
		gts['image']=imgs.to(device)
		if args.C:  # true  dataset中的设置((image)/255.-0.5)*2
			gts['write_image'] = (imgs.to(device) + 1.) / 2.   # 这一步还原到cv2.read

		gts['normal'] = outs['normal'].to(device)

		# colors:
		# imgs:  完全变形后的mesh
		# def1imgs: only 非刚性变形后的mesh
		# defVs:
		optimizer.zero_grad()
		avatar_color, loss = optNet.infer_mesh_color(neural_texture_model, TmpVs, Tmpfs, correct_pose,
																	lbs_weight_finetune, allow_nonrigid_mlp, vis_ws,
																	lbs_weight_root,
																	avatar_color,
																	dataset.H, dataset.W,dataset.frame_num,
																	ratio, frame_ids, mesh_infer_root,
																	args.nColor, gts)
		# obj_io.save_mesh_as_ply(osp.join(mesh_infer_root, str(frame_ids.item()) + '.ply'),
		# 						TmpVs.detach().cpu().numpy(), Tmpfs.detach().cpu().numpy(),
		# 						None,
		# 						avatar_color.detach().cpu().numpy())
		loss.backward()
		optimizer.step()

	obj_io.save_mesh_as_ply(osp.join(mesh_infer_root, "epoch_"+str(infer_epoch)+'.ply'),
							TmpVs.detach().cpu().numpy(), Tmpfs.detach().cpu().numpy(), None,
							avatar_color.detach().cpu().numpy())
	utils.save_model_tex(osp.join(args.rec_root, "neural_texture_model.pth"),
						 infer_epoch, neural_texture_model,optimizer,loss)




## 通过TmpVs avatar_normal 获得所有TmpVs的color
avatar_color=optNet.get_all_mesh_color(neural_texture_model, TmpVs, Tmpfs, args.rec_root,ratio)
avatar_color=torch.from_numpy(avatar_color).to(device)


### 2. (meshcolor,image color)->encoder->decoder->(mesh_color) 迭代处理 #33
not_color = None
feature_num=3
neural_rendering_model_train = neural_rendering_network(W, H, feature_num,norm='instance')
neural_rendering_model_train.to(device)
neural_rendering_model_train.train()
optimizer = torch.optim.Adam([{'params':[p for p in neural_rendering_model_train.parameters() if p.requires_grad]}], lr=config.get_float('train.learning_rate'))
# for parameter in neural_texture_model.parameters():#打印出参数矩阵及值

for finetune_epoch in range(0,70):
	print("===>>>fintune mesh color epoch:",finetune_epoch)
	for data_index, (frame_ids, outs) in enumerate(dataloader):
		if data_index * batch_size > args.frames if args.frames >= 0 else False:
			break

		imgs = outs['img']
		masks = outs['mask']
		if args.nColor:
			print(data_index * batch_size)
		else:  # false
			print(data_index * batch_size, end='	')
		frame_ids = frame_ids.long().to(device)
		gts['mask'] = masks.to(device)
		gts['image'] = imgs.to(device)


		gts['img_highres'] = (outs['img_highres'].to(device) + 1.) / 2.
		gts['mask_highres'] = outs['mask_highres'].to(device)
		gts['normal'] = outs['normal'].to(device)
		gts['seg'] = outs['seg'].to(device)


		optimizer.zero_grad()

		avatar_normal, avatar_color, not_color, loss = optNet.finetune_mesh_color(neural_texture_model, neural_rendering_model_train,TmpVs, Tmpfs, correct_pose,
																	lbs_weight_finetune, allow_nonrigid_mlp, vis_ws,
																	lbs_weight_root,
																	avatar_normal, avatar_color,not_color,
																	dataset.H, dataset.W,dataset.frame_num,
																	ratio, frame_ids, mesh_infer_root,
																	infer_epoch, args.nColor, gts)
		loss.backward()
		optimizer.step()



### 3. texture optimize ###
get_sharpen = Get_sharpen(kernel_name='n')
for infer_epoch in range(0,100):
	not_color = None
	print("===>>>optimize mesh color epoch:",infer_epoch)
	for data_index, (frame_ids, outs) in enumerate(dataloader):
		if data_index * batch_size > args.frames if args.frames >= 0 else False:
			break
		imgs = outs['img']
		masks = outs['mask']
		if args.nColor:
			print(data_index * batch_size)
		else:  # false
			print(data_index * batch_size, end='	')
		frame_ids = frame_ids.long().to(device)
		gts['mask'] = masks.to(device)
		gts['image']=imgs.to(device)
		if args.C:  # true  dataset中的设置((image)/255.-0.5)*2
			gts['write_image'] = (imgs.to(device) + 1.) / 2.   # 这一步还原到cv2.read


		gts['normal'] = outs['normal'].to(device)


		# colors:
		# imgs:  完全变形后的mesh
		# def1imgs: only 非刚性变形后的mesh
		# defVs:
		correct_pose = True
		optimizer.zero_grad()
		avatar_normal, avatar_color, loss = optNet.optimize_mesh_color(neural_texture_model, TmpVs, Tmpfs,correct_pose,lbs_weight_finetune, allow_nonrigid_mlp, vis_ws,lbs_weight_root,
																			   avatar_normal, avatar_color,
																			  dataset.H, dataset.W,dataset.frame_num,
																			   ratio, frame_ids, get_sharpen,mesh_optimize_root,
																			   infer_epoch, args.nColor,gts)
		obj_io.save_mesh_as_ply(osp.join(mesh_optimize_root, str(frame_ids.item()) + '.ply'),
								TmpVs.detach().cpu().numpy(), Tmpfs.detach().cpu().numpy(),
								avatar_normal.detach().cpu().numpy(),
								avatar_color.detach().cpu().numpy())
		loss.backward()
		optimizer.step()

	obj_io.save_mesh_as_ply(osp.join(mesh_optimize_root, "epoch_"+str(infer_epoch)+'.ply'),
							TmpVs.detach().cpu().numpy(), Tmpfs.detach().cpu().numpy(), avatar_normal.detach().cpu().numpy(),
							avatar_color.detach().cpu().numpy())
	utils.save_model_tex(osp.join(args.rec_root, "neural_texture_model.pth"),
						 infer_epoch, neural_texture_model,optimizer,loss)


###### 或者可以直接把当前帧warp上去（当然只有正面信息） ######
####### 以下方法时直接使用warp ######
# not_color = None
# # warp 正反两面细节
# for data_index, (frame_ids, outs) in enumerate(dataloader):
# 	if frame_ids.item()!=0 and frame_ids.item()!=100:
# 		continue
# 	if data_index*batch_size > args.frames if args.frames>=0 else False:
# 		break
# 	imgs=outs['img']
# 	masks=outs['mask']
# 	if args.nColor:
# 		print(data_index*batch_size)
# 	else:   # false
# 		print(data_index*batch_size,end='	')
# 	frame_ids=frame_ids.long().to(device)
# 	gts['mask']=masks.to(device)
# 	if args.C:   # true
# 		gts['image']=(imgs.to(device)+1.)/2.
#
# 	gts['img_highres']=(outs['img_highres'].to(device)+1.)/2.
# 	gts['mask_highres'] = outs['mask_highres'].to(device)
# 	gts['normal'] = outs['normal'].to(device)
# 	gts['seg'] = outs['seg'].to(device)
#
# 	correct_pose = True
# 	avatar_normal,avatar_color,not_color=optNet.finetune(TmpVs,Tmpfs,avatar_normal,avatar_color,not_color,
# 														 correct_pose,dataset.H,dataset.W,ratio,frame_ids,args.rec_root,endepoch,args.nColor,gts)
#
#
# # warp 剩余细节
# for data_index, (frame_ids, outs) in enumerate(dataloader):
# 	if frame_ids.item()%10!=0:
# 		continue
# 	if data_index*batch_size > args.frames if args.frames>=0 else False:
# 		break
# 	imgs=outs['img']
# 	masks=outs['mask']
# 	if args.nColor:
# 		print(data_index*batch_size)
# 	else:   # false
# 		print(data_index*batch_size,end='	')
# 	frame_ids=frame_ids.long().to(device)
# 	gts['mask']=masks.to(device)
# 	if args.C:   # true
# 		gts['image']=(imgs.to(device)+1.)/2.
#
# 	gts['img_highres']=(outs['img_highres'].to(device)+1.)/2.
# 	gts['mask_highres'] = outs['mask_highres'].to(device)
# 	gts['normal'] = outs['normal'].to(device)
# 	gts['seg'] = outs['seg'].to(device)
#
# 	# colors:
# 	# imgs:  完全变形后的mesh
# 	# def1imgs: only 非刚性变形后的mesh
# 	# defVs:
# 	print("===>>>before finetune：")
# 	correct_pose = True
# 	avatar_normal,avatar_color,not_color=optNet.finetune(TmpVs,Tmpfs,avatar_normal,avatar_color,not_color,
# 														 correct_pose,dataset.H,dataset.W,ratio,frame_ids,args.rec_root,endepoch,args.nColor,gts)
#




