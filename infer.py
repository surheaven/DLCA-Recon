import pdb
import datetime
import torch
import numpy as np
from dataset.dataset import getDatasetAndLoader
from model import getOptNet
from pyhocon import ConfigFactory,HOCONConverter
import argparse
import openmesh as om
import os
import os.path as osp
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
# batch_size=config.get_int('train.coarse.batch_size')
batch_size=args.batch_size
shuffle=False
dataset,dataloader=getDatasetAndLoader(osp.normpath(osp.join(args.rec_root,osp.pardir)),{'deformer':deformer_condlen,'renderer':renderer_condlen},batch_size,
						shuffle,config.get_int('train.num_workers'),
						False,False,False)

optNet,sdf_initialized=getOptNet(dataset,batch_size,None,None,resolutions,device,config)

print('load model: '+osp.join(args.rec_root,'latest.pth'))
optNet,dataset,curepoch=utils.load_model(osp.join(args.rec_root,'30.pth'),optNet,dataset,device)   # 可以换几何不一样的checkpoint
optNet.dataset=dataset
optNet.eval()

pytorch_total_params = sum(p.numel() for p in optNet.parameters())

print('Total - ', pytorch_total_params)



raster_settings = RasterizationSettings(
	image_size=(dataset.H,dataset.W), 
	blur_radius=0, 
	faces_per_pixel=1,
	perspective_correct=True,
	clip_barycentric_coords=False,
	cull_backfaces=False
	)

optNet.maskRender.rasterizer.raster_settings=raster_settings
optNet.maskRender.shader=HardPhongShader(device,optNet.maskRender.rasterizer.cameras)
optNet.pcRender=None
H=dataset.H
W=dataset.W
if 'train.fine.point_render' in config:
	raster_settings_silhouette = PointsRasterizationSettings(
		image_size=(H,W), 
		radius=config.get_float('train.fine.point_render.radius'),
		# radius=0.002,
		bin_size=64,
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
TmpVs,Tmpfs=optNet.discretizeSDF(ratio,None,0.)

mesh = om.TriMesh(TmpVs.detach().cpu().numpy(), Tmpfs.cpu().numpy())
om.write_mesh(osp.join(args.rec_root,'tmp.ply'),mesh)

lbs_weight_root=osp.join(args.rec_root,'lbs_weight')
os.makedirs(osp.join(args.rec_root,'colors'),exist_ok=True)
os.makedirs(osp.join(args.rec_root,'meshs'),exist_ok=True)
os.makedirs(osp.join(args.rec_root,'def1meshs'),exist_ok=True)
os.makedirs(osp.join(args.rec_root, 'totaldeformation_model'), exist_ok=True)
os.makedirs(lbs_weight_root, exist_ok=True)
os.makedirs(osp.join(args.rec_root,'colors1'),exist_ok=True)
os.makedirs(osp.join(args.rec_root,'normals1'),exist_ok=True)
os.makedirs(osp.join(args.rec_root,'masks1'),exist_ok=True)


if not args.nV:
	writer_meshs=cv2.VideoWriter(osp.join(args.rec_root,'meshs/video.mp4'),cv2.VideoWriter.fourcc('m', 'p', '4', 'v'),30.,(W,H))
	writer_def1meshs=cv2.VideoWriter(osp.join(args.rec_root,'def1meshs/video.mp4'),cv2.VideoWriter.fourcc('m', 'p', '4', 'v'),30.,(W,H))
	writer_colors=None
	writer_pcmasks=None
errors={}
errors['maskE']=-1.*np.ones((len(dataset)))
gts={}
correct_pose = True
lbs_weight_finetune = True
allow_nonrigid_mlp = True
vis_ws = False   # 只有最初的deform为true
for data_index, (frame_ids, outs) in enumerate(dataloader):
	if data_index*batch_size > args.frames if args.frames>=0 else False:
		break
	imgs=outs['img']
	masks=outs['mask']
	if args.nColor:
		print(data_index*batch_size)
	else:
		print(data_index*batch_size,end='	')
	frame_ids=frame_ids.long().to(device)
	gts['mask']=masks.to(device)
	if args.C:
		gts['image']=(imgs.to(device)+1.)/2.
	print(datetime.datetime.now())
	colors,imgs,def1imgs,defVs,colors1,normals1,masks1=optNet.infer(TmpVs,Tmpfs,correct_pose,lbs_weight_finetune,allow_nonrigid_mlp,vis_ws,lbs_weight_root,dataset.H,dataset.W,dataset.frame_num,ratio,frame_ids,args.rec_root,args.nColor,gts)

	for fid,img,def1img,defV in zip(frame_ids.cpu().numpy().reshape(-1),imgs,def1imgs,defVs):
		np.save(osp.join(args.rec_root,'meshs/%d.npy'%fid),defV.reshape(-1,3))
		if not args.nV:
			writer_meshs.write(img[:,:,[2,1,0]])
			writer_def1meshs.write(def1img[:,:,[2,1,0]])
		if not args.nI:
			cv2.imwrite(osp.join(args.rec_root,'meshs/%d.png'%fid),img[:,:,[2,1,0]])
			cv2.imwrite(osp.join(args.rec_root,'def1meshs/%d.png'%fid),def1img[:,:,[2,1,0]])
	if colors is not None:
		os.makedirs(osp.join(args.rec_root,'colors'),exist_ok=True)
		if not args.nV and writer_colors is None:
			writer_colors=cv2.VideoWriter(osp.join(args.rec_root,'colors/video.mp4'),cv2.VideoWriter.fourcc('m', 'p', '4', 'v'),30.,(W,H))
		if not args.nI:			
			for fid,color,color1,normal1,mask1 in zip(frame_ids.cpu().numpy().reshape(-1),colors,colors1,normals1,masks1):
				writer_colors.write(color) if not args.nV else None
				cv2.imwrite(osp.join(args.rec_root,'colors/%d.png'%fid),color)
				cv2.imwrite(osp.join(args.rec_root, 'colors1/%d.png' % fid), color1)
				cv2.imwrite(osp.join(args.rec_root, 'normals1/%d.png' % fid), normal1)
				cv2.imwrite(osp.join(args.rec_root, 'masks1/%d.png' % fid), mask1)




	errors['maskE'][frame_ids.cpu().numpy()]=gts['maskE']

if not args.nV:
	writer_meshs.release()
	writer_def1meshs.release()
	if writer_colors:
		writer_colors.release()
	if writer_pcmasks:
		writer_pcmasks.release()

with open(osp.join(args.rec_root,'errors.txt'),'w') as ff:
	ff.write('      mask\n')
	maskE=errors['maskE']
	for ind,e in enumerate(maskE.tolist()):
		if e>=0.:
			ff.write('%4d: %.4f\n'%(ind,e))
	maskE=maskE[maskE>=0.]
	ff.write('mask mean: %.4f, max: %.4f, min: %.4f, maxinds:'%(maskE.mean(),maskE.max(),maskE.min()))
	for ind in (-maskE).argsort()[:10]:
		ff.write('%d '%ind)

print('done')

