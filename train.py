import pdb

import torch
import numpy as np
from dataset.dataset import getDatasetAndLoader
from model import getOptNet
from pyhocon import ConfigFactory,HOCONConverter
import argparse
import trimesh
import cv2
import os
import os.path as osp
from MCAcc import Seg3dLossless
import utils
import model.Discriminator as Discriminator

parser = argparse.ArgumentParser(description='neu video body rec')
parser.add_argument('--gpu-ids',nargs='+',type=int,metavar='IDs',
					help='gpu ids')
parser.add_argument('--conf',default=None,metavar='M',
					help='config file')
parser.add_argument('--data',default=None,metavar='M',
					help='data root')
parser.add_argument('--model',default=None,metavar='M',
					help='pretrained scene model')
parser.add_argument('--model-rm-prefix',nargs='+',type=str,metavar='rm prefix', help='rm model prefix')
parser.add_argument('--sdf-model',default=None,metavar='M',
					help='substitute sdf model')
parser.add_argument('--save-folder',default=None,metavar='M',help='save folder')
args = parser.parse_args()


#point render
resolutions={'coarse':
[
	(14+1, 20+1, 8+1),
	(28+1, 40+1, 16+1),
	(56+1, 80+1, 32+1),
	(112+1, 160+1, 64+1),
	(224+1, 320+1, 128+1),
],
'medium':
[
	(18+1, 24+1, 12+1),
	(36+1, 48+1, 24+1),
	(72+1, 96+1, 48+1),
	(144+1, 192+1, 96+1),
	(288+1, 384+1, 192+1),
],
'fine':
[
	(20+1, 26+1, 14+1),
	(40+1, 52+1, 28+1),
	(80+1, 104+1, 56+1),
	(160+1, 208+1, 112+1),
	(320+1, 416+1, 224+1),
]
}

resolutions_higher = [
	(32+1, 32+1, 32+1),
	(64+1, 64+1, 64+1),
	(128+1, 128+1, 128+1),
	(256+1, 256+1, 256+1),
	(512+1, 512+1, 512+1),
]



config=ConfigFactory.parse_file(args.conf)
if len(args.gpu_ids):
	device=torch.device(args.gpu_ids[0])
else:
	device=torch.device(0)
data_root=args.data
if args.save_folder is None:
	print('please set save-folder...')
	assert(False)

global lbs_weight_root
save_root=osp.join(data_root,args.save_folder)
debug_root=osp.join(save_root,'debug')
front_mesh_root=osp.join(debug_root,'front_mesh_img')
back_mesh_root=osp.join(debug_root,'back_mesh_img')
side_mesh_root=osp.join(debug_root,'side_mesh_img')
lbs_weight_root_ori=osp.join(debug_root,'lbs_weight')
os.makedirs(save_root,exist_ok=True)
os.makedirs(debug_root,exist_ok=True)
os.makedirs(front_mesh_root,exist_ok=True)
os.makedirs(back_mesh_root,exist_ok=True)
os.makedirs(side_mesh_root,exist_ok=True)
os.makedirs(lbs_weight_root_ori,exist_ok=True)


# save the config file
with open(osp.join(save_root,'config.conf'),'w') as ff:
	ff.write(HOCONConverter.convert(config,'hocon'))
condlen={'deformer':config.get_int('mlp_deformer.condlen'),'renderer':config.get_int('render_net.condlen')}
batch_size=config.get_int('train.coarse.point_render.batch_size')
dataset,dataloader=getDatasetAndLoader(data_root,condlen,batch_size,
						config.get_bool('train.shuffle'),config.get_int('train.num_workers'),
						config.get_bool('train.opt_pose'),config.get_bool('train.opt_trans'),config.get_config('train.opt_camera'))

# bmins=[-0.8,-1.25,-0.4]
# bmaxs=[0.8,0.7,0.4]
# use adaptive box computation
bmins=None
bmaxs=None

if config.get_int('train.initial_iters')<=0:
	use_initial_sdf=True
else:
	use_initial_sdf=False
optNet,sdf_initialized=getOptNet(dataset,batch_size,bmins,bmaxs,resolutions['coarse'],device,config,use_initial_sdf)
optNet,dataloader=utils.set_hierarchical_config(config,'coarse',optNet,dataloader,resolutions['coarse'])

# 输入命令行时，如果有 --model和--sdf-model，直接load_model
# 使用pre-trained model 进行学习
curepoch=0
result_file=osp.join(args.data,args.save_folder)
args.model=osp.join(result_file,'latest.pth')
args.sdf_model=osp.join(result_file,'latest_sdf_idr' + '_%d_%d.ply' % (
config.get_int('sdf_net.multires'), config.get_int('train.skinner_pose_id')))

if args.model is not None and osp.isfile(args.model):
	print('load model: '+args.model,end='')
	if args.sdf_model is not None and osp.isfile(args.sdf_model):
		print(' and substitute sdf model with: '+args.sdf_model,end='')
		sdf_initialized=-1

	print()
	optNet,dataset,curepoch=utils.load_model(args.model,optNet,dataset,device,args.sdf_model,args.model_rm_prefix)
	print("continue training at epoch ",curepoch)

# 在 SMPL 模型中，初始姿势下的顶点位置是以模型的中心为原点的相对坐标系。
# 具体来说，x 轴是从模型的中心指向右侧，y 轴是从模型的中心指向上方，z 轴是从模型的中心指向前方。
# 根据初始化smplmesh的verts的min和max，来确定bbox的bmin和bmax
# smpl好像整体比例都差不多，根据shape定义
print('box:')
print(optNet.engine.b_min.view(-1).tolist())
print(optNet.engine.b_max.view(-1).tolist())
optNet.train()
###### optNet是generation ###### 上面

###### 新增加 Discriminator ######
discriminator = Discriminator.getDiscriNet(device, config.get_config('loss_coarse'))
discriminator.train()


# 如果之前已经重建出来一个人体的话。sdf_initialized被设置为-1，这里面就不进去了
# out:initial_sdf_idr_6_poseid.pth
print("sdf_pth ",osp.join(data_root,'initial_sdf_idr'+'_%d_%d.pth'%(config.get_int('sdf_net.multires'),config.get_int('train.skinner_pose_id'))))
if sdf_initialized>0:  # sdf_initialized=1200 代表sdf拟合的epoch数量
	optNet.initializeTmpSDF(sdf_initialized,osp.join(data_root,'initial_sdf_idr'+'_%d_%d.pth'%(config.get_int('sdf_net.multires'),config.get_int('train.skinner_pose_id'))),True)
	engine = Seg3dLossless(
			query_func=None, 
			b_min = optNet.engine.b_min,
			b_max = optNet.engine.b_max,
			resolutions=resolutions['coarse'],
			align_corners=False,
			balance_value=0.0, # be careful
			visualize=False,
			debug=False,
			use_cuda_impl=False,
			faster=False 
		).to(device)
	verts,faces=optNet.discretizeSDF(-1,engine)
	mesh = trimesh.Trimesh(verts.cpu().numpy(), faces.cpu().numpy())

	mesh.export(osp.join(data_root,'initial_sdf_idr'+'_%d_%d.ply'%(config.get_int('sdf_net.multires'),config.get_int('train.skinner_pose_id'))))
else:  # load latest model get t_middle_sdf_idr ply
	#optNet.initializeTmpSDF(1200, osp.join(data_root, 't_middle_sdf_idr' + '_%d_%d.pth' % (config.get_int('sdf_net.multires'), config.get_int('train.skinner_pose_type'))), True)
	engine = Seg3dLossless(
		query_func=None,
		b_min=optNet.engine.b_min,
		b_max=optNet.engine.b_max,
		resolutions=resolutions['coarse'],  # only改变resolutions 感觉没有啥变化
		align_corners=False,
		balance_value=0.0,  # be careful
		visualize=False,
		debug=False,
		use_cuda_impl=False,
		faster=False
	).to(device)
	# 离散化sdf
	ratio = {'sdfRatio': 1., 'deformerRatio': 1., 'renderRatio': 1.}
	verts, faces = optNet.discretizeSDF(ratio, engine)  # 这里面都是三角面，都是每个面的顶点
	#print("===>>>after sdf network,update mesh")
	# print("verts", verts.shape, verts)
	# print("faces", faces.shape, faces)
	mesh = trimesh.Trimesh(verts.cpu().numpy(), faces.cpu().numpy())  # 根据update的verts,faces生成mesh
	mesh.export(osp.join(data_root, 'latest_sdf_idr' + '_%d_%d.ply' % (config.get_int('sdf_net.multires'), config.get_int('train.skinner_pose_id'))))
	#print("===>>>get human model with cloth in epoch",curepoch)

learnable_ws=dataset.learnable_weights()


###### generation和discrimination的optimizer（更新参数的优化器）和scheduler（调整学习率的）
optimizer = torch.optim.Adam([{'params':learnable_ws},{'params':[p for p in optNet.parameters() if p.requires_grad]}], lr=config.get_float('train.learning_rate'))
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.get_list('train.scheduler.milestones'), gamma=config.get_float('train.scheduler.factor'))

optimizer_d = torch.optim.Adam([{'params':[p for p in discriminator.parameters() if p.requires_grad]}], lr=config.get_float('train.learning_rate'))
scheduler_d = torch.optim.lr_scheduler.MultiStepLR(optimizer_d, config.get_list('train.scheduler.milestones'), gamma=config.get_float('train.scheduler.factor'))


ratio={'sdfRatio':None,'deformerRatio':None,'renderRatio':None}
opt_times=0.
nepochs=config.get_int('train.nepoch')
sample_pix_num=config.get_int('train.sample_pix_num')
in_fine_hie=False

if config.get_int('train.medium.start_epoch') >= 0 and curepoch >= config.get_int('train.medium.start_epoch'):
	optNet, dataloader = utils.set_hierarchical_config(config, 'medium', optNet, dataloader, resolutions['medium'])
	print('enable medium hierarchical')
if config.get_int('train.fine.start_epoch') >= 0 and curepoch >= config.get_int('train.fine.start_epoch'):
	optNet, dataloader = utils.set_hierarchical_config(config, 'fine', optNet, dataloader, resolutions['fine'])
	print('enable fine hierarchical')
	in_fine_hie = True


def compute_bce(d_out, target):
	targets = d_out.new_full(size=d_out.size(), fill_value=target)
	loss = torch.nn.functional.binary_cross_entropy_with_logits(d_out, targets)
	return loss

def compute_grad2(d_out, x_in):
	batch_size = x_in.size(0)
	grad_dout = torch.autograd.grad(
		outputs=d_out.sum(), inputs=x_in,
		create_graph=True, retain_graph=True, only_inputs=True
	)[0]
	grad_dout2 = grad_dout.pow(2)
	assert (grad_dout2.size() == x_in.size())
	reg = grad_dout2.reshape(batch_size, -1).sum(1)
	return reg

def train_discriminator(discriminator,x_real,x_fake):
	loss_d_full = 0.
	x_real.requires_grad_()
	d_real = discriminator(x_real)  # [N_patch, 3, patch_size, patch_size]-->[N_patch,1]
	d_loss_real = compute_bce(d_real, 1)
	loss_d_full += d_loss_real
	reg = 10. * compute_grad2(d_real, x_real).mean()
	loss_d_full += reg
	x_fake.requires_grad_()
	d_fake = discriminator(x_fake)
	d_loss_fake = compute_bce(d_fake, 0)
	loss_d_full += d_loss_fake

	return loss_d_full


frame_num=dataset.frame_num
for epoch in range(curepoch,nepochs+1):
	lbs_weight_root=osp.join(lbs_weight_root_ori,str(epoch))
	os.makedirs(lbs_weight_root, exist_ok=True)
	if config.get_int('train.medium.start_epoch')>=0 and epoch==config.get_int('train.medium.start_epoch'):
		optNet,dataloader=utils.set_hierarchical_config(config,'medium',optNet,dataloader,resolutions['medium'])
		torch.cuda.empty_cache()
		print('enable medium hierarchical')
		utils.save_model(osp.join(save_root,"coarse.pth"),epoch,optNet,dataset)
	if config.get_int('train.fine.start_epoch')>=0 and epoch==config.get_int('train.fine.start_epoch'):
		optNet,dataloader=utils.set_hierarchical_config(config,'fine',optNet,dataloader,resolutions['fine'])
		print('enable fine hierarchical')
		torch.cuda.empty_cache()
		utils.save_model(osp.join(save_root,"medium.pth"),epoch,optNet,dataset)
		in_fine_hie=True

	###### 不同属性的衣物开启不一样 ######
	###### 从什么时候开始correct_pose呢？ ######
	correct_pose = False
	if epoch > 10:
		correct_pose = True
	######
	###### 从什么时候开始lbs_weight_finetune呢？ ######
	# 这个是不是只适用于宽松衣物
	# 消融实验：delayed_optimize
	lbs_weight_finetune = False
	if epoch > 20:
		lbs_weight_finetune = True
	######
	###### 从什么时候开始non_rigid_deformation(mlp)呢？ ###### 这个一直开着的
	allow_nonrigid_mlp = True
	if epoch > 20:
		allow_nonrigid_mlp = True
	######
	vis_ws = False

	###### 从什么时候开始gan结构的呢 ######
	# 暂且先不开，当generation有一定生成能力再开
	# 还是刚开始就一直开，一直对抗学习
	allow_d=False
	if epoch > 30:
		allow_d = False

	# for data_index, (frame_ids, imgs, masks, albedos) in enumerate(dataloader):
	for data_index, (frame_ids, outs) in enumerate(dataloader):
		# if frame_ids!=0:
		# 	continue
		#frame_list=list(range(20))+[180,222,252,276]
		# frame_list = list(range(20))
		# if frame_ids not in frame_list:
		# 	continue

		print('only run at frame ', frame_ids.item())
		frame_ids=frame_ids.long().to(device)
		optimizer.zero_grad()
		if allow_d:
			optimizer_d.zero_grad()

		ratio['sdfRatio']=1.
		ratio['deformerRatio']=opt_times/2500.+0.5
		ratio['renderRatio']=1.
		# 训练generation，里面叠加gan loss？？？
		loss,x_real,x_fake = optNet(epoch, correct_pose,lbs_weight_finetune,allow_nonrigid_mlp, allow_d, vis_ws,lbs_weight_root,outs, frame_num,
					  sample_pix_num, ratio, frame_ids, data_index, debug_root)
		# 训练discrimination
		loss_d=None
		loss_g=None
		if allow_d:
			# 固定G，训练D，所以x_fake.detach()
			# RuntimeError: set_sizes_and_strides is not allowed on a Tensor created from .data or .detach()
			# 解决方法：contiguous(深层拷贝变量，切断变量之间的联系)或者torch.no_grad(在这个函数里面重新用generator生成x_fake)
			# https://discuss.pytorch.org/t/runtimeerror-set-sizes-and-strides-is-not-allowed-on-a-tensor-created-from-data-or-detach/116910
			loss_d= train_discriminator(discriminator, x_real,x_fake.detach().contiguous())  #
			loss_d.backward()
			optimizer_d.step()
			# 固定D，计算G的gan_loss
			d_fake = discriminator(x_fake)
			loss_g = compute_bce(d_fake, 1)
			loss += loss_g*0.01   # 降低loss_g的权重



		# generation_loss 更新
		loss.backward()
		optNet.propagateTmpPsGrad(correct_pose, lbs_weight_finetune,allow_nonrigid_mlp,vis_ws,lbs_weight_root,frame_ids, frame_num, ratio)
		optimizer.step()   # 更新整个网络的参数 （optimizer只会优化放在里面的参数）
		# discrimation_loss 更新


		# data_index（这个指写入数据的顺序）和frame_id（这个就是数据的名字）不一定一样
		if data_index%1==0:
			outinfo='(%d/%d): loss = %.5f; color_loss: %.5f, eikonal_loss: %.5f'%(epoch,data_index,loss.item(),optNet.info['color_loss'],optNet.info['grad_loss'])+ \
					(' normal_loss: %.5f,'%optNet.info['normal_loss'] if 'normal_loss' in optNet.info else '')+ \
					(' def_loss: %.5f,'%optNet.info['def_loss'] if 'def_loss' in optNet.info else '')+ \
					(' offset_loss: %.5f,'%optNet.info['offset_loss'] if 'offset_loss' in optNet.info else '')+ \
					(' dct_loss: %.5f,'%optNet.info['dct_loss'] if 'dct_loss' in optNet.info else '')+ \
					(' lpips_loss: %.5f,'%optNet.info['lpips_loss'] if 'lpips_loss' in optNet.info else '')+ \
					(' color_loss_patch: %.5f,'%optNet.info['color_loss_patch'] if 'color_loss_patch' in optNet.info else '')+ \
					(' mrf_loss: %.5f,'%optNet.info['mrf_loss'] if 'mrf_loss' in optNet.info else '')+ \
					(' discriminator_loss: %.5f,'%loss_d if loss_d is not None else '')+ \
					(' generator_loss: %.5f,'%loss_g if loss_g is not None else '')

			outinfo+='\n'
			outinfo+='\tpc_sdf_l: %.5f'%(optNet.info['pc_loss_sdf'])
			outinfo+=';\tpc_norm_l: %.5f; '%(optNet.info['pc_loss_norm']) if 'pc_loss_norm' in optNet.info else '; '
			for k,v in optNet.info['pc_loss'].items():
				outinfo+=k+': %.5f\t'%v
			outinfo+='\n\trayInfo(%d,%d)\tinvInfo(%d,%d)\tratio: (%.2f,%.2f,%.2f)\tremesh: %.3f'%(*optNet.info['rayInfo'],*optNet.info['invInfo'],ratio['sdfRatio'],ratio['deformerRatio'],ratio['renderRatio'],optNet.info['remesh'])
			print(outinfo)

			###### 将loss保存到数组里面 ######



		opt_times+=1.

	###### 每过一段时间记录loss ######


	###### 每个epoch 每个step tmp渲染结果 ######
	### 内存不够 关闭 ###
	# front_img_files = sorted([f for f in os.listdir(front_mesh_root) if f.endswith('.jpg') or f.endswith('.png')])
	# back_img_files = sorted([f for f in os.listdir(back_mesh_root) if f.endswith('.jpg') or f.endswith('.png')])
	# side_img_files = sorted([f for f in os.listdir(side_mesh_root) if f.endswith('.jpg') or f.endswith('.png')])
	# img_num=len(front_img_files)
	# writer_front_mesh = cv2.VideoWriter(osp.join(front_mesh_root, 'video_%d.mp4' % epoch),
	# 									cv2.VideoWriter.fourcc('m', 'p', '4', 'v'), 30., (dataset.W, dataset.H))
	# writer_back_mesh = cv2.VideoWriter(osp.join(back_mesh_root, 'video%d.mp4' % epoch),
	# 								   cv2.VideoWriter.fourcc('m', 'p', '4', 'v'), 30., (dataset.W, dataset.H))
	# writer_side_mesh = cv2.VideoWriter(osp.join(side_mesh_root, 'video%d.mp4' % epoch),
	# 								   cv2.VideoWriter.fourcc('m', 'p', '4', 'v'), 30., (dataset.W, dataset.H))
	# for i in range(img_num):
	# 	# 遍历所有图像文件并写入视频
	# 	front_img_path = os.path.join(front_mesh_root, '%d.png' % i)
	# 	back_img_path = os.path.join(back_mesh_root, '%d.png' % i)
	# 	side_img_path = os.path.join(side_mesh_root, '%d.png' % i)
	# 	front_img = cv2.imread(front_img_path)
	# 	back_img = cv2.imread(back_img_path)
	# 	side_img = cv2.imread(side_img_path)
	# 	writer_front_mesh.write(front_img[:, :, [2, 1, 0]])
	# 	writer_back_mesh.write(back_img[:, :, [2, 1, 0]])
	# 	writer_side_mesh.write(side_img[:, :, [2, 1, 0]])
	#
	# writer_front_mesh.release()
	# writer_back_mesh.release()
	# writer_side_mesh.release()
	### 内存不够，关闭 ###
	### 每个epoch 每个step tmp渲染结果 ###

	if in_fine_hie:  # 只在epoch>=12时候才会绘制rgb+normal，但每一帧都会计算color和normal loss
		optNet.draw=True
	if epoch%30==0:
		utils.save_model(osp.join(save_root, "%d.pth"%epoch), epoch, optNet, dataset)
	utils.save_model(osp.join(save_root,"latest.pth"),epoch,optNet,dataset)
	utils.save_sdf_model(osp.join(save_root,'latest_sdf_idr' + '_%d_%d.ply' % (config.get_int('sdf_net.multires'), config.get_int('train.skinner_pose_id'))), epoch, optNet.sdf)
	if allow_d:
		scheduler_d.step()
	scheduler.step()   # 调整优化器的学习参数
