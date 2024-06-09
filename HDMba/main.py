# load Network
from models.AACNet import AACNet
from models.AIDTransformer import AIDTransformer
from models.Dehazeformer import DehazeFormer
from models.HDMba import HDMba

import time, math
import numpy as np
from torch.backends import cudnn
from torch import optim
import torch, warnings
from torch import nn
# from tensorboardX import SummaryWriter
import torchvision.utils as vutils
warnings.filterwarnings('ignore')
from option import opt, model_name, log_dir
from data_utils import *
from matplotlib import pyplot as plt
# torch.autograd.set_detect_anomaly(True)

print('log_dir :', log_dir)
print('model_name:', model_name)
models_ = {
	'AACNet': AACNet(),
	'AIDTransformer': AIDTransformer(),
	'DehazeFormer': DehazeFormer(),
	'HDMba': HDMba(),
}
loaders_ = \
 {'train': train_loader, 'test': test_loader}
start_time = time.time()
T = opt.steps


def lr_schedule_cosdecay(t, T, init_lr=opt.lr):
	lr = 0.5 * (1 + math.cos(t * math.pi / T)) * init_lr
	return lr


def train(net, loader_train, loader_test, optim, criterion):
	losses = []
	start_step = 0
	max_ssim = 0
	max_psnr = 0
	max_uqi = 0
	min_sam = 1
	ssims = []
	psnrs = []
	uqis = []
	sams = []
	if opt.resume and os.path.exists(opt.model_dir):
		print(f'resume from {opt.model_dir}')
		ckp = torch.load(opt.model_dir)
		losses = ckp['losses']
		net.load_state_dict(ckp['model'])
		start_step = ckp['step']
		max_ssim = ckp['max_ssim']
		max_psnr = ckp['max_psnr']
		min_sam = ckp['min_sam']
		max_uqi = ckp['max_uqi']
		psnrs = ckp['psnrs']
		ssims = ckp['ssims']
		uqis = ckp['uqis']
		sams = ckp['sams']
		print(f'start_step:{start_step} start training ---')
	else:
		print('train from scratch *** ')   # 没有训练好的网络，从头训练

	train_los = np.zeros(opt.steps)
	for step in range(start_step+1, opt.steps+1):
		net.train()
		lr = opt.lr
		if not opt.no_lr_sche:
			lr = lr_schedule_cosdecay(step, T)
			for param_group in optim.param_groups:
				param_group["lr"] = lr  
		x, y = next(iter(loader_train))
		x = x.to(opt.device)
		y = y.to(opt.device)
		out = net(x)
		# loss function
		loss1 = criterion[0](out, y)
		loss2 = criterion[1](out, y)
		loss = loss1 + 0.01*loss2
		loss.backward()

		optim.step()
		optim.zero_grad()
		losses.append(loss.item())
		train_los[step-1] = loss.item()
		print(
			f'\rtrain loss : {loss.item():.5f} |step :{step}/{opt.steps} |lr:{lr :.7f} |time_used :{(time.time() - start_time)  :.4f}s',
			end='', flush=True)

		if step % opt.eval_step == 0:
			with torch.no_grad():
				ssim_eval, psnr_eval, uqi_eval, sam_eval = test(net, loader_test)
			print(f'\nstep :{step} |ssim:{ssim_eval:.4f} |psnr:{psnr_eval:.4f} |uqi:{uqi_eval:.4f} |sam:{sam_eval:.4f}')
			ssims.append(ssim_eval)
			psnrs.append(psnr_eval)
			uqis.append(uqi_eval)
			sams.append(sam_eval)
			# if psnr_eval > max_psnr and ssim_eval > max_ssim and uqi_eval > max_uqi and min_sam > sam_eval:
			if psnr_eval > max_psnr and ssim_eval > max_ssim:
				max_ssim = max(max_ssim, ssim_eval)
				max_psnr = max(max_psnr, psnr_eval)
				max_uqi = max(max_uqi, uqi_eval)
				min_sam = min(min_sam, sam_eval)
				torch.save({
					'step': step,
					'max_psnr': max_psnr,
					'max_ssim': max_ssim,
					'max_uqi': max_uqi,
					'min_sam': min_sam,
					'ssims': ssims,
					'psnrs': psnrs,
					'uqis': uqis,
					'sams': sams,
					'losses': losses,
					'model': net.state_dict()
				}, opt.model_dir)
				print(f'\n model saved at step :{step}| max_psnr:{max_psnr:.4f}|max_ssim:{max_ssim:.4f}|max_uqi:{max_uqi:.4f} |min_sam:{min_sam:.4f}')

	iters = range(len(train_los))
	plt.figure()
	plt.plot(iters, train_los, 'g', label='train loss')
	plt.show()
	np.save(f'./numpy_files/{model_name}_{opt.steps}_losses.npy', losses)
	np.save(f'./numpy_files/{model_name}_{opt.steps}_ssims.npy', ssims)
	np.save(f'./numpy_files/{model_name}_{opt.steps}_psnrs.npy', psnrs)
	np.save(f'./numpy_files/{model_name}_{opt.steps}_uqis.npy', uqis)
	np.save(f'./numpy_files/{model_name}_{opt.steps}_sams.npy', sams)


def test(net, loader_test):  # verification
	net.eval()
	torch.cuda.empty_cache()
	ssims = []
	psnrs = []
	uqis = []
	sams = []
	for i, (inputs, targets) in enumerate(loader_test):
		inputs = inputs.to(opt.device)
		targets = targets.to(opt.device)
		pred = net(inputs)
		ssim1 = ssim(pred, targets).item()
		psnr1 = psnr(pred, targets)
		uqi1 = UQI(pred, targets)
		sam1 = SAM(pred, targets)
		# sam1 = calc_sam(pred, targets)
		ssims.append(ssim1)
		psnrs.append(psnr1)
		uqis.append(uqi1)
		sams.append(sam1)
	return np.mean(ssims), np.mean(psnrs), np.mean(uqis), np.mean(sams)


if __name__ == "__main__":
	loader_train = loaders_[opt.trainset]
	loader_test = loaders_[opt.testset]
	net = models_[opt.net]
	net = net.to(opt.device)
	if opt.device == 'cuda':
		net = torch.nn.DataParallel(net)
		cudnn.benchmark = True
	criterion = []
	criterion.append(nn.L1Loss().to(opt.device))   # L1损失被放入到criterion[0]
	criterion.append(nn.MSELoss().to(opt.device))

	optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, net.parameters()), lr=opt.lr, betas=(0.9, 0.999), eps=1e-08)
	optimizer.zero_grad()
	if torch.cuda.device_count() > 1:
		model = torch.nn.DataParallel(net)  # 前提是model已经在cuda上了
	train(net, loader_train, loader_test, optimizer, criterion)
	

