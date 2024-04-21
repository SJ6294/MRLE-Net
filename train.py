
# model py name, option
import os
import torch
import yaml

from utils import network_parameters, losses
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import time
import numpy as np
import random
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from tensorboardX import SummaryWriter
import utils.losses

from dataset import DataLowTrain, DataLowValid


from model.MRLENet import MRLENet
option = 'MRLENet'
val_limit = 160 # 160  # 140

## Set Seeds
torch.backends.cudnn.benchmark = True
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

## Load yaml configuration file``
with open('training_mas3k.yaml', 'r') as config:    # training.yaml
    opt = yaml.safe_load(config)
Train = opt['TRAINING']
OPT = opt['OPTIM']

## Build Model
print('==> Build the model')
model_restored = MRLENet(in_channels=3, wf=96, depth=4)  # 96

print(model_restored)


# parameter
p_number = network_parameters(model_restored)
model_restored.cuda()

## Training model path direction
mode = opt['MODEL']['MODE']

model_dir = os.path.join(Train['SAVE_DIR'], option, 'models')
utils.mkdir(model_dir)

train_dir = Train['TRAIN_DIR']
val_dir = Train['VAL_DIR']
image_save_dir = os.path.join(Train['SAVE_DIR'], option, 'save_images')

txt_dir = os.path.join(Train['SAVE_DIR'], 'abl.txt')

txt = open(txt_dir, "a")
## GPU
gpus = ','.join([str(i) for i in opt['GPU']])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
device_ids = [i for i in range(torch.cuda.device_count())]

if torch.cuda.device_count() > 1:
    print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")
if len(device_ids) > 1:
    model_restored = nn.DataParallel(model_restored, device_ids=device_ids)


## Optimizer
start_epoch = 1
new_lr = float(OPT['LR_INITIAL'])
optimizer = optim.Adam(model_restored.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)

## Scheduler (Strategy)
warmup_epochs = 3  
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, OPT['EPOCHS'] - warmup_epochs,
                                                        eta_min=float(OPT['LR_MIN']))
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
scheduler.step()

## Resume (Continue training by a pretrained model)
if Train['RESUME']:
    path_chk_rest = utils.get_last_path(model_dir, '_latest.pth')
    utils.load_checkpoint(model_restored, path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    utils.load_optim(optimizer, path_chk_rest)

    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------')

## Loss
Charloss = losses.CharbonnierLoss()
ssim_loss = losses.SSIM()
detail_loss = losses.detailLoss()
alpha= 0.84

## DataLoaders
print('==> Loading datasets')
train_dataset = DataLowTrain(train_dir, resize=352, img_options={'patch_size': Train['TRAIN_PS']})  
train_loader = DataLoader(dataset=train_dataset, batch_size=OPT['BATCH'],
                          shuffle=True, num_workers=0, drop_last=False)

val_dataset = DataLowValid(val_dir, resize=352, img_options={'patch_size': Train['VAL_PS']})
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0,
                        drop_last=False)

# Show the training configuration
print(f'''==> Training details:
------------------------------------------------------------------
    Restoration mode:   {mode}
    change_option:      {option}
    Train patches size: {str(Train['TRAIN_PS']) + 'x' + str(Train['TRAIN_PS'])} 
    Val patches size:   {str(Train['VAL_PS']) + 'x' + str(Train['VAL_PS'])}
    Model parameters:   {p_number}
    Start/End epochs:   {str(start_epoch) + '~' + str(OPT['EPOCHS'])}
    Batch sizes:        {OPT['BATCH']}
    Learning rate:      {OPT['LR_INITIAL']}
    GPU:                {'GPU' + str(device_ids)}''')
print('------------------------------------------------------------------')

txt.write('\n')
txt.write(f'''==> Training details:
------------------------------------------------------------------
    Restoration mode:   {mode}
    change_option:      {option}
    Train patches size: {str(Train['TRAIN_PS']) + 'x' + str(Train['TRAIN_PS'])}
    Val patches size:   {str(Train['VAL_PS']) + 'x' + str(Train['VAL_PS'])}
    Model parameters:   {p_number}
    Start/End epochs:   {str(start_epoch) + '~' + str(OPT['EPOCHS'])}
    Batch sizes:        {OPT['BATCH']}
    Learning rate:      {OPT['LR_INITIAL']}
    GPU:                {'GPU' + str(device_ids)}''')
txt.write('\n')
txt.write('------------------------------------------------------------------')

# Start training!
print('==> Training start: ')
best_psnr = 0
best_ssim = 0
best_epoch_psnr = 0
best_epoch_ssim = 0
total_start_time = time.time()

## Log
log_dir = os.path.join(Train['SAVE_DIR'], option, 'log')
utils.mkdir(log_dir)
writer = SummaryWriter(log_dir=log_dir, filename_suffix=f'_{mode}')

for epoch in range(start_epoch, OPT['EPOCHS'] + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    valid_loss = 0
    # train_id = 1
    total_epoch_loss = 0
    # total_valid_loss = 0

    model_restored.train()
    for i, data in enumerate(tqdm(train_loader), 0):

        optimizer.zero_grad()

        targets = data[0].cuda()
        inputs = data[1].cuda()
        restored_t = model_restored(inputs)

        filenameLs = data[2]

        # Compute loss
        loss =  alpha*(1-ssim_loss(restored_t, targets)) + (1-alpha)*(Charloss(restored_t, targets) + detail_loss(restored_t, targets)) # + 0.5*detail_loss(restored_t, targets)        
        
        # Back propagation
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    total_epoch_loss = epoch_loss / len(train_loader)
    
    ## Evaluation (Validation)
    if epoch >= val_limit :
        if epoch % Train['VAL_AFTER_EVERY'] == 0:
            model_restored.eval()
            psnr_val_rgb = []
            ssim_val_rgb = []

            with torch.no_grad():
                for ii, data_val in enumerate(val_loader, 0):
                    target = data_val[0].cuda()
                    input = data_val[1].cuda() 
                    filenames = data_val[2]

                    restored = model_restored(input)
                    restored = torch.clamp(restored,0,1) 

                    psnr_val_rgb.append(utils.torchPSNR(restored, target)) 
                    ssim_val_rgb.append(utils.torchSSIM(restored, target))

                    if epoch == 1 or epoch % Train['VAL_IMAGE_SAVE'] == 0:
                        target = target.permute(0, 2, 3, 1).cpu().detach().numpy()
                        input = input.permute(0, 2, 3, 1).cpu().detach().numpy()
                        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
                        
                        temp = np.concatenate((input[0]*255, restored[0]*255, target[0]*255), axis=1)
                        utils.mkdir("%s/%s" %(image_save_dir, epoch))
                        utils.save_img("%s/%s/%s.jpg" %(image_save_dir, epoch, filenames[0]), temp.astype(np.uint8))
                
            psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
            ssim_val_rgb = torch.stack(ssim_val_rgb).mean().item()

            # Save the best PSNR model of validation
            if psnr_val_rgb > best_psnr:
                best_psnr = psnr_val_rgb
                best_epoch_psnr = epoch
                torch.save({'epoch': epoch,
                            'state_dict': model_restored.state_dict(),
                            'optimizer': optimizer.state_dict()
                            }, os.path.join(model_dir, "model_bestPSNR.pth"))
            print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (
                epoch, psnr_val_rgb, best_epoch_psnr, best_psnr))
            txt.write('\n')
            txt.write("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (
                epoch, psnr_val_rgb, best_epoch_psnr, best_psnr))

            # Save the best SSIM model of validation
            if ssim_val_rgb > best_ssim:
                best_ssim = ssim_val_rgb
                best_epoch_ssim = epoch
                torch.save({'epoch': epoch,
                            'state_dict': model_restored.state_dict(),
                            'optimizer': optimizer.state_dict()
                            }, os.path.join(model_dir, "model_bestSSIM.pth"))
            print("[epoch %d SSIM: %.4f --- best_epoch %d Best_SSIM %.4f]" % (
                epoch, ssim_val_rgb, best_epoch_ssim, best_ssim))
            txt.write('\n')
            txt.write("[epoch %d SSIM: %.4f --- best_epoch %d Best_SSIM %.4f]" % (
                epoch, ssim_val_rgb, best_epoch_ssim, best_ssim))

            # # Save evey epochs of model
            # torch.save({'epoch': epoch,
            #             'state_dict': model_restored.state_dict(),
            #             'optimizer': optimizer.state_dict()
            #             }, os.path.join(model_dir, f"model_epoch_{epoch}.pth"))
        

            writer.add_scalar('val/PSNR', psnr_val_rgb, epoch)
            writer.add_scalar('val/SSIM', ssim_val_rgb, epoch)

            model_restored.train()

    scheduler.step()

    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.7f}".format(epoch, time.time() - epoch_start_time,
                                                                              total_epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")
    txt.write('\n')
    txt.write("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.7f}".format(epoch, time.time() - epoch_start_time,
                                                                              total_epoch_loss, scheduler.get_lr()[0]))

    # Save the last model
    torch.save({'epoch': epoch,
                'state_dict': model_restored.state_dict(),
                'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, "model_latest.pth"))
   

    writer.add_scalar('train/loss', epoch_loss, epoch)
    writer.add_scalar('train/lr', scheduler.get_lr()[0], epoch)
    # writer.add_scalars('loss', {'valid_loss':total_valid_loss, 
    #                             'train_loss':total_epoch_loss}, epoch)
    
writer.close()
txt.flush()
# option_txt.flush()

total_finish_time = (time.time() - total_start_time)  # seconds
print('Total training time: {:.1f} hours'.format((total_finish_time / 60 / 60)))
print('train_model option: %s'  % option)

