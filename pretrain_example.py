from dataloader_seismic import *
from fcn import *
from aux_functions import *

from torchinfo import summary
from pathlib import Path

root = os.getcwd()
root = str(Path(root).parent)

signature = 'prototype' #name for reference
pretask = 'frame_order' # jigsaw, rotation, frame_order
dataset = 'F3_netherlands' # F3_netherlands  Parihaka_NZPM

if pretask == 'jigsaw':
    n_classes = 9
    from train_jigsaw import *
elif pretask == 'rotation':
    n_classes = 5
    from train_rotation import *
elif pretask == 'frame_order':
    n_classes = 6
    from train_frame_order import *

else:
    raise ValueError(f'Not recognized pretask: {pretask}')

saving_logs_path = os.path.join(root,f'logs_folder/logs_{pretask}') 


args = {
    'dataset' : 'F3_netherlands', # F3_netherlands  Parihaka_NZPM
    'n_classes' : n_classes,
    'pretask'      : pretask,
    'batch_size': 25,     
    'n_channels' : 1,
    'num_workers' : 4,

    'epochs'    : 100,
    'lr' : 0.001, 
    'optim' : 'Adam', #SGD, Adam
    'scheduler' : 'StepLR', #StepLR, CosineAnnealingLR
    'step_size' : 5,
    'gamma' : 0.8,
    'eta_min' :  0.00001, 
    'weight_decay': 1e-4, # L2 penalty.
    'momentum': 0.9,      # Momentum.
}

if args['dataset']=='F3_netherlands':
    saving_models_path = os.path.join(root,f'saving_models_path/model_{pretask}')
    args['height'] = 448
    args['width'] = 448
    
elif args['dataset']=='Parihaka_NZPM': 
    saving_models_path = os.path.join(root,f'saving_models_path/model_{pretask}')
    args['height'] = 832
    args['width'] = 448

if torch.cuda.is_available():
    args['device'] = torch.device('cuda')
else:
    args['device'] = torch.device('cpu')
print(args['device'])

train_set = SeismicDataset(root=root, dataset_name=args['dataset'], split='train', task=args['pretask'])
val_set = SeismicDataset(root=root, dataset_name=args['dataset'], split='val', task=args['pretask'])

dataloader_train = DataLoader(train_set, 
                              batch_size=args['batch_size'],
                              shuffle=True,num_workers=args['num_workers'],drop_last=True)

dataloader_val = DataLoader(val_set, 
                              batch_size=args['batch_size'],
                              shuffle=True,num_workers=args['num_workers'])



name_model = f'{train_set.dataset_name}_{signature}_{pretask}'

# Instantiating architecture.
model_fcn = FCN(num_classes=args['n_classes'], in_channels=args['n_channels'],task=args['pretask']).to(args['device'])

# Printing architecture.
summary(model_fcn,input_size=(args['batch_size'],args['n_channels'],args['width'],args['height'] ))

# definição da criterion
criterion = set_criterion(task=args['pretask'],device=args['device'],dataset=args['dataset']) 

if args['optim'] == 'SGD':
    optimizer = torch.optim.SGD(model_fcn.parameters(), lr=args['lr'], weight_decay= args['weight_decay'], momentum= args['momentum'])
elif args['optim'] == 'Adam':
    optimizer = torch.optim.Adam(model_fcn.parameters(), lr=args['lr'], weight_decay= args['weight_decay'])

if args['scheduler'] == 'StepLR':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args['step_size'], gamma=args['gamma'])
elif args['scheduler'] == 'CosineAnnealingLR':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=args['eta_min'])

print(name_model)

#Train and save - Must adjust the name for each pretext task. ## train_frame, train_jigsaw, train_rotation ###
train_acc_all,train_acc_std_all,train_loss_all,train_loss_std_all, val_acc_all, val_acc_std_all,val_loss_all,val_loss_std_all =train_frame(dataloader_train,
                                                                            dataloader_val,
                                                                            model_fcn, 
                                                                            optimizer,
                                                                            criterion,
                                                                            args['epochs'], 
                                                                            args['device'], 
                                                                            scheduler,
                                                                            task=args['pretask'],
                                                                            save=True,
                                                                            saved_models_path=saving_models_path,
                                                                            name_model=name_model)
#Can be consulted in aux_functions
save_frame_log_results(saving_logs_path,name_model,
                          train_acc_all,train_acc_std_all,train_loss_all, train_loss_std_all,
                          val_acc_all, val_acc_std_all,val_loss_all, val_loss_std_all
                          )