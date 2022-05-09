from dataloader_seismic_v5 import *
from plots import *
from fcn_v2 import *
from aux_functions import *
from train_jigsaw import *

from torchinfo import summary

root = '/mnt/DADOS_CERGY_1/bruno/'


saving_logs_path = '/mnt/DADOS_CERGY_1/bruno/master_project/jigsaw_grid_search'

args = {
    'dataset' :  'Parihaka_NZPM' , # F3_netherlands  Parihaka_NZPM
    'tolerancia' : 100,
    'task'      : 'jigsaw',
    'gamma' : 0.8,
    'step_size' : 5,
    'weight_decay': 1e-4, # L2 penalty.
    'momentum': 0.9,      # Momentum.
    'batch_size' : 1,
    'batch_size_test_val' : 1,
    'epochs'    : 100,
    'n_channels' : 1,
    'num_workers' : 4,
    'jigsaw_classes' : 9,
}
   
if args['dataset']=='F3_netherlands':
    args['height'] = 128
    args['width'] = 192
    saved_models_path = '/mnt/DADOS_CERGY_1/bruno/saved_models_f3/jigsaw_grid_search'
    
elif args['dataset']=='Parihaka_NZPM': 
    args['height'] = 256
    args['width'] = 128
    saved_models_path = '/mnt/DADOS_CERGY_1/bruno/save_models_parih/jigsaw_grid_search'

if torch.cuda.is_available():
    args['device'] = torch.device('cuda')
else:
    args['device'] = torch.device('cpu')


train_set = SeismicDataset(root=root, dataset_name=args['dataset'], split='train',  task=args['task'] )
val_set = SeismicDataset(root=root, dataset_name=args['dataset'], split='val',  task=args['task'])

dataloader_train = DataLoader(train_set, batch_size=args['batch_size'],
                              shuffle=True, num_workers=args['num_workers'],drop_last=True)

dataloader_val = DataLoader(val_set, 
                              batch_size=args['batch_size_test_val'],
                              shuffle=True,num_workers=args['num_workers'])

grid = get_grid_search()

for idx in range(len(grid)):

    name_model = f'parih_first_jigsaw_grid_run_{idx}'
    args['lr'] = grid[f'grid_run_{idx}'][0]
    args['eta_min'] = grid[f'grid_run_{idx}'][1]
    args['optim'] = grid[f'grid_run_{idx}'][2]
    args['scheduler'] = grid[f'grid_run_{idx}'][3]

    
    # Instantiating architecture.
    model_fcn = FCN(num_classes=args['jigsaw_classes'], in_channels=args['n_channels'],task=args['task']).to(args['device'])

    # Printing architecture.
    if idx == 0:
        summary(model_fcn,input_size=(args['batch_size'],args['n_channels'],args['width'],args['height'] ))

    # definição da loss
    loss = set_loss(args['task'],args['device']) 
    #print(loss)

    # definição do otimizador
    if args['optim'] == 'SGD':
        optimizer = torch.optim.SGD(model_fcn.parameters(), lr=args['lr'], weight_decay= args['weight_decay'], momentum= args['momentum'])
    elif args['optim'] == 'Adam':
        optimizer = torch.optim.Adam(model_fcn.parameters(), lr=args['lr'], weight_decay= args['weight_decay'])
    #print(optimizer)

    #definindo o sheduler - deve ser acrescentado no train ainda
    if args['scheduler'] == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args['step_size'], gamma=args['gamma'])
    elif args['scheduler'] == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=args['eta_min'])

    print(name_model)
    #print(loss)
    print(optimizer)
    print(scheduler)

    train_acc_mean,train_acc_std,train_loss_mean,train_loss_std,val_acc_mean,val_acc_std,val_loss_mean,val_loss_std =train_jigsaw(
        dataloader_train, 
        dataloader_val,
        model_fcn, 
        optimizer,
        loss, 
        args['epochs'], 
        args['device'],
        scheduler=scheduler,
        tolerancia=args['tolerancia'],
        save=True,
        saved_models_path=saved_models_path, 
        name_model=name_model)


    save_jigsaw_log_results( saving_logs_path,name_model, 
                train_acc_mean,train_acc_std,train_loss_mean,train_loss_std,
                #test_acc_mean,test_acc_std,test_loss_mean,test_loss_std,
                val_acc_mean,val_acc_std,val_loss_mean,val_loss_std 
                )
    print("")



