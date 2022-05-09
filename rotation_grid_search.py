from dataloader_seismic_v5 import *
from fcn_v2 import *
from aux_functions import *
from train_rotation import *
from torchinfo import summary

root = '/mnt/DADOS_CERGY_1/bruno/'

saving_logs_path = '/mnt/DADOS_CERGY_1/bruno/master_project/rotation_grid_search'

args = {
    'dataset_name' : 'Parihaka_NZPM' , # F3_netherlands  Parihaka_NZPM
    'rot_classes' : 5,
    'task'      : 'rotation',
    'step_size' : 10,
    'gamma' : 0.8,
    'batch_size': 4,      # Mini-batch size. 
    'batch_size_test_val' : 1,
    'weight_decay': 1e-4, # L2 penalty.
    'momentum': 0.9,      # Momentum.
    'n_channels' : 1,
    'num_workers' : 4,
    
    'epochs'    : 150,

}

if args['dataset_name']=='F3_netherlands':
    saved_models_path = '/mnt/DADOS_CERGY_1/bruno/saved_models_f3/rotation_grid_search'
    args['height'] = 336
    args['width'] = 336
    
elif args['dataset_name']=='Parihaka_NZPM': 
    saved_models_path = '/mnt/DADOS_CERGY_1/bruno/save_models_parih/rotation_grid_search'
    args['height'] = 896
    args['width'] = 448

if torch.cuda.is_available():
    args['device'] = torch.device('cuda')
else:
    args['device'] = torch.device('cpu')

    
print(args['device'])

train_set = SeismicDataset(root=root, dataset_name=args['dataset_name'], split='train', task=args['task'])
val_set = SeismicDataset(root=root, dataset_name=args['dataset_name'], split='val', task=args['task'])

dataloader_train = DataLoader(train_set, 
                              batch_size=args['batch_size'],
                              shuffle=True,num_workers=args['num_workers'],drop_last=True)

dataloader_val = DataLoader(val_set, 
                              batch_size=args['batch_size_test_val'],
                              shuffle=True,num_workers=args['num_workers'])

grid = get_grid_search_rot_parih() #get_grid_search()

for idx in range(len(grid)):

    name_model = f'parih_rotation_second_grid_run_{idx}'
    args['lr'] = grid[f'grid_run_{idx}'][0]
    args['eta_min'] = grid[f'grid_run_{idx}'][1]
    args['optim'] = grid[f'grid_run_{idx}'][2]
    args['scheduler'] = grid[f'grid_run_{idx}'][3]


    # Instantiating architecture.
    model_fcn = FCN(num_classes=args['rot_classes'], in_channels=args['n_channels'],task=args['task']).to(args['device'])

    # Printing architecture.
    if idx == 0:
        summary(model_fcn,input_size=(args['batch_size'],args['n_channels'],args['width'],args['height'] ))


    # definição da loss
    loss = set_loss(args['task'],args['device']) 

    # definição do otimizador
    if args['optim'] == 'SGD':
        optimizer = torch.optim.SGD(model_fcn.parameters(), lr=args['lr'], weight_decay= args['weight_decay'], momentum= args['momentum'])
    elif args['optim'] == 'Adam':
        optimizer = torch.optim.Adam(model_fcn.parameters(), lr=args['lr'], weight_decay= args['weight_decay'])

    #definindo o sheduler - deve ser acrescentado no train ainda
    if args['scheduler'] == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args['step_size'], gamma=args['gamma'])
    elif args['scheduler'] == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=args['eta_min'])

    print(name_model)
    #print(loss)
    print(optimizer)
    print(scheduler)

    train_acc_all,train_acc_std_all,train_loss_all,train_loss_std_all, val_acc_all, val_acc_std_all,val_loss_all,val_loss_std_all =train_rotation(  dataloader_train,
                                                                                dataloader_val,
                                                                                model_fcn, 
                                                                                optimizer,
                                                                                loss,
                                                                                args['epochs'], 
                                                                                args['device'], 
                                                                                scheduler,
                                                                                task=args['task'],
                                                                                save=True,
                                                                                saved_models_path=saved_models_path,
                                                                                name_model=name_model)




    save_rotation_log_results(saving_logs_path,name_model,
                              train_acc_all,train_acc_std_all,train_loss_all, train_loss_std_all,
                              val_acc_all, val_acc_std_all,val_loss_all, val_loss_std_all
                              )