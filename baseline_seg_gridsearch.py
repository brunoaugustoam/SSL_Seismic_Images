from dataloader_seismic_v5 import *
from plots import *
from fcn_v2 import *
from aux_functions import *
from train_segmentation_grid import *


from torchinfo import summary

root = '/mnt/DADOS_CERGY_1/bruno/'
saving_logs_path = '/mnt/DADOS_CERGY_1/bruno/master_project/baseline_grid_logs'


args = {
    'dataset' :  'F3_netherlands' , # F3_netherlands  Parihaka_NZPM
    'width'     : 448,
    'task'      : 'segmentation',
    'batch_size' : 1,
    'num_workers' : 4,
    'n_channels' : 1,
    'train_type' : 'few_shot', # sup_ssl  few_shot fine_tune
    'p_few_shot' : 0.01, #1, 0.1, 0.01 porcentagens ### F3:916, 94, 9 /// # Parihaka: 823, 82, 8
    'train_set_size_f3' : 916,
    'train_set_size_parih' : 823,
    'epochs'    : 100, 
    'tolerancia' : 30,
    'step_size' : 20,
    'gamma'   :  0.5,
    'weight_decay': 1e-4, # L2 penalty.
    'momentum' : 0.9
}
dataset = args['dataset']

if torch.cuda.is_available():
    args['device'] = torch.device('cuda')
else:
    args['device'] = torch.device('cpu')
   
if args['dataset']=='F3_netherlands':
    n_few_shot = int(np.round(args['train_set_size_f3'] * args['p_few_shot'],0))
    saved_models_segmentation_path = '/mnt/DADOS_CERGY_1/bruno/saved_models_f3/baseline_grid'
    args['height'] =  448
    
elif args['dataset']=='Parihaka_NZPM': 
    n_few_shot = int(np.round(args['train_set_size_parih'] * args['p_few_shot'],0))
    saved_models_segmentation_path = '/mnt/DADOS_CERGY_1/bruno/save_models_parih/baseline_grid'
    args['height'] =  832

if args['dataset']=='F3_netherlands':
    args['n_classes'] = 10
elif args['dataset']=='Parihaka_NZPM': 
    args['n_classes'] = 6

train_set = SeismicDataset(root=root, dataset_name=args['dataset'], split='train', 
                            task=args['task'],train_type=args['train_type'],n_few_shot=n_few_shot)

val_set = SeismicDataset(root=root, dataset_name=args['dataset'], split='val', task=args['task'])

if args['dataset']=='F3_netherlands':
    certo = ['inline_607', 'crossline_1076', 'inline_178', 'crossline_724', 'inline_300', 'inline_150', 'inline_578', 'inline_664', 'inline_504']
    assert(train_set.sec_number_list[:9] == certo),'Not the right sections'
    
elif args['dataset']=='Parihaka_NZPM': 
    if args['p_few_shot'] == 0.01:
        certo = ['Inline_522', 'Inline_284', 'Inline_514', 'Inline_331', 'Crossline_596', 'Crossline_588', 'Crossline_208', 'Crossline_291']
    elif args['p_few_shot'] == 0.1:
        certo = ['Inline_522', 'Inline_284', 'Inline_514', 'Inline_331', 'Inline_210', 'Inline_90', 'Inline_299', 'Inline_181']
    assert(train_set.sec_number_list[:8] == certo),'Not the right sections'


dataloader_train = DataLoader(train_set, 
                              batch_size=args['batch_size'],
                              shuffle=True,num_workers=args['num_workers'])

dataloader_val = DataLoader(val_set, 
                              batch_size=args['batch_size'],
                              shuffle=True,num_workers=args['num_workers'])


grid = get_grid_search_baseline()

for idx in range(len(grid)):  

    freeze = False
    model = list(grid.keys())[0]

    name_model = f'{dataset}_first_{model}_base{idx}_1perc' 

    model_fcn = FCN(num_classes=args['n_classes'], in_channels=args['n_channels'],task=args['task']).to(args['device'])
    #Update Network for segmentation
    model_fcn.update_task(args['task'], args['n_classes'])
    model_fcn = model_fcn.to(args['device'])

    args['lr_backbone'] = grid[f'base_grid_{idx}'][0]
    if args['lr_backbone'] == 0.0:
        freeze = True
    args['lr_seg'] = grid[f'base_grid_{idx}'][1]


    if torch.cuda.is_available():
        args['device'] = torch.device('cuda')
    else:
        args['device'] = torch.device('cpu')

    # Printing architecture.
    if idx == 0:
        summary(model_fcn,input_size=(args['batch_size'],args['n_channels'],args['width'],args['height'] ))

    # definição da loss
    loss = set_loss(args['task'],args['device']) 

    # definição do otimizador
    optimizer = torch.optim.Adam([
        {'params': model_fcn.layer1.parameters(), 'lr':args['lr_backbone']}, 
        {'params': model_fcn.layer2.parameters(), 'lr':args['lr_backbone'] },
        {'params': model_fcn.layer3.parameters(), 'lr':args['lr_backbone'] },
        {'params': model_fcn.layer4.parameters(), 'lr':args['lr_backbone'] },
      
        {'params':model_fcn.segmenter.parameters(), 'lr':args['lr_seg'] }, ],
        lr=args['lr_seg'], weight_decay= args['weight_decay'])

    #definindo o sheduler - deve ser acrescentado no train ainda
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args['step_size'], gamma=args['gamma'])

    print('Run: {0} - Lr Backbone: {1}, Lr Segmenter: {2}'.format(idx, args['lr_backbone'],args['lr_seg']))
    print(f'name model {name_model}')

    train_iou_classes_all,train_iou_std_between_classes_all,train_iou_mean_all,train_loss_mean_all,train_loss_std_all,val_iou_classes_all,val_iou_std_between_classes_all,val_iou_mean_all,val_loss_mean_all,val_loss_std_all= train_segmentation(
        dataloader_train=dataloader_train,
        model=model_fcn,
        optimizer=optimizer,
        loss=loss,
        num_epochs=args['epochs'],
        dataloader_val= dataloader_val,
        device=args['device'],
        num_classes=args['n_classes'],
        scheduler=scheduler,
        task=args['task'],
        train_type=args['train_type'], 
        tolerancia=args['tolerancia'],
        save=True,
        saved_models_path=saved_models_segmentation_path,
        saving_logs_path=saving_logs_path,
        name_model=name_model,
        freeze=freeze,
        lr_fine= args['lr_seg'])

    #torch.save(model_fcn.state_dict(), os.path.join(saved_models_segmentation_path, name_model))

    save_segmentation_log_results(saving_logs_path,name_model, 
            train_iou_classes_all,train_iou_std_between_classes_all,train_iou_mean_all,train_loss_mean_all,train_loss_std_all,
            val_iou_classes_all,   val_iou_std_between_classes_all,   val_iou_mean_all,   val_loss_mean_all,   val_loss_std_all
        )


