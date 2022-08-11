from dataloader_seismic import *
from plots import *
from fcn import *
from aux_functions import *
from train_segmentation import *
from pathlib import Path
from torchinfo import summary

root = os.getcwd()
root = str(Path(root).parent)

dataset= 'F3_netherlands'       # 'F3_netherlands' 'Parihaka_NZPM'
pretask = 'frame_order'
n_few_shot = 1

signature = f'cross_{pretask}_{n_few_shot}shot'
load_path = os.path.join(root,'load_models/')

saved_models_segmentation_path = os.path.join(root,f'save_models_cross/fine_{pretask}')
saving_logs_path = os.path.join(root,f'pos_quali/fine_{pretask}_logs')


if pretask == 'jigsaw':
    pre_classes = 9
elif pretask == 'rotation':
    pre_classes = 5
elif pretask == 'frame_order':
    pre_classes = 6

args = {
    'dataset' :  dataset , # F3_netherlands  Parihaka_NZPM
    'task'      : 'segmentation',
    'batch_size' : 1,
    'num_workers' : 4,
    'n_channels' : 1,
    'train_type' : 'fine_tune', # sup_ssl  few_shot fine_tune
}

if torch.cuda.is_available():
    args['device'] = torch.device('cuda')
else:
    args['device'] = torch.device('cpu')
   
if args['dataset']=='F3_netherlands':
    args['height'] =  448
    args['n_classes'] = 10
    
elif args['dataset']=='Parihaka_NZPM': 
    args['height'] =  832
    args['n_classes'] = 6
    

for fold in range(0,5):  

    train_set = SeismicDataset(root=root, dataset_name=args['dataset'], split='train', 
                                task=args['task'],train_type=args['train_type'],n_few_shot=n_few_shot,
                                cross_val=True, fold=fold)
    dataloader_train = DataLoader(train_set, 
                                  batch_size=args['batch_size'],
                                  shuffle=True,num_workers=args['num_workers']) 


    #Name of the model in fine-tune
    name_model = f'{train_set.dataset_name}_{signature}_fold{fold}'


    # Instantiating architecture.
    args['pretask'] = pretask
    model_fcn = FCN(num_classes=pre_classes, in_channels=args['n_channels'],task=args['pretask']).to(args['device'])

    
    
    #Provide the name of the model to be loaded to be fine-tuned
    pretext_model = 'Parihaka_NZPM_prototype_frame_order_grid_run_19' #example, must be adjusted
    #After defining a model, load 
    model_fcn.load_state_dict(torch.load(os.path.join(load_path, pretext_model)))

    args = {
        'dataset' :  dataset , # F3_netherlands  Parihaka_NZPM
        'width'     : 448,
        'tolerancia' : 100,
        'task'      : 'segmentation',
        'step_size' : 20,
        'gamma'   :  0.5,
        'weight_decay': 1e-4, # L2 penalty.
        'batch_size' : 1,
        'epochs'    : 100, 
        'n_channels' : 1,
        'num_workers' : 4,
        'train_type' : 'fine_tune', # sup_ssl  few_shot fine_tune
    }

    args['lr_backbone'] = 0.0001
    args['lr_seg'] = 0.001


    if args['dataset']=='F3_netherlands':
        args['n_classes'] = 10
        args['height'] =  448
    elif args['dataset']=='Parihaka_NZPM': 
        args['n_classes'] = 6
        args['height'] =  832

    if torch.cuda.is_available():
        args['device'] = torch.device('cuda')
    else:
        args['device'] = torch.device('cpu')

    #Update Network for segmentation
    model_fcn.update_task(args['task'], args['n_classes'])
    model_fcn = model_fcn.to(args['device'])

    # Printing architecture.
    if fold == 0:
        summary(model_fcn,input_size=(args['batch_size'],args['n_channels'],args['width'],args['height'] ))

    # definição da loss
    criterion = set_criterion(args['task'],args['device']) #Cross Entropy Loss with wheights 

    # Custom optimizer
    optimizer = torch.optim.Adam([
        {'params': model_fcn.layer1.parameters(), 'lr':args['lr_backbone']}, 
        {'params': model_fcn.layer2.parameters(), 'lr':args['lr_backbone'] },
        {'params': model_fcn.layer3.parameters(), 'lr':args['lr_backbone'] },
        {'params': model_fcn.layer4.parameters(), 'lr':args['lr_backbone'] },
      
        {'params':model_fcn.segmenter.parameters(), 'lr':args['lr_seg'] }, ],
        lr=args['lr_seg'], weight_decay= args['weight_decay'])

    #definindo o sheduler - deve ser acrescentado no train ainda
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args['step_size'], gamma=args['gamma'])
    
    print(f'pretext model {pretext_model}')
    print('Run: {0} - Lr Backbone: {1}, Lr Segmenter: {2}'.format(fold, args['lr_backbone'],args['lr_seg']))

    train_iou_classes_all,train_iou_std_between_classes_all,train_iou_mean_all,train_loss_mean_all,train_loss_std_all= train_segmentation(
        dataloader_train=dataloader_train,
        model=model_fcn,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=args['epochs'],
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
        weighted=True)

    save_segmentation_log_few_shot(saving_logs_path,name_model, 
            train_iou_classes_all,train_iou_std_between_classes_all,train_iou_mean_all,train_loss_mean_all,train_loss_std_all,
        )


