import torch
from torchvision import transforms as T

import PIL
from PIL import Image

from skimage.feature import hog
import skimage
from scipy import ndimage as ndi
from scipy.ndimage.interpolation import rotate

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import numpy as np
from scipy.spatial.distance import cdist
from plots import *


def tensor_to_pil(tensor):
	if len(tensor.shape) == 3:
		image_array = np.transpose(tensor, (1, 2, 0))
		image_array = image_array.numpy()
	else:
		image_array = tensor.numpy()
	
	return T.ToPILImage()(image_array)


def rotate_tensor(img_nparray, rotations_list=None, plot=False):
    if rotations_list == None:
        rotations_list = [-8,-4,0,4,8]

    rot_label = int(np.random.randint(0,len(rotations_list), size=1))
    rot_angle = rotations_list[rot_label] 

    rotated_image = skimage.transform.rotate(img_nparray, angle= rot_angle, order=1)

    if plot:
        print(f"Rotation angle: {rot_angle} == Rot_label: {rot_label}")
        #if len(rotated_image.shape) == 3:
        #    plot_tensor_img_mask(rotated_image)
        #else:
        plot_array(rotated_image)


    return np.array(rotated_image), np.array(rot_label)

def random_crop(image, mask=np.array([]), final_size=(448,448), dataset='F3_netherlands'):
    """
    Todos esses 3 crops foram implementados para tensores 2D, depois implementar considerando 3D
    """ 
    assert len(image.shape) == 2, "Implemented for 2D only, 3D will be done soon" 
    
    height = image.shape[0] #450f3, 600parih
    width = image.shape[1] #600f3, 450f3
    if dataset=='F3_netherlands': 
        height_idx = int(np.random.randint(low=0,high=height-final_size[0], size=1))
        width_idx = int(np.random.randint(low=0,high=width-final_size[1], size=1))  

    elif dataset=='Parihaka_NZPM':
        height_idx = int(np.random.randint(low=0,high=height-final_size[0], size=1))
        width_idx = int(np.random.randint(low=0,high=width-final_size[1], size=1))  


    resize_img = image[height_idx:height_idx+final_size[0], width_idx:width_idx+final_size[1]]

    if mask.any():
        resize_mask = mask[height_idx:height_idx+final_size[0], width_idx:width_idx+final_size[1]]
        return resize_img, resize_mask
    else:
        return resize_img

def center_crop(image, mask=np.array([]),final_height=450,final_width=900):

    assert len(image.shape) == 2 , "Implemented for 2D only, 3D will be done soon" 

    #if len(image.shape) == 3:
    #    #image_array = np.transpose(image, (1, 2, 0))
    #    height_center = int(image.shape[1]/2)
    #    width_center = int(image.shape[2]/2)
    #else:
    height_center = int(image.shape[0]/2)
    width_center = int(image.shape[1]/2)

    half_height = int(final_height/2)
    half_width = int(final_width/2)
    
    resize_img = image[height_center-half_height :  height_center+half_height,
                          width_center-half_width  :  width_center+half_width]

    if mask.any():
        resize_mask = mask[height_center-half_height :  height_center+half_height,
                          width_center-half_width  :  width_center+half_width]
        return resize_img, resize_mask
    else:
        return resize_img

def crop(image, mask=np.array([]) ,top=0, left=0,final_height=450,final_width=600):

    assert len(image.shape) == 2, "Implemented for 2D only, 3D will be done soon" 

    half_height = int(final_height/2)
    half_width = int(final_width/2)
    
    resize_img = image[top  :  top  + final_height,
                          left :  left + final_width]
    if mask.any():
        resize_mask = mask[top  :  top  + final_height,
                              left :  left + final_width]
        return resize_img, resize_mask
    else:
        return resize_img

def generate_random_permutations(number_permutations=1000,n_tiles=9):
  p_hat = np.array(list(itertools.permutations(list(range(n_tiles)), n_tiles))) #all possible permutations

  for i in range(0,int(number_permutations/2)): 
    #1000 possible permutations at the end
    #each loop adds 2 random permutations with max hamming distance between them

    j = np.random.randint(0,p_hat.shape[0],size=1)
    p = p_hat[j] #one random permutation
    p_hat = np.delete(p_hat,j,axis=0) #remove permutation p, from position j from the queue

    D = cdist(p,p_hat, metric='hamming').mean(axis=0).flatten() #hamming distance between one random permutation p
                                                                #with respect to all other possible permutation

    max_dist_idx = np.array([D.argmax()])
    p2 = p_hat[max_dist_idx] #the permutation with highest hamming distance to p
    p_hat = np.delete(p_hat,max_dist_idx,axis=0) #remove permutation p, from position j from the queue
    
    if i == 0:
      final_permutations = p
      final_permutations = np.append(final_permutations,p2,axis=0)
    else:
      final_permutations = np.append(final_permutations,p,axis=0)
      final_permutations = np.append(final_permutations,p2,axis=0)
  print(f'permutatations shape: {final_permutations.shape}')
  return final_permutations

def create_jigsaw(image, possible_perm,size=(128,128),dataset='F3_netherlands', fix_perm=None, split='train'):
    if dataset == 'F3_netherlands':
        x1,x2 = 206,412 
        y1,y2 = 140,280
    
    elif dataset == 'Parihaka_NZPM':
        x1,x2 = 140,280 
        y1,y2 = 280,560

    #dataset == 'F3_Netherlands':
    #orig inline size: (462,951), here random cropped to before jigsaw: (450,600)
    #orig crossline size: (462,651) ->(450,600)  - mantem a proporcao 3:4
    # reduzidos a 90% do tamanho total disponivel por tile (150,200)
    #Parihaka_NZPM':
    #orig inline size: (782,1006) -> (450,600)  mantem a proporcao 3:4
    #orig crossline size: (590,1006) -> (450,600)
    assert split=='train' or split=='val' or split=='test', f"Not recognized split type {split}"

    tile1 = image[:y1, :x1]
    tile2 = image[:y1, x1:x2]
    tile3 = image[:y1, x2:]
    tile4 = image[y1:y2, :x1]
    tile5 = image[y1:y2, x1:x2]
    tile6 = image[y1:y2, x2:]
    tile7 = image[y2:, :x1]
    tile8 = image[y2:, x1:x2]
    tile9 = image[y2:, x2:]
    
    if split == 'train':
        tile1 = random_crop(tile1,final_size=size) 
        tile2 = random_crop(tile2,final_size=size) 
        tile3 = random_crop(tile3,final_size=size)
        tile4 = random_crop(tile4,final_size=size)
        tile5 = random_crop(tile5,final_size=size) 
        tile6 = random_crop(tile6,final_size=size) 
        tile7 = random_crop(tile7,final_size=size) 
        tile8 = random_crop(tile8,final_size=size) 
        tile9 = random_crop(tile9,final_size=size) 

    elif split == 'test' or split == 'val':
        tile1 = center_crop(tile1,final_height=size[0],final_width=size[1])
        tile2 = center_crop(tile2,final_height=size[0],final_width=size[1])
        tile3 = center_crop(tile3,final_height=size[0],final_width=size[1])
        tile4 = center_crop(tile4,final_height=size[0],final_width=size[1])
        tile5 = center_crop(tile5,final_height=size[0],final_width=size[1])
        tile6 = center_crop(tile6,final_height=size[0],final_width=size[1])
        tile7 = center_crop(tile7,final_height=size[0],final_width=size[1])
        tile8 = center_crop(tile8,final_height=size[0],final_width=size[1])
        tile9 = center_crop(tile9,final_height=size[0],final_width=size[1])


    else:
        raise ValueError(f'Not recognized split type {split}')

    dict_tiles = {0 : tile1, 1 : tile2, 2 : tile3,
              3 : tile4, 4 : tile5, 5 : tile6, 
              6 : tile7, 7 : tile8, 8 : tile9}

    if fix_perm != None:
        permutation = list(fix_perm)

    else:
        permutation_idx = np.random.randint(0,possible_perm.shape[0],size=1).item()
        permutation = list(possible_perm[permutation_idx])


    list_keys_orig = list(dict_tiles.keys())
    list_values_orig = list(dict_tiles.values())

    list_keys, list_values = [],[]
    for idx in permutation:
        list_keys.append(list_keys_orig[idx])
        list_values.append(list_values_orig[idx])

    #list_keys = [list_keys_orig[idx] for idx in permutation]
    #list_values = [list_values_orig[idx] for idx in permutation]

    return np.array(list_keys), np.array(list_values),np.array(list_keys_orig), np.array(list_values_orig),permutation_idx


def set_loss(task, device,weighted=False,mask=None):
    assert task == None or task == 'segmentation' or task == 'rotation' or task == 'hog' or task == 'jigsaw', f'Not recognized task: {task}'
    
    if task == None or task == 'segmentation' or task == 'rotation' or task == 'jigsaw':
        if weighted:
            assert mask!=None,"Must provide mask"
            #If provided, the optional argument weight should be a 1D Tensor assigning weight to each of the classes. 
            #This is particularly useful when you have an unbalanced training set.
            mask_t = mask.squeeze()
            vals,counts = np.unique(mask_t, return_counts=True)
            weights = 1/counts
            weights
            loss = nn.CrossEntropyLoss().to(device)
        else:
            loss = nn.CrossEntropyLoss().to(device)
        
    elif task =='hog':
        loss = nn.MSELoss().to(device)
        
    else:
        raise ValueError(f'Not recognized task: {task}')
        
    return loss


def save_rotation_log_results(saving_logs_path,name_model,
                            train_acc,train_acc_std,train_loss, train_loss_std,
                            val_acc, val_acc_std,val_loss, val_loss_std
                        ):
    
    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'train_acc'+'.txt'),  train_acc  )
    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'train_acc_std'+'.txt'),  train_acc_std  )
    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'train_loss'+'.txt'), train_loss)
    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'train_loss_std'+'.txt'), train_loss_std)

    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'val_acc'+'.txt'),   val_acc  )
    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'val_acc_std'+'.txt'),    val_acc_std  )
    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'val_loss'+'.txt'),    val_loss  )
    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'val_loss_std'+'.txt'), val_loss_std)


def save_segmentation_log_results(saving_logs_path,name_model,
    train_iou_classes_all, train_iou_std_between_classes_all, train_iou_mean_all, train_loss_mean_all, train_loss_std_all, 
    val_iou_classes_all,   val_iou_std_between_classes_all,   val_iou_mean_all,   val_loss_mean_all,   val_loss_std_all,
    #test_iou_classes_all,  test_iou_std_between_classes_all,  test_iou_mean_all,  test_loss_mean_all,  test_loss_std_all
    ):
    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'train_iou_classes_all'+'.txt'         )     , train_iou_classes_all    )
    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'train_iou_std_between_classes_all'+'.txt'         )     , train_iou_std_between_classes_all    )
    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'train_iou_mean_all'+'.txt'          )   , train_iou_mean_all       )
    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'train_loss_mean_all'+'.txt'     )  , train_loss_mean_all  )
    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'train_loss_std_all'+'.txt'          ) ,  train_loss_std_all      )

    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'val_iou_classes_all'+'.txt'           ), val_iou_classes_all       )
    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'val_iou_std_between_classes_all'+'.txt'         )     , val_iou_std_between_classes_all    )
    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'val_iou_mean_all'+'.txt'            ),  val_iou_mean_all      )
    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'val_loss_mean_all'+'.txt'  ) ,  val_loss_mean_all )
    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'val_loss_std_all'+'.txt'            ),  val_loss_std_all      )

    #np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'test_iou_classes_all'+'.txt'          ) ,  test_iou_classes_all       )
    #np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'test_iou_std_between_classes_all'+'.txt'         )     , test_iou_std_between_classes_all    )
    #np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'test_iou_mean_all'+'.txt'           ) ,  test_iou_mean_all       )
    #np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'test_loss_mean_all'+'.txt' ) ,  test_loss_mean_all  )
    #np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'test_loss_std_all'+'.txt'           ) ,  test_loss_std_all       )

def save_segmentation_log_few_shot(saving_logs_path,name_model,
    train_iou_classes_all, train_iou_std_between_classes_all, train_iou_mean_all, train_loss_mean_all, train_loss_std_all, 
    ):
    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'train_iou_classes_all'+'.txt'         )     , train_iou_classes_all    )
    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'train_iou_std_between_classes_all'+'.txt'         )     , train_iou_std_between_classes_all    )
    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'train_iou_mean_all'+'.txt'          )   , train_iou_mean_all       )
    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'train_loss_mean_all'+'.txt'     )  , train_loss_mean_all  )
    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'train_loss_std_all'+'.txt'          ) ,  train_loss_std_all      )


def save_hog_log_results(saving_logs_path,name_model,train_loss,train_std, 
                            val_loss,val_std, 
                            #test_loss,test_loss_std
                            ):
    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'train_loss'+'.txt'), train_loss)
    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'train_std'+'.txt'),  train_std  )

    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'val_loss'+'.txt'),   val_loss  )
    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'val_std'+'.txt'),    val_std  )

    #np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'test_loss'+'.txt'),  test_loss  )
    #np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'test_loss_std'+'.txt'),  test_loss_std  )


def save_jigsaw_log_results(saving_logs_path,name_model, 
            train_acc_mean,train_acc_std,train_loss_mean,train_loss_std,
            #test_acc_mean,test_acc_std,test_loss_mean,test_loss_std,
            val_acc_mean,val_acc_std,val_loss_mean,val_loss_std ):

    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'train_acc_mean'+'.txt'         )     , train_acc_mean    )
    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'train_acc_std'+'.txt'         )     , train_acc_std    )
    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'train_loss_mean'+'.txt'          )   , train_loss_mean       )
    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'train_loss_std'+'.txt'     )  , train_loss_std  )

    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'val_acc_mean'+'.txt'           ), val_acc_mean       )
    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'val_acc_std'+'.txt'         )     , val_acc_std    )
    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'val_loss_mean'+'.txt'            ),  val_loss_mean      )
    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'val_loss_std'+'.txt'  ) ,  val_loss_std )

    #np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'test_acc_mean'+'.txt'          ) ,  test_acc_mean       )
    #np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'test_acc_std'+'.txt'         )     , test_acc_std    )
    #np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'test_loss_mean'+'.txt'           ) ,  test_loss_mean       )
    #np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'test_loss_std'+'.txt' ) ,  test_loss_std  )

def get_grid_search():
    'Lr - Inicial   Lr - Final/Min  Optimizer   Schedulers'
    grid = {
    'grid_run_0' : [0.05000,   0.00072, 'SGD' ,'StepLR'],
    'grid_run_1' : [0.05000, 0.00072, 'SGD' ,'CosineAnnealingLR'],
    'grid_run_2' : [0.05000, 0.00072, 'Adam',    'StepLR'],
    'grid_run_3' : [0.05000, 0.00072, 'Adam',    'CosineAnnealingLR'],

    'grid_run_4' : [0.01000, 0.00014 ,'SGD' ,'StepLR'],
    'grid_run_5' : [0.01000, 0.00014 ,'SGD' ,'CosineAnnealingLR'],
    'grid_run_6' : [0.01000, 0.00014 ,'Adam',    'StepLR'],
    'grid_run_7' : [0.01000, 0.00014 ,'Adam',    'CosineAnnealingLR'],

    'grid_run_8' :  [0.00500 ,  0.000072,    'SGD' ,'StepLR'],
    'grid_run_9' : [0.00500,   0.000072,    'SGD' ,'CosineAnnealingLR'],
    'grid_run_10' : [0.00500,   0.000072,    'Adam',    'StepLR'],
    'grid_run_11' : [0.00500,   0.000072,    'Adam',    'CosineAnnealingLR'],

    'grid_run_12' : [0.00100   , 0.000014  ,  'SGD' ,'StepLR'],
    'grid_run_13' : [0.00100   , 0.000014  ,  'SGD' ,'CosineAnnealingLR'],
    'grid_run_14' : [0.00100   , 0.000014  ,  'Adam',    'StepLR'],
    'grid_run_15' : [0.00100   , 0.000014  ,  'Adam',    'CosineAnnealingLR'],

    'grid_run_16' : [0.00050 ,   0.000007  ,  'SGD' ,'StepLR'],
    'grid_run_17' : [0.00050 ,   0.000007  ,  'SGD' ,'CosineAnnealingLR'],
    'grid_run_18' : [0.00050 ,   0.000007  ,  'Adam',    'StepLR'],
    'grid_run_19' : [0.00050 ,   0.000007  ,  'Adam',    'CosineAnnealingLR'],
    }
    return grid

def get_grid_search_rot_parih():
    'Lr - Inicial   Lr - Final/Min  Optimizer   Schedulers'
    grid = {
    'grid_run_0' : [0.01000, 0.00014 ,'Adam',    'StepLR'],#grid_run_6
    'grid_run_1' : [0.00500,   0.000072,    'Adam',    'StepLR'],#grid_run_10
    'grid_run_2' : [0.00100   , 0.000014  ,  'Adam',    'StepLR'],#grid_run_14
    'grid_run_3' : [0.00050 ,   0.000007  ,  'Adam',    'StepLR'],#grid_run_18
    'grid_run_4' : [0.00050 ,   0.000007  ,  'Adam',    'CosineAnnealingLR'],#grid_run_19
    'grid_run_5' : [0.000050 ,   0.0000007  ,  'Adam',    'StepLR'],#grid_run_20
    'grid_run_6' : [0.000050 ,   0.0000007  ,  'Adam',    'CosineAnnealingLR'],#grid_run_21
    }
    return grid

def get_grid_search_finetune_jigsaw_f3():
    'Model -  Lrbackbone   Lrsegmenter - %train just for record'
    grid = {
        'finetune_grid_0' : ['grid_run_9', 0.001,   0.01, 0.01],
        'finetune_grid_1' : ['grid_run_9', 0.0005,   0.005, 0.01],
        'finetune_grid_2' : ['grid_run_9', 0.0001,   0.001, 0.01],
        'finetune_grid_3' : ['grid_run_9', 0.0,   0.001, 0.01],

        'finetune_grid_4' : ['grid_run_6', 0.001,   0.01, 0.01],
        'finetune_grid_5' : ['grid_run_6', 0.0005,   0.005, 0.01],
        'finetune_grid_6' : ['grid_run_6', 0.0001,   0.001, 0.01],
        'finetune_grid_7' : ['grid_run_6', 0.0,   0.001, 0.01],

        'finetune_grid_8' : ['grid_run_15', 0.001,   0.01, 0.01],
        'finetune_grid_9' : ['grid_run_15', 0.0005,   0.005, 0.01],
        'finetune_grid_10' : ['grid_run_15', 0.0001,   0.001, 0.01],
        'finetune_grid_11' : ['grid_run_15',0.0,   0.001 , 0.01],

        'finetune_grid_12' : ['grid_run_1', 0.001,   0.01, 0.01],
        'finetune_grid_13' : ['grid_run_1',0.0005,   0.005 , 0.01],
        'finetune_grid_14' : ['grid_run_1', 0.0001,   0.001, 0.01],
        'finetune_grid_15' : ['grid_run_1',0.0,   0.001 , 0.01],

        'finetune_grid_16' : ['grid_run_0', 0.001,   0.01, 0.01],
        'finetune_grid_17' : ['grid_run_0', 0.0005,   0.005, 0.01],
        'finetune_grid_18' : ['grid_run_0', 0.0001,   0.001, 0.01],
        'finetune_grid_19' : ['grid_run_0', 0.0,   0.001, 0.01],

        'finetune_grid_20' : ['grid_run_11', 0.001,   0.01, 0.01],
        'finetune_grid_21' : ['grid_run_11',0.0005,   0.005 , 0.01],
        'finetune_grid_22' : ['grid_run_11',0.0001,   0.001 , 0.01],
        'finetune_grid_23' : ['grid_run_11',0.0,   0.001 , 0.01],

        'finetune_grid_24' : ['grid_run_5', 0.001,   0.01, 0.01],
        'finetune_grid_25' : ['grid_run_5', 0.0005,   0.005, 0.01],
        'finetune_grid_26' : ['grid_run_5', 0.0001,   0.001, 0.01],
        'finetune_grid_27' : ['grid_run_5', 0.0,   0.001, 0.01],

        'finetune_grid_28' : ['grid_run_7', 0.001,   0.01, 0.01],
        'finetune_grid_29' : ['grid_run_7', 0.0005,   0.005, 0.01],
        'finetune_grid_30' : ['grid_run_7', 0.0001,   0.001, 0.01],
        'finetune_grid_31' : ['grid_run_7', 0.0,   0.001, 0.01],

        'finetune_grid_32' : ['grid_run_8', 0.001,   0.01, 0.01],
        'finetune_grid_33' : ['grid_run_8', 0.0005,   0.005, 0.01],
        'finetune_grid_34' : ['grid_run_8', 0.0001,   0.001, 0.01],
        'finetune_grid_35' : ['grid_run_8', 0.0,   0.001, 0.01],

        }
    return grid

def get_grid_search_finetune_jigsaw_parih():
    'Model -  Lrbackbone   Lrsegmenter - %train just for record'
    grid = {
        'finetune_grid_0' : ['grid_run_5', 0.001,   0.01, 0.01],
        'finetune_grid_1' : ['grid_run_5', 0.0005,   0.005, 0.01],
        'finetune_grid_2' : ['grid_run_5', 0.0001,   0.001, 0.01],
        'finetune_grid_3' : ['grid_run_5', 0.0,   0.001, 0.01],

        'finetune_grid_4' : ['grid_run_4', 0.001,   0.01, 0.01],
        'finetune_grid_5' : ['grid_run_4', 0.0005,   0.005, 0.01],
        'finetune_grid_6' : ['grid_run_4', 0.0001,   0.001, 0.01],
        'finetune_grid_7' : ['grid_run_4', 0.0,   0.001, 0.01],

        'finetune_grid_8' : ['grid_run_8', 0.001,   0.01, 0.01],
        'finetune_grid_9' : ['grid_run_8', 0.0005,   0.005, 0.01],
        'finetune_grid_10' : ['grid_run_8', 0.0001,   0.001, 0.01],
        'finetune_grid_11' : ['grid_run_8',0.0,   0.001 , 0.01],

        'finetune_grid_12' : ['grid_run_6', 0.001,   0.01, 0.01],
        'finetune_grid_13' : ['grid_run_6',0.0005,   0.005 , 0.01],
        'finetune_grid_14' : ['grid_run_6', 0.0001,   0.001, 0.01],
        'finetune_grid_15' : ['grid_run_6',0.0,   0.001 , 0.01],

        'finetune_grid_16' : ['grid_run_7', 0.001,   0.01, 0.01],
        'finetune_grid_17' : ['grid_run_7', 0.0005,   0.005, 0.01],
        'finetune_grid_18' : ['grid_run_7', 0.0001,   0.001, 0.01],
        'finetune_grid_19' : ['grid_run_7', 0.0,   0.001, 0.01],

        'finetune_grid_20' : ['grid_run_1', 0.001,   0.01, 0.01],
        'finetune_grid_21' : ['grid_run_1',0.0005,   0.005 , 0.01],
        'finetune_grid_22' : ['grid_run_1',0.0001,   0.001 , 0.01],
        'finetune_grid_23' : ['grid_run_1',0.0,   0.001 , 0.01],

        'finetune_grid_24' : ['grid_run_2', 0.001,   0.01, 0.01],
        'finetune_grid_25' : ['grid_run_2', 0.0005,   0.005, 0.01],
        'finetune_grid_26' : ['grid_run_2', 0.0001,   0.001, 0.01],
        'finetune_grid_27' : ['grid_run_2', 0.0,   0.001, 0.01],

        'finetune_grid_28' : ['grid_run_11', 0.001,   0.01, 0.01],
        'finetune_grid_29' : ['grid_run_11', 0.0005,   0.005, 0.01],
        'finetune_grid_30' : ['grid_run_11', 0.0001,   0.001, 0.01],
        'finetune_grid_31' : ['grid_run_11', 0.0,   0.001, 0.01],

        }
    return grid

def get_grid_search_finetune_rotation_f3():
    'Model -  Lrbackbone   Lrsegmenter - %train just for record'
    grid = {
        'finetune_grid_0' : ['grid_run_14', 0.001,   0.01, 0.01],
        'finetune_grid_1' : ['grid_run_14', 0.0005,   0.005, 0.01],
        'finetune_grid_2' : ['grid_run_14', 0.0001,   0.001, 0.01],
        'finetune_grid_3' : ['grid_run_14', 0.0,   0.001, 0.01],

        'finetune_grid_4' : ['grid_run_10', 0.001,   0.01, 0.01],
        'finetune_grid_5' : ['grid_run_10', 0.0005,   0.005, 0.01],
        'finetune_grid_6' : ['grid_run_10', 0.0001,   0.001, 0.01],
        'finetune_grid_7' : ['grid_run_10', 0.0,   0.001, 0.01],

        'finetune_grid_8' : ['grid_run_15', 0.001,   0.01, 0.01],
        'finetune_grid_9' : ['grid_run_15', 0.0005,   0.005, 0.01],
        'finetune_grid_10' : ['grid_run_15', 0.0001,   0.001, 0.01],
        'finetune_grid_11' : ['grid_run_15',0.0,   0.001 , 0.01],

        'finetune_grid_12' : ['grid_run_6', 0.001,   0.01, 0.01],
        'finetune_grid_13' : ['grid_run_6',0.0005,   0.005 , 0.01],
        'finetune_grid_14' : ['grid_run_6', 0.0001,   0.001, 0.01],
        'finetune_grid_15' : ['grid_run_6',0.0,   0.001 , 0.01],

        'finetune_grid_16' : ['grid_run_19', 0.001,   0.01, 0.01],
        'finetune_grid_17' : ['grid_run_19', 0.0005,   0.005, 0.01],
        'finetune_grid_18' : ['grid_run_19', 0.0001,   0.001, 0.01],
        'finetune_grid_19' : ['grid_run_19', 0.0,   0.001, 0.01],

        'finetune_grid_20' : ['grid_run_11', 0.001,   0.01, 0.01],
        'finetune_grid_21' : ['grid_run_11',0.0005,   0.005 , 0.01],
        'finetune_grid_22' : ['grid_run_11',0.0001,   0.001 , 0.01],
        'finetune_grid_23' : ['grid_run_11',0.0,   0.001 , 0.01],

        'finetune_grid_24' : ['grid_run_18', 0.001,   0.01, 0.01],
        'finetune_grid_25' : ['grid_run_18', 0.0005,   0.005, 0.01],
        'finetune_grid_26' : ['grid_run_18', 0.0001,   0.001, 0.01],
        'finetune_grid_27' : ['grid_run_18', 0.0,   0.001, 0.01],

        'finetune_grid_28' : ['grid_run_7', 0.001,   0.01, 0.01],
        'finetune_grid_29' : ['grid_run_7', 0.0005,   0.005, 0.01],
        'finetune_grid_30' : ['grid_run_7', 0.0001,   0.001, 0.01],
        'finetune_grid_31' : ['grid_run_7', 0.0,   0.001, 0.01],

        }
    return grid

def get_grid_search_finetune_rotation_parih():
    'Model -  Lrbackbone   Lrsegmenter - %train just for record'
    grid = {
        'finetune_grid_0' : ['second_grid_run_2', 0.001,   0.01, 0.01],
        'finetune_grid_1' : ['second_grid_run_2', 0.0005,   0.005, 0.01],
        'finetune_grid_2' : ['second_grid_run_2', 0.0001,   0.001, 0.01],
        'finetune_grid_3' : ['second_grid_run_2', 0.0,   0.001, 0.01],

        'finetune_grid_4' : ['second_grid_run_3', 0.001,   0.01, 0.01],
        'finetune_grid_5' : ['second_grid_run_3', 0.0005,   0.005, 0.01],
        'finetune_grid_6' : ['second_grid_run_3', 0.0001,   0.001, 0.01],
        'finetune_grid_7' : ['second_grid_run_3', 0.0,   0.001, 0.01],

        'finetune_grid_8' : ['first_grid_run_10', 0.001,   0.01, 0.01],
        'finetune_grid_9' : ['first_grid_run_10', 0.0005,   0.005, 0.01],
        'finetune_grid_10' : ['first_grid_run_10', 0.0001,   0.001, 0.01],
        'finetune_grid_11' : ['first_grid_run_10',0.0,   0.001 , 0.01],

        'finetune_grid_12' : ['first_grid_run_14', 0.001,   0.01, 0.01],
        'finetune_grid_13' : ['first_grid_run_14',0.0005,   0.005 , 0.01],
        'finetune_grid_14' : ['first_grid_run_14', 0.0001,   0.001, 0.01],
        'finetune_grid_15' : ['first_grid_run_14',0.0,   0.001 , 0.01],

        'finetune_grid_16' : ['first_grid_run_18', 0.001,   0.01, 0.01],
        'finetune_grid_17' : ['first_grid_run_18', 0.0005,   0.005, 0.01],
        'finetune_grid_18' : ['first_grid_run_18', 0.0001,   0.001, 0.01],
        'finetune_grid_19' : ['first_grid_run_18', 0.0,   0.001, 0.01],

        }
    return grid

def get_grid_search_baseline():
    ' Lrbackbone   Lrsegmenter - %train just for record'
    grid = {
        'base_grid_0' : [0.001,   0.01, 0.01],
        'base_grid_1' : [0.0005,   0.005, 0.01],
        'base_grid_2' : [0.0001,   0.001, 0.01],
        'base_grid_3' : [0.001,   0.001, 0.01],
        }
    return grid




def window_inline_f3(pred0,pred1,pred2,pred3,pred4, mode='mode', plot=False):
    assert mode == 'mean' or mode=='mode', f'Not recognized calculation mode:{mode}'

    pos0 = (0,112)
    pos1 = (112,224)
    pos2 = (224,336)
    pos3 = (336,448)
    pos4 = (448,560)
    pos5 = (560,672)
    pos6 = (672,784)
    pos7 = (784,896)

    final = np.zeros(shape=(448,896))
    assert final.shape==(448,896), "Wrong shape"
    if mode == 'mode':    
        for row in range(final.shape[0]):
            for column in range(final.shape[1]):
                if column < pos0[1]: #0-112 - pred0
                    x =  pred0[row][column-pos0[0]]
                    vals,counts = np.unique(x, return_counts=True)
                    final[row][column] = vals[counts.argmax()]

                elif column >=pos1[0] and column <pos1[1]: #112-224 pred0,pred1
                    x =  pred0[row][column-pos0[0]], pred1[row][column-pos1[0]]
                    vals,counts = np.unique(x, return_counts=True)
                    final[row][column] = vals[counts.argmax()]

                elif column >=pos2[0] and column <pos2[1]: #224-336 pred0,pred1,pred2
                    x =  pred0[row][column-pos0[0]], pred1[row][column-pos1[0]], pred2[row][column-pos2[0]]
                    vals,counts = np.unique(x, return_counts=True)
                    final[row][column] = vals[counts.argmax()]

                elif column >=pos3[0] and column <pos3[1]: #336-448 pred0,pred1,pred2,pred3
                    x =  pred0[row][column-pos0[0]] , pred1[row][column-pos1[0]], pred2[row][column-pos2[0]], pred3[row][column-pos3[0]]
                    vals,counts = np.unique(x, return_counts=True)
                    final[row][column] = vals[counts.argmax()]

                elif column >=pos4[0] and column <pos4[1]: #448-560 pred1,pred2,pred3,pred4
                    x = final[row][column], pred1[row][column-pos1[0]] , pred2[row][column-pos2[0]], pred3[row][column-pos3[0]], pred4[row][column-pos4[0]]
                    vals,counts = np.unique(x, return_counts=True)
                    final[row][column] = vals[counts.argmax()]

                elif column >=pos5[0] and column <pos5[1]: #560-672 pred2,pred3,pred4
                    x =  pred2[row][column-pos2[0]], pred3[row][column-pos3[0]], pred4[row][column-pos4[0]]
                    vals,counts = np.unique(x, return_counts=True)
                    final[row][column] = vals[counts.argmax()]

                elif column >=pos6[0] and column <pos6[1]: #672-784 pred3,pred4
                    x =  pred3[row][column-pos3[0]] , pred4[row][column-pos4[0]]
                    vals,counts = np.unique(x, return_counts=True)
                    final[row][column] = vals[counts.argmax()]

                elif column >=pos7[0] and column <pos7[1]: #784-896 pred4
                    x =  pred4[row][column-pos4[0]]
                    vals,counts = np.unique(x, return_counts=True)
                    final[row][column] = vals[counts.argmax()]
    elif mode =='mean':
        for row in range(final.shape[0]):
            for column in range(final.shape[1]):
                if column < pos0[1]: #0-112 - pred0
                    x =  pred0[row][column-pos0[0]]
                    final[row][column] = np.mean(x)
                    
                elif column >=pos1[0] and column <pos1[1]: #112-224 pred0,pred1
                    x =  pred0[row][column-pos0[0]], pred1[row][column-pos1[0]]
                    final[row][column] = np.mean(x)
                    
                elif column >=pos2[0] and column <pos2[1]: #224-336 pred0,pred1,pred2
                    x =  pred0[row][column-pos0[0]], pred1[row][column-pos1[0]], pred2[row][column-pos2[0]]
                    final[row][column] = np.mean(x)
                    
                elif column >=pos3[0] and column <pos3[1]: #336-448 pred0,pred1,pred2,pred3
                    x =  pred0[row][column-pos0[0]] , pred1[row][column-pos1[0]], pred2[row][column-pos2[0]], pred3[row][column-pos3[0]]
                    final[row][column] = np.mean(x)
                    
                elif column >=pos4[0] and column <pos4[1]: #448-560 pred1,pred2,pred3,pred4
                    x =  pred1[row][column-pos1[0]] , pred2[row][column-pos2[0]], pred3[row][column-pos3[0]], pred4[row][column-pos4[0]]
                    final[row][column] = np.mean(x)
                    
                elif column >=pos5[0] and column <pos5[1]: #560-672 pred2,pred3,pred4
                    x =  pred2[row][column-pos2[0]], pred3[row][column-pos3[0]], pred4[row][column-pos4[0]]
                    final[row][column] = np.mean(x)
                    
                elif column >=pos6[0] and column <pos6[1]: #672-784 pred3,pred4
                    x =  pred3[row][column-pos3[0]] , pred4[row][column-pos4[0]]
                    final[row][column] = np.mean(x)
                    
                elif column >=pos7[0] and column <pos7[1]: #784-896 pred4
                    x =  pred4[row][column-pos4[0]]
                    final[row][column] = np.mean(x)

    if plot:
        colors = ["gold", "lawngreen", "lightseagreen", "orange", "blue", "sienna", "violet", "lightseagreen", "darkorange","red"]
        cmap_seismic = LinearSegmentedColormap.from_list("mycmap", colors)
        fig, ax = plt.subplots(1, 1, figsize=(9, 4))
        m = ax.imshow(final, cmap=cmap_seismic,alpha=0.7,vmin=0,vmax=len(colors)-1)
        plt.colorbar(m)
        plt.show()
    return final


def window_crossline_f3(pred0,pred1,pred2, mode='mean', plot=False):
    assert mode == 'mean' or mode=='mode', f'Not recognized calculation mode:{mode}'

    pos0 = (0,76) #0
    pos1 = (76,152) #0,1
    pos2 = (152,448) # 0,1,2
    pos3 = (448,524) #1,2
    pos4 = (524,600) #2

    final = np.zeros(shape=(448,600))
    assert final.shape==(448,600), "Wrong shape"
    if mode == 'mode':
        for row in range(final.shape[0]):
            for column in range(final.shape[1]):
                if column < pos0[1]: #0-76 - pred0
                    x =  pred0[row][column-pos0[0]]
                    vals,counts = np.unique(x, return_counts=True)
                    final[row][column] = vals[counts.argmax()]
                    
                elif column >=pos1[0] and column <pos1[1]: #76-152 pred0,pred1
                    x =  pred0[row][column-pos0[0]], pred1[row][column-pos1[0]]
                    vals,counts = np.unique(x, return_counts=True)
                    final[row][column] = vals[counts.argmax()]
                    
                elif column >=pos2[0] and column <pos2[1]: #152-448 pred0,pred1,pred2
                    x =  pred0[row][column-pos0[0]], pred1[row][column-pos1[0]], pred2[row][column-pos2[0]]
                    vals,counts = np.unique(x, return_counts=True)
                    final[row][column] = vals[counts.argmax()]
                    
                elif column >=pos3[0] and column <pos3[1]: #448-524 pred1,pred2
                    x = pred1[row][column-pos1[0]] ,pred2[row][column-pos2[0]]
                    vals,counts = np.unique(x, return_counts=True)
                    final[row][column] = vals[counts.argmax()]
                    
                elif column >=pos4[0] and column <pos4[1]: #524-600 pred2
                    x =   pred2[row][column-pos2[0]]
                    vals,counts = np.unique(x, return_counts=True)
                    final[row][column] = vals[counts.argmax()]
    elif mode == 'mean':
        for row in range(final.shape[0]):
            for column in range(final.shape[1]):
                if column < pos0[1]: #0-76 - pred0
                    x =  pred0[row][column-pos0[0]]
                    final[row][column] = np.mean(x)
                    
                elif column >=pos1[0] and column <pos1[1]: #76-152 pred0,pred1
                    x =  pred0[row][column-pos0[0]], pred1[row][column-pos1[0]]
                    final[row][column] = np.mean(x)
                    
                elif column >=pos2[0] and column <pos2[1]: #152-448 pred0,pred1,pred2
                    x =  pred0[row][column-pos0[0]], pred1[row][column-pos1[0]], pred2[row][column-pos2[0]]
                    final[row][column] = np.mean(x)
                    
                elif column >=pos3[0] and column <pos3[1]: #448-524 pred1,pred2
                    x =  pred1[row][column-pos1[0]] ,pred2[row][column-pos2[0]]
                    final[row][column] = np.mean(x)
                    
                elif column >=pos4[0] and column <pos4[1]: #524-600 pred2
                    x =   pred2[row][column-pos2[0]]
                    final[row][column] = np.mean(x)



    if plot:
        colors = ["gold", "lawngreen", "lightseagreen", "orange", "blue", "sienna", "violet", "lightseagreen", "darkorange","red"]
        cmap_seismic = LinearSegmentedColormap.from_list("mycmap", colors)
        fig, ax = plt.subplots(1, 1, figsize=(9, 4))
        m = ax.imshow(final, cmap=cmap_seismic,alpha=0.7,vmin=0,vmax=len(colors)-1)
        plt.colorbar(m)
        plt.show()


    return final

def make_cross_val_folds_f3(sections_list,masks_list,sec_number_list,fold,split,fold_size=0.2):
    assert fold >= 0 and fold <= 4, f'Not Valid Fold: {fold}'
    assert fold_size >=0 and fold_size<=1.0, f'Invalid Fold size: {fold_size}'

    limit = int(np.round((len(sections_list) * fold_size),0))

    if fold ==0: #train 0:60, test 60-80, val 80-100
        if split=='train':
            sections_list = sections_list[:limit*3]
            masks_list = masks_list[:limit*3]
            sec_number_list = sec_number_list[:limit*3]
        elif split=='test':   
            sections_list = sections_list[limit*3:limit*4]
            masks_list = masks_list[limit*3:limit*4]
            sec_number_list = sec_number_list[limit*3:limit*4]
        elif split=='val':
            sections_list = sections_list[limit*4:]
            masks_list = masks_list[limit*4:]
            sec_number_list = sec_number_list[limit*4:]

    elif fold ==1: #train 0:40 e 80-100, test 40-60, val 60-80
        if split=='train':
            sections_list = sections_list[:limit*2] + sections_list[limit*4:]
            masks_list = masks_list[:limit*2] + masks_list[limit*4:]
            sec_number_list = sec_number_list[:limit*2] + sec_number_list[limit*4:]
        elif split=='test':   
            sections_list = sections_list[limit*2:limit*3]
            masks_list = masks_list[limit*2:limit*3]
            sec_number_list = sec_number_list[limit*2:limit*3]
        elif split=='val':
            sections_list = sections_list[limit*3:limit*4]
            masks_list = masks_list[limit*3:limit*4]
            sec_number_list = sec_number_list[limit*3:limit*4]

    elif fold ==2: #train 0:20 e 60-100, test 20-40, val 40-60
        if split=='train':
            sections_list = sections_list[:limit] + sections_list[limit*3:]
            masks_list = masks_list[:limit] + masks_list[limit*3:]
            sec_number_list = sec_number_list[:limit] + sec_number_list[limit*3:]
        elif split=='test':   
            sections_list = sections_list[limit:limit*2]
            masks_list = masks_list[limit:limit*2]
            sec_number_list = sec_number_list[limit:limit*2]
        elif split=='val':
            sections_list = sections_list[limit*2:limit*3]
            masks_list = masks_list[limit*2:limit*3]
            sec_number_list = sec_number_list[limit*2:limit*3]
    
    elif fold ==3: #train 20:80, test 80-100, val 0-20
        if split=='train':
            sections_list = sections_list[limit:limit*4]
            masks_list = masks_list[limit:limit*4]
            sec_number_list = sec_number_list[limit:limit*4]
        elif split=='test':   
            sections_list = sections_list[limit*4:]
            masks_list = masks_list[limit*4:]
            sec_number_list = sec_number_list[limit*4:]
        elif split=='val':
            sections_list = sections_list[:limit]
            masks_list = masks_list[:limit]
            sec_number_list = sec_number_list[:limit]

    elif fold ==4: #train 40-100, test 0-20, val 20-40
        if split=='train':
            sections_list = sections_list[limit*2:]
            masks_list = masks_list[limit*2:] 
            sec_number_list = sec_number_list[limit*2:]
        elif split=='test':   
            sections_list = sections_list[:limit]
            masks_list = masks_list[:limit]
            sec_number_list = sec_number_list[:limit]
        elif split=='val':
            sections_list = sections_list[limit:limit*2]
            masks_list = masks_list[limit:limit*2]
            sec_number_list = sec_number_list[limit:limit*2]


    return sections_list,masks_list,sec_number_list


def verify_folds_f3(split, fold, sec_number_list):

    if fold ==0:
        if split == 'train':
            sections = ['inline_607', 'crossline_1076', 'inline_178', 'crossline_724'] + ['crossline_619','crossline_995', 'crossline_1133', 'inline_293']
        if split == 'val':
            sections = ['inline_240', 'inline_670', 'crossline_610','crossline_722',]+['inline_462','crossline_924', 'inline_627', 'inline_294']
        if split == 'test':
            sections = ['crossline_887', 'crossline_472', 'crossline_432', 'crossline_571']+['crossline_540', 'crossline_1157','inline_363', 'crossline_1016']
        assert sec_number_list[:4] + sec_number_list[-4:] == sections, f"Selected fold does not match with the pre-set"
    
    if fold ==1:
        if split == 'train':
            sections = ['inline_607', 'crossline_1076', 'inline_178', 'crossline_724']+['inline_462','crossline_924', 'inline_627', 'inline_294']
        if split == 'val':
            sections = ['crossline_887', 'crossline_472', 'crossline_432', 'crossline_571']+['crossline_540', 'crossline_1157','inline_363', 'crossline_1016']
        if split == 'test':
            sections = ['crossline_611', 'inline_255', 'crossline_1039', 'crossline_606',]+['crossline_619', 'crossline_995', 'crossline_1133', 'inline_293']
        assert sec_number_list[:4] + sec_number_list[-4:] == sections, f"Selected fold does not match with the pre-set"
    
    if fold ==2:
        if split == 'train':
            sections = ['inline_607', 'crossline_1076', 'inline_178', 'crossline_724']+['inline_462','crossline_924', 'inline_627', 'inline_294']+['crossline_1149', 'inline_316', 'crossline_638', 'crossline_1050']
            assert sec_number_list[:4] + sec_number_list[-4:]+ sec_number_list[549:553] == sections, f"Selected fold does not match with the pre-set"
        if split == 'val':
            sections = ['crossline_611', 'inline_255', 'crossline_1039', 'crossline_606',]+['crossline_619', 'crossline_995', 'crossline_1133', 'inline_293']
            assert sec_number_list[:4] + sec_number_list[-4:] == sections, f"Selected fold does not match with the pre-set"
        if split == 'test':
            sections = ['inline_643', 'crossline_1196', 'crossline_816', 'inline_451']+[ 'crossline_531', 'crossline_385', 'crossline_1005', 'inline_165']
            assert sec_number_list[:4] + sec_number_list[-4:] == sections, f"Selected fold does not match with the pre-set"

    if fold ==3:
        if split == 'train':
            sections = ['inline_643', 'crossline_1196', 'crossline_816', 'inline_451']+['crossline_540', 'crossline_1157','inline_363', 'crossline_1016']
        if split == 'val':
            sections = ['inline_607', 'crossline_1076', 'inline_178', 'crossline_724']+ ['inline_342', 'crossline_900', 'crossline_919','crossline_1233']
        if split == 'test':
            sections = ['inline_240', 'inline_670', 'crossline_610','crossline_722',]+['inline_462','crossline_924', 'inline_627', 'inline_294']
        assert sec_number_list[:4] + sec_number_list[-4:] == sections, f"Selected fold does not match with the pre-set"

    if fold ==4:
        if split == 'train':
            sections = ['crossline_611', 'inline_255', 'crossline_1039', 'crossline_606',] + ['inline_462','crossline_924', 'inline_627', 'inline_294']
        if split == 'val':
            sections = ['inline_643', 'crossline_1196', 'crossline_816', 'inline_451']+[ 'crossline_531', 'crossline_385', 'crossline_1005', 'inline_165']
        if split == 'test':
            sections = ['inline_607', 'crossline_1076', 'inline_178', 'crossline_724']+ ['inline_342', 'crossline_900', 'crossline_919','crossline_1233']
        assert sec_number_list[:4] + sec_number_list[-4:] == sections, f"Selected fold does not match with the pre-set"


def make_cross_val_folds_parih(sections_list,sec_number_list,fold,split,fold_size=0.2):
    assert fold >= 0 and fold <= 4, f'Not Valid Fold: {fold}'
    assert fold_size >=0 and fold_size<=1.0, f'Invalid Fold size: {fold_size}'

    limit = int(np.round((len(sections_list) * fold_size),0))

    if fold ==0: #train 0:60, test 60-80, val 80-100
        if split=='train':
            sections_list = sections_list[:limit*3]
            sec_number_list = sec_number_list[:limit*3]
        elif split=='test':   
            sections_list = sections_list[limit*3:limit*4]
            sec_number_list = sec_number_list[limit*3:limit*4]
        elif split=='val':
            sections_list = sections_list[limit*4:]
            sec_number_list = sec_number_list[limit*4:]

    elif fold ==1: #train 0:40 e 80-100, test 40-60, val 60-80
        if split=='train':
            sections_list = sections_list[:limit*2] + sections_list[limit*4:]
            sec_number_list = sec_number_list[:limit*2] + sec_number_list[limit*4:]
        elif split=='test':   
            sections_list = sections_list[limit*2:limit*3]
            sec_number_list = sec_number_list[limit*2:limit*3]
        elif split=='val':
            sections_list = sections_list[limit*3:limit*4]
            sec_number_list = sec_number_list[limit*3:limit*4]

    elif fold ==2: #train 0:20 e 60-100, test 20-40, val 40-60
        if split=='train':
            sections_list = sections_list[:limit] + sections_list[limit*3:]
            sec_number_list = sec_number_list[:limit] + sec_number_list[limit*3:]
        elif split=='test':   
            sections_list = sections_list[limit:limit*2]
            sec_number_list = sec_number_list[limit:limit*2]
        elif split=='val':
            sections_list = sections_list[limit*2:limit*3]
            sec_number_list = sec_number_list[limit*2:limit*3]
    
    elif fold ==3: #train 20:80, test 80-100, val 0-20
        if split=='train':
            sections_list = sections_list[limit:limit*4]
            sec_number_list = sec_number_list[limit:limit*4]
        elif split=='test':   
            sections_list = sections_list[limit*4:]
            sec_number_list = sec_number_list[limit*4:]
        elif split=='val':
            sections_list = sections_list[:limit]
            sec_number_list = sec_number_list[:limit]

    elif fold ==4: #train 40-100, test 0-20, val 20-40
        if split=='train':
            sections_list = sections_list[limit*2:]
            sec_number_list = sec_number_list[limit*2:]
        elif split=='test':   
            sections_list = sections_list[:limit]
            sec_number_list = sec_number_list[:limit]
        elif split=='val':
            sections_list = sections_list[limit:limit*2]
            sec_number_list = sec_number_list[limit:limit*2]


    return sections_list,sec_number_list

def verify_folds_parih(split, fold, sec_number_list):

    if fold ==0:
        if split == 'train':
            sections = ['Inline_65', 'Inline_435', 'Inline_310', 'Inline_499',] + [ 'Inline_272', 'Inline_211', 'Crossline_65', 'Inline_204']
        if split == 'val':
            sections = ['Inline_379', 'Crossline_660', 'Inline_359', 'Inline_144']+['Crossline_53', 'Crossline_161', 'Crossline_314', 'Crossline_103']
        if split == 'test':
            sections = ['Inline_525', 'Inline_276', 'Crossline_127', 'Crossline_128',]+[ 'Inline_288', 'Crossline_597', 'Crossline_696', 'Crossline_358']
        assert sec_number_list[:4] + sec_number_list[-4:] == sections, f"Selected fold does not match with the pre-set"
    
    if fold ==1:
        if split == 'train':
            sections = ['Inline_65', 'Inline_435', 'Inline_310', 'Inline_499',]+['Crossline_53', 'Crossline_161', 'Crossline_314', 'Crossline_103']
        if split == 'val':
            sections =  ['Inline_525', 'Inline_276', 'Crossline_127', 'Crossline_128',]+[ 'Inline_288', 'Crossline_597', 'Crossline_696', 'Crossline_358']
        if split == 'test':
            sections = ['Inline_299', 'Inline_119', 'Crossline_426', 'Crossline_165']+[ 'Inline_272', 'Inline_211', 'Crossline_65', 'Inline_204']
        assert sec_number_list[:4] + sec_number_list[-4:] == sections, f"Selected fold does not match with the pre-set"
    
    if fold ==2:
        if split == 'train':
            sections = ['Inline_65', 'Inline_435', 'Inline_310', 'Inline_499',] +['Crossline_53', 'Crossline_161', 'Crossline_314', 'Crossline_103']+['Crossline_630', 'Crossline_577', 'Inline_555', 'Crossline_16']
            assert sec_number_list[:4] + sec_number_list[-4:]+ sec_number_list[495:499]
        if split == 'val':
            sections =  ['Inline_299', 'Inline_119', 'Crossline_426', 'Crossline_165']+[ 'Inline_272', 'Inline_211', 'Crossline_65', 'Inline_204']
            assert sec_number_list[:4] + sec_number_list[-4:] == sections, f"Selected fold does not match with the pre-set"
        if split == 'test':
            sections = ['Crossline_234', 'Inline_161', 'Inline_419', 'Crossline_427',]+['Inline_569', 'Crossline_323', 'Inline_202', 'Inline_514']
            assert sec_number_list[:4] + sec_number_list[-4:] == sections, f"Selected fold does not match with the pre-set"

    if fold ==3:
        if split == 'train':
            sections = ['Crossline_234', 'Inline_161', 'Inline_419', 'Crossline_427',]+['Inline_288', 'Crossline_597', 'Crossline_696', 'Crossline_358']
        if split == 'val':
            sections = ['Inline_65', 'Inline_435', 'Inline_310', 'Inline_499',]+ ['Crossline_252', 'Crossline_50', 'Crossline_583', 'Crossline_444']
        if split == 'test':
            sections = ['Inline_379', 'Crossline_660', 'Inline_359', 'Inline_144']+['Crossline_53', 'Crossline_161', 'Crossline_314', 'Crossline_103']
        assert sec_number_list[:4] + sec_number_list[-4:] == sections, f"Selected fold does not match with the pre-set"

    if fold ==4:
        if split == 'train':
            sections = ['Inline_299', 'Inline_119', 'Crossline_426', 'Crossline_165'] + ['Crossline_53', 'Crossline_161', 'Crossline_314', 'Crossline_103']
        if split == 'val':
            sections = ['Crossline_234', 'Inline_161', 'Inline_419', 'Crossline_427',]+['Inline_569', 'Crossline_323', 'Inline_202', 'Inline_514']
        if split == 'test':
            sections = ['Inline_65', 'Inline_435', 'Inline_310', 'Inline_499',] + ['Crossline_252', 'Crossline_50', 'Crossline_583', 'Crossline_444']
        assert sec_number_list[:4] + sec_number_list[-4:] == sections, f"Selected fold does not match with the pre-set"


def window_inline_parih(pred0,pred1,pred2,pred3,pred4, mode='mode', plot=False):
    assert mode == 'mean' or mode=='mode', f'Not recognized calculation mode:{mode}'
    #pred shape= 832,448
    pos0 = (0,112)
    pos1 = (112,224)
    pos2 = (224,336)
    pos3 = (336,448)
    pos4 = (448,560)
    pos5 = (560,672)
    pos6 = (672,784)
    pos7 = (784,896)

    final = np.zeros(shape=(1006,782))
    assert final.shape==(1006,782), "Wrong shape"
    if mode == 'mode':    
        for row in range(final.shape[0]):
            for column in range(final.shape[1]):
                if column < pos0[1]: #0-112 - pred0
                    x =  pred0[row][column-pos0[0]]
                    vals,counts = np.unique(x, return_counts=True)
                    final[row][column] = vals[counts.argmax()]

                elif column >=pos1[0] and column <pos1[1]: #112-224 pred0,pred1
                    x =  pred0[row][column-pos0[0]], pred1[row][column-pos1[0]]
                    vals,counts = np.unique(x, return_counts=True)
                    final[row][column] = vals[counts.argmax()]

                elif column >=pos2[0] and column <pos2[1]: #224-336 pred0,pred1,pred2
                    x =  pred0[row][column-pos0[0]], pred1[row][column-pos1[0]], pred2[row][column-pos2[0]]
                    vals,counts = np.unique(x, return_counts=True)
                    final[row][column] = vals[counts.argmax()]

                elif column >=pos3[0] and column <pos3[1]: #336-448 pred0,pred1,pred2,pred3
                    x =  pred0[row][column-pos0[0]] , pred1[row][column-pos1[0]], pred2[row][column-pos2[0]], pred3[row][column-pos3[0]]
                    vals,counts = np.unique(x, return_counts=True)
                    final[row][column] = vals[counts.argmax()]

                elif column >=pos4[0] and column <pos4[1]: #448-560 pred1,pred2,pred3,pred4
                    x = final[row][column], pred1[row][column-pos1[0]] , pred2[row][column-pos2[0]], pred3[row][column-pos3[0]], pred4[row][column-pos4[0]]
                    vals,counts = np.unique(x, return_counts=True)
                    final[row][column] = vals[counts.argmax()]

                elif column >=pos5[0] and column <pos5[1]: #560-672 pred2,pred3,pred4
                    x =  pred2[row][column-pos2[0]], pred3[row][column-pos3[0]], pred4[row][column-pos4[0]]
                    vals,counts = np.unique(x, return_counts=True)
                    final[row][column] = vals[counts.argmax()]

                elif column >=pos6[0] and column <pos6[1]: #672-784 pred3,pred4
                    x =  pred3[row][column-pos3[0]] , pred4[row][column-pos4[0]]
                    vals,counts = np.unique(x, return_counts=True)
                    final[row][column] = vals[counts.argmax()]

                elif column >=pos7[0] and column <pos7[1]: #784-896 pred4
                    x =  pred4[row][column-pos4[0]]
                    vals,counts = np.unique(x, return_counts=True)
                    final[row][column] = vals[counts.argmax()]
    elif mode =='mean':
        for row in range(final.shape[0]):
            for column in range(final.shape[1]):
                if column < pos0[1]: #0-112 - pred0
                    x =  pred0[row][column-pos0[0]]
                    final[row][column] = np.mean(x)
                    
                elif column >=pos1[0] and column <pos1[1]: #112-224 pred0,pred1
                    x =  pred0[row][column-pos0[0]], pred1[row][column-pos1[0]]
                    final[row][column] = np.mean(x)
                    
                elif column >=pos2[0] and column <pos2[1]: #224-336 pred0,pred1,pred2
                    x =  pred0[row][column-pos0[0]], pred1[row][column-pos1[0]], pred2[row][column-pos2[0]]
                    final[row][column] = np.mean(x)
                    
                elif column >=pos3[0] and column <pos3[1]: #336-448 pred0,pred1,pred2,pred3
                    x =  pred0[row][column-pos0[0]] , pred1[row][column-pos1[0]], pred2[row][column-pos2[0]], pred3[row][column-pos3[0]]
                    final[row][column] = np.mean(x)
                    
                elif column >=pos4[0] and column <pos4[1]: #448-560 pred1,pred2,pred3,pred4
                    x =  pred1[row][column-pos1[0]] , pred2[row][column-pos2[0]], pred3[row][column-pos3[0]], pred4[row][column-pos4[0]]
                    final[row][column] = np.mean(x)
                    
                elif column >=pos5[0] and column <pos5[1]: #560-672 pred2,pred3,pred4
                    x =  pred2[row][column-pos2[0]], pred3[row][column-pos3[0]], pred4[row][column-pos4[0]]
                    final[row][column] = np.mean(x)
                    
                elif column >=pos6[0] and column <pos6[1]: #672-784 pred3,pred4
                    x =  pred3[row][column-pos3[0]] , pred4[row][column-pos4[0]]
                    final[row][column] = np.mean(x)
                    
                elif column >=pos7[0] and column <pos7[1]: #784-896 pred4
                    x =  pred4[row][column-pos4[0]]
                    final[row][column] = np.mean(x)

    if plot:
        colors = ["gold", "lawngreen", "lightseagreen", "orange", "blue", "sienna", "violet", "lightseagreen", "darkorange","red"]
        cmap_seismic = LinearSegmentedColormap.from_list("mycmap", colors)
        fig, ax = plt.subplots(1, 1, figsize=(9, 4))
        m = ax.imshow(final, cmap=cmap_seismic,alpha=0.7,vmin=0,vmax=len(colors)-1)
        plt.colorbar(m)
        plt.show()
    return final
