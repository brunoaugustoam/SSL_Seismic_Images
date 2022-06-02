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
from pathlib import Path

from plots_demo import *

def normalize_1(section):
    """
    Normalizes array between (-1,1)
    """
    section = ( (section - np.min(section)) / (np.max(section) - np.min(section) ) )
    section = 2*section -1
    return section


def rotate_tensor(img_nparray, rotations_list=None, plot=False):
    """ Rotates the given tensor into the given angle list.  """
    from plots import plot_array
    if rotations_list == None:
        rotations_list = [-8,-4,0,4,8]

    rot_label = int(np.random.randint(0,len(rotations_list), size=1))
    rot_angle = rotations_list[rot_label] 

    rotated_image = skimage.transform.rotate(img_nparray, angle= rot_angle, order=1)

    if plot:
        print(f"Rotation angle: {rot_angle} == Rot_label: {rot_label}")
        plot_array(rotated_image)


    return np.array(rotated_image), np.array(rot_label)

def random_crop(image, mask=np.array([]), final_size=(448,448), dataset='F3_netherlands'):
    """
    Crops the image and mask in a random crop
    """ 
    assert len(image.shape) == 2, "Implemented for 2D only" 
    
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
    """
    Crops the image and mask in a center crop
    """ 
    assert len(image.shape) == 2 , "Implemented for 2D only, 3D will be done soon" 

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
    """
    Custom crop
    """

    assert len(image.shape) == 2, "Implemented for 2D" 

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

    return np.array(list_keys), np.array(list_values),np.array(list_keys_orig), np.array(list_values_orig),permutation_idx


def set_criterion(task, device,weighted=False,mask=None,dataset='Parihaka_NZPM'):
    assert task == None or task == 'segmentation' or task == 'rotation' or task == 'hog' or task == 'jigsaw' or task =='inpainting' or task=='presentation', f'Not recognized task: {task}'
    
    if task == None or task == 'segmentation' or task == 'rotation' or task == 'jigsaw' or task=='presentation':
        if weighted:
            assert mask!=None,"Must provide mask"
            #If provided, the optional argument weight should be a 1D Tensor assigning weight to each of the classes. 
            #This is particularly useful when you have an unbalanced training set.
            mask_t = mask.squeeze()
            vals,counts = np.unique(mask_t, return_counts=True)
            if dataset == 'Parihaka_NZPM':
                for i in range(6):
                    if i not in vals:
                        vals=np.insert(vals,i,i)
                        counts =np.insert(counts,i,1)
            elif dataset == 'F3_netherlands':
                for i in range(10):
                    if i not in vals:
                        vals=np.insert(vals,i,i)
                        counts =np.insert(counts,i,1)

            norm = counts/counts.max()
            weights = 1/norm
            weights = torch.from_numpy(weights).float()
            criterion = nn.CrossEntropyLoss(weight=weights).to(device)
        else:
            criterion = nn.CrossEntropyLoss().to(device)
        
    elif task =='hog' or task =='inpainting':
        criterion = nn.MSELoss().to(device)
        
    else:
        raise ValueError(f'Not recognized task: {task}')
        
    return criterion


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

def save_segmentation_log_few_shot(saving_logs_path,name_model,
    train_iou_classes_all, train_iou_std_between_classes_all, train_iou_mean_all, train_loss_mean_all, train_loss_std_all, 
    ):
    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'train_iou_classes_all'+'.txt'         )     , train_iou_classes_all    )
    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'train_iou_std_between_classes_all'+'.txt'         )     , train_iou_std_between_classes_all    )
    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'train_iou_mean_all'+'.txt'          )   , train_iou_mean_all       )
    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'train_loss_mean_all'+'.txt'     )  , train_loss_mean_all  )
    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'train_loss_std_all'+'.txt'          ) ,  train_loss_std_all      )




def save_jigsaw_log_results(saving_logs_path,name_model, 
            train_acc_mean,train_acc_std,train_loss_mean,train_loss_std,
            val_acc_mean,val_acc_std,val_loss_mean,val_loss_std ):

    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'train_acc_mean'+'.txt'         )     , train_acc_mean    )
    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'train_acc_std'+'.txt'         )     , train_acc_std    )
    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'train_loss_mean'+'.txt'          )   , train_loss_mean       )
    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'train_loss_std'+'.txt'     )  , train_loss_std  )

    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'val_acc_mean'+'.txt'           ), val_acc_mean       )
    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'val_acc_std'+'.txt'         )     , val_acc_std    )
    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'val_loss_mean'+'.txt'            ),  val_loss_mean      )
    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'val_loss_std'+'.txt'  ) ,  val_loss_std )
