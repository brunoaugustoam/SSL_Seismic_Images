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
#from plots import *
from pathlib import Path

def normalize_1(section):
    """
    Normalizes array between (-1,1)
    """
    section = ( (section - np.min(section)) / (np.max(section) - np.min(section) ) )
    section = 2*section -1
    return section


def rotate_tensor(img_nparray, rotations_list=None, plot=False):
    """ Rotates the given tensor into the given angle list.  """
    from plots import *
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

def random_crop_inpainting(section,final_size=(192,192),plot=False,dataloader_crop=False):
    """
    Random crop for inpaiting task. 
    Retuns the cropped section to be used as input to the network and the tile to be used as label
    """
    cropped_section = section.copy()

    height = cropped_section.shape[0]
    width = cropped_section.shape[1]

    height_idx = int(np.random.randint(low=25,high=height-final_size[0]-25, size=1))
    width_idx = int(np.random.randint(low=25,high=width-final_size[1]-25, size=1))  

    crop = cropped_section[height_idx:height_idx+final_size[0], width_idx:width_idx+final_size[1]]
    cropped_tile = crop.copy()
    for row in range(height_idx,height_idx+final_size[0]):
        for column in range(width_idx,width_idx+final_size[1]):
            cropped_section[row][column] = 0
    
    if plot:
        plot_inpainting_sec(cropped_section,cropped_tile)

    if dataloader_crop:
        return cropped_section,cropped_tile,height_idx,width_idx ,final_size
    else:
        return cropped_section,cropped_tile

def center_crop_inpainting(section,final_size=(192,192),plot=False,dataloader_crop=False):
    """
    Center crop for inpaiting task. 
    Retuns the cropped section to be used as input to the network and the tile to be used as label
    """
    cropped_section = section.copy()
    
    height_center = int(cropped_section.shape[0]/2)
    width_center = int(cropped_section.shape[1]/2)

    half_height = int(final_size[0]/2)
    half_width = int(final_size[1]/2)
    
    crop = cropped_section[height_center-half_height :  height_center+half_height,
                          width_center-half_width  :  width_center+half_width]
    cropped_tile = crop.copy()
    
    for row in range(height_center-half_height , height_center+half_height):
        for column in range(width_center-half_width ,width_center+half_width):
            cropped_section[row][column] = 0
    cropped_sec = cropped_section.copy()
    if plot:
        plot_inpainting_sec(cropped_sec,cropped_tile)

    if dataloader_crop:
        return cropped_sec,cropped_tile,height_center,width_center ,final_size
    
    return cropped_sec,cropped_tile

def crop_inpainting_dataloader(section_orig,height_idx, width_idx,final_size=(128,128),split='train',plotar=False):
    """
    This way will only work for batch_size==1, otherwise needs to be rearranged
    For that must buld a loop to iterate the entire batch, perform the crops and build it again
    """
    assert split== 'train' or split== 'val' or split=='test', 'Not recognized split type'
    cropped_section = section_orig.squeeze().detach().numpy()
    cropped_section = cropped_section.copy()
    if split== 'train':
        top = int(height_idx.cpu().detach().numpy())
        left = int(width_idx.cpu().detach().numpy())
        
        cropped_tile = cropped_section[top  :  top  + final_size[0],
                          left :  left + final_size[1]]
        
    elif split== 'val' or split=='test':
        height_center = int(height_idx.cpu().detach().numpy())
        width_center = int(width_idx.cpu().detach().numpy())
        half_height = int(final_size[0]/2)
        half_width = int(final_size[1]/2)
        
        cropped_tile = cropped_section[height_center-half_height :  height_center+half_height,
                          width_center-half_width  :  width_center+half_width]
        
    else:
        raise ValueError(f'Not recognized split {split}')
    if plotar:
        plot_inpainting_sec(cropped_section,cropped_tile)
        
    cropped_tile = np.expand_dims(cropped_tile, axis=0)
    cropped_tile = np.expand_dims(cropped_tile, axis=0)
    cropped_tile_copy = cropped_tile.copy()
    
    assert list(cropped_tile_copy.squeeze().shape) == list(final_size), f"dims not match {cropped_tile_copy.shape} and {list(final_size)}"
    
    return torch.from_numpy(cropped_tile_copy)

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


def save_hog_log_results(saving_logs_path,name_model,train_loss_mean,train_loss_std, 
                            val_loss_mean,val_loss_std, 
                            #test_loss,test_loss_std
                            ):
    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'train_loss_mean'+'.txt'), train_loss_mean)
    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'train_loss_std'+'.txt'),  train_loss_std  )

    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'val_loss_mean'+'.txt'),   val_loss_mean  )
    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'val_loss_std'+'.txt'),    val_loss_std  )

    #np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'test_loss'+'.txt'),  test_loss  )
    #np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'test_loss_std'+'.txt'),  test_loss_std  )


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



def save_inpainting_log_results( saving_logs_path,name_model, 
                                train_loss_mean,train_loss_std,
                                val_loss_mean,val_loss_std ):
    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'train_loss_mean'+'.txt'), train_loss_mean)
    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'train_loss_std'+'.txt'),  train_loss_std  )

    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'val_loss_mean'+'.txt'),   val_loss_mean  )
    np.savetxt(os.path.join(saving_logs_path, name_model+'_'+'val_loss_std'+'.txt'),    val_loss_std  )


def window_inline_f3(pred0,pred1,pred2,pred3,pred4, mode='mode', plot=False,label=False):
    assert mode == 'mean' or mode=='mode', f'Not recognized calculation mode:{mode}'
    if mode == 'mean':
        print("WARNING: You are using windows with mean on the ovelap sectios, this might not be adequate for segmetnation tasks")

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
        if label:
            m = ax.imshow(final, cmap=cmap_seismic,alpha=0.7,vmin=0,vmax=len(colors)-1)
        else:
            m = ax.imshow(final, cmap='Greys',vmin=-1,vmax=1)
        plt.colorbar(m)
        plt.show()
    return final


def window_crossline_f3(pred0,pred1,pred2, mode='mean', plot=False,label=False):
    assert mode == 'mean' or mode=='mode', f'Not recognized calculation mode:{mode}'
    if mode == 'mean':
        print("WARNING: You are using windows with mean on the ovelap sectios, this might not be adequate for segmetnation tasks")

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
        if label:
            m = ax.imshow(final, cmap=cmap_seismic,alpha=0.7,vmin=0,vmax=len(colors)-1)
        else:
            m = ax.imshow(final, cmap='Greys',vmin=-1,vmax=1)

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
           sections = ['crossline_820', 'crossline_787', 'inline_340', 'crossline_612'] + ['inline_202', 'crossline_803', 'crossline_1054', 'crossline_1239']
            #sections = ['Inline_65', 'Inline_435', 'Inline_310', 'Inline_499',] + [ 'Inline_272', 'Inline_211', 'Crossline_65', 'Inline_204']
        if split == 'val':
           sections = ['inline_239', 'inline_623', 'crossline_705', 'crossline_1064'] + ['crossline_536', 'inline_321', 'crossline_574', 'crossline_899']
           # sections = ['Inline_379', 'Crossline_660', 'Inline_359', 'Inline_144']+['Crossline_53', 'Crossline_161', 'Crossline_314', 'Crossline_103']
        if split == 'test':
           sections = ['crossline_404', 'crossline_1011', 'crossline_518', 'crossline_1050'] + ['inline_444', 'crossline_914', 'crossline_309', 'crossline_948']
        assert sec_number_list[:4] + sec_number_list[-4:] == sections, f"Selected fold does not match with the pre-set"
    
    if fold ==1:
        if split == 'train':
            sections = ['crossline_820', 'crossline_763', 'inline_340', 'crossline_612'] + ['inline_202', 'crossline_305', 'crossline_1054', 'crossline_1239']
        if split == 'val':
            sections = ['crossline_404', 'crossline_1011', 'crossline_518', 'crossline_1050'] + ['inline_444', 'crossline_914', 'crossline_309', 'crossline_948']
        if split == 'test':
            sections = ['inline_555', 'crossline_1102', 'crossline_1092', 'crossline_1144'] + ['crossline_953', 'inline_469', 'inline_389', 'inline_323']
        assert sec_number_list[:4] + sec_number_list[-4:] == sections, f"Selected fold does not match with the pre-set"
    
    if fold ==2:
        if split == 'train':
            sections = ['crossline_906', 'crossline_763', 'crossline_402', 'crossline_1164'] + ['inline_202', 'crossline_305', 'inline_145', 'crossline_1239']
        if split == 'val':
            sections = ['inline_555', 'crossline_1102', 'crossline_1092', 'crossline_1144'] + ['crossline_953', 'inline_469', 'inline_389', 'inline_323']
        if split == 'test':
            sections = ['crossline_791', 'crossline_1126', 'crossline_871', 'crossline_779'] + ['crossline_1210', 'inline_365', 'crossline_1012', 'crossline_503']
        assert sec_number_list[:4] + sec_number_list[-4:] == sections, f"Selected fold does not match with the pre-set"

    if fold ==3:
        if split == 'train':
            sections = ['crossline_913', 'inline_316', 'crossline_743', 'crossline_694'] + ['crossline_1012', 'crossline_897', 'inline_453', 'crossline_503']
        if split == 'val':
            sections = ['crossline_446', 'crossline_341', 'inline_195', 'crossline_809'] + ['crossline_422', 'crossline_558', 'inline_202', 'crossline_1239']
        if split == 'test':
            sections = ['inline_239', 'inline_623', 'crossline_705', 'crossline_1064'] + ['crossline_536', 'inline_321', 'crossline_574', 'crossline_899']
        assert sec_number_list[:4] + sec_number_list[-4:] == sections, f"Selected fold does not match with the pre-set"

    if fold ==4:
        if split == 'train':
            sections = ['crossline_906', 'crossline_763', 'crossline_402', 'crossline_1164'] + ['inline_389', 'crossline_305', 'inline_145', 'inline_323']
        if split == 'val':
            sections = ['crossline_791', 'crossline_1126', 'crossline_871', 'crossline_779'] + ['crossline_1210', 'inline_365', 'crossline_1012', 'crossline_503']
        if split == 'test':
            sections = ['crossline_446', 'crossline_341', 'inline_195', 'crossline_809'] + ['crossline_422', 'crossline_558', 'inline_202', 'crossline_1239']
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
           sections = ['Inline_117', 'Inline_87', 'Inline_497', 'Inline_442'] + ['Inline_387', 'Crossline_252', 'Inline_265', 'Inline_226']
        if split == 'val':
           sections = ['Crossline_129', 'Crossline_220', 'Crossline_442', 'Crossline_780'] + ['Inline_273', 'Crossline_290', 'Crossline_477', 'Inline_494']
        if split == 'test':
           sections = ['Inline_198', 'Inline_535', 'Crossline_626', 'Crossline_185'] + ['Inline_347', 'Crossline_21', 'Inline_288', 'Crossline_29']
        assert sec_number_list[:4] + sec_number_list[-4:] == sections, f"Selected fold does not match with the pre-set"
    
    if fold ==1:
        if split == 'train':
           sections = ['Crossline_309', 'Inline_87', 'Inline_497', 'Inline_442'] + ['Inline_387', 'Crossline_252', 'Inline_265', 'Inline_226']
        if split == 'val':
           sections = ['Inline_198', 'Inline_535', 'Crossline_626', 'Crossline_185'] + ['Inline_347', 'Crossline_21', 'Inline_288', 'Crossline_29']
        if split == 'test':
           sections = ['Crossline_320', 'Crossline_569', 'Crossline_328', 'Crossline_356'] + ['Inline_24', 'Crossline_543', 'Inline_272', 'Crossline_769']
        assert sec_number_list[:4] + sec_number_list[-4:] == sections, f"Selected fold does not match with the pre-set"
    
    if fold ==2:
        if split == 'train':
           sections = ['Crossline_309', 'Inline_87', 'Inline_497', 'Inline_442'] + ['Inline_387', 'Crossline_252', 'Crossline_435', 'Inline_226']
        if split == 'val':
           sections = ['Crossline_320', 'Crossline_569', 'Crossline_328', 'Crossline_356'] + ['Inline_24', 'Crossline_543', 'Inline_272', 'Crossline_769']
        if split == 'test':
           sections = ['Inline_33', 'Crossline_457', 'Crossline_346', 'Inline_59'] + ['Crossline_601', 'Crossline_113', 'Inline_569', 'Crossline_452']
        assert sec_number_list[:4] + sec_number_list[-4:] == sections, f"Selected fold does not match with the pre-set"

    if fold ==3:
        if split == 'train':
           sections = ['Inline_376', 'Inline_0', 'Inline_35', 'Inline_97'] + ['Crossline_113', 'Inline_569', 'Inline_181', 'Crossline_452']
        if split == 'val':
           sections = ['Crossline_253', 'Crossline_325', 'Crossline_546', 'Crossline_286'] + ['Inline_480', 'Inline_387', 'Crossline_252', 'Inline_226']
        if split == 'test':
           sections =['Crossline_129', 'Crossline_220', 'Crossline_442', 'Crossline_780'] + ['Inline_273', 'Crossline_290', 'Crossline_477', 'Inline_494']
        assert sec_number_list[:4] + sec_number_list[-4:] == sections, f"Selected fold does not match with the pre-set"

    if fold ==4:
        if split == 'train':
            sections = ['Crossline_309', 'Crossline_69', 'Inline_454', 'Crossline_322'] + ['Crossline_543', 'Inline_272', 'Crossline_435', 'Crossline_769']
        if split == 'val':
            sections = ['Inline_33', 'Crossline_457', 'Crossline_346', 'Inline_59'] + ['Crossline_601', 'Crossline_113', 'Inline_569', 'Crossline_452']
        if split == 'test':
            sections = ['Crossline_253', 'Crossline_325', 'Crossline_546', 'Crossline_286'] + ['Inline_480', 'Inline_387', 'Crossline_252', 'Inline_226']
        assert sec_number_list[:4] + sec_number_list[-4:] == sections, f"Selected fold does not match with the pre-set"


def window_inline_parih(pred0,pred1,pred2,pred3,pred4,pred5,plot=False,label=False):

    #pred shape= 832,448
    #horizontais
    h0 = 167
    h1 = 334
    h2 = 448
    h3 = 615
    #h4 = (615,782)

    #verticais
    v0 = 174
    v1 = 832
    #v2 = (832,1006)
    ###AINDA PRECISA AJUSTAR OS DESLOCAMENTOS TODOS
    final = np.zeros(shape=(1006,782))
    assert final.shape==(1006,782), "Wrong shape"
    for row in range(final.shape[0]):
        for column in range(final.shape[1]):
            if row < v0:
                if column < h0: #pred 0 
                    x =  pred0[row][column]
                    vals,counts = np.unique(x, return_counts=True)
                    final[row][column] = vals[counts.argmax()]

                elif column >=h0 and column <h1: #pred0,pred1
                    x =  pred0[row][column-h0], pred1[row][column-h1]
                    vals,counts = np.unique(x, return_counts=True)
                    final[row][column] = vals[counts.argmax()]

                elif column >=h1 and column <h2: # pred0,pred1,pred2
                    x =  pred0[row][column-h0], pred1[row][column-h1], pred2[row][column-h2]
                    vals,counts = np.unique(x, return_counts=True)
                    final[row][column] = vals[counts.argmax()]

                elif column >=h2 and column <h3: #pred1,pred2
                    x =  pred1[row][column-h1], pred2[row][column-h2]
                    vals,counts = np.unique(x, return_counts=True)
                    final[row][column] = vals[counts.argmax()]

                elif column >=h3: #pred2,
                    x = pred2[row][column-h2]
                    vals,counts = np.unique(x, return_counts=True)
                    final[row][column] = vals[counts.argmax()]

            elif row >= v0 and row <= v1:
                if column < h0: #pred 0 and pred3
                    x =  pred0[row-v0][column], pred3[row-v0][column]
                    vals,counts = np.unique(x, return_counts=True)
                    final[row][column] = vals[counts.argmax()]

                elif column >=h0 and column <h1: #pred0,pred1
                    x =  pred0[row-v0][column-h0], pred1[row][column-h1]
                    vals,counts = np.unique(x, return_counts=True)
                    final[row][column] = vals[counts.argmax()]

                elif column >=h1 and column <h2: # pred0,pred1,pred2
                    x =  pred0[row-v0][column-h0], pred1[row][column-h1], pred2[row][column-h2]
                    vals,counts = np.unique(x, return_counts=True)
                    final[row][column] = vals[counts.argmax()]

                elif column >=h2 and column <h3: #pred1,pred2
                    x =  pred1[row][column-h1], pred2[row][column-h2]
                    vals,counts = np.unique(x, return_counts=True)
                    final[row][column] = vals[counts.argmax()]

                elif column >=h3: #pred2,
                    x = pred2[row][column-h2]
                    vals,counts = np.unique(x, return_counts=True)
                    final[row][column] = vals[counts.argmax()]


    if plot:
        colors = ['steelblue','darkturquoise', 'lightcoral', 'mediumseagreen','lavender','wheat']
        cmap_seismic = LinearSegmentedColormap.from_list("mycmap", colors)
        fig, ax = plt.subplots(1, 1, figsize=(9, 4))
        if label:
            m = ax.imshow(final, cmap=cmap_seismic,alpha=0.7,vmin=0,vmax=len(colors)-1)
        else:
            m = ax.imshow(final, cmap='Greys',vmin=-1,vmax=1)
        plt.colorbar(m)
        plt.show()
    return final
