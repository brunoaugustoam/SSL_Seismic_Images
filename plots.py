from aux_functions import *

import torch
from torch import nn, optim
from torchvision import transforms as T
import numpy as np

import PIL
from PIL import Image 

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib.patches as patches



def plot_tensor_img_mask(seismic_image, image_mask= None, section_number='insert_number', mask_transparency = 0.7, colors=None):
    section_number=section_number
    if type(seismic_image) == torch.Tensor and len(seismic_image.shape) ==3:
        image_array = np.transpose(seismic_image, (1, 2, 0)) 
        if image_mask != None:
            mask_array = np.transpose(image_mask, (1, 2, 0)) 
            mask = mask_array.cpu().detach().numpy()
        print(f'Section: {section_number}, being: {image_array.shape}')
        image_array = image_array.cpu().detach().numpy()
        
    elif type(seismic_image) == np.ndarray and len(seismic_image.shape) ==3:
        image_array = np.transpose(seismic_image, (1, 2, 0)) 
        if type(image_mask) != None:
            if type(image_mask) == torch.Tensor and len(image_mask.shape) ==3:
                mask = np.transpose(image_mask, (1, 2, 0)) 
    else:
        image_array = seismic_image

        
    fig = plt.figure(figsize=(16,9)) 
    image_array = (image_array +1)/2 * 255
    image = T.ToPILImage()(image_array)
    plt.tight_layout()

    ax = plt.subplot(1, 1, 1)
    a = plt.imshow(image, cmap="Greys", vmin=-1, vmax=1)
    #a = plt.imshow(image, cmap="Greys")
    plt.colorbar(a)

    if colors == None:
        colors = ["darkorange", "gold", "lawngreen", "lightseagreen", "orange", "blue", "sienna", "violet", "lightseagreen", "red"]
    cmap_teste = LinearSegmentedColormap.from_list("mycmap", colors)
    
    #m = plt.imshow(mask, alpha=mask_transparency, cmap="Pastel1",vmin=0,vmax=9)
    if image_mask!= None:
        m = plt.imshow(image_mask, alpha=mask_transparency, cmap=cmap_teste,vmin=0,vmax=9)
        #criar o listed cmp e passar acima
        plt.colorbar(m)
    ax.set_title('{}'.format(str(section_number)))
    ax.axis('off')



def plot_pil(input_image, name='insert_number',rotation=None):
    if rotation != None:
        print(f"Rotation Label: {rotation}")
    
    if type(input_image) == np.ndarray:
        input_image = (input_image +1)/2 * 255
        input_image = T.ToPILImage()(input_image)

    elif type(input_image) == torch.Tensor:
        input_image = tensor_to_pil(input_image)
    else:
        raise ValueError(f'Not recognized type {type(input_image)}')

    fig = plt.figure(figsize=(16,9)) 
    plt.tight_layout()
    ax = plt.subplot(1, 1, 1)
    ax.set_title(name)
    plt.imshow(input_image, cmap="Greys", vmin=-1, vmax=1)

def tensor_to_pil(tensor):
    if len(tensor.shape) == 3:
        image_array = np.transpose(tensor, (1, 2, 0))
        image_array = image_array.numpy()
    else:
        image_array = tensor.numpy()
    image_array = (image_array +1)/2 * 255
    return T.ToPILImage()(image_array)


def plot_loss(epochs, train_loss_mean_all, train_loss_std_all,val_loss_mean_all,val_loss_std_all,test_loss_mean_all, test_loss_std_all,title="Loss", save=False, path=None, name=None):

    fig, ax = plt.subplots(1,1, figsize=(10,6))
    ax.plot(np.arange(epochs), train_loss_mean_all, label='Train', alpha=0.9)
    ax.plot(np.arange(epochs), val_loss_mean_all, label='Val', alpha=0.9)
    ax.plot(np.arange(epochs), test_loss_mean_all, label='Test', alpha=0.9)

    ax.fill_between(np.arange(epochs), train_loss_mean_all-2*train_loss_std_all, train_loss_mean_all+2*train_loss_std_all, alpha=0.2) #2 desvios padrões
    ax.fill_between(np.arange(epochs), val_loss_mean_all-2*val_loss_std_all, val_loss_mean_all+2*val_loss_std_all, alpha=0.2) #2 desvios padrões
    ax.fill_between(np.arange(epochs), test_loss_mean_all-2*test_loss_std_all, test_loss_mean_all+2*test_loss_std_all, alpha=0.2) #2 desvios padrões

    #ax.set_ylim([-0.1, 1.0])
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Epochs', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.set_xticks(np.arange(0,epochs+1,5))
    ax.grid()
    ax.legend()
    if save:
        plt.savefig(os.path.join(path, name))

def plot_accu(epochs, train_accu,val_accu,test_accu, title="Accuracy", save=False, path=None, name=None):
    'Enter array not list'
    train_accu = train_accu*100
    val_accu = val_accu*100
    test_accu = test_accu*100

    fig, ax = plt.subplots(1,1, figsize=(10,6))
    #ylim = plt.ylim()
    ax.plot(np.arange(epochs), train_accu, label='Train', alpha=0.9)
    ax.plot(np.arange(epochs), val_accu, label='Validation', alpha=0.9)
    ax.plot(np.arange(epochs), test_accu, label='Test', alpha=0.9, c='k')
    ax.set_title(f'{title} - {name}', fontsize=16)
    ax.set_xlabel('Epochs', fontsize=14)
    ax.set_ylabel('Accuracy %', fontsize=14)
    #ax.set_ylim(ylim)
    ax.set_xticks(np.arange(0,epochs+1,5))
    ax.set_yticks(np.arange(0, 100,5))
    ax.grid()
    ax.legend()
    if save:
        plt.savefig(os.path.join(path, name))

def plot_compare(epochs, list_of_lists,list_of_labels,alpha=0.8, title="insert title", ylabel = "insert ylabel", xlabel = "inser xlabel",fill=False,list_of_std=None,save=False, path=None, name=None):
    it = iter(list_of_lists)
    the_len = len(next(it))
    
    fig, ax = plt.subplots(1,1, figsize=(10,6))
    if len(list_of_lists.shape) == 1:
        for idx in range(0,len(list_of_lists)):
            item = np.array(list_of_lists[idx])
            ax.plot(np.arange(1,epochs+1), item, label=list_of_labels[idx], alpha=alpha)
            if fill:
                std = np.array(list_of_std[idx])
                ax.fill_between(np.arange(1,epochs+1), item-2*std, item+2*std, alpha=0.2) 

    elif len(list_of_lists.shape) == 2:
        for idx in range(0,list_of_lists.shape[0]):
            item = np.array(list_of_lists[idx])
            ax.plot(np.arange(1,epochs+1), item, label=list_of_labels[idx], alpha=alpha)
            if fill:
                std = np.array(list_of_std[idx])
                ax.fill_between(np.arange(1,epochs+1), item-2*std, item+2*std, alpha=0.2) 
    else:
        raise ValueError("Tensor of higer dimension should be rearranged ")

    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xticks(np.arange(0,epochs+1,np.round(epochs/10,4)))

    if list_of_lists.ravel()[list_of_lists.ravel().argmax()] >4:
        plt.ylim(0, 4)
    elif ylabel=='Accu':
        plt.ylim(0, 1)
    ax.grid()
    ax.legend()
    if save:
        plt.savefig(os.path.join(path, name))
    plt.show()

def plot_confusion_matrix(cm, classes,save=False, path=None, name=None, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if save:
        plt.savefig(os.path.join(path, name))


def get_plot_conf_matrix(predictions, labels,classes=None, title='Confusion matrix', save=False, path=None, name=None,normalize=False):
        
    labels = torch.IntTensor(labels)
    predictions = torch.IntTensor(predictions)

    labels = torch.flatten(labels)
    predictions = torch.flatten(predictions)
    
    cm = confusion_matrix(labels,predictions )
    plt.figure(figsize=(10,10))
    plot_confusion_matrix(cm, classes,save, path,name, normalize=normalize, title=title, cmap=plt.cm.Blues)

    



def plot_batch(secao, iterator, name_list, mask=None, task=None,dataset='F3_netherlands',alpha=0.6):
    assert task == None or task == 'segmentation' or task == 'rotation' or task == 'jigsaw', f'Not recognized task: {task}'

    if task == 'jigsaw':
        raise ValueError(f"Not yet implemented task {task}")

    else:
        pass
            
    image_array = secao[iterator,:,:,:].cpu().detach().numpy()
    image_array = np.transpose(image_array, (1, 2, 0)) 
    image_array = (image_array +1)/2 * 255
    image_array = T.ToPILImage()(image_array)

    if task == None or task=='segmentation':
        mask_array  = mask[iterator,:,:,:].cpu().detach().numpy()
        mask_array = np.transpose(mask_array, (1, 2, 0))
        mask_array = mask_array.squeeze()

    else:
        pass


    if task == None or task == 'rotation' or task == 'segmentation':
        fig = plt.figure(figsize=(16,9)) 
    else:
        raise ValueError(f"Not yet implemented task {task}")


    if task == "rotation":
        plt.title(f'{name_list[iterator]} - Rotation Label: {mask[iterator]}')
    else:
        plt.title(name_list[iterator])

    if task == None or task == 'rotation' or task=='segmentation':
        ax = plt.subplot(1, 1, 1)
        a = plt.imshow(image_array, cmap="Greys", vmin=-1, vmax=1)
 
        plt.colorbar(a)
        plt.tight_layout()  
    else:
        raise ValueError(f"Not yet implemented task {task}")

    if task == None or task=='segmentation':
        if dataset=='F3_netherlands':
            colors = ["gold", "lawngreen", "lightseagreen", "orange", "blue", "sienna", "violet", "lightseagreen", "darkorange","red"]
        elif dataset =='Parihaka_NZPM':
            colors = ['steelblue','darkturquoise', 'lightcoral', 'mediumseagreen','lavender','wheat']
        cmap_teste = LinearSegmentedColormap.from_list("mycmap", colors)
        m = plt.imshow(mask_array, alpha=alpha, cmap=cmap_teste,vmin=0,vmax=len(colors)-1)
        plt.colorbar(m)


def plot_segmentation_results(images,mask,y_hat,i,name,dataset, save=False, path=None, name_model=None):
    """Input as tensors will be converted inside"""
    
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    if dataset=='F3_netherlands':
        colors = ["gold", "lawngreen", "lightseagreen", "orange", "blue", "sienna", "violet", "lightseagreen", "darkorange","red"]
    elif dataset =='Parihaka_NZPM':
        colors = ['steelblue','darkturquoise', 'lightcoral', 'mediumseagreen','lavender','wheat']
    
    cmap_seismic = LinearSegmentedColormap.from_list("mycmap", colors)
    
    section = images.detach().cpu().numpy().squeeze()
    ax[0].imshow(section, cmap="Greys", vmin=-1, vmax=1)
    ax[0].set_yticks([])
    ax[0].set_xticks([])
    ax[0].set_title(name[i])
    
    labs = mask.detach().cpu().numpy().squeeze(1)
    ax[1].imshow(labs[i], cmap=cmap_seismic,vmin=0,vmax=len(colors)-1)
    ax[1].set_yticks([])
    ax[1].set_xticks([])
    ax[1].set_title('Mask')
    
    preds = y_hat.data.max(1)[1].cpu().numpy()
    ax[2].imshow(preds[i],cmap=cmap_seismic,vmin=0,vmax=len(colors)-1)
    ax[2].set_yticks([])
    ax[2].set_xticks([])
    ax[2].set_title('Prediction')

    plt.colorbar(ax[2].imshow(preds[i],cmap=cmap_seismic,vmin=0,vmax=len(colors)-1))
    
    if save:
        plt.savefig(os.path.join(path, name_model))
    plt.show()



def plot_segy_label_dataloader(seismic_image, number, label_mask=None,plot_type='over',iterator=0,title='Title', task='segmentation'):
    if task == 'segmentation':
        assert(seismic_image.shape == label_mask.shape), 'labels and image with different shape'
    assert(plot_type == 'img' or plot_type == 'both' or plot_type == 'over'), f"Plot type not recognized: {plot_type}, use 'img', 'over' or 'both' instead "
    
    fig = plt.figure(figsize=(16,9))

    ax = fig.add_subplot(121)
    if plot_type == 'both': # and plot_type != 'over':
        ax1 = fig.add_subplot(122)

    image_array = seismic_image[iterator,:,:,:].cpu().detach().numpy().squeeze()
    if task == 'segmentation':
        mask_array  = label_mask[iterator,:,:,:].cpu().detach().numpy().squeeze()


    seis = ax.imshow(image_array, cmap="Greys", vmin=-1, vmax=1)
    

    colors = ['steelblue','darkturquoise', 'lightcoral', 'mediumseagreen','lavender','wheat']
    cmap_seismic = LinearSegmentedColormap.from_list("mycmap", colors)

    ax.invert_xaxis()
    ax.set_title(f'Section: {title}')

    if plot_type == 'img':
        plt.colorbar(seis, ax=ax)
        
    elif plot_type == 'over' and task == 'segmentation':
        lab = plt.imshow(mask_array,cmap=cmap_seismic,vmin=0,vmax=len(colors)-1, alpha=0.3)
        fig.colorbar(lab, ax=ax)
        
    elif plot_type == 'both':
        if task == 'segmentation':
            lab = plt.imshow(mask_array,cmap=cmap_seismic,vmin=0,vmax=len(colors)-1)
            fig.colorbar(lab, ax=ax1)
        ax1.invert_xaxis()
        ax1.set_title(f'Mask: {title}')

    else:
        raise ValueError(f'Not recognized plot type {plot_type}')

def plot_jigsaw(labels,tiles,sec_name):
    print(f'Permutation: {labels}')
    print(sec_name)
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(12, 12),sharex=True, sharey=True)
    for i, ax in zip(range(tiles.shape[0]), axs.ravel()):
        ax.imshow(tiles[i], cmap="Greys", vmin=-1, vmax=1)
        ax.set_title(labels[i])

    plt.show()

def plot_jigsaw_loader(labels,tiles,sec_name,iterator,dataset='F3_netherlands'):
    print(f'Permutation: {labels[iterator]}')
    print(sec_name[iterator])
    if dataset=='F3_netherlands':
        fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(12, 12),sharex=True, sharey=True)
    elif dataset=='Parihaka_NZPM':
        fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(12, 18),sharex=True, sharey=True)
    for i, ax in zip(range(tiles.shape[1]), axs.ravel()):
            ax.imshow(tiles[iterator][i], cmap="Greys", vmin=-1, vmax=1)
            ax.set_title(labels[iterator][i])

    plt.show()