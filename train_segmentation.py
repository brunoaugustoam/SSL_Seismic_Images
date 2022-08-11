
import torch
from torch import nn, optim

import os, time
import numpy as np
#from tqdm import tqdm

from sklearn import metrics

from plots import *
from aux_functions import *

def evaluate_iouf1(prediction_tensor, labels_tensor,class_array):
    prds = prediction_tensor.cpu().numpy().squeeze()
    labs = labels_tensor.detach().cpu().numpy().squeeze()

    iou_mean = metrics.jaccard_score(labs.ravel(), prds.ravel(), average='macro')
    iou_classes = metrics.jaccard_score(labs.ravel(), prds.ravel(), average=None, labels=class_array)
    #f1 = metrics.f1_score(labs.ravel(), prds.ravel(), average='macro')

    #return np.array(iou, dtype=np.float32), np.array(f1, dtype=np.float32)
    #return np.array(iou_mean), np.array(iou_classes)
    return iou_mean, iou_classes

def train_segmentation(dataloader_train, model, optimizer, criterion, num_epochs, device, num_classes,dataloader_val=None ,scheduler=None, task='segmentation', train_type='sup_ssl',tolerancia=40,save=False,saved_models_path=None, saving_logs_path=None,name_model=None,freeze=False,weighted=True):
    assert task=='segmentation' , "Not recognized task or wrong implemented task for this function: {task}"
    assert train_type=='sup_ssl' or train_type=='few_shot' or train_type=='fine_tune', f"Not recognized train type type {train_type}"
    if train_type == 'sup_ssl':
      assert dataloader_val != None

    if save:
      assert saved_models_path != None and name_model != None and saving_logs_path != None, 'None path type not valid'
    # Training procedure.
    print('training on', device)
    highest_iou = 0
    counter = 0
    lr = scheduler.get_last_lr()
    

    # Setting network for training mode.
    model.train()


    # Lists for losses and metrics.
    train_iou_classes_all, train_iou_std_between_classes_all, train_iou_mean_all = [],[],[]
    train_loss_mean_all, train_loss_std_all = [],[]

    val_iou_classes_all, val_iou_std_between_classes_all, val_iou_mean_all = [],[],[]
    val_loss_mean_all, val_loss_std_all = [],[]

    test_iou_classes_all, test_iou_std_between_classes_all, test_iou_mean_all = [],[],[]
    test_loss_mean_all, test_loss_std_all = [],[]

    start = time.time()
    #pred_list, semantic_label_list = [],[]
    
    # Iterating over epochs.
    for epoch in range(num_epochs):

      model.train()
      if  train_type == 'fine_tune' and freeze==True:
        model.conv1.eval()
        model.layer1.eval()
        model.layer2.eval()
        model.layer3.eval()
        model.layer4.eval()

      
      print(f"Training epoch {epoch+1} / {num_epochs}" )

      loss_per_epoch,iou_mean_per_epoch, iou_classes_per_epoch = [],[],[]
      mean_iou_classes_per_epoch = []
      
      n = 0 #zero n every epoch to get epoch size
      plotar=False

      #First Load the mini batch and iterate over it
      for idx, (train_images, train_labels,name) in enumerate(dataloader_train):

        if idx % 30 ==0 :
          print(f'{idx}/{ len(dataloader_train.dataset.sections) /dataloader_train.batch_size}')

        #Redefine Loss
        if weighted:
          criterion = set_criterion(task, device,weighted=True,mask=train_labels,dataset=dataloader_train.dataset.dataset_name)
        
        proof = train_images.detach().cpu().numpy().squeeze()
        assert np.max(proof) <= 1 and np.min(proof) >= -1, f"Unexpected values interval:{proof}" 

        #Cast images to device
        train_images = train_images.to(device)
        train_labels = train_labels.to(device)

        #zero the gradients
        optimizer.zero_grad()
        
        #Predict
        #y_hat = model(train_images)
        y_hat = model(train_images)

        #get prediction of max value per pixel
        #pred = y_hat.data.max(1)[1].cpu().numpy()
        pred = y_hat.data.max(1)[1]

        #list of predictions and Labels
        #pred_list.append(pred.cpu().numpy()) 
        #semantic_label_list.append(train_labels.cpu().numpy())
        
        #Compute Loss and sum it up
        l = criterion(y_hat.squeeze(1), train_labels.squeeze(1).long()).sum()
        #ajustar para dar squeeze soh na dimensao do n_Channels e nao no batch size

        #allocate on gpu
        y_hat = y_hat.to(device)

        #Get Gradients
        l.backward()

        #Uptade
        optimizer.step()

        #loss per epochs
        loss_per_epoch.append(l.item())

        #obtain array of iou and f1 for each class in entire batch
        iou_mean, iou_classes = evaluate_iouf1(pred, train_labels,dataloader_train.dataset.classes)
        iou_mean_per_epoch.append(iou_mean)
        iou_classes_per_epoch.append(iou_classes)
        
        #sum number of all seen images over one epoch
        n += train_labels.size()[0]

      #call lr scheduler
      if scheduler != None:
        scheduler.step()
      if scheduler.get_last_lr() != lr:
        print(f'Lr updated to: {scheduler.get_last_lr()}')

      iou_classes_per_epoch = np.array(iou_classes_per_epoch)
      iou_classes_per_epoch_T = np.transpose(iou_classes_per_epoch, (1,0))

      
      for idx in range(iou_classes_per_epoch_T.shape[0]):
          mean_iou_classes_per_epoch.append(np.mean(iou_classes_per_epoch_T[idx]))


      train_iou_classes_all.append(mean_iou_classes_per_epoch)
      train_iou_std_between_classes_all.append(np.std(iou_classes_per_epoch))
      #train iou mean per epoch and general
      iou_mean_per_epoch = np.array(np.mean(iou_mean_per_epoch))
      train_iou_mean_all.append(iou_mean_per_epoch)


      #loss per epoch
      loss_per_epoch = np.array(loss_per_epoch)
      train_loss_mean_all.append(np.mean(loss_per_epoch))
      #loss std per epoch
      train_loss_std_all.append(np.std(loss_per_epoch))


      #Print relevant info
      print(f'epoch {epoch + 1}/{num_epochs}, train loss {np.mean(loss_per_epoch):.4}, mean iou over all classes: {np.mean(iou_mean_per_epoch):.4f} ,time {time.time() - start:.1f} sec, n={n}')
      #if epoch % 5 == 0:
        #plotar=True

      #after entire epoch, evaluate on validation set
      if train_type=='sup_ssl' or train_type=='few_shot':
        val_iou_mean_per_class, val_iou_std_between_classes,val_iou_mean, val_loss_mean,  val_loss_std,  val_pred_list, val_semantic_label_list =  val_segmentation(dataloader_val, model, criterion, device, num_classes, plotar=plotar,weighted=weighted)
        
        val_iou_classes_all.append(val_iou_mean_per_class)
        val_iou_std_between_classes_all.append(val_iou_std_between_classes)
        val_iou_mean_all.append(val_iou_mean)
        val_loss_mean_all.append(val_loss_mean)
        val_loss_std_all.append(val_loss_std)
        

        if val_iou_mean > highest_iou:
          print(f'highest_iou: {highest_iou} -> {val_iou_mean}')
          highest_iou = val_iou_mean
          counter = 0
          if save:
            torch.save(model.state_dict(), os.path.join(saved_models_path, name_model))

            save_segmentation_log_results(saving_logs_path,name_model, 
              train_iou_classes_all,train_iou_std_between_classes_all,train_iou_mean_all,train_loss_mean_all,train_loss_std_all,
              val_iou_classes_all,val_iou_std_between_classes_all,val_iou_mean_all,val_loss_mean_all,val_loss_std_all) #,
              #test_iou_classes_all,test_iou_std_between_classes_all,test_iou_mean_all,test_loss_mean_all,test_loss_std_all)
                  

        elif val_iou_mean <= highest_iou:
          counter += 1
          print(f"counter at {counter}/{tolerancia}")
          if counter > tolerancia:
            break

      elif train_type=='fine_tune':
        if iou_mean_per_epoch > highest_iou:
          print(f'highest_iou: {highest_iou} -> {iou_mean_per_epoch}')
          highest_iou = iou_mean_per_epoch
          counter = 0
          if save:
            torch.save(model.state_dict(), os.path.join(saved_models_path, name_model))

            save_segmentation_log_few_shot(saving_logs_path,name_model, 
              train_iou_classes_all,train_iou_std_between_classes_all,train_iou_mean_all,train_loss_mean_all,train_loss_std_all,
              )
                  
        elif iou_mean_per_epoch <= highest_iou:
          counter += 1
          print(f"counter at {counter}/{tolerancia}")
          if counter > tolerancia:
            break


      print("")
    if train_type=='sup_ssl'or train_type=='few_shot':
      return( np.array(train_iou_classes_all), np.array(train_iou_std_between_classes_all), np.array(train_iou_mean_all), np.array(train_loss_mean_all), np.array(train_loss_std_all), 
              np.array(val_iou_classes_all),   np.array(val_iou_std_between_classes_all) ,  np.array(val_iou_mean_all) ,  np.array(val_loss_mean_all),   np.array(val_loss_std_all) 
              )
    elif train_type=='fine_tune':
      return( np.array(train_iou_classes_all), np.array(train_iou_std_between_classes_all), np.array(train_iou_mean_all), np.array(train_loss_mean_all), np.array(train_loss_std_all), 
        )


def test_segmentation(dataloader_test, model, criterion, device,num_classes, printar=False, plotar=False, save=False, path=None, name_model=None,weighted=True,task='segmentation',late_fusion=False,saving_activation_path=None): 

    test_pred_list, test_semantic_label_list = [],[]
    loss_per_epoch,iou_mean_per_epoch, iou_classes_per_epoch = [],[],[]
    mean_iou_classes_per_epoch = []
    n = 0 
    probs_late = []

    model.eval()
    with torch.no_grad():
      for idx, (test_images, test_labels, name) in enumerate(dataloader_test):  

        if idx % 30 ==0 :
          print(f'{idx}/{ len(dataloader_test.dataset.sections) /dataloader_test.batch_size}') 

        if weighted:
          criterion = set_criterion(task, device,weighted=True,mask=test_labels,dataset=dataloader_test.dataset.dataset_name)

        #Cast images to device
        test_images =test_images.to(device)
        test_labels =test_labels.to(device)

        if test_images.shape[0] != dataloader_test.batch_size:
          print('Batch size does not match dataset size')
          test_images.unsqueeze()
          test_labels.unsqueeze()

        #Predict
        y_hat = model(test_images)

        if late_fusion:
          probs = y_hat.detach().cpu().numpy().squeeze(0)
          probs_late.append(probs)

        #get prediction of max value per pixel
        pred = y_hat.data.max(1)[1] #.type(torch.int32)

        #list with predictions and labels
        test_pred_list.append(pred.cpu().numpy())
        test_semantic_label_list.append(test_labels.cpu().numpy())

        #Compute Loss     
        l = criterion(y_hat.squeeze(1), test_labels.squeeze(1).long()).sum()

        #allocate on device
        y_hat = y_hat.to(device)

        #loss per epochs
        loss_per_epoch.append(l.item())

        #obtain array of iou and f1 for each class in entire batch
        iou_mean, iou_classes = evaluate_iouf1(pred, test_labels,dataloader_test.dataset.classes)
        iou_classes_per_epoch.append(iou_classes)
        iou_mean_per_epoch.append(iou_mean)

        #sum number of all seen images over all epochs
        n += test_labels.size()[0]

    #loss per epoch
    loss_per_epoch = np.array(loss_per_epoch)
    iou_classes_per_epoch = np.array(iou_classes_per_epoch)
    iou_classes_per_epoch_T = np.transpose(iou_classes_per_epoch, (1,0))

    
    for idx in range(iou_classes_per_epoch_T.shape[0]):
        mean_iou_classes_per_epoch.append(np.mean(iou_classes_per_epoch_T[idx]))

    if late_fusion:
      activ_path = os.path.join(saving_activation_path,name_model)
      activ_late = np.asarray(activ_late, dtype=np.float16) 
      preds_late = np.asarray(test_pred_list, dtype=np.float16) 

      np.savez_compressed(activ_path,activation=activ_late,preds=preds_late) 

    if printar:
      print("test IoU")
      for classe in range(iou_classes_per_epoch_T.shape[0]):
        print(f"Classe:{classe} - mean: {iou_classes_per_epoch[classe].mean():.3f}")

    if plotar:
      plot_segmentation_results(test_images,test_labels,y_hat,0,name,dataloader_test.dataset.dataset_name, save, path, name_model)

    print("")
    print(f'test loss {np.mean(loss_per_epoch):.5f},  test mean iou general: {np.round(np.mean(iou_mean_per_epoch),4)}')

    return  np.array(mean_iou_classes_per_epoch), np.array(np.std(iou_classes_per_epoch)), np.array(np.mean(iou_mean_per_epoch)), np.mean(loss_per_epoch), np.std(loss_per_epoch),  np.array(test_pred_list), np.array(test_semantic_label_list)


def val_segmentation(dataloader_val, model, criterion, device,num_classes, printar=False, plotar=False,weighted=True,task='segmentation'): 

    loss_per_epoch,iou_mean_per_epoch, iou_classes_per_epoch = [],[],[]
    val_pred_list, val_semantic_label_list = [],[]
    mean_iou_classes_per_epoch = []
    n = 0 

    model.eval()
    with torch.no_grad():
      for idx, (val_images, val_labels, name) in enumerate(dataloader_val):  

        if weighted:
            criterion = set_criterion(task, device,weighted=True,mask=val_labels,dataset=dataloader_val.dataset.dataset_name)

        if idx % 20 ==0 :
          print(f'{idx}/{ len(dataloader_val.dataset.sections) /dataloader_val.batch_size}') 

        #Cast images to device
        val_images =val_images.to(device)
        val_labels =val_labels.to(device)

        if val_images.shape[0] != dataloader_val.batch_size:
          print('Batch size does not match dataset size')
          val_images.unsqueeze()
          val_labels.unsqueeze()

        #Predict
        y_hat = model(val_images)

        #get prediction of max value per pixel
        pred = y_hat.data.max(1)[1] #.type(torch.int32)

        #list with predictions and labels
        val_pred_list.append(pred.cpu().numpy())
        val_semantic_label_list.append(val_labels.cpu().numpy())

        #Compute Loss     
        l = criterion(y_hat.squeeze(1), val_labels.squeeze(1).long()).sum()

        #allocate on device
        y_hat = y_hat.to(device)

        #loss per epochs
        loss_per_epoch.append(l.item())

        #obtain array of iou and f1 for each class in entire batch
        iou_mean, iou_classes = evaluate_iouf1(pred, val_labels,dataloader_val.dataset.classes)

        iou_classes_per_epoch.append(iou_classes)
        iou_mean_per_epoch.append(iou_mean)

        #sum number of all seen images over all epochs
        n += val_labels.size()[0]

    #loss per epoch
    loss_per_epoch = np.array(loss_per_epoch)
    iou_classes_per_epoch = np.array(iou_classes_per_epoch)
    iou_classes_per_epoch_T = np.transpose(iou_classes_per_epoch, (1,0))

    mean_iou_classes_per_epoch = []
    for idx in range(iou_classes_per_epoch_T.shape[0]):
        mean_iou_classes_per_epoch.append(np.mean(iou_classes_per_epoch_T[idx]))

 
    if printar:
      print("val IoU")
      for classe in range(iou_classes_per_epoch_T.shape[0]):
        print(f"Classe:{classe} - mean: {iou_classes_per_epoch[classe].mean():.3f}")

    if plotar:
      plot_segmentation_results(val_images,val_labels,y_hat,0,name,dataset=dataloader_val.dataset.dataset_name)

    print("")
    print(f'val loss {np.mean(loss_per_epoch):.5f},  val mean iou general: {np.round(np.mean(iou_mean_per_epoch),4)}')

    return np.array(mean_iou_classes_per_epoch) ,  np.array(np.std(iou_classes_per_epoch)), np.array(np.mean(iou_mean_per_epoch)), np.mean(loss_per_epoch), np.std(loss_per_epoch),  np.array(val_pred_list), np.array(val_semantic_label_list)




