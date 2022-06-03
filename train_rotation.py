
import torch
from torch import nn, optim
import os, time
import numpy as np

def train_rotation(dataloader_train, dataloader_validation, model, optimizer, criterion, num_epochs, device, scheduler=None, task=None, tolerancia = 100,save=False,saved_models_path=None, name_model=None):
    assert task=='rotation' , "Not recognized task or implemented task for this function: {task}"
    if save:
      assert saved_models_path != None and name_model != None, 'No path or name specified'

    # Training procedure.
    print('training on', device)

    # Setting network for training mode.
    model.train()
    start = time.time()
    smallest_loss = np.inf
    counter = 0
    lr = scheduler.get_last_lr()

    # Lists for losses and metrics.
    train_loss_all, train_loss_std_all, train_acc_all, train_acc_std_all  = [], [], [], []
    val_loss_all, val_loss_std_all, val_acc_all, val_acc_std_all = [],[], [], []

    # Iterating over epochs.
    for epoch in range(num_epochs):
      
      model.train()
      print(f"Training epoch {epoch+1} / {num_epochs}" )
      #Training Procedure
      loss_per_epoch, acc_per_epoch =[], []
      train_loss_sum, train_acc_sum = 0.0, 0.0 
      n = 0 

      #First Load the mini batch and iterate over it
      for idx, (train_images, train_labels,name) in enumerate(dataloader_train):

        if idx % 2 ==0 :
          print(f'{idx}/{ len(dataloader_train.dataset.sections) /dataloader_train.batch_size}')
        
        #Cast images to device
        train_images = train_images.to(device)
        train_labels = train_labels.to(device)

        #zero the gradients
        optimizer.zero_grad()
        
        #Predict
        y_hat = model(train_images)

        #get prediction of max value
        _,pred = torch.max(y_hat, axis=1) 

        #Compute Loss and sum it up
        l = criterion(y_hat, train_labels).sum()

        #allocate on gpu
        y_hat = y_hat.to(device)

        #Get Gradients
        l.backward()
        #Uptade
        optimizer.step()

        ##return item of the tensor l and sum up over all batch images
        train_loss_sum += l.item() 

        #loss per epochs
        loss_per_epoch.append(l.item())


        #get accuracy and sum over all images of bathc
        acc_sum = (y_hat.argmax(axis=1) == train_labels).sum().item() 
        acc_per_epoch.append(acc_sum/train_labels.size()[0])

        #sum number of all seen images over one epoch
        n += train_labels.size()[0]
      
      #call lr scheduler
      if scheduler != None:
        scheduler.step()
        if scheduler.get_last_lr() != lr:
            print(f'Lr updated to: {scheduler.get_last_lr()}')

      #acc 
      acc_per_epoch = np.array(acc_per_epoch)
      acc_mean = np.mean(acc_per_epoch)
      acc_std = np.std(acc_per_epoch)
      
      #loss std per epoch
      loss_mean= np.array(np.mean(loss_per_epoch))
      loss_std= np.array(np.std(loss_per_epoch))

      #Save for all epochs
      train_acc_all.append(acc_mean) 
      train_acc_std_all.append(np.std(np.array(acc_std)))
      train_loss_all.append(loss_mean)
      train_loss_std_all.append(loss_std)

      #Print relevant info
      print(f'train loss {loss_mean:.5}, train acc {acc_mean:.4f}, time {time.time() - start:.1f} sec, n={n}')
      print("")


      #after entire epoch, evaluate on validation set
      val_acc,val_acc_std, val_loss, val_loss_std, val_predictions, val_labels = validade_rotation(dataloader_validation, model, criterion, device)
      val_acc_all.append(val_acc)
      val_acc_std_all.append(val_acc_std)
      val_loss_all.append(val_loss)
      val_loss_std_all.append(val_loss_std)


      if val_loss < smallest_loss:
        smallest_loss = val_loss
        counter = 0
        if save:
            torch.save(model.state_dict(), os.path.join(saved_models_path, name_model))

      else:
        counter += 1
        print(f"counter at {counter}/ {tolerancia}")
        if counter > tolerancia:
          break


      print("")
    return (np.array(train_acc_all), np.array(train_acc_std_all), np.array(train_loss_all), np.array(train_loss_std_all), 
            np.array(val_acc_all), np.array(val_acc_std_all), np.array(val_loss_all), np.array(val_loss_std_all)
            )

def test_rotation(dataloader_test, model, criterion, device): 

    """Evaluate accuracy of a model on the given data set."""

    acc_sum,  loss_sum = 0.0, 0
    loss_per_epoch, acc_per_epoch = [], []
    pred_list, rotation_label_list= [],[]
    n = 0


    model.eval()
    with torch.no_grad():
      for idx, (test_images, test_labels,name) in enumerate((dataloader_test)):   

        if idx % 2 ==0 :
          print(f'{idx}/{ len(dataloader_test.dataset.sections) /dataloader_test.batch_size}') 


        #Cast images to device
        test_images =test_images.to(device)
        test_labels =test_labels.to(device)

        #Predict
        y_hat = model(test_images)
        #get prediction of max value
        _,pred = torch.max(y_hat, axis=1)

        #list with predictions and labels
        pred_list.append(pred.cpu().numpy())
        rotation_label_list.append(test_labels.cpu().numpy())

        #allocate on device
        y_hat = y_hat.to(device)
        
        l = criterion(y_hat, test_labels).sum()

        #return item of the tensor l and sum up over all batch images
        loss_sum += l.item()

        #loss per epochs
        loss_per_epoch.append(l.item())

        #get accuracy and sum over all images of batch
        acc_sum = (y_hat.argmax(axis=1) == test_labels).sum().item()
        acc_per_epoch.append(acc_sum/test_labels.size()[0])


        #sum number of all seen images over all epochs
        n += test_labels.size()[0]

    
    #test_loss_mean
    loss_per_epoch = np.array(loss_per_epoch)
    loss_mean = np.mean(loss_per_epoch)
    loss_std = np.std(loss_per_epoch)

    #accuracy mean per epoch 
    acc_per_epoch = np.array(acc_per_epoch)
    acc_mean = np.mean(acc_per_epoch)
    acc_std = np.std(acc_per_epoch)
    
    print("")
    print(f'test loss {loss_mean:.5f},  test acc {acc_mean:.5f}')

    return acc_mean, acc_std, loss_mean,  loss_std,  np.hstack(pred_list), np.hstack(rotation_label_list)

def validade_rotation(dataloader_validation, model, criterion, device): 

    """Evaluate accuracy of a model on the given data set."""

    #acc_sum, n, l = torch.Tensor([0]), 0, 0
    acc_sum,  loss_sum = 0.0, 0
    loss_per_epoch, acc_per_epoch = [], []
    pred_list, rotation_label_list= [],[]
    n = 0


    model.eval()
    with torch.no_grad():
      for idx, (val_images, val_labels,name) in enumerate((dataloader_validation)):   

        if idx % 50 ==0 :
          print(f'{idx}/{ len(dataloader_validation.dataset.sections) /dataloader_validation.batch_size}') 


        #Cast images to device
        val_images =val_images.to(device)
        val_labels =val_labels.to(device)

        #Predict
        y_hat = model(val_images)
        #get prediction of max value
        _,pred = torch.max(y_hat, axis=1)


        #list with predictions and labels
        pred_list.append(pred.cpu().numpy())
        rotation_label_list.append(val_labels.cpu().numpy())

        #allocate on device
        y_hat = y_hat.to(device)
        
        l = criterion(y_hat, val_labels).sum()

        #return item of the tensor l and sum up over all batch images
        loss_sum += l.item()

        #loss per epochs
        loss_per_epoch.append(l.item())

        #get accuracy and sum over all images of batch
        acc_sum = (y_hat.argmax(axis=1) == val_labels).sum().item()
        acc_per_epoch.append(acc_sum/val_labels.size()[0])


        #sum number of all seen images over all epochs
        n += val_labels.size()[0]

    #val_loss_mean
    loss_per_epoch = np.array(loss_per_epoch)
    loss_mean = np.mean(loss_per_epoch)
    loss_std = np.std(loss_per_epoch)

    #val_acc_mean
    acc_per_epoch = np.array(acc_per_epoch)
    acc_mean = np.mean(acc_per_epoch)
    acc_std = np.std(acc_per_epoch)
    
    print("")
    print(f'val loss {loss_mean:.5f},  val acc {acc_mean:.5f}')

    return acc_mean, acc_std, loss_mean,  loss_std,  np.hstack(pred_list), np.hstack(rotation_label_list)


