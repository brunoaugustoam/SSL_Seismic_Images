
import torch
from torch import nn, optim

import os, time
import numpy as np
from aux_functions import *

def train_jigsaw(dataloader_train, dataloader_val, model, optimizer, criterion, num_epochs, device, scheduler=None,tolerancia = 40,save=False,saved_models_path=None, saving_logs_path=None,name_model=None):
    

    model.train()
    start = time.time()
    smallest_loss = np.inf
    counter = 0
    lr = scheduler.get_last_lr()

    train_acc_all, train_acc_std_all, train_loss_all, train_loss_std_all = [],[],[],[]
    #test_acc_mean_all, test_acc_std_all, test_loss_mean_all, test_loss_std_all = [],[],[],[]
    val_acc_mean_all, val_acc_std_all, val_loss_mean_all, val_loss_std_all = [],[],[],[]
    
    for epoch in range(num_epochs):

        model.train()
        print(f"Training epoch {epoch+1} / {num_epochs}" )
        pred_list,label_list,loss_per_epoch,acc_per_epoch = [],[],[],[]
        loss_sum,acc_sum = 0.0,0.0
        n=0

        for idx,(labels,tiles,labels_orig,tiles_orig,permutation_idx,sec_name) in enumerate(dataloader_train):

            if idx % 100==0:
                print( f'{idx}/{len(dataloader_train.dataset.sections)/dataloader_train.batch_size} ')

            #(1,9,135,180) -> tiles.reshape(9,1,135,180)
            train_images = tiles.permute(dims=(1,0,2,3)) 
            
            train_labels = labels.permute(dims=(1,0)) 

            #Cast images to device
            train_images = train_images.to(device) #uma imagem [9,1,135,180], sendo 9 os tiles
            train_labels = train_labels.to(device) #uma lista da permutacao utilizada

            optimizer.zero_grad()

            #Predict
            y_hat = model(train_images) #
            pred = y_hat.data.max(1)[1] #uma 'lista' de 9 indices correspondendo à permutacao

            #list of predictions and Labels
            pred_list.append(pred.cpu().numpy()) 
            label_list.append(train_labels.cpu().numpy())

            #Compute Loss and sum it up
            l = criterion(y_hat.squeeze(1), train_labels.squeeze(1).long()).sum()
            loss_sum += l.item() 
            loss_per_epoch.append(l.item())
            #ajustar para dar squeeze soh na dimensao do n_Channels e nao no batch size

            #allocate on gpu
            y_hat = y_hat.to(device) #atualizar o 'set_loss' pra CrossEntropy

            #Get Gradients
            l.backward()

            #Uptade
            optimizer.step()

            #Cacular a acuracia
            acc_sum = (y_hat.argmax(axis=1) == train_labels.permute(dims=(1,0))).sum().item() 
            acc_per_epoch.append(acc_sum/9)

            n += train_labels.size()[0]

        #call lr scheduler
        if scheduler != None:
            scheduler.step()
            if scheduler.get_last_lr() != lr:
                print(f'Lr updated to: {scheduler.get_last_lr()}')

        #acc_mean = acc_sum/ n
        acc_per_epoch = np.array(acc_per_epoch)
        train_acc_mean = np.mean(acc_per_epoch)
        train_acc_std = np.std(acc_per_epoch)

        #loss_mean = loss_sum / n
        loss_per_epoch = np.array(loss_per_epoch)
        train_loss_mean = np.mean(loss_per_epoch)
        train_loss_std = np.std(loss_per_epoch)

        #Print relevant info
        print(f'epoch {epoch + 1}/{num_epochs}, train loss {train_loss_mean:.5f} +- {train_loss_std:.5f}, train acc {train_acc_mean:.4f} +- {train_acc_std:.4f}, time {time.time() - start:.1f} sec, n={n}')
        print("")

        val_acc_mean, val_acc_std, val_loss_mean, val_loss_std, val_pred_list, val_label_list = val_jigsaw(dataloader_val,
									                                                                       model, 
									                                                                       criterion, 
									                                                                       device )

        #test_acc_mean, test_acc_std, test_loss_mean, test_loss_std, test_pred_list, test_label_list = test_jigsaw(dataloader_test,
        #                                                                                                   model, 
        #                                                                                                   optimizer,
        #                                                                                                   loss, 
        #                                                                                                   device)

        if val_loss_mean < smallest_loss:
            print(f'smallest_loss: {smallest_loss} -> {val_loss_mean}')
            smallest_loss = val_loss_mean
            counter = 0
            if save:
                torch.save(model.state_dict(), os.path.join(saved_models_path, name_model))

        elif val_loss_mean >= smallest_loss:
            counter += 1
            print(f"counter at {counter}/{tolerancia}")
            if counter > tolerancia:
                break



        train_acc_all.append(train_acc_mean)
        train_acc_std_all.append(train_acc_std)
        train_loss_all.append(train_loss_mean)
        train_loss_std_all.append(train_loss_std)

        #test_acc_mean_all.append(test_acc_mean)
        #test_acc_std_all.append(test_acc_std)
        #test_loss_mean_all.append(test_loss_mean)
        #test_loss_std_all.append(test_loss_std)

        val_acc_mean_all.append(val_acc_mean)
        val_acc_std_all.append(val_acc_std)
        val_loss_mean_all.append(val_loss_mean)
        val_loss_std_all.append(val_loss_std)

        '''save_jigsaw_log_results( saving_logs_path,name_model, 
                                        train_acc_mean,train_acc_std,train_loss_mean,train_loss_std,
                                        test_acc_mean,test_acc_std,test_loss_mean,test_loss_std,
                                        val_acc_mean,val_acc_std,val_loss_mean,val_loss_std 
                                        )'''

    return ( np.array(train_acc_all), np.array(train_acc_std_all), np.array(train_loss_all),  np.array(train_loss_std_all) ,
        #np.array(test_acc_mean_all), np.array(test_acc_std_all), np.array(test_loss_mean_all),  np.array(test_loss_std_all),
        np.array(val_acc_mean_all), np.array(val_acc_std_all), np.array(val_loss_mean_all),  np.array(val_loss_std_all)
        )


def test_jigsaw(dataloader_test, model, criterion, device, val=False):

    pred_list, label_list, loss_per_epoch, acc_per_epoch = [],[],[],[]
    loss_sum , acc_sum = 0.0, 0.0
    n=0

    model.eval()
    with torch.no_grad():
        for idx,(labels,tiles,labels_orig,tiles_orig,permutation_idx,sec_name) in enumerate(dataloader_test):
            if idx % 100 ==0 :
                print(f'{idx}/{ len(dataloader_test.dataset.sections) /dataloader_test.batch_size}')

            test_images = tiles.permute(dims=(1,0,2,3))
            test_labels = labels.permute(dims=(1,0)) 

            #Cast images to device
            test_images = test_images.to(device) #uma imagem [9,1,135,180], sendo 9 os tiles
            test_labels = test_labels.to(device) #uma lista da permutacao utilizada

            y_hat = model(test_images) 
            pred = y_hat.data.max(1)[1] #uma 'lista' de 9 indices correspondendo a permutacao
            #list of predictions and Labels
            pred_list.append(pred.cpu().numpy())
            label_list.append(test_labels.cpu().numpy())

            #Compute Loss and sum it up
            l = criterion(y_hat.squeeze(1), test_labels.squeeze(1).long()).sum()
            loss_sum += l.item() 
            loss_per_epoch.append(l.item())
            #ajustar para dar squeeze soh na dimensao do n_Channels e nao no batch size

            #allocate on gpu
            y_hat = y_hat.to(device) #atualizar o 'set_loss' pra CrossEntropy

            #Cacular a acuracia
            acc_sum = (y_hat.argmax(axis=1) == test_labels.permute(dims=(1,0))).sum().item() 
            acc_per_epoch.append(acc_sum/9)

            n += test_labels.size()[0]


    #acc_mean = acc_sum/ n
    acc_per_epoch = np.array(acc_per_epoch)
    acc_mean = np.mean(acc_per_epoch)
    acc_std = np.std(acc_per_epoch)

    #loss_mean = loss_sum / n
    loss_per_epoch = np.array(loss_per_epoch)
    loss_mean = np.mean(loss_per_epoch)
    loss_std = np.std(loss_per_epoch)

    #Print relevant info
    if val:
        print(f'val loss {loss_mean:.5} +- {loss_std:.5}, val acc {acc_mean:.4f} +- {acc_std:.4f}')
    else:
        print(f'test loss {loss_mean:.5} +- {loss_std:.5}, test acc {acc_mean:.4f} +- {acc_std:.4f}')
    print("")
    return np.array(acc_mean), np.array(acc_std), np.array(loss_mean),  np.array(loss_std),  np.array(pred_list), np.array(label_list)

def val_jigsaw(dataloader_val, model, criterion, device):

    pred_list, label_list, loss_per_epoch, acc_per_epoch = [],[],[],[]
    loss_sum , acc_sum = 0.0, 0.0
    n=0

    model.eval()
    with torch.no_grad():
        for idx,(labels,tiles,labels_orig,tiles_orig,permutation_idx,sec_name) in enumerate(dataloader_val):
            if idx % 100 ==0 :
                print(f'{idx}/{ len(dataloader_val.dataset.sections) /dataloader_val.batch_size}')

            val_images = tiles.permute(dims=(1,0,2,3))
            val_labels = labels.permute(dims=(1,0)) 

            #Cast images to device
            val_images = val_images.to(device) #uma imagem [9,1,135,180], sendo 9 os tiles
            val_labels = val_labels.to(device) #uma lista da permutacao utilizada

            y_hat = model(val_images) 
            pred = y_hat.data.max(1)[1] #uma 'lista' de 9 indices correspondendo a permutacao
            #list of predictions and Labels
            pred_list.append(pred.cpu().numpy())
            label_list.append(val_labels.cpu().numpy())

            #Compute Loss and sum it up
            l = criterion(y_hat.squeeze(1), val_labels.squeeze(1).long()).sum()
            loss_sum += l.item() 
            loss_per_epoch.append(l.item())
            #ajustar para dar squeeze soh na dimensao do n_Channels e nao no batch size

            #allocate on gpu
            y_hat = y_hat.to(device) #atualizar o 'set_loss' pra CrossEntropy

            #Cacular a acuracia
            acc_sum = (y_hat.argmax(axis=1) == val_labels.permute(dims=(1,0))).sum().item() 
            acc_per_epoch.append(acc_sum/9)

            n += val_labels.size()[0]


    #acc_mean = acc_sum/ n
    acc_per_epoch = np.array(acc_per_epoch)
    acc_mean = np.mean(acc_per_epoch)
    acc_std = np.std(acc_per_epoch)

    #loss_mean = loss_sum / n
    loss_per_epoch = np.array(loss_per_epoch)
    loss_mean = np.mean(loss_per_epoch)
    loss_std = np.std(loss_per_epoch)

    #Print relevant info
    print(f'val loss {loss_mean:.5} +- {loss_std:.5}, val acc {acc_mean:.4f} +- {acc_std:.4f}')
    print("")

    return np.array(acc_mean), np.array(acc_std), np.array(loss_mean),  np.array(loss_std),  np.array(pred_list), np.array(label_list)