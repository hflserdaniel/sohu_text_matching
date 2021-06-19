from model import BertClassifierSingleModel, NezhaClassifierSingleModel, SBERTSingleModel, SNEZHASingleModel, BertClassifierTextCNNSingleModel
from data import SentencePairDatasetWithType, SentencePairDatasetForSBERT
from utils import focal_loss, FGM
from transformers import AdamW, get_linear_schedule_with_warmup
from config import Config

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
from sklearn import metrics
from tensorboardX import SummaryWriter

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,4,5,6,7'    # recommended for NEZHA
# os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,4'

def train(model, device, epoch, train_dataloader, test_dataloader, save_dir, optimizer, scheduler=None,  criterion_type='CE', model_type='bert', print_every=100, eval_every=500, writer=None, use_fgm=False):
    print("Training at epoch {}".format(epoch))
    if use_fgm:
        print("Using fgm for adversial attack")

    est_batch = len(train_dataloader.dataset) / (train_dataloader.batch_size)
    model.train()

    # for multiple GPU support
    model = torch.nn.DataParallel(model)
    
    assert criterion_type == 'CE' or criterion_type == 'FL'
    if criterion_type == 'CE':
        criterion = nn.CrossEntropyLoss()
    elif criterion_type == 'FL':
        criterion = focal_loss()

    if use_fgm:
        fgm = FGM(model)

    total_loss = []
    total_gt_a, total_preds_a =  [], []
    total_gt_b, total_preds_b = [], []

    # the following commented code is compatitable with multitasks (e.g. 6 subtasks with designated dataset)
    # however, the model's performance seems to be influenced by 1 precent in task b
    # for idx, batch in enumerate(train_dataloader):
    #     # for SentencePairDatasetWithType, types would be returned
    #     input_ids, input_types, labels, types = batch
    #     input_ids = input_ids.to(device)
    #     input_types = input_types.to(device)
    #     # labels should be flattened
    #     labels = labels.to(device).view(-1)
        
    #     optimizer.zero_grad()
        
    #     # the probs given by the model
    #     all_probs = model(input_ids, input_types)
    #     num_tasks = len(all_probs)

    #     all_masks = [(types==task_id).numpy() for task_id in range(num_tasks)]
    #     all_output = [all_probs[task_id][all_masks[task_id]] for task_id in range(num_tasks)]
    #     all_labels = [labels[all_masks[task_id]] for task_id in range(num_tasks)]
        
    #     # calculate the loss and BP
    #     # TODO: different weights for each task?
    #     all_loss = None
    #     for task_id in range(num_tasks):
    #         if all_masks[task_id].sum() != 0:
    #             if all_loss is None:
    #                 all_loss = criterion(all_output[task_id], all_labels[task_id])
    #             else:
    #                 all_loss += criterion(all_output[task_id], all_labels[task_id])
    #     all_loss.backward()
        
    #     # code for fgm adversial training 
    #     if use_fgm:
    #         fgm.attack()
    #         # adv_probs_a, adv_probs_b = model(input_ids, input_types)
    #         adv_all_probs = model(input_ids, input_types)
    #         adv_all_output = [adv_all_probs[task_id][all_masks[task_id]] for task_id in range(num_tasks)]
    #         # calculate the loss and BP
    #         adv_all_loss = None
    #         for task_id in range(num_tasks):
    #             if all_masks[task_id].sum() != 0:
    #                 if adv_all_loss is None:
    #                     adv_all_loss = criterion(adv_all_output[task_id], all_labels[task_id])
    #                 else:
    #                     adv_all_loss += criterion(adv_all_output[task_id], all_labels[task_id])
    #         adv_all_loss.backward()
    #         fgm.restore()

    #     optimizer.step()
    #     if scheduler is not None:
    #         scheduler.step()
        

    #     all_gt = [all_labels[task_id].cpu().numpy().tolist() if all_masks[task_id].sum()!=0 else [] for task_id in range(num_tasks)]
    #     all_preds = [all_output[task_id].argmax(axis=1).cpu().numpy().tolist() if all_masks[task_id].sum()!=0 else [] for task_id in range(num_tasks)]

    #     gt_a, preds_a = [], []
    #     for task_id in range(0, num_tasks, 2):
    #         gt_a += all_gt[task_id]
    #         preds_a += all_preds[task_id]

    #     gt_b, preds_b = [], []
    #     for task_id in range(1, num_tasks, 2):
    #         gt_b += all_gt[task_id]
    #         preds_b += all_preds[task_id]

    #     total_preds_a += preds_a
    #     total_gt_a += gt_a
    #     total_preds_b += preds_b
    #     total_gt_b += gt_b
    #     total_loss.append(all_loss.item())
    #     # print('a', preds_a, gt_a)
    #     # print('b', preds_b, gt_b)
        
    #     acc_a = metrics.accuracy_score(gt_a, preds_a) if len(gt_a)!=0 else 0
    #     f1_a = metrics.f1_score(gt_a, preds_a, zero_division=0)
    #     acc_b = metrics.accuracy_score(gt_b, preds_b) if len(gt_b)!=0 else 0
    #     f1_b = metrics.f1_score(gt_b, preds_b, zero_division=0)

    #     # learning rate for bert is the second (the last) parameter group
    #     writer.add_scalar('train/learning_rate', optimizer.param_groups[-1]['lr'], global_step=epoch*est_batch+idx)
    #     writer.add_scalar('train/loss', all_loss.item(), global_step=epoch*est_batch+idx)
    #     writer.add_scalar('train/acc_a', acc_a, global_step=epoch*est_batch+idx)
    #     writer.add_scalar('train/acc_b', acc_b, global_step=epoch*est_batch+idx)
    #     writer.add_scalar('train/f1_a', f1_a, global_step=epoch*est_batch+idx)
    #     writer.add_scalar('train/f1_b', f1_b, global_step=epoch*est_batch+idx)

    #     # print the loss and accuracy score if reach print_every 
    #     if (idx+1) % print_every == 0:
    #         print("\tBatch: {} / {:.0f}, Loss: {:.6f}".format(idx, est_batch, all_loss.item()))
    #         print("\t\t Task A\tAcc: {:.6f}, F1: {:.6f}".format(acc_a, f1_a))
    #         print("\t\t Task B\tAcc: {:.6f}, F1: {:.6f}".format(acc_b, f1_b))

    #     # evaluate the model if reach eval_every, instead of evaluate after the whole epoch
    #     global best_dev_loss, best_dev_f1
    #     if (idx+1) % eval_every == 0:
    #         dev_loss, dev_acc_a, dev_acc_b, dev_f1_a, dev_f1_b = eval(model, device, test_dataloader, criterion_type)
    #         dev_f1 = (dev_f1_a + dev_f1_b) / 2 
    #         writer.add_scalar('eval/loss', dev_loss, global_step=epoch*est_batch+idx)
    #         writer.add_scalar('eval/acc_a', dev_acc_a, global_step=epoch*est_batch+idx)
    #         writer.add_scalar('eval/acc_b', dev_acc_b, global_step=epoch*est_batch+idx)
    #         writer.add_scalar('eval/f1_a', dev_f1_a, global_step=epoch*est_batch+idx)
    #         writer.add_scalar('eval/f1_b', dev_f1_b, global_step=epoch*est_batch+idx)
    #         # in practice, better loss is preferred instead of better f1 score,
    #         # which could be resulted from random overfitting on the valid set
    #         # 0517: save the model's state_dict instead of the whole model (mainly for NEZHA's sake)
    #         if (dev_loss < best_dev_loss or dev_f1 > best_dev_f1):
    #             if dev_loss < best_dev_loss:
    #                 best_dev_loss = dev_loss
    #                 torch.save(model.state_dict(), save_dir + model_type + '_epoch_{}_{}_'.format(epoch, task_type) + 'loss')
    #                 print("----------BETTER LOSS, MODEL SAVED-----------")
    #             if dev_f1 > best_dev_f1:
    #                 best_dev_f1 = dev_f1
    #                 torch.save(model.state_dict(), save_dir + model_type + '_epoch_{}_{}_'.format(epoch, task_type) + 'f1')
    #                 print("----------BETTER F1, MODEL SAVED-----------")
    
    for idx, batch in enumerate(train_dataloader):
        # for SentencePairDatasetWithType, types would be returned
        input_ids, input_types, labels, types = batch
        input_ids = input_ids.to(device)
        input_types = input_types.to(device)
        # labels should be flattened
        labels = labels.to(device).view(-1)
        
        optimizer.zero_grad()
        
        # the probs given by the model
        probs_a, probs_b = model(input_ids, input_types)
        
        mask_a, mask_b = (types==0).numpy(), (types==1).numpy()
        output_a, labels_a = probs_a[mask_a], labels[mask_a]
        output_b, labels_b = probs_b[mask_b], labels[mask_b]
        
        # calculate the loss and BP
        # loss_a = criterion(output_a, labels_a) if mask_a.sum()!=0 else None
        # loss_b = criterion(output_b, labels_b) if mask_b.sum()!=0 else None
        # so-called multi-task training
        # TODO: different weights for each task?
        if mask_a.sum()==0:
            loss = criterion(output_b, labels_b)
        elif mask_b.sum()==0:
            loss = criterion(output_a, labels_a)
        else:
            loss = criterion(output_a, labels_a) + criterion(output_b, labels_b)
        # print(loss.item())
        loss.backward()
        
        # code for fgm adversial training 
        if use_fgm:
            fgm.attack()
            adv_probs_a, adv_probs_b = model(input_ids, input_types)
            # calculate the loss and BP
            adv_output_a, adv_output_b = adv_probs_a[mask_a], adv_probs_b[mask_b]
            if mask_a.sum()==0:
                adv_loss = criterion(adv_output_b, labels_b)
            elif mask_b.sum()==0:
                adv_loss = criterion(adv_output_a, labels_a)
            else:
                adv_loss = criterion(adv_output_a, labels_a) + criterion(adv_output_b, labels_b)
            adv_loss.backward()
            fgm.restore()

        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        gt_a = labels_a.cpu().numpy().tolist()
        preds_a = output_a.argmax(axis=1).cpu().numpy().tolist() if len(gt_a)!=0 else []

        gt_b = labels_b.cpu().numpy().tolist()
        preds_b = output_b.argmax(axis=1).cpu().numpy().tolist() if len(gt_b)!=0 else []
        
        total_preds_a += preds_a
        total_gt_a += gt_a
        total_preds_b += preds_b
        total_gt_b += gt_b
        total_loss.append(loss.item())
        # print('a', preds_a, gt_a)
        # print('b', preds_b, gt_b)
        
        acc_a = metrics.accuracy_score(gt_a, preds_a) if len(gt_a)!=0 else 0
        f1_a = metrics.f1_score(gt_a, preds_a, zero_division=0)
        acc_b = metrics.accuracy_score(gt_b, preds_b) if len(gt_b)!=0 else 0
        f1_b = metrics.f1_score(gt_b, preds_b, zero_division=0)

        writer.add_scalar('train/learning_rate', optimizer.param_groups[-1]['lr'], global_step=epoch*est_batch+idx)
        writer.add_scalar('train/loss', loss.item(), global_step=epoch*est_batch+idx)
        writer.add_scalar('train/acc_a', acc_a, global_step=epoch*est_batch+idx)
        writer.add_scalar('train/acc_b', acc_b, global_step=epoch*est_batch+idx)
        writer.add_scalar('train/f1_a', f1_a, global_step=epoch*est_batch+idx)
        writer.add_scalar('train/f1_b', f1_b, global_step=epoch*est_batch+idx)

        # print the loss and accuracy score if reach print_every 
        if (idx+1) % print_every == 0:
            print("\tBatch: {} / {:.0f}, Loss: {:.6f}".format(idx, est_batch, loss.item()))
            print("\t\t Task A\tAcc: {:.6f}, F1: {:.6f}".format(acc_a, f1_a))
            # if (f1_a == 0):
            #     print(metrics.precision_recall_fscore_support(gt_a, preds_a, zero_division=0))
            
            print("\t\t Task B\tAcc: {:.6f}, F1: {:.6f}".format(acc_b, f1_b))
            # if (f1_b == 0):
            #     print(metrics.precision_recall_fscore_support(gt_b, preds_b, zero_division=0))
        # evaluate the model if reach eval_every, instead of evaluate after the whole epoch
        global best_dev_loss, best_dev_f1
        if (idx+1) % eval_every == 0:
            dev_loss, dev_acc_a, dev_acc_b, dev_f1_a, dev_f1_b = eval(model, device, test_dataloader, criterion_type)
            dev_f1 = (dev_f1_a + dev_f1_b) / 2 
            writer.add_scalar('eval/loss', dev_loss, global_step=epoch*est_batch+idx)
            writer.add_scalar('eval/acc_a', dev_acc_a, global_step=epoch*est_batch+idx)
            writer.add_scalar('eval/acc_b', dev_acc_b, global_step=epoch*est_batch+idx)
            writer.add_scalar('eval/f1_a', dev_f1_a, global_step=epoch*est_batch+idx)
            writer.add_scalar('eval/f1_b', dev_f1_b, global_step=epoch*est_batch+idx)
            if (dev_loss < best_dev_loss or dev_f1 > best_dev_f1):
                if dev_loss < best_dev_loss:
                    best_dev_loss = dev_loss
                    # torch.save(model, save_dir + model_type + '_epoch_{}_{}_'.format(epoch, task_type) + 'loss')
                    torch.save(model.state_dict(), save_dir + model_type + '_epoch_{}_{}_'.format(epoch, task_type) + 'loss')                 
                    print("----------BETTER LOSS, MODEL SAVED-----------")
                if dev_f1 > best_dev_f1:
                    best_dev_f1 = dev_f1
                    # torch.save(model, save_dir + model_type + '_epoch_{}_{}_'.format(epoch, task_type) + 'f1')
                    torch.save(model.state_dict(), save_dir + model_type + '_epoch_{}_{}_'.format(epoch, task_type) + 'f1')
                    print("----------BETTER F1, MODEL SAVED-----------")

    loss = np.array(total_loss).mean()
    # Setting average=None to return class-specific scores
    # 0502 BUG FIXED: do not use 'macro', DO NOT require class-specific metrics!
    # macro_f1 = metrics.f1_score(total_gt, total_preds, average='macro')
    f1_a = metrics.f1_score(total_gt_a, total_preds_a, zero_division=0)
    f1_b = metrics.f1_score(total_gt_b, total_preds_b, zero_division=0)
    f1 = (f1_a + f1_b) / 2
    print("Average f1 on training set: {:.6f}, f1_a: {:.6f}, f1_b: {:.6f}".format(f1, f1_a, f1_b))

    return loss, f1, f1_a, f1_b


def eval(model, device, test_dataloader, criterion_type='CE'):
    print("Evaluating")
    model.eval()
    # if called while training, then model parallel is already done
    # model = torch.nn.DataParallel(model)

    assert criterion_type == 'CE' or criterion_type == 'FL'
    if criterion_type == 'CE':
        criterion = nn.CrossEntropyLoss()
    elif criterion_type == 'FL':
        criterion = focal_loss()

    total_loss = []
    total_gt_a, total_preds_a =  [], []
    total_gt_b, total_preds_b = [], []

    for idx, batch in enumerate(test_dataloader):
        input_ids, input_types, labels, types = batch
        input_ids = input_ids.to(device)
        input_types = input_types.to(device)
        # labels should be flattened
        labels = labels.to(device).view(-1)

        # the probs given by the model, without grads 
        with torch.no_grad():
            # the probs given by the model
            probs_a, probs_b = model(input_ids, input_types)
            mask_a, mask_b = (types==0).numpy(), (types==1).numpy()
            output_a, labels_a = probs_a[mask_a], labels[mask_a]
            output_b, labels_b = probs_b[mask_b], labels[mask_b]

            if mask_a.sum()==0:
                loss = criterion(output_b, labels_b)
            elif mask_b.sum()==0:
                loss = criterion(output_a, labels_a)
            else:
                loss = criterion(output_a, labels_a) + criterion(output_b, labels_b)

            gt_a = labels_a.cpu().numpy().tolist()
            preds_a = output_a.argmax(axis=1).cpu().numpy().tolist() if len(gt_a)!=0 else []

            gt_b = labels_b.cpu().numpy().tolist()
            preds_b = output_b.argmax(axis=1).cpu().numpy().tolist() if len(gt_b)!=0 else []
            
            total_preds_a += preds_a
            total_gt_a += gt_a
            total_preds_b += preds_b
            total_gt_b += gt_b
            total_loss.append(loss.item())

    loss = np.array(total_loss).mean()
    acc_a = metrics.accuracy_score(total_gt_a, total_preds_a) if len(total_gt_a)!=0 else 0
    f1_a = metrics.f1_score(total_gt_a, total_preds_a, zero_division=0)
    if (f1_a == 0):
        print("F1_a = 0, checking precision, recall, fscore and support...")
        print(metrics.precision_recall_fscore_support(total_gt_a, total_preds_a, zero_division=0))
    
    acc_b = metrics.accuracy_score(total_gt_b, total_preds_b) if len(total_gt_b)!=0 else 0
    f1_b = metrics.f1_score(total_gt_b, total_preds_b, zero_division=0)
    if (f1_b == 0):
        print("F1_b = 0, checking precision, recall, fscore and support...")
        print(metrics.precision_recall_fscore_support(total_gt_b, total_preds_b, zero_division=0))

    # Setting average=None to return class-specific scores
    # macro_f1 = metrics.f1_score(total_gt, total_preds, average='macro')
    # f1 = metrics.f1_score(total_gt, total_preds)
    
    # print loss and classification report    
    print("Loss on dev set: ", loss)
    print("F1 on dev set: {:.6f}, f1_a: {:.6f}, f1_b: {:.6f}".format((f1_a+f1_b)/2, f1_a, f1_b))

    # return loss, acc, macro_f1
    return loss, acc_a, acc_b, f1_a, f1_b


if __name__ == '__main__':
    config = Config()
    device = config.device
    pretrained = config.pretrained
    model_type = config.model_type
    use_fgm = config.use_fgm

    save_dir = config.save_dir
    data_dir = config.data_dir
    # whether to shuffle the pos of source and target to augment data
    shuffle_order = config.shuffle_order
    # whether to use the positive case in task b for task a (positives)
    # and to use the negativate case in task a for task b (negatives)
    aug_data = config.aug_data
    # method for clipping long seqeunces, 'head' or 'tail'
    clip_method = config.clip_method

    task_type = config.task_type
    task_a = ['短短匹配A类',  '短长匹配A类', '长长匹配A类']
    task_b = ['短短匹配B类',  '短长匹配B类', '长长匹配B类']

    # hypter parameters here
    epochs = config.epochs
    lr = config.lr
    classifer_lr = config.classifier_lr
    weight_decay = config.weight_decay
    hidden_size = config.hidden_size
    train_bs = config.train_bs
    eval_bs = config.eval_bs

    print_every = config.print_every
    eval_every = config.eval_every

    train_data_dir, dev_data_dir = [], []
    # integrate the two tasks into one dataset using task_type = 'ab'
    if 'a' in task_type:
        for task in task_a:
            train_data_dir.append(data_dir + task + '/train.txt')
            train_data_dir.append(data_dir + task + '/train_r2.txt')
            train_data_dir.append(data_dir + task + '/train_r3.txt')
            train_data_dir.append(data_dir + task + '/train_rematch.txt')   # training file in rematch
            dev_data_dir.append(data_dir + task + '/valid.txt')
            dev_data_dir.append(data_dir + task + '/valid_rematch.txt')     # valid file in rematch

    if 'b' in task_type:
        for task in task_b:
            train_data_dir.append(data_dir + task + '/train.txt')
            train_data_dir.append(data_dir + task + '/train_r2.txt')
            train_data_dir.append(data_dir + task + '/train_r3.txt')
            train_data_dir.append(data_dir + task + '/train_rematch.txt')   # training file in rematch
            dev_data_dir.append(data_dir + task + '/valid.txt')
            dev_data_dir.append(data_dir + task + '/valid_rematch.txt')     # valid file in rematch

    # toy dataset for testing
    if config.load_toy_dataset:
        train_data_dir = [
            '../data/sohu2021_open_data/短短匹配A类/valid.txt', 
            '../data/sohu2021_open_data/短短匹配B类/valid.txt',
            '../data/sohu2021_open_data/短长匹配A类/valid.txt',
            '../data/sohu2021_open_data/短长匹配B类/valid.txt',
            '../data/sohu2021_open_data/长长匹配A类/valid.txt', 
            '../data/sohu2021_open_data/长长匹配B类/valid.txt']
        dev_data_dir = ['../data/sohu2021_open_data/短短匹配A类/valid.txt',
                        '../data/sohu2021_open_data/短短匹配B类/valid.txt',]


    print("Loading pretrained Model from {}...".format(pretrained))
    # integrating SBERT model into a unified training framework
    if 'sbert' in model_type.lower():
        print("Using SentenceBERT model and dataset")
        if 'nezha' in model_type.lower():
            model = SNEZHASingleModel(bert_dir=pretrained, hidden_size=hidden_size)
        else:
            model = SBERTSingleModel(bert_dir=pretrained, hidden_size=hidden_size)
        model.to(device)
        print("Loading Training Data...")
        print(train_data_dir)
        # augment the data with shuffle_order=True (changing order of source and target)
        # or with aug_data=True (neg cases in A -> B, pos cases in B -> A)
        train_dataset = SentencePairDatasetForSBERT(train_data_dir, True, pretrained, shuffle_order, aug_data=aug_data, clip=clip_method)
        train_dataloader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True)

        print("Loading Dev Data...")
        test_dataset = SentencePairDatasetForSBERT(dev_data_dir, True, pretrained, clip=clip_method)
        test_dataloader = DataLoader(test_dataset, batch_size=eval_bs, shuffle=False)

    # 0517: for training, load weights from pretrained with from_pretrained=True (by default)
    # for larger model, adjust the hidden_size according to its config
    # distinguish model architectures or pretrained models according to model_type
    else:
        print("Using BERT model and dataset")
        if 'nezha' in model_type.lower():
            print("Using NEZHA pretrained model")
            model = NezhaClassifierSingleModel(bert_dir=pretrained, hidden_size=hidden_size)
        elif 'cnn' in model_type.lower():
            print("Adding TextCNN after BERT output")
            model = BertClassifierTextCNNSingleModel(bert_dir=pretrained, hidden_size=hidden_size)
        else:
            print("Using conventional BERT model with linears")
            model = BertClassifierSingleModel(bert_dir=pretrained, hidden_size=hidden_size)
        model.to(device)

        print("Loading Training Data...")
        print(train_data_dir)
        # augment the data with shuffle_order=True (changing order of source and target)
        # or with aug_data=True (neg cases in A -> B, pos cases in B -> A)
        train_dataset = SentencePairDatasetWithType(train_data_dir, True, pretrained, shuffle_order, aug_data=aug_data, clip=clip_method)
        train_dataloader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True)

        print("Loading Dev Data...")
        test_dataset = SentencePairDatasetWithType(dev_data_dir, True, pretrained, clip=clip_method)
        test_dataloader = DataLoader(test_dataset, batch_size=eval_bs, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, correct_bias=False)
    # 0514 setting different lr for bert encoder and classifier
    # TODO: verify the large lr works for classifiers 
    # optimizer = AdamW([
    #     {"params": model.all_classifier.parameters(), "lr": classifer_lr},
    #     {"params": model.bert.parameters()}],
    #     lr=lr)
        
    # for p in optimizer.param_groups:
    #     outputs = ''
    #     for k, v in p.items():
    #         if k is 'params':
    #             outputs += (k + ': ' + str(v[0].shape).ljust(30) + ' ')
    #         else:
    #             outputs += (k + ': ' + str(v).ljust(10) + ' ')
    #     print(outputs)

    total_steps = len(train_dataloader) * epochs

    # TODO: using ReduceLROnPlateau instead of linear scheduler
    if config.use_scheduler:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_training_steps = total_steps,
            num_warmup_steps = config.num_warmup_steps,
        )
    else:
        scheduler = None

    print("Training on Task {}...".format(task_type))
    writer = SummaryWriter('runs/{}'.format(model_type + '_' + task_type))
    
    best_dev_loss = 999
    best_dev_f1 = 0
    for epoch in range(epochs):
        train_loss, train_f1, train_f1_a, train_f1_b = train(model, device, epoch, train_dataloader, test_dataloader, \
                                                              save_dir, optimizer, scheduler=scheduler, model_type=model_type, \
                                                              print_every=print_every, eval_every=eval_every, writer=writer, use_fgm=use_fgm)
