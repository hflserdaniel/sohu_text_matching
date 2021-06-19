from model import BertClassifierSingleModel, NezhaClassifierSingleModel, SBERTSingleModel, SNEZHASingleModel, BertClassifierTextCNNSingleModel

from data import SentencePairDatasetWithType, SentencePairDatasetForSBERT
from config import Config

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
from sklearn import metrics
from tqdm import tqdm

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

def infer(model, device, dev_dataloader, test_dataloader, search_thres=True, threshold_fixed_a=0.5, threshold_fixed_b=0.5, save_valid=True):
    print("Inferring")
    model.eval()

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    total_gt_a, total_preds_a, total_probs_a =  [], [], []
    total_gt_b, total_preds_b, total_probs_b = [], [], []

    print("Model running on dev set...")
    for idx, batch in enumerate(tqdm(dev_dataloader)):
        input_ids, input_types, labels, types = batch
        input_ids = input_ids.to(device)
        input_types = input_types.to(device)
        # labels should be flattened
        labels = labels.to(device).view(-1)

        with torch.no_grad():
            all_probs = model(input_ids, input_types)
            num_tasks = len(all_probs)

            all_masks = [(types==task_id).numpy() for task_id in range(num_tasks)]
            all_output = [all_probs[task_id][all_masks[task_id]] for task_id in range(num_tasks)]
            all_labels = [labels[all_masks[task_id]] for task_id in range(num_tasks)]

            all_gt = [all_labels[task_id].cpu().numpy().tolist() if all_masks[task_id].sum()!=0 else [] for task_id in range(num_tasks)]
            all_preds = [all_output[task_id].argmax(axis=1).cpu().numpy().tolist() if all_masks[task_id].sum()!=0 else [] for task_id in range(num_tasks)]

            gt_a, preds_a, probs_a = [], [], []
            for task_id in range(0, num_tasks, 2):
                gt_a += all_gt[task_id]
                preds_a += all_preds[task_id]
                probs_a += [prob[-1] for prob in nn.functional.softmax(all_output[task_id], dim=1).cpu().numpy().tolist()]

            gt_b, preds_b, probs_b = [], [], []
            for task_id in range(1, num_tasks, 2):
                gt_b += all_gt[task_id]
                preds_b += all_preds[task_id]
                probs_b += [prob[-1] for prob in nn.functional.softmax(all_output[task_id], dim=1).cpu().numpy().tolist()]

            total_gt_a += gt_a
            total_preds_a += preds_a
            total_probs_a += probs_a

            total_gt_b += gt_b
            total_preds_b += preds_b
            total_probs_b += probs_b

    if search_thres:
        # search for the optimal threshold
        print("Searching for the best threshold on valid dataset...")
        thresholds = np.arange(0.2, 0.9, 0.01)
        fscore_a = np.zeros(shape=(len(thresholds)))
        fscore_b = np.zeros(shape=(len(thresholds)))
        print('Length of sequence: {}'.format(len(thresholds)))
        
        print("Original F1 Score for Task A: {}".format(str(metrics.f1_score(total_gt_a, total_preds_a, zero_division=0))))
        if len(total_gt_a) != 0:
            print("\tClassification Report\n")
            print(metrics.classification_report(total_gt_a, total_preds_a))

        print("Original F1 Score for Task B: {}".format(str(metrics.f1_score(total_gt_b, total_preds_b, zero_division=0))))
        if len(total_gt_b) != 0:
            print("\tClassification Report\n")
            print(metrics.classification_report(total_gt_b, total_preds_b))
            
        for index, thres in enumerate(tqdm(thresholds)):
            y_pred_prob_a = (np.array(total_probs_a) > thres).astype('int')
            fscore_a[index] = metrics.f1_score(total_gt_a, y_pred_prob_a.tolist(), zero_division=0)

            y_pred_prob_b = (np.array(total_probs_b) > thres).astype('int')
            fscore_b[index] = metrics.f1_score(total_gt_b, y_pred_prob_b.tolist(), zero_division=0)

        # record the optimal threshold for task A
        # print(fscore_a)
        index_a = np.argmax(fscore_a)
        threshold_opt_a = round(thresholds[index_a], ndigits=4)
        f1_score_opt_a = round(fscore_a[index_a], ndigits=6)
        print('Best Threshold for Task A: {} with F-Score: {}'.format(threshold_opt_a, f1_score_opt_a))
        # print("\nThreshold Classification Report\n")
        # print(metrics.classification_report(total_gt_a, (np.array(total_probs_a) > threshold_opt_a).astype('int').tolist()))

        # record the optimal threshold for task B
        index_b = np.argmax(fscore_b)
        threshold_opt_b = round(thresholds[index_b], ndigits=4)
        f1_score_opt_b = round(fscore_b[index_b], ndigits=6)
        print('Best Threshold for Task B: {} with F-Score: {}'.format(threshold_opt_b, f1_score_opt_b))
        # print("\nThreshold Classification Report\n")
        # print(metrics.classification_report(total_gt_b, (np.array(total_probs_b) > threshold_opt_b).astype('int').tolist()))

        if save_valid:
            y_pred_prob_a = (np.array(total_probs_a) > threshold_opt_a).astype('int')
            y_pred_prob_b = (np.array(total_probs_b) > threshold_opt_b).astype('int')
            # index of valid and valid_rematch
            # ssa, sla, lla = y_pred_prob_a[0:3395], y_pred_prob_a[3395:7681], y_pred_prob_a[7681:]
            # gt_ssa, gt_sla, gt_lla = total_gt_a[0:3395], total_gt_a[3395:7681], total_gt_a[7681:]
            # ssb, slb, llb = y_pred_prob_b[0:3393], y_pred_prob_b[3393:7684], y_pred_prob_b[7684:]
            # gt_ssb, gt_slb, gt_llb = total_gt_b[0:3393], total_gt_b[3393:7684], total_gt_b[7684:]

            # valid_rematch only
            ssa, sla, lla = y_pred_prob_a[0:1750], y_pred_prob_a[1750:4380], y_pred_prob_a[4380:]
            gt_ssa, gt_sla, gt_lla = total_gt_a[0:1750], total_gt_a[1750:4380], total_gt_a[4380:]
            ssb, slb, llb = y_pred_prob_b[0:1750], y_pred_prob_b[1750:4385], y_pred_prob_b[4385:]
            gt_ssb, gt_slb, gt_llb = total_gt_b[0:1750], total_gt_b[1750:4385], total_gt_b[4385:]
            print("f1 on ssa: ", metrics.f1_score(gt_ssa, ssa))
            print("f1 on sla: ", metrics.f1_score(gt_sla, sla))
            print("f1 on lla: ", metrics.f1_score(gt_lla, lla))
            print("f1 on ssb: ", metrics.f1_score(gt_ssb, ssb))
            print("f1 on slb: ", metrics.f1_score(gt_slb, slb))
            print("f1 on llb: ", metrics.f1_score(gt_llb, llb))
            
            np.save('../valid_output/{}_pred_a.npy'.format(model_type), y_pred_prob_a)
            np.save('../valid_output/{}_pred_b.npy'.format(model_type), y_pred_prob_b)
            np.save('../valid_output/gt_a.npy', np.array(total_gt_a))
            np.save('../valid_output/gt_b.npy', np.array(total_gt_b))

    total_ids_a, total_probs_a = [], []
    total_ids_b, total_probs_b = [], []
    for idx, batch in enumerate(tqdm(test_dataloader)):
        input_ids, input_types, ids, types = batch
        input_ids = input_ids.to(device)
        input_types = input_types.to(device)
        
        # the probs given by the model, without grads 
        with torch.no_grad():
            # probs_a, probs_b = model(input_ids, input_types)
            # mask_a, mask_b = (types==0).numpy(), (types==1).numpy()

            all_probs = model(input_ids, input_types)
            num_tasks = len(all_probs)
            # mask_a, mask_b = (types==0).numpy(), (types==1).numpy()
            all_masks = [(types==task_id).numpy() for task_id in range(num_tasks)]
            all_output = [all_probs[task_id][all_masks[task_id]] for task_id in range(num_tasks)]

            total_ids_a += [id for id in ids if id.endswith('a')]
            total_ids_b += [id for id in ids if id.endswith('b')]

            gt_a, preds_a, probs_a = [], [], []
            for task_id in range(0, num_tasks, 2):
                probs_a += [prob[-1] for prob in nn.functional.softmax(all_output[task_id], dim=1).cpu().numpy().tolist()]

            gt_b, preds_b, probs_b = [], [], []
            for task_id in range(1, num_tasks, 2):
                probs_b += [prob[-1] for prob in nn.functional.softmax(all_output[task_id], dim=1).cpu().numpy().tolist()]

            total_probs_a += probs_a
            total_probs_b += probs_b

    # positive if the prob passes the original threshold of 0.5
    total_fixed_preds_a = (np.array(total_probs_a) > threshold_fixed_a).astype('int').tolist()
    total_fixed_preds_b = (np.array(total_probs_b) > threshold_fixed_b).astype('int').tolist()
    
    if search_thres:
        # positive if the prob passes the optimal threshold
        total_preds_a = (np.array(total_probs_a) > threshold_opt_a).astype('int').tolist()
        total_preds_b = (np.array(total_probs_b) > threshold_opt_b).astype('int').tolist()
    else:
        total_preds_a = None
        total_preds_b = None

    return total_ids_a, total_preds_a, total_fixed_preds_a, \
           total_ids_b, total_preds_b, total_fixed_preds_b 

if __name__=='__main__':
    config = Config()
    device = config.device
    dummy_pretrained = config.dummy_pretrained
    model_type = config.infer_model_name

    save_dir = config.infer_model_dir
    model_name = config.infer_model_name
    hidden_size = config.hidden_size
    output_dir= config.infer_output_dir
    output_filename = config.infer_output_filename
    data_dir = config.data_dir
    task_a = ['短短匹配A类',  '短长匹配A类', '长长匹配A类']
    task_b = ['短短匹配B类',  '短长匹配B类', '长长匹配B类']
    task_type = config.infer_task_type

    infer_bs = config.infer_bs
    search_thres = config.infer_search_thres
    threshold_fixed_a = config.infer_fixed_thres_a
    threshold_fixed_b = config.infer_fixed_thres_b
    # method for clipping long seqeunces, 'head' or 'tail'
    clip_method = config.infer_clip_method
    
    dev_data_dir, test_data_dir = [], []
    if 'a' in task_type:
        for task in task_a:
            # dev_data_dir.append(data_dir + task + '/valid.txt')
            dev_data_dir.append(data_dir + task + '/valid_rematch.txt')
            test_data_dir.append(data_dir + task + '/test_with_id_rematch.txt')
    if 'b' in task_type:
        for task in task_b:
            # dev_data_dir.append(data_dir + task + '/valid.txt')
            dev_data_dir.append(data_dir + task + '/valid_rematch.txt')
            test_data_dir.append(data_dir + task + '/test_with_id_rematch.txt')

    print("Loading Bert Model from {}...".format(save_dir + model_name))
    # distinguish model architectures or pretrained models according to model_type
    if 'sbert' in model_type.lower():
        print("Using SentenceBERT model and dataset")
        if 'nezha' in model_type.lower():
            model = SNEZHASingleModel(bert_dir=dummy_pretrained, from_pretrained=False, hidden_size=hidden_size)
        else:
            model = SBERTSingleModel(bert_dir=dummy_pretrained, from_pretrained=False, hidden_size=hidden_size)
        
        model_dict = torch.load(save_dir + model_name)
        # model_dict = torch.load(save_dir + model_name)
        # weights will be saved in module when DataParallel
        model.load_state_dict({k.replace('module.','') : v for k, v in model_dict.items()})
        model.to(device)

        print("Loading Dev Data...")
        dev_dataset = SentencePairDatasetForSBERT(dev_data_dir, True, dummy_pretrained, clip=clip_method)
        dev_dataloader = DataLoader(dev_dataset, batch_size=infer_bs, shuffle=False)

        print("Loading Test Data...")
        # for test dataset, is_train should be set to False, thus get ids instead of labels
        test_dataset = SentencePairDatasetForSBERT(test_data_dir, False, dummy_pretrained, clip=clip_method)
        test_dataloader = DataLoader(test_dataset, batch_size=infer_bs, shuffle=False)

    else:
        print("Using BERT model and dataset")
        if 'nezha' in model_type.lower():
            print("Using NEZHA pretrained model")
            model = NezhaClassifierSingleModel(bert_dir=dummy_pretrained, from_pretrained=False, hidden_size=hidden_size)
        elif 'cnn' in model_type.lower():
            print("Adding TextCNN after BERT output")
            model = BertClassifierTextCNNSingleModel(bert_dir=dummy_pretrained, from_pretrained=False, hidden_size=hidden_size)
        else:
            print("Using conventional BERT model with linears")
            model = BertClassifierSingleModel(bert_dir=dummy_pretrained, from_pretrained=False, hidden_size=hidden_size)

        model_dict = torch.load(save_dir + model_name)
        # weights will be saved in module when DataParallel
        model.load_state_dict({k.replace('module.','') : v for k, v in model_dict.items()})
        model.to(device)

        # model_dict = torch.load(save_dir + model_name).module.state_dict()
        # model.load_state_dict(model_dict)
        # model.to(device)

        print("Loading Dev Data...")
        dev_dataset = SentencePairDatasetWithType(dev_data_dir, True, dummy_pretrained, clip=clip_method)
        dev_dataloader = DataLoader(dev_dataset, batch_size=infer_bs, shuffle=False)

        print("Loading Test Data...")
        # for test dataset, is_train should be set to False, thus get ids instead of labels
        test_dataset = SentencePairDatasetWithType(test_data_dir, False, dummy_pretrained, clip=clip_method)
        test_dataloader = DataLoader(test_dataset, batch_size=infer_bs, shuffle=False)

    total_ids_a, total_preds_a, total_fixed_preds_a, total_ids_b, total_preds_b, total_fixed_preds_b = infer(model, device, dev_dataloader, test_dataloader, search_thres, threshold_fixed_a, threshold_fixed_b)
    
    with open(output_dir + 'fixed_' + output_filename, 'w') as f_out:
        for id, pred in zip(total_ids_a, total_fixed_preds_a):   
            f_out.writelines(str(id) + ',' + str(pred) + '\n')
        for id, pred in zip(total_ids_b, total_fixed_preds_b):   
            f_out.writelines(str(id) + ',' + str(pred) + '\n')

    if total_preds_a is not None:
        with open(output_dir + output_filename, 'w') as f_out:
            for id, pred in zip(total_ids_a, total_preds_a):   
                f_out.writelines(str(id) + ',' + str(pred) + '\n')
            for id, pred in zip(total_ids_b, total_preds_b):   
                f_out.writelines(str(id) + ',' + str(pred) + '\n')