from model import BertClassifierSingleModel, NezhaClassifierSingleModel, SBERTSingleModel, SNEZHASingleModel, BertClassifierTextCNNSingleModel
from data import SentencePairDatasetWithType, SentencePairDatasetForSBERT

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time

def infer_final(model, device, test_dataloader, threshold_fixed_a=0.5, threshold_fixed_b=0.5):
    print("Inferring for final stage")
    model.eval()

    # as only one GPU is available for the final stage
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)

    total_ids_a, total_probs_a = [], []
    total_ids_b, total_probs_b = [], []
    for idx, batch in enumerate(tqdm(test_dataloader)):
        input_ids, input_types, ids, types = batch
        input_ids = input_ids.to(device)
        input_types = input_types.to(device)
        
        # the probs given by the model, without grads 
        with torch.no_grad():
            all_probs = model(input_ids, input_types)
            num_tasks = len(all_probs)

            all_masks = [(types==task_id).numpy() for task_id in range(num_tasks)]
            all_output = [all_probs[task_id][all_masks[task_id]] for task_id in range(num_tasks)]

            total_ids_a += [id for id in ids if id.endswith('a')]
            total_ids_b += [id for id in ids if id.endswith('b')]

            probs_a, probs_b = [], []
            for task_id in range(0, num_tasks, 2):
                probs_a += [prob[-1] for prob in nn.functional.softmax(all_output[task_id], dim=1).cpu().numpy().tolist()]

            for task_id in range(1, num_tasks, 2):
                probs_b += [prob[-1] for prob in nn.functional.softmax(all_output[task_id], dim=1).cpu().numpy().tolist()]

            total_probs_a += probs_a
            total_probs_b += probs_b

    # positive if the prob passes the original threshold of 0.5
    total_fixed_preds_a = (np.array(total_probs_a) > threshold_fixed_a).astype('int').tolist()
    total_fixed_preds_b = (np.array(total_probs_b) > threshold_fixed_b).astype('int').tolist()
    
    total_preds_a = None
    total_preds_b = None

    return total_ids_a, total_preds_a, total_fixed_preds_a, \
           total_ids_b, total_preds_b, total_fixed_preds_b 

if __name__=='__main__':
    s_time = time.time()

    parser = ArgumentParser()
    parser.add_argument("-i","--input", type=str, required=True, help="输入文件")
    parser.add_argument("-o","--output", type=str, required=True, help="输出文件")
    args = parser.parse_args()
    input_dir = args.input
    output_dir= args.output

    device = 'cuda'
    data_dir = '../data/sohu2021_open_data/'
    save_dir = '../checkpoints/rematch/'
    result_dir = '../results/final/'
    bert_tokenizer_config = '../data/dummy_bert/'   # as NEZHA, MACBERT and ROBERTA share the same tokenizer vocabulary
    ernie_tokenizer_config = '../data/dummy_ernie/'  # unfortunately, ERNIE has its unique vocabulary, should load dataset again

    # only use test dataloader for final stage
    # the test file will be in one file
    test_data_dir = [input_dir]
    bert_model_configs = [
        # model_name, dummy_pretrained, threshold_fixed_a, threshold_fixed_b, infer_bs
        ('0520_roberta_80k_same_lr_zy_epoch_1_ab_loss', '../data/dummy_bert/', 0.4, 0.3, 128),
        ('0518_macbert_same_lr_epoch_1_ab_loss', '../data/dummy_bert/', 0.37, 0.39, 128),
        ('0523_roberta_dataaug_epoch_0_ab_loss', '../data/dummy_bert/', 0.41, 0.48, 128)
    ]

    ernie_model_configs = [
        ('0523_ernie_epoch_1_ab_loss', '../data/dummy_ernie/', 0.42, 0.39, 128),
    ]

    sbert_model_configs = [
        ('0520_roberta_sbert_same_lr_epoch_1_ab_loss', '../data/dummy_bert/', 0.4, 0.36, 128)
    ]

    # We will first infer for the bert-style models
    if len(bert_model_configs) != 0:
        print("Loading Test Data for BERT models...")
        test_dataset = SentencePairDatasetWithType(test_data_dir, False, bert_tokenizer_config)

    for model_config in bert_model_configs:
        model_name, dummy_pretrained, threshold_fixed_a, threshold_fixed_b, infer_bs = model_config
        test_dataloader = DataLoader(test_dataset, batch_size=infer_bs, shuffle=False)
        print("Loading Bert Model from {}...".format(save_dir + model_name))
        # distinguish model architectures or pretrained models according to model_type
        if 'nezha' in model_name.lower():
            print("Using NEZHA pretrained model")
            model = NezhaClassifierSingleModel(bert_dir=dummy_pretrained, from_pretrained=False)
        elif 'cnn' in model_name.lower():
            print("Adding TextCNN after BERT output")
            model = BertClassifierTextCNNSingleModel(bert_dir=dummy_pretrained, from_pretrained=False)
        else:
            print("Using conventional BERT model with linears")
            model = BertClassifierSingleModel(bert_dir=dummy_pretrained, from_pretrained=False)

        model_dict = torch.load(save_dir + model_name)
        # weights will be saved in module when DataParallel
        model.load_state_dict({k.replace('module.','') : v for k, v in model_dict.items()})
        model.to(device)

        # total_ids_a, total_preds_a, total_fixed_preds_a, total_ids_b, total_preds_b, total_fixed_preds_b = infer(model, device, dev_dataloader, test_dataloader, search_thres, threshold_fixed_a, threshold_fixed_b)
        total_ids_a, total_preds_a, total_fixed_preds_a, total_ids_b, total_preds_b, total_fixed_preds_b = infer_final(model, device, test_dataloader, threshold_fixed_a, threshold_fixed_b)
        
        with open(result_dir + 'final_' + '{}.csv'.format(model_name), 'w') as f_out:
            for id, pred in zip(total_ids_a, total_fixed_preds_a):   
                f_out.writelines(str(id) + ',' + str(pred) + '\n')
            for id, pred in zip(total_ids_b, total_fixed_preds_b):   
                f_out.writelines(str(id) + ',' + str(pred) + '\n')

    # infer for the ernie models, dataset should be reloaded for ernie's vocabulary
    if len(ernie_model_configs) != 0:
        print("Loading Test Data for ERNIE models...")
        test_dataset = SentencePairDatasetWithType(test_data_dir, False, ernie_tokenizer_config)

    for model_config in ernie_model_configs:
        model_name, dummy_pretrained, threshold_fixed_a, threshold_fixed_b, infer_bs = model_config
        test_dataloader = DataLoader(test_dataset, batch_size=infer_bs, shuffle=False)
        print("Loading Bert Model from {}...".format(save_dir + model_name))
        # distinguish model architectures or pretrained models according to model_type
        if 'nezha' in model_name.lower():
            print("Using NEZHA pretrained model")
            model = NezhaClassifierSingleModel(bert_dir=dummy_pretrained, from_pretrained=False)
        elif 'cnn' in model_name.lower():
            print("Adding TextCNN after BERT output")
            model = BertClassifierTextCNNSingleModel(bert_dir=dummy_pretrained, from_pretrained=False)
        else:
            print("Using conventional BERT model with linears")
            model = BertClassifierSingleModel(bert_dir=dummy_pretrained, from_pretrained=False)

        model_dict = torch.load(save_dir + model_name)
        # weights will be saved in module when DataParallel
        model.load_state_dict({k.replace('module.','') : v for k, v in model_dict.items()})
        model.to(device)

        # total_ids_a, total_preds_a, total_fixed_preds_a, total_ids_b, total_preds_b, total_fixed_preds_b = infer(model, device, dev_dataloader, test_dataloader, search_thres, threshold_fixed_a, threshold_fixed_b)
        total_ids_a, total_preds_a, total_fixed_preds_a, total_ids_b, total_preds_b, total_fixed_preds_b = infer_final(model, device, test_dataloader, threshold_fixed_a, threshold_fixed_b)
        
        with open(result_dir + 'final_' + '{}.csv'.format(model_name), 'w') as f_out:
            for id, pred in zip(total_ids_a, total_fixed_preds_a):   
                f_out.writelines(str(id) + ',' + str(pred) + '\n')
            for id, pred in zip(total_ids_b, total_fixed_preds_b):   
                f_out.writelines(str(id) + ',' + str(pred) + '\n')

    # infer for SBERT models 
    if len(sbert_model_configs) != 0:
        print("Loading Test Data for SBERT models...")
        # for test dataset, is_train should be set to False, thus get ids instead of labels
        test_dataset = SentencePairDatasetForSBERT(test_data_dir, False, bert_tokenizer_config)

    for model_config in sbert_model_configs:
        model_name, dummy_pretrained, threshold_fixed_a, threshold_fixed_b, infer_bs = model_config
        test_dataloader = DataLoader(test_dataset, batch_size=infer_bs, shuffle=False)
        print("Loading SentenceBert Model from {}...".format(save_dir + model_name))
        # distinguish model architectures or pretrained models according to model_type
        if 'nezha' in model_name.lower():
            model = SNEZHASingleModel(bert_dir=dummy_pretrained, from_pretrained=False)
        else:
            model = SBERTSingleModel(bert_dir=dummy_pretrained, from_pretrained=False)
        
        model_dict = torch.load(save_dir + model_name)
        # weights will be saved in module when training on multiple GPUs with DataParallel
        model.load_state_dict({k.replace('module.','') : v for k, v in model_dict.items()})
        model.to(device)
            
        # total_ids_a, total_preds_a, total_fixed_preds_a, total_ids_b, total_preds_b, total_fixed_preds_b = infer(model, device, dev_dataloader, test_dataloader, search_thres, threshold_fixed_a, threshold_fixed_b)
        total_ids_a, total_preds_a, total_fixed_preds_a, total_ids_b, total_preds_b, total_fixed_preds_b = infer_final(model, device, test_dataloader, threshold_fixed_a, threshold_fixed_b)
        
        with open(result_dir + 'final_' + '{}.csv'.format(model_name), 'w') as f_out:
            for id, pred in zip(total_ids_a, total_fixed_preds_a):   
                f_out.writelines(str(id) + ',' + str(pred) + '\n')
            for id, pred in zip(total_ids_b, total_fixed_preds_b):   
                f_out.writelines(str(id) + ',' + str(pred) + '\n')

    # finally, merge all the output files in output_dir
    print("Merging the model outputs...")
    result_list = [filename for filename in os.listdir(result_dir) if filename.endswith('.csv')]
    result_dict = {}
    for name in result_list:
        with open(result_dir + name, "r", encoding="utf-8") as fr:
            for line in fr:
                words = line.strip().split(",")
                if words[0] == "id":
                    continue
                if words[0] not in result_dict:
                    result_dict[words[0]] = [words[1]]
                else:
                    result_dict[words[0]].append(words[1])

    # merging the outputs into final csv file
    with open(output_dir, "w", encoding="utf-8") as fw:
        fw.write("id,label"+"\n")
        for k, v in result_dict.items():
            tmp = {}
            for ele in v:
                if ele in tmp:
                    tmp[ele] += 1
                else:
                    tmp[ele] = 1
            tmp = sorted(tmp.items(), key=lambda d: d[1], reverse=True)
            fw.write(",".join([k, tmp[0][0]]) + "\n")

    e_time = time.time()
    print("Time taken: ", e_time - s_time)