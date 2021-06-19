import numpy as np
from sklearn.metrics import f1_score
from itertools import combinations
from tqdm import tqdm
# from collections import defaultdict

def merge_on_valid(model_names, verbose=False):
    # len of valid_a: 4971, len of valid_b: 4969
    # total_preds_a, total_preds_b = [0]*4971, [0]*4969
    total_preds_a, total_preds_b = [0]*6911, [0]*6914
    # positive if the vote exceeds the threshold (>)
    threshold = len(model_names)/2
    for model in model_names:
        # print("processing model {}".format(model))
        preds_a, preds_b = np.load('{}_pred_a.npy'.format(model)), np.load('{}_pred_b.npy'.format(model))
        preds_a, preds_b = preds_a.tolist(), preds_b.tolist()
        assert len(total_preds_a)==len(preds_a) and len(total_preds_b)==len(preds_b)
        for idx, pred_a in enumerate(preds_a):
            total_preds_a[idx] += pred_a
        for idx, pred_b in enumerate(preds_b):
            total_preds_b[idx] += pred_b
        # print(len(preds_a), len(preds_b))
        # print(type(preds_b)) 

    total_preds_a, total_preds_b = np.array(total_preds_a), np.array(total_preds_b)
    vote_a, vote_b = (total_preds_a>threshold).astype('int'), (total_preds_b>threshold).astype('int')
    gt_a, gt_b = np.load(valid_dir + 'gt_a.npy'), np.load(valid_dir + 'gt_b.npy')
    # print(len(vote_a), len(vote_b))
    # print(len(gt_a), len(gt_b))

    f1a, f1b = f1_score(gt_a, vote_a), f1_score(gt_b, vote_b)
    ssa, ssb = f1_score(gt_a[:1750], vote_a[:1750]), f1_score(gt_b[:1750], vote_b[:1750])
    sla, slb = f1_score(gt_a[1750:4380], vote_a[1750:4380]), f1_score(gt_b[1750:4385], vote_b[1750:4385])
    lla, llb = f1_score(gt_a[4380:], vote_a[4380:]), f1_score(gt_b[4385:], vote_b[4385:])

    if verbose:
        print("f1a: {}, f1b: {}".format(f1a, f1b))
        print("ssa: {}, ssb: {}".format(ssa, ssb))
        print("sla: {}, slb: {}".format(sla, slb))
        print("lla: {}, llb: {}".format(lla, llb))

    return f1a, f1b, ssa, ssb, sla, slb, lla, llb

if __name__ == '__main__':
    valid_dir = '../valid_output/'
    total_model_names = [
        '0518_roberta_same_lr_epoch_1_ab_loss',
        '0520_roberta_diff_lr_epoch_1_ab_loss',
        '0520_roberta_tcnn_diff_lr_epoch_1_ab_loss',
        '0520_roberta_80k_same_lr_zy_epoch_1_ab_loss',
        '0522_roberta_80k_fl_epoch_1_ab_loss',
        '0519_nezha_same_lr_epoch_1_ab_f1',
        '0519_nezha_same_lr_epoch_0_ab_loss',
        '0518_nezha_diff_lr_zy_epoch_1_ab_loss',
        '0518_macbert_same_lr_epoch_1_ab_loss',
        '0520_macbert_sbert_same_lr_epoch_1_ab_loss',
        '0520_roberta_sbert_same_lr_epoch_1_ab_loss',
        '0522_roberta_80k_tcnn_epoch_1_ab_loss',
        '0523_roberta_dataaug_epoch_0_ab_loss',
        '0523_ernie_epoch_1_ab_loss'
    ]
    total_model_dir = [valid_dir + model_name for model_name in total_model_names]
    f1a, f1b, *_ = merge_on_valid(total_model_dir)
    print("total merge: f1 {}, f1a {}, f1b {}".format(((f1a+f1b)/2), f1a, f1b))
    print()

    for size in [3,5,7,9,11]:
        print("searching the best merge of {} models".format(size))
        records = []
        combs = combinations(total_model_dir, size)
        best_f1 = 0
        best_comb = None
        for comb in tqdm(combs):
            f1a, f1b, *_ = merge_on_valid(list(comb))
            if (f1a + f1b)/2 > best_f1:
                best_f1 = (f1a + f1b)/2
                best_comb = comb
            records.append((list(comb), (f1a+f1b)/2))
        print("best f1 and model list:")
        print(best_f1, best_comb)
        merge_on_valid(list(best_comb), True)

        print("top5 candidates list:")
        records.sort(key=lambda x:x[-1], reverse=True)
        for i in range(5):
            print(records[i])
        print()