import os

if __name__ == '__main__':
    result_dir = '../results/rematch/'

    target_files = [
        '0520_roberta_80k_same_lr_zy_epoch_1_ab_loss',
        '0518_macbert_same_lr_epoch_1_ab_loss',
        '0520_roberta_sbert_same_lr_epoch_1_ab_loss',
        '0523_roberta_dataaug_epoch_0_ab_loss',
        '0523_ernie_epoch_1_ab_loss'
    ]
    # 0.7931380664848722

    # target_files = [
    #     '0518_roberta_same_lr_epoch_1_ab_loss',
    #     '0519_nezha_same_lr_epoch_1_ab_f1',
    #     '0518_nezha_diff_lr_zy_epoch_1_ab_loss',
    #     '0518_macbert_same_lr_epoch_1_ab_los',
    #     '0523_roberta_dataaug_epoch_0_ab_loss'
    # ]
    # # 0.7930518678397445

    result_list = [file_name+'.csv' for file_name in target_files]
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

    with open(result_dir+"merge.csv", "w", encoding="utf-8") as fw:
        fw.write("id,label"+"\n")
        for k, v in result_dict.items():
            tmp = {}
            for ele in v:
                if ele in tmp:
                    tmp[ele] += 1
                else:
                    tmp[ele] = 1
            tmp = sorted(tmp.items(), key=lambda d: d[1], reverse=True)
            # print(tmp)
            fw.write(",".join([k, tmp[0][0]]) + "\n")