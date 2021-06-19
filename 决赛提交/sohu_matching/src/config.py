class Config():
    def __init__(self):
        self.device= 'cuda'
        self.model_type = '0523_roberta_80k_6tasks'
        self.task_type = 'ab'

        self.save_dir = '/data1/wangchenyue/sohu_matching/checkpoints/rematch/'
        self.data_dir = '/data1/wangchenyue/sohu_matching/data/sohu2021_open_data/'
        self.load_toy_dataset = False

        # self.pretrained = '/data1/wangchenyue/Downloads/chinese-macbert-base/'
        # self.pretrained = '/data1/wangchenyue/Downloads/nezha-base-wwm/'
        # self.pretrained = '/data1/wangchenyue/Downloads/chinese-roberta-wwm-ext/'
        # self.pretrained = '/data1/wangchenyue/Downloads/roberta-base-finetuned-chinanews-chinese/'
        # self.pretrained = '/data1/wangchenyue/Downloads/chinese-bert-wwm-ext/'
        # self.pretrained = '/data1/wangchenyue/Downloads/roberta-base-word-chinese'
        # self.pretrained = '/data1/wangchenyue/Downloads/ernie-1.0/'
        self.pretrained = '/data1/wangchenyue/Downloads/DSP/roberta-wwm-rematch/checkpoint-80000/'
        
        self.epochs = 3
        self.lr = 2e-5
        self.classifier_lr = 1e-3
        self.use_scheduler = True
        self.weight_decay = 1e-3
        self.num_warmup_steps = 2000

        # for larger models, e.g.roberta-large hidden_size = 1024, otherwise 768
        self.hidden_size = 768
        # for sbert, train_bs = 16, eval_bs = 32, otherwise 32/64
        self.train_bs = 32
        self.eval_bs = 64
        self.criterion = 'CE'
        self.print_every = 50
        self.eval_every = 500

        # whether to shffle the order in training data as augmentation
        self.shuffle_order = False
        self.aug_data = False
        # how to clip the long sequences, 'head': using the first sentences, 'tail': using the last sentences
        # 'head' is reportedly better than 'tail'
        self.clip_method = 'head'

        # whether to use fgm for adversial attack in training
        self.use_fgm = False

        # settings for inference
        # self.infer_model_dir = '../checkpoints/0502/'
        self.infer_model_dir = '/data1/wangchenyue/sohu_matching/checkpoints/rematch/'
        self.infer_model_name = '0525_roberta_6tasks_epoch_1_ab_loss'
        # fake pretrained model dir containing config.json and vocab.txt, for tokenzier and model initialization
        self.dummy_pretrained = '../data/dummy_bert/'
        # self.dummy_pretrained = '../data/dummy_ernie/'
        # self.dummy_pretrained = '../data/dummy_nezha/'
        # infer_task_type should match the last letter in infer_model_name
        self.infer_task_type = self.infer_model_name.split('_')[-2]
        self.infer_output_dir = '/data1/wangchenyue/sohu_matching/results/rematch/'
        self.infer_output_filename = '{}.csv'.format(self.infer_model_name)
        self.infer_clip_method = 'head'
        # for NEZHA, infer_bs=64, otherwise 256 
        self.infer_bs = 256
        self.infer_fixed_thres_a = 0.45
        self.infer_fixed_thres_b = 0.35
        self.infer_search_thres = True
