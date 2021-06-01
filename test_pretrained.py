# -*- coding:utf-8 -*-
import os
import time
import argparse
import sys
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from utils.metric_util import per_class_iu, fast_hist_crop
from dataloader.pc_dataset import get_SemKITTI_label_name
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data

from utils.load_save_util import load_checkpoint

from dataloader.dataset_semantickitti import get_model_class, collate_fn_BEV
from dataloader.pc_dataset import get_pc_model_class

import warnings

warnings.filterwarnings("ignore")


def main(args):
    pytorch_device = torch.device('cuda:0')

    config_path = args.config_path

    configs = load_config_data(config_path)

    dataset_config = configs['dataset_params']
    test_dataloader_config = configs['test_data_loader']

    test_batch_size = test_dataloader_config['batch_size']

    model_config = configs['model_params']
    train_hypers = configs['train_params']

    grid_size = model_config['output_shape']
    num_class = model_config['num_class']
    ignore_label = dataset_config['ignore_label']

    model_load_path = train_hypers['model_load_path']
    model_save_path = train_hypers['model_save_path']

    SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

    print("load checkpoint")

    my_model = model_builder.build(model_config)
    if os.path.exists(model_load_path):
        my_model = load_checkpoint(model_load_path, my_model)

    my_model.to(pytorch_device)
   
   
    print("init model")

    data_path = test_dataloader_config["data_path"]
    test_imageset = test_dataloader_config["imageset"]
    test_ref = test_dataloader_config["return_ref"]

    label_mapping = dataset_config["label_mapping"]

    SemKITTI = get_pc_model_class(dataset_config['pc_dataset_type'])

    test_pt_dataset = SemKITTI(data_path, imageset=test_imageset,
                              return_ref=test_ref, label_mapping=label_mapping)

    test_dataset = get_model_class(dataset_config['dataset_type'])(
        test_pt_dataset,
        grid_size=grid_size,
        fixed_volume_space=dataset_config['fixed_volume_space'],
        max_volume_space=dataset_config['max_volume_space'],
        min_volume_space=dataset_config['min_volume_space'],
        ignore_label=dataset_config["ignore_label"],
    )

    print("init loader")

    print(test_dataloader_config)

    test_dataset_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                     batch_size=test_batch_size,
                                                     collate_fn=collate_fn_BEV,
                                                     shuffle=test_dataloader_config["shuffle"],
                                                     num_workers=test_dataloader_config["num_workers"],
                                                     )
    # test_dataset_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, collate_fn=collate_fn_BEV, shuffle=False, num_workers=1)
    
    my_model.eval()

    print("loading data")

    with torch.no_grad():
        for i_iter, (_, test_vox_label, test_grid, _, test_pt_fea) in enumerate(test_dataset_loader):
            print("iteration:", i_iter)
        
            test_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in test_pt_fea]
            test_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in test_grid]
            test_label_tensor = test_vox_label.type(torch.LongTensor).to(pytorch_device)

            # test_pt_fea_ten = [i.type(torch.FloatTensor).to(pytorch_device) for i in test_pt_fea]
            # test_grid_ten = [i.to(pytorch_device) for i in test_grid]
            # test_label_tensor = test_vox_label.type(torch.LongTensor).to(pytorch_device)

            # print("test_pt_fea_ten =", test_pt_fea_ten)
            # print("test_grid_ten =", test_grid_ten)
            # print("test_batch_size =", test_batch_size)

            # print("eval model")
            predict_labels = my_model(test_pt_fea_ten, test_grid_ten, test_batch_size)
            predict_labels = torch.argmax(predict_labels, dim=1)
            predict_labels = predict_labels.cpu().detach().numpy()
            
            print("predictions:")
            print(predict_labels)
                



if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='config/semantickitti.yaml')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)
