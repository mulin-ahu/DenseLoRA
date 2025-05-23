import torch
import torch.nn.functional as F

# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import copy
import json
import os
import re
import sys
import argparse

import fire

import torch

sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
from peft import PeftModel
from tqdm import tqdm
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib as mpl
from scipy.stats import norm
import shutil


def parse_args():
    parser = {}
    parser['dataset'] = "boolq"
    parser['model'] = 'LLaMA3-8B'
    parser['adapter'] = 'LoRA'
    parser['base_model'] = '/mnt/disk2/ml/meta_llama/llama3/'
    parser['lora_weights_init'] = './finetuned_result/llama3_8b/lora_32_2_init'
    parser['lora_weights'] = './finetuned_result/llama3_8b/lora_32_2'
    parser['batch_size'] = 1
    parser['load_8bit'] = False

    return parser


def load_model(args):

    base_model = args['base_model']
    if not base_model:
        raise ValueError(f'can not find base model name by the value: {args["model"]}')

    lora_weights_init = args['lora_weights_init']
    lora_weights = args['lora_weights']
    if not lora_weights:
        raise ValueError(f'can not find lora weight, the value is: {lora_weights}')

    load_8bit = args['load_8bit']
    print("lora_weights_init", lora_weights_init)
    print("lora_weights", lora_weights)

    #tokenizer = AutoTokenizer.from_pretrained(base_model)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="balanced_low_0",
        trust_remote_code=True,
    ) # fix zwq

    model_peft = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float16,
        device_map={"":0}
    )

    model_init = PeftModel.from_pretrained(
        model,
        lora_weights_init,
        torch_dtype=torch.float16,
        device_map={"": 0}
    )
    return model_peft, model_init


def main(
        load_8bit: bool = False,
        base_model: str = "",
        lora_weights: str = "tloen/alpaca-lora-7b",
        share_gradio: bool = False,
):
    args = parse_args()
    base_model = args['base_model']
    if not base_model:
        raise ValueError(f'can not find base model name by the value: {args["model"]}')
    if not lora_weights:
        raise ValueError(f'can not find lora weight, the value is: {lora_weights}')
    lora_weights_init = args['lora_weights_init']
    lora_weights = args['lora_weights']

    load_8bit = args['load_8bit']

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="balanced_low_0",
        trust_remote_code=True,
    )  # fix zwq

    model_peft = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float16,
        device_map={"": 0}
    )
    key_dict = {}
    for key, module in model_peft.model.named_modules():
        key_dict[key] = module

    model_init = PeftModel.from_pretrained(
        model,
        lora_weights_init,
        torch_dtype=torch.float16,
        device_map={"": 0}
    )
    key_dict_init = {}
    for key, module in model_init.model.named_modules():
        key_dict_init[key] = module

    #lora_a = key_dict['model.layers.0.self_attn.q_proj.lora_A']
    #lora_b = key_dict['model.layers.0.self_attn.q_proj.lora_B']
    #lora_ = key_dict['model.layers.0.self_attn.q_proj.lora_A']
    #lora_init = key_dict_init['model.layers.0.self_attn.q_proj.lora_A']

    '''
    result = lora_b.weight @ lora_a.weight #torch.abs(lora_b.weight @ lora_a.weight)[:32,:32]
    Ua, sa, Va = np.linalg.svd(result.cpu().detach().numpy())
    weighta = Ua[:32,:32]
    #Ub, sb, Vb = np.linalg.svd(lora_b.weight.cpu().detach().numpy())
    #weightb = Ub[:32, :32]
    #result = torch.where(result>0.01, result, torch.zeros_like(result))
    max_value = 1 # torch.max(result)
    #weight = torch.div(result, max_value).cpu().detach().numpy()
    plt.imshow(np.abs(weighta), interpolation="nearest", cmap='RdBu', origin='lower')
    plt.colorbar(shrink=.92)
    plt.xticks(())
    plt.yticks(())
    plt.savefig("model_layers_6" + ".png")
    plt.show()
    '''
    for key in key_dict:
        '''
        if 'lora_A' in key:
            #if 'lora_dropout' not in key and 'lora_encoder' not in key and "lora_decoder" not in key:
            print(key)
            lora_a = key_dict[key]
            lora_b = key_dict[key[:-1] + 'B']
            result = lora_b.weight @ lora_a.weight  # torch.abs(lora_b.weight @ lora_a.weight)[:32,:32]
            Ua, sa, Va = np.linalg.svd(result.cpu().detach().numpy())
            weighta = Ua[:32, :32]
            plt.imshow(np.abs(weighta), cmap='viridis')
            plt.colorbar()
            name = "result_svd_" + key.split('.')[-2] + '_' + key.split('.')[2]
            plt.savefig(name + ".png")
            plt.show()
            plt.close()

            plt.imshow(np.abs(result.cpu().detach().numpy())[:32,:32], cmap='viridis')
            plt.colorbar()
            name = "result_" + key.split('.')[-2] + '_' + key.split('.')[2]
            plt.savefig(name + ".png")
            plt.show()
            plt.close()

            if key.split('.')[2] != '0':
                break

        '''
        if 'lora_' in key and 'lora_decoder' not in key and 'lora_encoder' not in key and "lora_dropout" not in key:
            print("key", key)

            name = "main" + key.split('.')[-2] + '_' + key.split('.')[2] + '_' + key.split('.')[-1]
            module = key_dict[key]
            module_init = key_dict_init[key]
            result = torch.abs(torch.sub(module.weight, module_init.weight))[:32,:32]
            #max_value = torch.max(result)
            #weight = torch.div(result, max_value).cpu().detach().numpy()
            weight = result.cpu().detach().numpy()
            plt.imshow(weight, cmap='viridis', vmin=0, vmax=0.15)
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=12)
            #plt.xticks(())
            #plt.yticks(())
            plt.tick_params(axis='y', labelsize=14)
            plt.tick_params(axis='x', labelsize=14)
            plt.savefig(name + ".png")
            plt.show()
            plt.close()
            if key.split('.')[2] != '0':
                break

                #print("module.numpy", weight)


if __name__ == "__main__":
    main()
