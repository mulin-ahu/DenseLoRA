# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,3 torchrun --nproc_per_node=2 --master_port=3192 finetune.py \
#CUDA_VISIBLE_DEVICES=$4 python finetune.py \
# "q_proj", "k_proj", "v_proj"
#WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=3192 finetune.py \
#CUDA_VISIBLE_DEVICES=$4 python finetune.py \

CUDA_VISIBLE_DEVICES=0 python readbin.py
