<h1 align="center">
    <p> DenseLoRA: Dense Low-Rank Adaptation of Large Language Models <br> [ACL 2025]</p>
</h1>


## Useful Links

- An amazing tutorial about implementing DoRA from scratch by Sebastian Raschka, see https://magazine.sebastianraschka.com/p/lora-and-dora-from-scratch 
- An amazing blog post from Answer.AI about QDoRA/FSDP which allows finetuning LLMs on consumer-level GPUs, see https://www.answer.ai/posts/2024-04-26-fsdp-qdora-llama3.html

## Quick Start and some tricks regarding finetuning with DenseLoRA
### Setup
Install dependencies
```bash
pip install -r requirements.txt
```

### Training(finetune.py)
An example could be as follows:
```bash
sh llama3_8B_denselora.sh 16 32 ./finetuned_result/llama3-8b/ 0 
```

### Inerence
An example could be as follows:
```bash
sh llama3_8B_denselora_eval.sh ./finetuned_result/llama3-87b/ 0
```

## Citation
If you find DoRA useful, please consider giving a star and citation:
```bibtex
@article{liu2024dora,
  title={DoRA: Weight-Decomposed Low-Rank Adaptation},
  author={Liu, Shih-Yang and Wang, Chien-Yi and Yin, Hongxu and Molchanov, Pavlo and Wang, Yu-Chiang Frank and Cheng, Kwang-Ting and Chen, Min-Hung},
  journal={arXiv preprint arXiv:2402.09353},
  year={2024}
}
```

## Acknowledgement
This repo benefits from LLM-Adapters, DoRA. Thanks for their wonderful works.

