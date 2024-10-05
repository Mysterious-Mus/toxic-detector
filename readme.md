# ToxicDetector

## Abstract 

Large language models (LLMs) like ChatGPT and Gemini have significantly advanced natural language processing, enabling various applications such as chatbots and automated content generation. However, these models can be exploited by malicious individuals who craft toxic prompts to elicit harmful or unethical responses. These individuals often employ jailbreaking techniques to bypass safety mechanisms, highlighting the need for robust toxic prompt detection methods. Existing detection techniques, both blackbox and whitebox, face challenges related to the diversity of toxic prompts, scalability, and computational efficiency. In response, we propose ToxicDetector, a lightweight greybox method designed to efficiently detect toxic prompts in LLMs. ToxicDetector leverages LLMs to create toxic concept prompts, uses embedding vectors to form feature vectors, and employs a Multi-Layer Perceptron (MLP) classifier for prompt classification. Our evaluation, conducted on various versions of the LLama models (LLama-3, LLama-2, and LLama-1), demonstrates that ToxicDetector achieves high accuracy (96.07%) and low false positive rates (3.29%), outperforming state-of-the-art methods. Additionally, ToxicDetector's processing time of 0.084 seconds per prompt makes it highly suitable for real-time applications.  ToxicDetector achieves high accuracy, efficiency, and scalability, making it a practical method for toxic prompt detection in LLMs.

## Citation

You can cite us using the following BibTeX entry:

```bibtex
@inproceedings{liu2024efficient,
  title={Efficient Detection of Toxic Prompts in Large Language Models},
  author={Liu, Yi and Yu, Junzhe and Sun, Huijia and Shi, Ling and Deng, Gelei and Chen, Yuqi and Liu, Yang},
  booktitle={Proceedings of the 39th IEEE/ACM International Conference on Automated Software Engineering (ASE 2024)},
  year={2024}
}
```

## Experiments

### Environment Setup

#### Create a virtual environment

```shell
conda create -n toxicdetector python=3.11
conda activate toxicdetector
```

#### Install CUDA and pytorch according to your environment

For CUDA installation, please follow the instructions on the [official website](https://developer.nvidia.com/cuda-downloads).

For pytorch installation, please follow the instructions on the [official website](https://pytorch.org/get-started/locally/).

#### Install other dependencies

```shell
pip install -r requirements.txt
```

#### Download the models needed

Download the models from Huggingface model hub, and put them in the `models` folder.

You may download only the pytorch checkpoint files.

The structure of the `models` folder should be like this:

```
models
├── llama2-7B-chat-hf
│   ├── config.json
│   ├── pytorch_model.bin
│   └── ...
...
```

### Train and Evaluate ToxicDetector

Our sample experiment folder is at `experiments/Information_Leakage`.

```shell
python prompt_classify.py experiments/Information_Leakage
```

After running the above command, the statistics of the experiment will be put in the `experiments/Information_Leakage/stats` folder.

### Experimenting on different settings

You can create your own experiment folder by following the structure of the `experiments/Information_Leakage` folder.

The config file `experiments/Information_Leakage/config.yaml` contains the settings for the experiment and necessary comments for the settings.