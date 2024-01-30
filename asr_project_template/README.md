# ASR project and Implementation of Conformer model.

## Installation guide
Before starting working with this repo, make sure you have followed this installation guide.

First of all, prepare the environment. Sufficient requirements are listed in the corresponding requirements file.
```shell
pip install -r requirements.txt
```
There is also a KenLM language model that is required if you want to use LM for decoding. Do not forget to install necessary library for that, by run:
```shell
pip install https://github.com/kpu/kenlm/archive/master.zip
python ../language_model/lm.py
```
Then you need to clone this repository:
```shell
!git clone https://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@github.com/Pe4enkazAMI/ASR
```

To run the code, make sure to edit ../hw_asr/config.json and provide data.
The following code runs model inference
```python
import wandb
wandb.login(key="UR KEY")
!python train.py -r ../yanformerbest3/YanformerX3.pth
```

## Details 
In the essence, this project is an implementation of Conformer neural network, mostly this repo follows the paper [https://arxiv.org/abs/2005.08100](Conformer: Convolution-augmented Transformer for Speech Recognition).

However, during the implementation I have conducted an ablation study of several techniques used in the paper. 

Fisrtly, I gor rid of so called Variational Noise without loss of quality. 

Next, I made a few experiments with positional encoding, which resulted in exclusion of it from the pipeline as it had no impact on quality. 

Finally, I conducted many experiments regarding optimization setup. 

During this project, I have experimented with the whole set of hyperparameters configurations presented in the paper.

Detailed report can be found here https://wandb.ai/bayesian_god/asr_project/reports/ASR-YANFORMER--Vmlldzo1NzkxNjY1?accessToken=ca45b1evl1og0kifnqhe7oug4frn7bt0vzka3cwk1piej2mz1qxl9nfoxh9vi0zn

Final scores are:

СER (test-other): 0.11
WER (test-other): 0.24