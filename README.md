# Taichi

## Introduction
This is an anonymous_GitHub_repository for ACL Rolling Review 2023.

## Environment Configuration
python version == 3.7.13 \
pip install -r requirement.txt

## Data
Please download data from [here](https://pan.baidu.com/s/1gImSonez6yWVAOJms2_w0A?pwd=bm9q)  

## How to run Taichi?
python main.py \
--dataset sst2 \
--adv_attack textfooler 

## How to evaluete Taichi?
cd Textattack/trec

textattack attack \
--model-from-file "my_model.py" \
--dataset-from-file "my_dataset.py" \
--recipe textfooler  \
--num-examples 1000 \
--log-to-csv "output/textfooler_test.csv" 
