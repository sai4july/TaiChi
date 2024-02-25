# Taichi

## Environment Configuration
python version == 3.7.13 \
pip install -r requirement.txt

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
