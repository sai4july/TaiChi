import os
import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    # -- Data Params --
    parser.add_argument("--task", type=str, default="sst2")
    parser.add_argument("--dataset", type=str, default="sst2")
    parser.add_argument("--save_path", type=str, default="./trained_model/sst2")
    parser.add_argument("--model_type", type=str, default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--num_labels", type=float, default=2)
    parser.add_argument("--freeze", type=bool, default=False) # whether freeze pre-trained model's parameters
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_len",type=int,default=50)
    parser.add_argument("--batch_size",type=int,default=16)
    parser.add_argument("--use_cuda", type=bool, default=True)
    parser.add_argument("--num_workers",type=int,default=os.cpu_count())
    parser.add_argument("--log_steps",type=int,default= 75) 
    parser.add_argument("--eval_steps",type=int,default= 75) 
    

    # --MTL Params--
    parser.add_argument("--backbone_ori",type=str,default="") 
    parser.add_argument("--backbone_adv",type=str,default="")
    parser.add_argument("--adv_attack",type=str,default="pwws") # type of adv_examples
    parser.add_argument("--beta",type=float,default= 1) # contrast hyper-pramas
    parser.add_argument("--gamma",type=float,default= 1) # hyper-pramas for KL loss
    parser.add_argument("--T",type=float,default=1) # temperatur param for contrast learning task


    # --TEST--
    parser.add_argument("--test", type=bool, default=False)
    parser.add_argument("--attack", type=str, default="pwws") # pwws/textfooler/deepwordbug/textbugger
    parser.add_argument("--test_model", type=str, default="#")


    return parser