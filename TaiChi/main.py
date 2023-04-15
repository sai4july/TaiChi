import torch.nn as nn
import datetime
from utils import *
import torch
from training import Trainer
from logger import get_logger
from transformers import AdamW,get_linear_schedule_with_warmup,BertModel,AutoConfig


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings
        
class Backbone(nn.Module):
    def __init__(self, num_labels, BERT_MODEL_NAME, type = "clean", freeze_bert=args.freeze):
        super().__init__()
        self.type = type
        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(BERT_MODEL_NAME, output_hidden_states=True)
        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME)
        self.pool = MeanPooling()
        self.fc = nn.Linear(self.config.hidden_size,num_labels)  
        self._init_weights(self.fc)
        # loss
        self.loss_CE = nn.CrossEntropyLoss()

        if freeze_bert:
            print("freezing bert parameters")
            for param in self.bert.parameters():
                param.requires_grad = False

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def feature(self,ids,mask):
        outputs = self.bert(input_ids=ids,attention_mask=mask)
        last_hidden_states = outputs[0]
        feature = self.pool(last_hidden_states,mask)
        return feature

    def forward(self, input_ids=None, attention_mask=None, smooth = False, labels=None):

        feature = self.feature(input_ids,attention_mask)
        logits  = self.fc(feature)

        if labels is not None:

   
                loss_CE = self.loss_LSCE(
                    logits.view(-1, self.num_labels), labels
                )
                
                loss = loss_CE

                return {'loss':loss,"logits":logits,"feature":feature}

        else:
            return {"logits":logits,"feature":feature}



if __name__ == "__main__":
    if not args.test:
        parser = get_parser()
        args = parser.parse_args()
        name = getName()
        logger = get_logger(log_file=f"{name}_{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.txt")
        logger.info(f"{args.dataset} args: {args}")

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)


        train_data, dev_data,test_data, attack_data, adv_data = load_data(args.dataset)
        train_dataset = Bert_dataset(train_data)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
        dev_dataset = Bert_dataset(dev_data)
        dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
        test_dataset = Bert_dataset(test_data)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        attack_dataset = Bert_dataset(attack_data)
        attack_loader = DataLoader(attack_dataset, batch_size=args.batch_size, shuffle=False)
        adv_dataset = Bert_dataset(adv_data)
        adv_loader = DataLoader(adv_dataset, batch_size=args.batch_size, shuffle=False)
        

        logger.info(f"train_data:{len(train_data)},dev_data:{len(dev_data)},test_data:{len(test_data)}")

        # model = BertForSequenceClassification.from_pretrained(args.model_type,num_labels=args.num_labels) 
        backbone_ori = Backbone(args.num_labels,args.model_type)
        backbone_adv = Backbone(args.num_labels,args.model_type,"adv")

        if args.backbone_ori:
            backbone_ori.load_state_dict(torch.load(args.backbone_ori))
            logger.info(f"Pretrained backbone_ori [{args.backbone_ori}] has been loaded.")
        if args.backbone_adv:
            backbone_adv.load_state_dict(torch.load(args.backbone_adv))
            logger.info(f"Pretrained backbone_adv [{args.backbone_adv}] has been loaded.")

        optimizer_ori = AdamW(backbone_ori.parameters(), lr=args.lr, eps=1e-8)
        optimizer_adv = AdamW(backbone_adv.parameters(), lr=args.lr, eps=1e-8)  
        
        scheduler_ori = get_linear_schedule_with_warmup(optimizer_ori, 
                                                num_warmup_steps=0, 
                                                num_training_steps=len(train_loader)*args.epochs)

        scheduler_adv = get_linear_schedule_with_warmup(optimizer_adv, 
                                        num_warmup_steps=0, 
                                        num_training_steps=len(train_loader)*args.epochs)

                                        
        trainer = Trainer(
            backbone_ori,
            optimizer_ori,    
            scheduler_ori,
            backbone_adv,
            optimizer_adv,    
            scheduler_adv,
            args.epochs,
            args.log_steps,
            args.eval_steps,
            args.use_cuda,
            logger
        )

        trainer.train(train_loader,dev_loader,attack_loader,adv_loader)

        # evaluate test dataset #
        acc = trainer.predict(test_loader)
        attack_acc = trainer.predict(attack_loader)
        acc_adv = trainer.predict(test_loader,"adv")
        attack_acc_adv = trainer.predict(attack_loader,"adv")
        logger.info(f"[CLEAN] test acc = {acc:.4f}, attack_acc_{args.attack} = {attack_acc:.4f}")
        logger.info(f"[ADV] test acc = {acc_adv:.4f}, attack_acc_{args.attack} = {attack_acc_adv:.4f}")

    else:
        # evaluate test dataset #
        test_data,adv_data = load_test_data()
        adv_dataset = Bert_dataset(adv_data)
        adv_loader = DataLoader(adv_dataset, batch_size=args.batch_size, shuffle=False)
        test_dataset = Bert_dataset(test_data)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        model = Backbone(args.num_labels,args.model_type)
        model.load_state_dict(torch.load(args.test_model))
        model.cuda()
        acc = predict(test_loader,model)
        adv_acc = predict(adv_loader,model)
        print(f"task: {args.task}\nmodel: {args.test_model.split('/')[-1]}\nattack: {args.attack}\nacc = {acc:.4f}\nadv_acc = {adv_acc:.4f}.")