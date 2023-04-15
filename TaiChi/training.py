import os
import time
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from utils import *
from tqdm.contrib import tzip
from collections import defaultdict

class Trainer:
    def __init__(
        self,
        backbone1,
        optimizer1,
        scheduler1,
        backbone2,
        optimizer2,
        scheduler2,
        n_epochs,
        log_steps,
        eval_steps,
        use_cuda=True,
        logger=None,
    ):
        self.backbone1 = backbone1
        self.optimizer1 = optimizer1
        self.scheduler1 = scheduler1
        self.backbone2 = backbone2
        self.optimizer2 = optimizer2
        self.scheduler2 = scheduler2
        self.n_epochs = n_epochs
        self.log_steps = log_steps
        self.eval_steps = eval_steps
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        self.logger = logger

        self.best_score = 0

        self.best_attack_acc = 0.0

        self.best_acc = 0.0

        self.name = getName()

        if self.device == "cuda":
            self.backbone1.cuda()
            self.backbone2.cuda()

    def train(self, train_loader, val_loader, attack_loader, adv_loader):
        """Train model for self.n_epochs.
        Train and validation (if validation loader given) losses stored in self.losses

        :param train_loader: train loader of input data
        :param val_loader: validation loader of input data
        """

        self.backbone1.train()
        self.backbone2.train()
        if val_loader is not None:
            init_val_acc_ori = self.evaluate(val_loader)
            init_attack_acc_ori = self.evaluate(attack_loader)
            init_val_acc_adv = self.evaluate(val_loader,"adv")
            init_attack_acc_adv = self.evaluate(attack_loader,"adv")
            self.logger.info(f"Init_val_acc_ori: {init_val_acc_ori:5f}, Init_attack_acc_of_{args.attack}_ori: {init_attack_acc_ori:5f}")
            self.logger.info(f"Init_val_acc_adv: {init_val_acc_adv:5f}, Init_attack_acc_of_{args.attack}_adv: {init_attack_acc_adv:5f}")
            self.best_score_ori = init_val_acc_ori + init_attack_acc_ori
            self.best_score_adv = init_val_acc_adv + init_attack_acc_adv
            self.best_acc_ori = init_val_acc_ori
            self.best_attack_acc_ori = init_attack_acc_ori
            self.best_acc_adv = init_val_acc_adv
            self.best_attack_acc_adv = init_attack_acc_adv
            
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)

        num_training_steps = self.n_epochs * len(train_loader)

        global_step = 0

        # loss set
        loss_clean = torch.tensor(0)
        loss_adv = torch.tensor(0)
        loss_KL = torch.tensor(0)
        
        self.logger.info(f"Training model for {self.n_epochs} epochs..")

        train_start = time.time()
        for epoch in range(self.n_epochs):
            epoch_start = time.time()
            for ori,adv in tzip(train_loader,adv_loader):
    
                ori = tuple(data.to(self.device) for data in ori) 
                adv = tuple(data.to(self.device) for data in adv) 
    
                inputs_ids, inputs_masks,token_type_ids,inputs_labels = ori
                adv_inputs_ids, adv_inputs_masks,adv_token_type_ids,adv_inputs_labels = adv
      
                self.optimizer1.zero_grad()
                self.optimizer2.zero_grad()

                ## The Cross Entropy Part for CLEAN (MAIN TASK)
                clean_outputs_ori = self.backbone1(inputs_ids, inputs_masks, smooth = args.smooth, labels = inputs_labels)
                loss_clean = clean_outputs_ori['loss']
                clean_logits_ori = clean_outputs_ori['logits']
                clean_feature_ori = clean_outputs_ori['feature']

                loss_clean.backward(retain_graph=True)
                loss_clean.backward()

                ## The Cross Entropy Part for ADV (MAIN TASK)
                adv_outputs_adv = self.backbone2(adv_inputs_ids, adv_inputs_masks, smooth = args.smooth, labels = adv_inputs_labels)
                loss_adv = adv_outputs_adv['loss']
                adv_logits_adv = adv_outputs_adv['logits']
                adv_feature_adv = adv_outputs_adv['feature']

                loss_adv.backward(retain_graph=True)
                loss_adv.backward()


                """
                The common part
                """

                clean_outputs_adv = self.backbone1(adv_inputs_ids, adv_inputs_masks)
                clean_logits_adv = clean_outputs_adv['logits']
                clean_feature_adv = clean_outputs_adv['feature']

                adv_outputs_ori = self.backbone2(inputs_ids, inputs_masks)
                adv_logits_ori = adv_outputs_ori['logits']
                adv_feature_ori = adv_outputs_ori['feature']

                """
                The inner interactive tasks
                """

                ## Contrastive Learning

                loss_CTR_ori = self.CTR_loss(clean_feature_ori,clean_feature_adv,args.T) 
                loss_CTR_ori.backward(retain_graph=True)
                loss_CTR_ori.backward()
                
                loss_CTR_adv = self.CTR_loss(adv_feature_adv,adv_feature_ori,args.T) 
                loss_CTR_adv.backward(retain_graph=True)
                loss_CTR_adv.backward()


                """
                The outer interactive tasks
                """
                
                loss_KL_ori_pair = self.KL_loss(adv_logits_ori, clean_logits_ori)
                loss_KL_adv_pair = self.KL_loss(clean_logits_adv, adv_logits_adv)
                loss_KL = args.gamma * loss_KL_adv_pair 
                loss_KL = args.gamma * loss_KL_ori_pair + args.gamma * loss_KL_adv_pair 
                loss_KL.backward()

            
                if self.log_steps and global_step  % self.log_steps == 0:
                    logger_infomation = f"[Train] epoch:{epoch+1}/{self.n_epochs}, step: {global_step}/{num_training_steps}"
                    logger_infomation += f", clean_loss：{loss_clean:.5f}, adv_loss：{loss_adv:.5f}"
                    if args.KL:
                        logger_infomation += f", outer_KL_loss:{loss_KL.item():.5f}"
                        logger_infomation += f", KL_loss(ori pair):{loss_KL_ori_pair.item():.5f}"
                        logger_infomation += f", KL_loss(adv pair):{loss_KL_adv_pair.item():.5f}"
                    logger_infomation += '.'
                    self.logger.info(logger_infomation)

                
                self.optimizer1.step()
                self.optimizer2.step()
               
                self.scheduler1.step()
                self.scheduler2.step()
                
                global_step += 1
    
                # Evaluate on validation set
                if val_loader is not None and self.eval_steps > 0 and global_step != 0 and \
                    (global_step % self.eval_steps == 0 or global_step == (num_training_steps - 1)):
                    train_acc_ori = self.evaluate(train_loader)
                    val_acc_ori= self.evaluate(val_loader)
                    attack_acc_ori = self.evaluate(attack_loader)

                    train_acc_adv = self.evaluate(train_loader,"adv")
                    val_acc_adv= self.evaluate(val_loader,"adv")
                    attack_acc_adv = self.evaluate(attack_loader,"adv")

                    self.logger.info(f"[ORI] [Evaluate] epoch:{epoch+1}/{self.n_epochs}, step: {global_step}/{num_training_steps}, train_acc_ori:{train_acc_ori:.5f}, val_acc_ori:{val_acc_ori:.5f}, attack_acc_{args.attack}_ori: {attack_acc_ori:.5f}")    

                    self.logger.info(f"[ADV] [Evaluate] epoch:{epoch+1}/{self.n_epochs}, step: {global_step}/{num_training_steps}, train_acc_adv:{train_acc_adv:.5f}, val_acc_adv:{val_acc_adv:.5f}, attack_acc_{args.attack}_adv: {attack_acc_adv:.5f}")  

                    
                    self.backbone1.train()
                    self.backbone2.train()

                    if val_acc_ori > self.best_acc_ori:
                        self.best_acc_ori = val_acc_ori
                        self.save_model()
                        self.logger.info(f"[ORI] [MODEL UPDATE] attack_acc:{attack_acc_ori:.5f}, val_acc:{val_acc_ori:.5f}")
                        self.logger.info(f"[ORI] [BEST ACC] best accuracy performance has been updated: {self.best_acc_ori:.5f} -> {val_acc_ori:.5f}, attack_acc_{args.attack}:{attack_acc_ori:.5f}")

                    if attack_acc_adv > self.best_attack_acc_adv:
                        self.best_attack_acc_adv = attack_acc_adv
                        self.save_model("adv")
                        self.logger.info(f"[ADV] [MODEL UPDATE] attack_acc:{attack_acc_adv:.5f}, val_acc:{val_acc_adv:.5f}")
                        self.logger.info(f"[ADV] [BEST ATTACK_ACC] best attack_accuracy performance has been updated: {self.best_attack_acc_adv:.5f} -> {attack_acc_adv:.5f}, val_acc:{val_acc_adv:.5f}")


            epoch_time = time.time() - epoch_start
            s = (
                f"[Epoch {epoch + 1}] "
            )

            if val_loader is not None:
                val_acc_ori= self.evaluate(val_loader)
                attack_acc_ori= self.evaluate(attack_loader)
                s += (
                    f" ---- [ORI] val_acc = {val_acc_ori:.4f}, attack_acc = {attack_acc_ori:.4f}"
                )

                val_acc_adv= self.evaluate(val_loader,"adv")
                attack_acc_adv= self.evaluate(attack_loader,"adv")
                s += (
                    f" ---- [ADV] val_acc = {val_acc_adv:.4f}, attack_acc = {attack_acc_adv:.4f}"
                )
                
            s += f" [{epoch_time:.1f}s]"
            self.logger.info(s)

        train_time = int(time.time() - train_start)
        self.logger.info(f"-- Training done in {train_time}s.")


    def evaluate(self, data_loader,type="clean"):
        self.backbone1.eval()
        self.backbone2.eval()

        y_list = []
        y_hat_list = []

        for batch in tqdm(data_loader):
            batch = tuple(data.to(self.device) for data in batch)
            inputs_ids, inputs_masks,token_type_ids,inputs_labels = batch
            with torch.no_grad():
                if type == "clean":
                    preds = self.backbone1(input_ids = inputs_ids, attention_mask=inputs_masks) 
                else:
                    preds = self.backbone2(input_ids = inputs_ids, attention_mask=inputs_masks) 
            y_list.extend(inputs_labels.detach().cpu().numpy())
            y_hat_list.extend(preds['logits'].detach().cpu().numpy())

        y_list = np.array(y_list)
        y_hat_list = np.array(y_hat_list)
        preds = np.argmax(y_hat_list, axis=1).flatten() # shape = (1, :)
        labels = y_list.flatten()
        acc = np.sum(preds==labels) / len(y_list)
        return acc
    
    def predict(self,data_loader,type="clean"):
        cls = self.backbone1

        if not args.test:
            if type == "avg":   
                cls.load_state_dict(torch.load(f'{args.save_path}/{type}_mix_{args.alpha}_{self.name}_best_acc_model.pkl'))
            else:
                cls.load_state_dict(torch.load(f'{args.save_path}/{type}_{self.name}_best_acc_model.pkl'))
        else:
            cls.load_state_dict(torch.load(args.adv_model))
            
        cls.eval()
        y_list = []
        y_hat_list = []
        for batch in tqdm(data_loader):
            batch = tuple(data.to(self.device) for data in batch)
            inputs_ids, inputs_masks,token_type_ids,inputs_labels = batch
            with torch.no_grad():
                preds = cls(input_ids = inputs_ids, attention_mask=inputs_masks)
            y_list.extend(inputs_labels.detach().cpu().numpy())
            y_hat_list.extend(preds['logits'].detach().cpu().numpy())

        y_list = np.array(y_list)
        y_hat_list = np.array(y_hat_list)
        preds = np.argmax(y_hat_list, axis=1).flatten() # shape = (1, :)
        labels = y_list.flatten()
        acc = np.sum(preds==labels) / len(y_list)
        return acc
        
    def save_model(self,type="clean"):
        if type == "clean":
            cls = self.backbone1
        else:
            cls = self.backbone2
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
        torch.save(cls.state_dict(),f'{args.save_path}/{type}_{self.name}_best_acc_model.pkl')
    
    def compute_accuracy(self,logits,lable):
        preds = np.argmax(logits.detach().cpu().numpy(), axis=1).flatten()
        labels = lable.detach().cpu().numpy().flatten()
        acc = np.sum(preds==labels) / len(labels)
        return acc

    def CTR_loss(self, x, x_aug, T):
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss
    

    def KL_loss(self,Y_adv,Y_clean):
        # generate random probabilities
        q = nn.functional.softmax(Y_adv, dim=-1)
        p = nn.functional.softmax(Y_clean, dim=-1)

        # calculate kl loss
        kl_loss = nn.KLDivLoss(reduction="batchmean")
        loss = kl_loss(q.log(), p)
        return loss
    

    
