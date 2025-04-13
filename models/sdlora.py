import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy

import timm
from backbone.lora import LoRA_ViT_timm
import torch.distributed as dist

import os

num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, True)

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
            # model = nn.parallel.DistributedDataParallel(model, device_ids=[self._device], output_device=self._device, find_unused_parameters=True)
        # if len(self._multiple_gpus) > 1:
        #     self._network = self._network.module

        self._train(self.train_loader, self.test_loader)

        # to test
        # self._network.to(self._device)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def update_network(self, index=True):
        # if use VIT-B-16
        model = timm.create_model("vit_base_patch16_224",pretrained=True, num_classes=0)

        # if use DINO
        # model = timm.create_model('vit_base_patch16_224_dino', pretrained=True, num_classes=0)

        # SD-LoRA-RR
        '''
        if self._cur_task >=4 and self._cur_task <8:
            rank = 8 #8
        elif self._cur_task >=8:
            rank = 6 #6
        # elif self._cur_task >=8:
        #     rank = 4
        else:
            rank = 10
        '''
        rank=10
        model = LoRA_ViT_timm(vit_model=model.eval(), r=rank, num_classes=10, index=index, increment= self.args['increment'], filepath=self.args['filepath'], 
        cur_task_index= self._cur_task)
        model.out_dim = 768
        return model

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._cur_task == 0:
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=self.args["init_lr"],
                # weight_decay=self.args["init_weight_decay"],
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"]
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)

        else:
            if len(self._multiple_gpus) > 1:
                self._network = self._network.module
            self._network.backbone = self.update_network(index=False)
            if len(self._multiple_gpus) > 1:
                self._network = nn.DataParallel(self._network, self._multiple_gpus)       
            self._network.to(self._device) 

            optimizer = optim.SGD(
                self._network.parameters(),
                lr=self.args["lrate"],
                momentum=0.9,
            )  # 1e-5
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=self.args["milestones"], gamma=self.args["lrate_decay"]
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

        save_lora_name = self.args['filepath']

        if len(self._multiple_gpus) > 1:
            self._network.module.backbone.save_lora_parameters(save_lora_name, self._cur_task)
            self._network.module.save_fc(save_lora_name, self._cur_task)
        else:
            self._network.backbone.save_lora_parameters(save_lora_name, self._cur_task)
            self._network.save_fc(save_lora_name, self._cur_task)

    def get_optimizer(self):
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()), 
                momentum=0.9, 
                lr=self.init_lr,
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                # lr=self.init_lr, 
                self.args["lrate"],
                # weight_decay=self.weight_decay
                betas=(0.9, 0.999)
            )
            
        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=self.init_lr, 
                weight_decay=self.weight_decay
            )

        return optimizer
    
    def get_scheduler(self, optimizer):
        if self.args["scheduler"] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.args['tuned_epoch'], eta_min=self.args['min_lr'])
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == 'constant':
            scheduler = None

        return scheduler



    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["init_epoch"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]
                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epoch"],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epoch"],
                    losses / len(train_loader),
                    train_acc,
                )

            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["epochs"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                # logits = self._network(inputs)["logits"]
                logits, ortho_loss = self._network(inputs, ortho_loss=True)
                logits = logits['logits'] 
                

                fake_targets = targets - self._known_classes
                loss_clf = F.cross_entropy(
                    logits[:, self._known_classes :], fake_targets
                )
                # print('@@@@@@@@@@@@@@loss2', loss_clf, torch.mean(ortho_loss))

                # loss = loss_clf + 10* torch.mean(ortho_loss)
                loss = loss_clf

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)
