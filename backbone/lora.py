# Sheng Wang at Feb 22 2023

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# from safetensors import safe_open
# from safetensors.torch import save_file
from timm.models.vision_transformer import VisionTransformer as timm_ViT
from torch import Tensor
from torch.nn.parameter import Parameter

from backbone.base_vit import ViT
import os
from backbone.linears import SimpleLinear
import gc
import torch.nn.utils as utils
import copy

class _LoRALayer(nn.Module):
    def __init__(self, w: nn.Module, w_a: nn.Module, w_b: nn.Module):
        super().__init__()
        self.w = w
        self.w_a = w_a
        self.w_b = w_b

    def forward(self, x):
        x = self.w(x) + self.w_b(self.w_a(x))
        return x


class LoRA_ViT(nn.Module):
    """Applies low-rank adaptation to a vision transformer.
    Args:
        vit_model: a vision transformer model, see base_vit.py
        r: rank of LoRA
        num_classes: how many classes the model output, default to the vit model
        lora_layer: which layer we apply LoRA.
    Examples::
        >>> model = ViT('B_16_imagenet1k')
        >>> lora_model = LoRA_ViT(model, r=4)
        >>> preds = lora_model(img)
        >>> print(preds.shape)
        torch.Size([1, 1000])
    """
    def __init__(self, vit_model: ViT, r: int, num_classes: int = 0, lora_layer=None):
        super(LoRA_ViT, self).__init__()

        assert r > 0
        base_vit_dim = vit_model.transformer.blocks[0].attn.proj_q.in_features
        dim = base_vit_dim
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(vit_model.transformer.blocks)))
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []
        # lets freeze first
        for param in vit_model.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(vit_model.transformer.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_q_linear = blk.attn.proj_q
            w_v_linear = blk.attn.proj_v
            w_a_linear_q = nn.Linear(dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, dim, bias=False)
            w_a_linear_v = nn.Linear(dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.proj_q = _LoRALayer(w_q_linear, w_a_linear_q, w_b_linear_q)
            blk.attn.proj_v = _LoRALayer(w_v_linear, w_a_linear_v, w_b_linear_v)

        self.reset_parameters()
        self.lora_vit = vit_model
        if num_classes > 0:
            self.lora_vit.fc = nn.Linear(vit_model.fc.in_features, num_classes)

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.lora_vit(x)


class _LoRA_qkv_timm(nn.Module):
    """
    In timm it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """
    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x)) #* self.scaling_factor
        new_v = self.linear_b_v(self.linear_a_v(x)) #* self.scaling_factor
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim :] += new_v
        return qkv
    
class _LoRA_qkv_timm_train(nn.Module):
    def __init__(self, qkv, linear_a_q, linear_b_q, linear_a_v, linear_b_v, #linear_a_q1, linear_b_q1, linear_a_v1, linear_b_v1,
        task_id, saved_A, saved_B, t_layer_i, rank, scaling_factor, scaling_factor_prev, eval1=False):
        super().__init__()
        self.linear_a_q = linear_a_q.cuda()
        self.linear_b_q = linear_b_q.cuda()
        self.linear_a_v = linear_a_v.cuda()
        self.linear_b_v = linear_b_v.cuda()

        self.scaling_factor = scaling_factor.cuda()
        self.scaling_factor_prev = scaling_factor_prev.cuda()

        self.task_id = task_id
        self.qkv = qkv
        self.dim = qkv.in_features
        self.saved_A = saved_A
        self.saved_B = saved_B
        self.t_layer_i = t_layer_i
        self.rank = rank
        self.eval = eval1

    def forward(self, x):

        w_a_linear_q = nn.Linear(self.dim, self.rank, bias=False)
        w_b_linear_q = nn.Linear(self.rank, self.dim, bias=False)
        w_a_linear_v = nn.Linear(self.dim, self.rank, bias=False)
        w_b_linear_v = nn.Linear(self.rank, self.dim, bias=False)
         
        new_q, new_v = 0, 0
        for i in range(self.task_id):
        # for i in range(0):
            saved_A_i, saved_B_i = self.saved_A['saved_A_'+str(i)], self.saved_B['saved_B_'+str(i)]
            Q, V = list(enumerate(zip(saved_A_i,saved_B_i)))[self.t_layer_i*2: self.t_layer_i*2+2]
            _, (A_q, B_q) = Q
            _, (A_v, B_v) = V

            w_a_linear_q.weight = Parameter(A_q.weight)
            w_a_linear_q.weight.requires_grad = False 
            w_a_linear_q.to(x.device)
            w_b_linear_q.weight = Parameter(B_q.weight)
            w_b_linear_q.weight.requires_grad = False 
            w_b_linear_q.to(x.device)
            w_a_linear_v.weight = Parameter(A_v.weight)
            w_a_linear_v.weight.requires_grad = False 
            w_a_linear_v.to(x.device)
            w_b_linear_v.weight = Parameter(B_v.weight)
            w_b_linear_v.weight.requires_grad = False  
            w_b_linear_v.to(x.device)


            if i ==0 :
                new_q = self.scaling_factor_prev[i]( w_b_linear_q(w_a_linear_q(x))/ (torch.norm(w_b_linear_q.weight)* torch.norm(w_a_linear_q.weight) )  )
                new_v = self.scaling_factor_prev[i]( w_b_linear_v(w_a_linear_v(x))/ (torch.norm(w_b_linear_v.weight)* torch.norm(w_a_linear_v.weight) )  )
            else:

                new_q += self.scaling_factor_prev[i]( w_b_linear_q(w_a_linear_q(x))/ (torch.norm(w_b_linear_q.weight)* torch.norm(w_a_linear_q.weight) )  )
                new_v += self.scaling_factor_prev[i]( w_b_linear_v(w_a_linear_v(x))/ (torch.norm(w_b_linear_v.weight)* torch.norm(w_a_linear_v.weight) )  )

        new_q += self.scaling_factor[0]( self.linear_b_q(self.linear_a_q(x)) )
        new_v += self.scaling_factor[0]( self.linear_b_v(self.linear_a_v(x)) )
        qkv = self.qkv(x) 
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim :] += new_v
        return qkv

class _LoRA_qkv_timm_eval(nn.Module):
    """
    In timm it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """
    def __init__(self, task_id, qkv: nn.Module, saved_A, saved_B, t_layer_i, rank, scaling_factor,  scaling_factor_prev, save_file):
        super().__init__()
        self.task_id = task_id
        self.qkv = qkv
        self.dim = qkv.in_features
        self.saved_A = saved_A
        self.saved_B = saved_B
        self.t_layer_i = t_layer_i
        self.rank = rank

        self.save_file = save_file
        self.scaling_factor = scaling_factor.cuda()
        self.scaling_factor_prev = scaling_factor_prev.cuda()


    def forward(self, x):
        new_q, new_v = 0, 0

        w_a_linear_q = nn.Linear(self.dim, self.rank, bias=False)
        w_b_linear_q = nn.Linear(self.rank, self.dim, bias=False)
        w_a_linear_v = nn.Linear(self.dim, self.rank, bias=False)
        w_b_linear_v = nn.Linear(self.rank, self.dim, bias=False)


        file_path = self.save_file+'scaling_factor'+str(self.task_id-1)+'.pt'
        scaling_param = torch.load(file_path)
        for i in range(self.task_id):
            saved_A_i, saved_B_i = self.saved_A['saved_A_'+str(i)], self.saved_B['saved_B_'+str(i)]
            Q, V = list(enumerate(zip(saved_A_i,saved_B_i)))[self.t_layer_i*2: self.t_layer_i*2+2]
            _, (A_q, B_q) = Q
            _, (A_v, B_v) = V

            w_a_linear_q.weight = Parameter(A_q.weight)
            w_b_linear_q.weight = Parameter(B_q.weight)
            w_a_linear_v.weight = Parameter(A_v.weight)
            w_b_linear_v.weight = Parameter(B_v.weight)

            if i ==0 :
                new_q = self.scaling_factor_prev[i]( w_b_linear_q(w_a_linear_q(x))/ (torch.norm(w_b_linear_q.weight)* torch.norm(w_a_linear_q.weight) )  )
                new_v = self.scaling_factor_prev[i]( w_b_linear_v(w_a_linear_v(x))/ (torch.norm(w_b_linear_v.weight)* torch.norm(w_a_linear_v.weight) )  )
            else:
                new_q += self.scaling_factor_prev[i]( w_b_linear_q(w_a_linear_q(x))/ (torch.norm(w_b_linear_q.weight)* torch.norm(w_a_linear_q.weight) )  )
                new_v += self.scaling_factor_prev[i]( w_b_linear_v(w_a_linear_v(x))/ (torch.norm(w_b_linear_v.weight)* torch.norm(w_a_linear_v.weight) )  )

        new_q = self.scaling_factor[0]( w_b_linear_q(w_a_linear_q(x)) )
        new_v = self.scaling_factor[0]( w_b_linear_v(w_a_linear_v(x)) )
 
        qkv = self.qkv(x) 
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim :] += new_v
        return qkv
    


class ParameterWrapper(nn.Module):
    def __init__(self, param):
        super(ParameterWrapper, self).__init__()
        self.param = param
    
    def forward(self, x):
        # print('x, param', x.device(), self.pram.device())
        return x * self.param
    
class MyLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MyLinear, self).__init__()
        self.linear_b_q = nn.Linear(input_dim, output_dim, bias=False)
        self.linear_b_q = utils.weight_norm(self.linear_b_q)

    def forward(self, x):
        return self.linear_b_q(x)


class LoRA_ViT_timm(nn.Module):
    def __init__(self, vit_model: timm_ViT, r: int, num_classes: int = 0, increment=10, filepath = './', lora_layer=None, eval=False, index=True, cur_task_index=None):
        super(LoRA_ViT_timm, self).__init__()

        assert r > 0
        self.rank =r
        self.base_vit = copy.deepcopy(vit_model)
        


        if not eval:
            self.save_file = filepath
            self.increment = increment
            print('save_file', self.save_file)


        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(vit_model.blocks)))


        self.w_As, self.w_Bs = [], []  # These are linear layers

        
        if index:
            print('Initialize task-id and curtask id')
            self.task_id, self.cur_id = 0,0
        
        if cur_task_index != None:
            # print('Update the network!!!', cur_task_index)
            self.task_id = cur_task_index

        # freeze the saved part
        for param in self.base_vit.parameters():
            param.requires_grad = False


        for param in vit_model.parameters():
            param.requires_grad = False

        saved_lora_A, saved_lora_B = {}, {}
        for i in range(self.task_id):
            file_path = self.save_file+'lora_w_a_'+str(i)+'.pt'
            saved_lora_A['saved_A_'+str(i)] = torch.load(file_path)
            file_path = self.save_file+'lora_w_b_'+str(i)+'.pt'
            saved_lora_B['saved_B_'+str(i)] = torch.load(file_path)

        scaling_factor = nn.Parameter(torch.Tensor([0.8]))
        self.wrapped_param = nn.ModuleList([ParameterWrapper(scaling_factor)])
        self.wrapped_param_prev = nn.ModuleList([ParameterWrapper(nn.Parameter(torch.Tensor([0.8]))) for _ in range(20)])

        # Do the surgery 
        for t_layer_i, blk in enumerate(vit_model.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)

            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)

            if not eval:
                blk.attn.qkv = _LoRA_qkv_timm_train(
                    w_qkv_linear, w_a_linear_q, w_b_linear_q, w_a_linear_v, w_b_linear_v, 
                    self.task_id, saved_lora_A, saved_lora_B, t_layer_i, self.rank , self.wrapped_param, self.wrapped_param_prev, eval1=False
                )
            else:
                blk.attn.qkv = _LoRA_qkv_timm_eval(self.task_id, w_qkv_linear, saved_lora_A, saved_lora_B, t_layer_i, self.rank, self.wrapped_param, self.wrapped_param_prev, self.save_file) 

        self.reset_parameters()
        self.lora_vit = vit_model
        if not eval:
            self.lora_vit.head = torch.nn.Identity()
        else:
            self.reset_lora_vit_head()



    def reset_lora_vit_head(self):
        task_incremental = self.increment
        self.lora_vit.head = self.generate_fc(768, (self.task_id)*task_incremental).cuda()
        temp_weights = torch.load(self.save_file+'CLs_weight'+str(self.task_id-1)+'.pt') 
        temp_bias = torch.load(self.save_file+'CLs_bias'+str(self.task_id-1)+'.pt') 

        self.lora_vit.head.weight.data = temp_weights.data.cuda()
        self.lora_vit.head.bias.data = temp_bias.data.cuda()


    # This part is only used during the evaluation
    def reset(self, eval=False):
        self.__init__(self.base_vit, self.rank, lora_layer=None, eval=eval, index=False)

    def reset_parameters(self) -> None:
        # if self.task_id ==0: 
            for w_A in self.w_As:
                nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
                # nn.init.kaiming_uniform_(w_A.linear_b_q.weight, a=math.sqrt(5) )
            for w_B in self.w_Bs:
                nn.init.zeros_(w_B.weight)


    def save_wrap_param(self, filename):
        if self.task_id ==1:   
            scaling_param = torch.zeros(20,20)
        else:
            scaling_param = torch.load(filename + 'scaling_factor'+str(self.task_id-2)+'.pt')
        i = self.task_id-1
        # print('save i', i)
        for j in range(i+1):
            if j == i:
                scaling_param[i][j] = self.wrapped_param[0].param.clone()
            else:
                scaling_param[i][j] = self.wrapped_param_prev[j].param.clone()  
        torch.save(scaling_param, filename + 'scaling_factor'+str(self.task_id-1)+'.pt')
        
    def save_lora_parameters(self, filename: str, task_id) -> None:
        self.task_id += 1
        if not os.path.exists(filename):
           os.makedirs(filename)
        torch.save(self.w_As, filename + 'lora_w_a_'+str(task_id)+'.pt')
        torch.save(self.w_Bs, filename + 'lora_w_b_'+str(task_id)+'.pt')

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def load_eval_vit(self):
        self.lora_vit = copy.deepcopy(self.base_vit)
        saved_lora_A, saved_lora_B = {}, {}
        for i in range(self.task_id):
            file_path = self.save_file+'lora_w_a_'+str(i)+'.pt'
            saved_lora_A['saved_A_'+str(i)] = torch.load(file_path)
            file_path = self.save_file+'lora_w_b_'+str(i)+'.pt'
            saved_lora_B['saved_B_'+str(i)] = torch.load(file_path)

        # for param in self.eval_vit.parameters():
        for param in self.lora_vit.parameters():
            param.requires_grad = False
        
        # for t_layer_i, blk in enumerate(self.eval_vit.blocks):
        for t_layer_i, blk in enumerate(self.lora_vit.blocks):
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            blk.attn.qkv = _LoRA_qkv_timm_eval(self.task_id, w_qkv_linear, saved_lora_A, saved_lora_B, t_layer_i, self.rank)    
        self.reset_lora_vit_head()

    def compute_ortho_loss(self):
        loss = torch.tensor(0).float().cuda()
        # print('task_id', self.task_id)
        for i in range(self.task_id):
            file_path = self.save_file+'lora_w_a_'+str(i)+'.pt'
            if os.path.exists(file_path):
                w_As = torch.load(file_path)
                num_layer = len(self.w_As)
                for j in range(num_layer):
                    temp = torch.matmul(w_As[j].weight.to(self.w_As[j].weight.device), self.w_As[j].weight.t())
                    temp = torch.sum(torch.square(temp))
                    loss = loss.to(self.w_As[j].weight.device)
                    loss += temp
        return loss
    
    def forward(self, x: Tensor, loss= False, eval=False) -> Tensor:
        if eval:
            self.reset(eval=True)
            return self.lora_vit(x)
        elif loss:
            loss = self.compute_ortho_loss()
            return self.lora_vit(x), loss
        else:
            return self.lora_vit(x)
