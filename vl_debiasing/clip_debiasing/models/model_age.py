from abc import ABC
from typing import Tuple, Callable, Any, Union, List, Dict

import clip
import torch
import torch.nn as nn
from torch.nn import Module

from clip_debiasing import Dotdict

import torch.nn.functional as F

import random
# class ImageMask(nn.Module):
#     def __init__(self):
#         super(ImageMask, self).__init__()
#         self.weights = nn.Parameter(torch.Tensor(3, 224, 224))
        
#     def forward(self,x):
#         # print(self.weights)
#         return x * self.weights
    
class MLP(nn.Module):
    def __init__(self, D_in, D_out, n_layer=1):
        super().__init__()
        if n_layer == 1:
            self.linear = nn.Linear(D_in, D_out)
        elif n_layer == 2:
            self.linear = nn.Sequential(
                nn.Linear(D_in, 256),
                nn.ReLU(),
                nn.Linear(256, D_out)
            )
        # elif n_layer == 2:
        #     self.linear = nn.Sequential(
        #         nn.Linear(D_in, D_out),
        #         nn.LayerNorm(D_out),
        #         nn.ReLU(),
        #         nn.Linear(D_out, D_out),
        #         nn.LayerNorm(D_out)
        #     )
        
    def forward(self,x):
        # print(self.linear.weight.grad)
        x = self.linear(x)
        x = F.relu(x)
        return x


class ClipLike(Module, ABC):
    """Essentially a type stub for specifying what a clip-like model supports"""

    visual: Any
    logit_scale: Any
    dtype: torch.dtype
    positional_embedding: Any
    text_projection: Any
    token_embedding: Any
    visual: Any

    def transformer(self, text_features) -> Any:
        pass

    def ln_final(self, text_features) -> Any:
        pass

    def encode_image(self, images) -> torch.Tensor:
        pass

    def encode_text(self, tokenized_texts) -> torch.Tensor:
        pass


VALID_CLIP_MODELS = [
    "openai/CLIP/RN50",
    "openai/CLIP/RN101",
    "openai/CLIP/RN50x4",
    "openai/CLIP/ViT-B/16",
    "openai/CLIP/ViT-B/32",
    "openai/CLIP/ViT-L/14",
]

VALID_MODELS = (
    # openai clips
    VALID_CLIP_MODELS
)


def model_loader(model_name, device=None, jit=False) -> Tuple[ClipLike, Callable[[Any], torch.Tensor],
                                                              Callable[[Any], torch.LongTensor], str]:
    """Returns cliplike model, preprocessing function for images, tokenizer, and modelname/alias"""
    # Some models aren't compatible with the tokens we generate (they have mismatching dimensions),

    if model_name not in VALID_MODELS:
        raise NotImplementedError(
            f"{model_name} not found, should be on of..", VALID_MODELS
        )

    if model_name.startswith("openai/CLIP/"):
        arch_str = model_name.replace("openai/CLIP/", "")
        model, preprocess = clip.load(arch_str, device=device, jit=jit)
        tokenizer = clip.tokenize
        alias_name = "debiased-clip-" + "-".join(model_name.split("/")[2:]).lower()
    elif model_name.startswith("m-bain/frozen-in-time/"):
        raise NotImplementedError
    elif model_name.startswith("facebookresearch/SLIP/"):
        raise NotImplementedError
    else:
        raise NotImplementedError

    return model, preprocess, tokenizer, alias_name


class DebiasedCLIP(nn.Module):

    def __init__(self, arch_str, device, debiasing_modules=True, **_kwargs,):
        super().__init__()

        # self.logit_scale = clip_model.logit_scale
        self.dtype = torch.float32

        self.clip, self.preprocess = clip.load(arch_str, device=device)
        
        for param in self.parameters():
            param.requires_grad = False # freeze all parameters
        
        if debiasing_modules:
            self.debiaser = MLP(512, 512, 2).to(device) 
            # self.modality_adapter = ImageMask().to(device)
            self.modality_adapter = MLP(512, 512, 2).to(device)
            # self.debiaser.load_state_dict(torch.load("debiaser.pth"))
            # self.modality_adapter.load_state_dict(torch.load("adapter.pth"))

            for param in list(self.debiaser.parameters()) + list(self.modality_adapter.parameters()):
                param.requires_grad = True 

            self.queue_size = 65536
            self.momentum = 0.995
            self.alpha = 0.4

            self.register_buffer("image_queue", torch.randn(512, self.queue_size))
            self.register_buffer("text_queue", torch.randn(512, self.queue_size))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

            self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
            self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
            self.temp = nn.Parameter(torch.ones([]) * 0.07)
            
            self.debiaser_m = MLP(512, 512, 2).to(device)
            self.modality_adapter_m = MLP(512, 512, 2).to(device)

            self.model_pairs = [[self.debiaser,self.debiaser_m],
                                [self.modality_adapter,self.modality_adapter_m],
                            ]
            self.copy_params()
        

    def encode_text(self, text):
        with torch.no_grad():
            text_embeddings = self.clip.encode_text(text) # from shape (bs, 1, 77) to (bs, 77)
        return self.debiaser(text_embeddings.float()) + text_embeddings

    def encode_image(self, image):
        # aligned_image = self.modality_adapter(image)
        with torch.no_grad():
            image_embeddings = self.clip.encode_image(image)
    
        return self.debiaser(self.modality_adapter(image_embeddings.float()) + image_embeddings) + image_embeddings
        
        # return self.clip.encode_image(image)

    def forward(self, image, text1, text2, text3, epoch):
        # initialise text feats with zeros
        
        with torch.no_grad():
            self._momentum_update()
            # biased_word1_features = F.normalize(self.clip.encode_text(word1), dim=-1)
            # biased_word2_features = F.normalize(self.clip.encode_text(word2), dim=-1)
            # counterfact_text_features = F.normalize(self.clip.encode_text(text2), dim=-1)

            original_image_features = F.normalize(self.clip.encode_image(image), dim=-1)
            original_text_features = F.normalize(self.clip.encode_text(text1), dim=-1)

            image_embeddings = self.clip.encode_image(image)
            debiased_image_features_m = F.normalize(self.debiaser_m(self.modality_adapter_m(image_embeddings.float()) + image_embeddings) + image_embeddings, dim=-1)

            text_embeddings = self.clip.encode_text(text1)
            debiased_text_features_m = F.normalize(self.debiaser_m(text_embeddings.float()) + text_embeddings, dim=-1)

            original_image_features_all = torch.cat([original_image_features.t(),self.image_queue.clone().detach()],dim=1)
            original_text_features_all = torch.cat([original_text_features.t(),self.text_queue.clone().detach()],dim=1)

            sim_i2t_targets = F.softmax(original_image_features.float() @ original_text_features_all.float() / self.temp, dim=1)
            sim_t2i_targets = F.softmax(original_text_features.float() @ original_image_features_all.float() / self.temp, dim=1)

            sim_i2t_targets_m = F.softmax(debiased_image_features_m.float() @ original_text_features_all.float() / self.temp, dim=1)
            sim_t2i_targets_m = F.softmax(debiased_text_features_m.float() @ original_image_features_all.float() / self.temp, dim=1)

            if epoch >= 5: # learn a basic modality adapter and debiaser first
                sim_i2t_targets = self.alpha * sim_i2t_targets_m + (1 - self.alpha) * sim_i2t_targets
                sim_t2i_targets = self.alpha * sim_t2i_targets_m + (1 - self.alpha) * sim_t2i_targets

        fair_image_features = F.normalize(self.encode_image(image), dim=-1)

        fair_text1_features = F.normalize(self.encode_text(text1), dim=-1)
        fair_text2_features = F.normalize(self.encode_text(text2), dim=-1)
        fair_text3_features = F.normalize(self.encode_text(text3), dim=-1)

        biased_text1_features = fair_text1_features - original_text_features
        # biased_text2_features = fair_text2_features - counterfact_text_features
        biased_image_features = fair_image_features - original_image_features

        # text_sim_loss = -torch.nn.CosineSimilarity(dim=-1)(fair_text1_features, fair_text2_features).mean()
        
        # print(fair_text1_features.shape, biased_word1_features.shape)
        # print(torch.nn.CosineSimilarity(dim=-1)(fair_text1_features, biased_word1_features).shape)

        # text_diff_loss = torch.abs(torch.nn.CosineSimilarity(dim=-1)(fair_text1_features, biased_text1_features)).mean() + torch.abs(torch.nn.CosineSimilarity(dim=-1)(fair_text2_features, biased_text2_features)).mean()
        text_diff_loss = 0.5 * (torch.abs(torch.nn.CosineSimilarity(dim=-1)(fair_image_features, biased_image_features)).mean() + torch.abs(torch.nn.CosineSimilarity(dim=-1)(fair_image_features, biased_text1_features)).mean())

        prob = random.random()
        fair_text_random = fair_text1_features if prob < 1/3 else fair_text2_features if prob < 2/3 else fair_text3_features#

        fair_i2t = fair_image_features.float() @ original_text_features_all.float() / self.temp
        fair_t2i = fair_text_random.float() @ original_image_features_all.float() / self.temp

        # fair_logits = norm_fair_image_features.float() @ norm_fair_text_features.t().float()

        # loss_i2t = -torch.sum(F.log_softmax(fair_logits, dim=1)*original_logits,dim=1).mean()


        loss_i2t = -torch.sum(F.log_softmax(fair_i2t, dim=1)*sim_i2t_targets,dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(fair_t2i, dim=1)*sim_t2i_targets,dim=1).mean()

        loss_ita = (loss_i2t+loss_t2i)/2

        # delta = epoch * 0.1 / 100 # 0 to 0.1

        # loss = text_sim_loss * 0.2 + text_diff_loss * 0.1 + loss_ita * 0.7

        # loss = text_sim_loss * (0.2 + 2 * delta) + text_diff_loss * (0.1 + 2 * delta) + loss_ita * (0.7 - 3 * delta)
        # initial: 0.2, 0.1, 0.7
        # final: 0.4, 0.2, 0.4
        # loss = text_sim_loss * 0.3 + loss_i2t * 0.7
        self._dequeue_and_enqueue(original_image_features, original_text_features)
        # print(text_sim_loss, text_diff_loss, loss_ita)
        return loss_ita, text_diff_loss


    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feats, text_feats):
        # gather keys before updating queue
        # image_feats = concat_all_gather(image_feat)
        # text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        # assert self.queue_size % batch_size == 0  # for simplicity
        if ptr + batch_size > self.queue_size:
            self.image_queue[:, ptr:] = image_feats[:self.queue_size - ptr].T
            self.text_queue[:, ptr:] = text_feats[:self.queue_size - ptr].T
            self.image_queue[:, :ptr + batch_size - self.queue_size] = image_feats[self.queue_size - ptr:].T
            self.text_queue[:, :ptr + batch_size - self.queue_size] = text_feats[self.queue_size - ptr:].T
        else:
            self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
            self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                
    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    
