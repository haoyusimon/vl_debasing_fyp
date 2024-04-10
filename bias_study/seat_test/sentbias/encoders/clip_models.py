from abc import ABC
from typing import Tuple, Callable, Any, Union, List, Dict

import clip
import torch
import torch.nn as nn
from torch.nn import Module


import torch.nn.functional as F
from sklearn.decomposition import TruncatedSVD
import numpy as np

import random
# from scipy.special import digamma
# import scipy.spatial as ss
from math import log
from itertools import permutations

# class ImageMask(nn.Module):
#     def __init__(self):
#         super(ImageMask, self).__init__()
#         self.weights = nn.Parameter(torch.Tensor(3, 224, 224))
        
#     def forward(self,x):
#         # print(self.weights)
#         return x * self.weights
    
class Dotdict(dict):
    def __getattr__(self, __name: str) -> Any:
        return super().get(__name)

    def __setattr__(self, __name: str, __value: Any) -> None:
        return super().__setitem__(__name, __value)

    def __delattr__(self, __name: str) -> None:
        return super().__delitem__(__name)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)
        
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

    def forward(self, image, text1, text2, word1, word2, epoch):
        # initialise text feats with zeros
        
        with torch.no_grad():
            self._momentum_update()
            # biased_word1_features = F.normalize(self.clip.encode_text(word1), dim=-1)
            # biased_word2_features = F.normalize(self.clip.encode_text(word2), dim=-1)
            counterfact_text_features = F.normalize(self.clip.encode_text(text2), dim=-1)

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

        biased_text1_features = fair_text1_features - original_text_features
        # biased_text2_features = fair_text2_features - counterfact_text_features
        biased_image_features = fair_image_features - original_image_features

        text_sim_loss = -torch.nn.CosineSimilarity(dim=-1)(fair_text1_features, fair_text2_features).mean()
        
        # print(fair_text1_features.shape, biased_word1_features.shape)
        # print(torch.nn.CosineSimilarity(dim=-1)(fair_text1_features, biased_word1_features).shape)

        # text_diff_loss = torch.abs(torch.nn.CosineSimilarity(dim=-1)(fair_text1_features, biased_text1_features)).mean() + torch.abs(torch.nn.CosineSimilarity(dim=-1)(fair_text2_features, biased_text2_features)).mean()
        text_diff_loss = 0.5 * (torch.abs(torch.nn.CosineSimilarity(dim=-1)(fair_image_features, biased_image_features)).mean() + torch.abs(torch.nn.CosineSimilarity(dim=-1)(fair_image_features, biased_text1_features)).mean())


        fair_text_random = fair_text1_features if random.random() < 0.5 else fair_text2_features # maybe sum up the fair_text from different directions?

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
        return loss_ita, text_sim_loss, text_diff_loss


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

# def Mixed_KSG(x,y,k=5):
#     assert len(x)==len(y), "Lists should have same length"
#     assert k <= len(x)-1, "Set k smaller than num. samples - 1"
#     N = len(x)
#     if x.ndim == 1:
#         x = x.reshape((N,1))
#     dx = len(x[0])   	
#     if y.ndim == 1:
#         y = y.reshape((N,1))
#     dy = len(y[0])
#     data = np.concatenate((x,y),axis=1)

#     tree_xy = ss.cKDTree(data)
#     tree_x = ss.cKDTree(x)
#     tree_y = ss.cKDTree(y)

#     knn_dis = [tree_xy.query(point,k+1,p=float('inf'))[0][k] for point in data]
#     ans = 0

#     for i in range(N):
#         kp, nx, ny = k, k, k
#         if knn_dis[i] == 0:
#             kp = len(tree_xy.query_ball_point(data[i],1e-15,p=float('inf')))
#             nx = len(tree_x.query_ball_point(x[i],1e-15,p=float('inf')))
#             ny = len(tree_y.query_ball_point(y[i],1e-15,p=float('inf')))
#         else:
#             nx = len(tree_x.query_ball_point(x[i],knn_dis[i]-1e-15,p=float('inf')))
#             ny = len(tree_y.query_ball_point(y[i],knn_dis[i]-1e-15,p=float('inf')))
#         ans += (digamma(kp) + log(N) - digamma(nx) - digamma(ny))/N
#     return ans

class CLIP_clipped(nn.Module):

    def __init__(self, arch_str, device, hidden_dim, m, debiasing_modules=True, **_kwargs,):
        super().__init__()

        # self.logit_scale = clip_model.logit_scale
        self.dtype = torch.float32

        
        self.clip, self.preprocess = clip.load(arch_str, device=device)

        # # gender
        # self.keep_ind = torch.tensor([  2,   5,   6,   7,   9,  10,  12,  16,  17,  18,  21,  24,  25,  27,
        #  29,  30,  31,  33,  36,  37,  40,  41,  42,  43,  45,  47,  48,  49,
        #  51,  53,  54,  62,  63,  64,  65,  66,  68,  73,  76,  80,  84,  85,
        #  88,  89,  90,  91,  98, 100, 107, 109, 111, 112, 113, 116, 117, 118,
        # 119, 120, 125, 127, 132, 134, 135, 139, 142, 143, 144, 147, 148, 151,
        # 153, 156, 158, 160, 162, 164, 165, 166, 169, 172, 173, 174, 176, 180,
        # 181, 182, 183, 184, 185, 186, 188, 189, 190, 191, 193, 194, 196, 200,
        # 202, 203, 205, 208, 209, 211, 215, 216, 219, 220, 221, 222, 223, 225,
        # 233, 234, 235, 236, 239, 245, 248, 249, 252, 253, 256, 257, 260, 263,
        # 264, 268, 272, 274, 276, 278, 279, 281, 282, 284, 285, 286, 291, 293,
        # 294, 295, 298, 299, 300, 301, 302, 303, 305, 306, 310, 313, 316, 317,
        # 318, 319, 321, 324, 327, 328, 329, 330, 331, 332, 335, 336, 340, 344,
        # 345, 346, 348, 349, 350, 351, 352, 354, 356, 357, 358, 359, 360, 361,
        # 362, 363, 364, 365, 366, 367, 369, 370, 372, 373, 377, 378, 379, 380,
        # 381, 383, 385, 386, 393, 395, 398, 402, 404, 408, 410, 412, 413, 415,
        # 417, 420, 426, 427, 431, 433, 435, 436, 437, 438, 441, 442, 443, 444,
        # 446, 451, 453, 454, 458, 459, 460, 463, 469, 472, 474, 476, 478, 479,
        # 480, 481, 484, 485, 487, 488, 489, 491, 492, 493, 494, 495, 497, 499,
        # 501, 503, 504, 508])

        # age
        self.keep_ind = torch.tensor([  2,   4,   5,   6,   8,   9,  10,  11,  12,  16,  17,  18,  19,  21,
         24,  25,  26,  27,  28,  29,  30,  31,  33,  37,  39,  40,  41,  43,
         47,  48,  49,  51,  53,  54,  59,  61,  62,  64,  65,  66,  68,  69,
         72,  73,  76,  80,  84,  85,  88,  89,  90,  91,  96,  98, 107, 109,
        111, 114, 118, 120, 125, 127, 134, 135, 139, 142, 143, 144, 147, 148,
        151, 152, 153, 155, 156, 158, 160, 164, 165, 166, 169, 172, 173, 174,
        175, 176, 177, 179, 180, 181, 182, 185, 186, 188, 189, 191, 193, 194,
        196, 200, 203, 204, 206, 207, 208, 209, 211, 214, 215, 216, 219, 220,
        221, 223, 229, 230, 233, 234, 235, 239, 241, 242, 245, 246, 248, 249,
        252, 256, 260, 263, 264, 271, 272, 276, 279, 285, 286, 291, 294, 295,
        298, 299, 300, 302, 303, 305, 306, 310, 313, 316, 317, 318, 319, 321,
        322, 327, 328, 329, 330, 331, 336, 337, 340, 343, 344, 345, 346, 349,
        350, 351, 352, 354, 356, 357, 358, 359, 361, 362, 363, 364, 367, 368,
        369, 370, 372, 378, 379, 380, 386, 388, 391, 392, 393, 394, 395, 397,
        398, 402, 404, 405, 408, 410, 412, 413, 415, 418, 420, 421, 424, 425,
        427, 430, 431, 435, 436, 437, 438, 440, 441, 442, 443, 445, 446, 449,
        451, 453, 454, 458, 459, 460, 463, 466, 467, 469, 474, 476, 478, 479,
        480, 484, 485, 486, 487, 488, 491, 493, 494, 496, 497, 499, 501, 502,
        503, 504, 507, 508])
        # self.keep_ind = None
        
        # if self.keep_ind == None:
        #     ds = FairFace(iat_type='gender', mode="train_val", equal_split=False, transforms=self.preprocess)
        #     dl = DataLoader(ds, batch_size=10, num_workers=6)
        #     x = defaultdict(lambda : [])
        #     y = []
        #     for batch in tqdm(dl):
        #         # print(batch['img'])
        #         with torch.no_grad():
        #             image_embeddings = self.clip.encode_image(batch['img'].to('cuda'))
        #         for dim in range(hidden_dim):
        #             x[dim] += image_embeddings[:, dim].tolist()
        #         y += batch["iat_label"].tolist()
        #         # break
        #         # print(Mixed_KSG(np.array(x[0]), np.array(y)))
        #         # print(Mixed_KSG(np.array(x[1]), np.array(y)))
        #         # print(x)
        #         # print(y)
        #     scores = []
        #     for dim in range(hidden_dim): # change for ViT-L/14 which has hidden size of 768
        #         scores.append(Mixed_KSG(np.array(x[dim]), np.array(y)))
        #     scores = torch.FloatTensor(scores)
        #     remaining_inds = torch.topk(scores, m, largest=False).indices
        #     remaining_inds = remaining_inds.sort().values
        #     self.keep_ind = remaining_inds
        # print(self.keep_ind)

        # x = torch.stack([feats[ind] for feats, ind in zip(x, keep_ind)], dim=0)
    def encode_text(self, text):
        with torch.no_grad():
            text_embeddings = self.clip.encode_text(text) # from shape (bs, 1, 77) to (bs, 77)
        # x = torch.stack([feats[ind] for feats, ind in zip(x, keep_ind)], dim=0)
        return text_embeddings[:, self.keep_ind]

    def encode_image(self, image):
        # aligned_image = self.modality_adapter(image)
        with torch.no_grad():
            image_embeddings = self.clip.encode_image(image)
    
        return image_embeddings[:, self.keep_ind]
    
class CLIP_prompt(nn.Module):

    def __init__(self, arch_str, device, **_kwargs,):

        def get_embeddings(text, clip_model, normalize=True):
            text_tokens = clip.tokenize(text)

            clip_model.to(device)
            clip_model.eval()
            with torch.no_grad():
                text_tokens = text_tokens.to(device)
                text_embeddings = clip_model.encode_text(text_tokens).float().cpu()
                if normalize:
                    text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            # clip_model.cpu()
            return text_embeddings
        # Helper functions for debiasing 
        def get_proj_matrix(embeddings):
            tSVD = TruncatedSVD(n_components=len(embeddings))
            embeddings_ = tSVD.fit_transform(embeddings)
            basis = tSVD.components_.T

            # orthogonal projection
            proj = np.linalg.inv(np.matmul(basis.T, basis))
            proj = np.matmul(basis, proj)
            proj = np.matmul(proj, basis.T)
            proj = np.eye(proj.shape[0]) - proj
            return proj

        def get_A(z_i, z_j):
            z_i = z_i[:, None]
            z_j = z_j[:, None]
            return np.matmul(z_i, z_i.T) + np.matmul(z_j, z_j.T) - np.matmul(z_i, z_j.T) - np.matmul(z_j, z_i.T)

        def get_M(embeddings, S):
            d = embeddings.shape[1]
            M = np.zeros((d, d))
            for s in S:
                M  += get_A(embeddings[s[0]], embeddings[s[1]])
            return M / len(S)
        
        super().__init__()

        # self.logit_scale = clip_model.logit_scale
        self.dtype = torch.float32
        self.device = device

        self.lam = 1000

        self.clip, self.preprocess = clip.load(arch_str, device=device)

        # spurious_prompt = ['A photo of a male.', 'A photo of a male celebrity.', 'A photo of a man.', 'A photo of David.',
        #                    'A photo of a female.', 'A photo of a female celebrity.', 'A photo of a woman.', 'A photo of Amy.']
        spurious_prompt = ['A photo of a male.',
                           'A photo of a female.']

        # candidate_prompt = ['A photo of a male celebrity with dark hair.',
        #                     'A photo of a female celebrity with dark hair.',
        #                     'A photo of a male celebrity with blond hair.',
        #                     'A photo of a female celebrity with blond hair.']
# template,concept
# This is the face of a {} person,good
#  This is a photo of a {} person,kind
#               This person is {},successful
#                     A {} person,noncriminal
#                 A {} individual,nonviolent
#           This individual is {},moral
#                   Photo of a {},rich
#                              {},honest
#                    This is a {},smart
#       This is the photo of a {},evil
#                            A {},unkind
#                                ,unsuccessful
#                                ,criminal
#                                ,violent
#                                ,immoral
#                                ,poor
#                                ,dishonest
#                                ,dumb
#                                ,clever
#                                ,stupid
#                                ,hardworking
#                                ,lazy
#                                ,nasty
#                                ,trustworthy
#                                ,untrustworthy
#                                ,caring
#                                ,dangerous
#                                ,heroic
#                                ,villainous
        
        candidate_prompt = []
        S = []
        concepts = ["good", "evil"]
        # concepts = ["good", "kind", "successful", "noncriminal", "nonviolent", "moral", "rich", "honest", "smart", "evil", "unkind", "unsuccessful", "criminal", "violent", "immoral", "poor", "dishonest", "dumb", "clever", "stupid", "hardworking", "lazy", "nasty", "trustworthy", "caring", "dangerous", "heroic", "villainous"]
        for idx, word in enumerate(concepts):
            candidate_prompt += [f'A photo of a male {word} person.', f'A photo of a female {word} person.']
            article = 'an' if word in ["evil", "immoral", "unkind", "unsuccessful", "honest", "untrustworthy"] else 'a'
            candidate_prompt += [f'A photo of {article} {word} man.', f'A photo of {article} {word} woman.']
            S += [[2*idx, 2*idx + 1], [2*idx + 2, 2*idx + 3]]

        spurious_embeddings = get_embeddings(spurious_prompt,
                                             self.clip,
                                             normalize=True)
        
        spurious_embeddings = spurious_embeddings.numpy()
        P0 = get_proj_matrix(spurious_embeddings)

        # Calculate Embedding of Positive Pairs
        candidate_embeddings = get_embeddings(candidate_prompt,
                                             self.clip,
                                             normalize=True)
        candidate_embeddings = candidate_embeddings.numpy()

        # Closed Form Optimum
        print('Solve Closed Form Optimum')
        self.M = get_M(candidate_embeddings, S)
        self.G = self.lam * self.M + np.eye(self.M.shape[0])
        self.P = np.matmul(P0, np.linalg.inv(self.G))

        print(self.P)
        # x = torch.stack([feats[ind] for feats, ind in zip(x, keep_ind)], dim=0)
    def encode_text(self, text):
        with torch.no_grad():
            text_embeddings = self.clip.encode_text(text) # from shape (bs, 1, 77) to (bs, 77)

        text_embeddings = np.matmul(text_embeddings.cpu(), self.P.T)
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        text_embeddings = torch.tensor(text_embeddings).float().to(self.device)
        return text_embeddings

    def encode_image(self, image):
        # aligned_image = self.modality_adapter(image)
        with torch.no_grad():
            image_embeddings = self.clip.encode_image(image)
    
        return image_embeddings
    
class CLIP_prompt_age(nn.Module):

    def __init__(self, arch_str, device, **_kwargs,):

        def get_embeddings(text, clip_model, normalize=True):
            text_tokens = clip.tokenize(text)

            clip_model.to(device)
            clip_model.eval()
            with torch.no_grad():
                text_tokens = text_tokens.to(device)
                text_embeddings = clip_model.encode_text(text_tokens).float().cpu()
                if normalize:
                    text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            # clip_model.cpu()
            return text_embeddings
        # Helper functions for debiasing 
        def get_proj_matrix(embeddings):
            tSVD = TruncatedSVD(n_components=len(embeddings))
            embeddings_ = tSVD.fit_transform(embeddings)
            basis = tSVD.components_.T

            # orthogonal projection
            proj = np.linalg.inv(np.matmul(basis.T, basis))
            proj = np.matmul(basis, proj)
            proj = np.matmul(proj, basis.T)
            proj = np.eye(proj.shape[0]) - proj
            return proj

        def get_A(z_i, z_j):
            z_i = z_i[:, None]
            z_j = z_j[:, None]
            return np.matmul(z_i, z_i.T) + np.matmul(z_j, z_j.T) - np.matmul(z_i, z_j.T) - np.matmul(z_j, z_i.T)

        def get_M(embeddings, S):
            d = embeddings.shape[1]
            M = np.zeros((d, d))
            for s in S:
                M  += get_A(embeddings[s[0]], embeddings[s[1]])
            return M / len(S)
        
        super().__init__()

        # self.logit_scale = clip_model.logit_scale
        self.dtype = torch.float32
        self.device = device

        self.lam = 1000

        self.clip, self.preprocess = clip.load(arch_str, device=device)

        # spurious_prompt = ['A photo of a male.', 'A photo of a male celebrity.', 'A photo of a man.', 'A photo of David.',
        #                    'A photo of a female.', 'A photo of a female celebrity.', 'A photo of a woman.', 'A photo of Amy.']
        # spurious_prompt = ['A photo of a young man.', 'A photo of a young woman.', 'A photo of a young person.',
        #                    'A photo of a midddle-aged man.', 'A photo of a middle-aged woman.', 'A photo of a middle-aged person.',
        #                    'A photo of an old man.', 'A photo of an old woman.', 'A photo of an old person.']

        spurious_prompt = ['A photo of a young man.', 'A photo of a child.', 
                           'A photo of a midddle-aged man.', 'A photo of an adult.',
                           'A photo of an old man.', 'A photo of an elder.']
        

        # spurious_prompt = ['A photo of a young man.',
        #                    'A photo of a midddle-aged man.', 
        #                    'A photo of an old man.']
        

        # candidate_prompt = ['A photo of a male celebrity with dark hair.',
        #                     'A photo of a female celebrity with dark hair.',
        #                     'A photo of a male celebrity with blond hair.',
        #                     'A photo of a female celebrity with blond hair.']
# template,concept
# This is the face of a {} person,good
#  This is a photo of a {} person,kind
#               This person is {},successful
#                     A {} person,noncriminal
#                 A {} individual,nonviolent
#           This individual is {},moral
#                   Photo of a {},rich
#                              {},honest
#                    This is a {},smart
#       This is the photo of a {},evil
#                            A {},unkind
#                                ,unsuccessful
#                                ,criminal
#                                ,violent
#                                ,immoral
#                                ,poor
#                                ,dishonest
#                                ,dumb
#                                ,clever
#                                ,stupid
#                                ,hardworking
#                                ,lazy
#                                ,nasty
#                                ,trustworthy
#                                ,untrustworthy
#                                ,caring
#                                ,dangerous
#                                ,heroic
#                                ,villainous
        
        candidate_prompt = []
        S = []
        # concepts = ["good", "evil"]

        # concepts = ["good", "kind", "successful", "noncriminal", "nonviolent", "evil", "unkind", "unsuccessful", "criminal", "violent"] # the best
                    
        concepts = ["good", "kind", "successful", "noncriminal", "nonviolent", "moral", "rich", "honest", "smart", "evil", "unkind", "unsuccessful", "criminal", "violent", "immoral", "poor", "dishonest", "dumb", "clever", "stupid", "hardworking", "lazy", "nasty", "trustworthy", "caring", "dangerous", "heroic", "villainous"]
        for idx, word in enumerate(concepts):
            # candidate_prompt += [f'A photo of a young {word} person.', f'A photo of a middle-aged {word} person.']
            # candidate_prompt += [f'A photo of a middle-aged {word} person.',  f'A photo of an old {word} person.']
            # candidate_prompt += [f'A photo of a young {word} person.', f'A photo of an old {word} person.']

            # candidate_prompt += [f'A photo of a young {word} male.', f'A photo of a middle-aged {word} male.'] # the best
            # candidate_prompt += [f'A photo of a middle-aged {word} male.',  f'A photo of an old {word} male.']
            # candidate_prompt += [f'A photo of a young {word} male.', f'A photo of an old {word} male.']
            # candidate_prompt += [f'A photo of a young {word} female.', f'A photo of a middle-aged {word} female.']
            # candidate_prompt += [f'A photo of a middle-aged {word} female.',  f'A photo of an old {word} female.']
            # candidate_prompt += [f'A photo of a young {word} female.', f'A photo of an old {word} female.']

            article = 'an' if word in ["evil", "immoral", "unkind", "unsuccessful", "honest", "untrustworthy"] else 'a'
            prompts = [f'0A photo of a young {word} person.', f'1A photo of a middle-aged {word} person.', f'2A photo of an old {word} person.', f'3A photo of {article} {word} child.', f'4A photo of {article} {word} adult.', f'5A photo of {article} {word} elder.']
            for pair in list(permutations(prompts, 2)):
                candidate_prompt += [pair[0][1:], pair[1][1:]]
                S += [[15*idx + int(pair[0][0]), 15*idx + int(pair[1][0])]]
            # andidate_prompt += [list(pair) for pair in list(permutations(prompts, 2))]
            # print([list(pair) for pair in list(permutations(prompts, 2))])

            # candidate_prompt += [f'A photo of a young {word} male.', f'A photo of a middle-aged {word} male.'] # the best
            # candidate_prompt += [f'A photo of a middle-aged {word} male.',  f'A photo of an old {word} male.']
            # candidate_prompt += [f'A photo of a young {word} male.', f'A photo of an old {word} male.']
            # candidate_prompt += [f'A photo of a young {word} female.', f'A photo of a middle-aged {word} female.']
            # candidate_prompt += [f'A photo of a middle-aged {word} female.',  f'A photo of an old {word} female.']
            # candidate_prompt += [f'A photo of a young {word} female.', f'A photo of an old {word} female.']


            # article = 'an' if word in ["evil", "immoral", "unkind", "unsuccessful", "honest", "untrustworthy"] else 'a'
            # candidate_prompt += [f'A photo of {article} {word} young man.', f'A photo of {article} {word} middle-aged man.', f'A photo of {article} {word} old man.']
            # candidate_prompt += [f'A photo of {article} {word} young woman.', f'A photo of {article} {word} middle-aged woman.', f'A photo of {article} {word} old woman.']


            # S += [[15*idx + pair[0], 15*idx + pair[1]] for pair in list(permutations(range(6), 2))]
            # print(S)
            # S += [[3*idx, 3*idx + 1], [3*idx + 1, 3*idx + 2], [3*idx + 0, 3*idx + 2]]

            # S += [[6*idx, 6*idx + 1], [6*idx + 1, 6*idx + 2], [6*idx + 0, 6*idx + 2], [6*idx + 3, 6*idx + 4], [6*idx + 4, 6*idx + 5], [6*idx + 3, 6*idx + 5]]

        spurious_embeddings = get_embeddings(spurious_prompt,
                                             self.clip,
                                             normalize=True)
        # print(candidate_prompt)
        # print(S)
        spurious_embeddings = spurious_embeddings.numpy()
        P0 = get_proj_matrix(spurious_embeddings)

        # Calculate Embedding of Positive Pairs
        candidate_embeddings = get_embeddings(candidate_prompt,
                                             self.clip,
                                             normalize=True)
        candidate_embeddings = candidate_embeddings.numpy()

        # Closed Form Optimum
        print('Solve Closed Form Optimum')
        self.M = get_M(candidate_embeddings, S)
        self.G = self.lam * self.M + np.eye(self.M.shape[0])
        self.P = np.matmul(P0, np.linalg.inv(self.G))

        print(self.P)
        # x = torch.stack([feats[ind] for feats, ind in zip(x, keep_ind)], dim=0)
    def encode_text(self, text):
        with torch.no_grad():
            text_embeddings = self.clip.encode_text(text) # from shape (bs, 1, 77) to (bs, 77)

        text_embeddings = np.matmul(text_embeddings.cpu(), self.P.T)
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        text_embeddings = torch.tensor(text_embeddings).float().to(self.device)
        return text_embeddings

    def encode_image(self, image):
        # aligned_image = self.modality_adapter(image)
        with torch.no_grad():
            image_embeddings = self.clip.encode_image(image)
    
        return image_embeddings
    