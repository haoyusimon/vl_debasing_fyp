from abc import ABC
from typing import Tuple, Callable, Any, Union, List, Dict

import clip
import torch
import torch.nn as nn
from torch.nn import Module

from clip_debiasing import Dotdict
from torch.utils.data import DataLoader

import torch.nn.functional as F

import scipy.spatial as ss
from scipy.special import digamma
from math import log
import numpy.random as nr
import numpy as np
from clip_debiasing.datasets import IATDataset, FairFace, FACET

from collections import defaultdict
from tqdm import tqdm

def Mixed_KSG(x,y,k=5):
    assert len(x)==len(y), "Lists should have same length"
    assert k <= len(x)-1, "Set k smaller than num. samples - 1"
    N = len(x)
    if x.ndim == 1:
        x = x.reshape((N,1))
    dx = len(x[0])   	
    if y.ndim == 1:
        y = y.reshape((N,1))
    dy = len(y[0])
    data = np.concatenate((x,y),axis=1)

    tree_xy = ss.cKDTree(data)
    tree_x = ss.cKDTree(x)
    tree_y = ss.cKDTree(y)

    knn_dis = [tree_xy.query(point,k+1,p=float('inf'))[0][k] for point in data]
    ans = 0

    for i in range(N):
        kp, nx, ny = k, k, k
        if knn_dis[i] == 0:
            kp = len(tree_xy.query_ball_point(data[i],1e-15,p=float('inf')))
            nx = len(tree_x.query_ball_point(x[i],1e-15,p=float('inf')))
            ny = len(tree_y.query_ball_point(y[i],1e-15,p=float('inf')))
        else:
            nx = len(tree_x.query_ball_point(x[i],knn_dis[i]-1e-15,p=float('inf')))
            ny = len(tree_y.query_ball_point(y[i],knn_dis[i]-1e-15,p=float('inf')))
        ans += (digamma(kp) + log(N) - digamma(nx) - digamma(ny))/N
    return ans

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


class CLIP_clipped(nn.Module):

    def __init__(self, arch_str, device, hidden_dim, m, debiasing_modules=True, attribute='gender', **_kwargs,):
        super().__init__()

        # self.logit_scale = clip_model.logit_scale
        self.dtype = torch.float32

        self.clip, self.preprocess = clip.load(arch_str, device=device)

        # for gender, m=256
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

        # for age, m=256
        # self.keep_ind = torch.tensor([  2,   4,   5,   6,   8,   9,  10,  11,  12,  16,  17,  18,  19,  21,
        #  24,  25,  26,  27,  28,  29,  30,  31,  33,  37,  39,  40,  41,  43,
        #  47,  48,  49,  51,  53,  54,  59,  61,  62,  64,  65,  66,  68,  69,
        #  72,  73,  76,  80,  84,  85,  88,  89,  90,  91,  96,  98, 107, 109,
        # 111, 114, 118, 120, 125, 127, 134, 135, 139, 142, 143, 144, 147, 148,
        # 151, 152, 153, 155, 156, 158, 160, 164, 165, 166, 169, 172, 173, 174,
        # 175, 176, 177, 179, 180, 181, 182, 185, 186, 188, 189, 191, 193, 194,
        # 196, 200, 203, 204, 206, 207, 208, 209, 211, 214, 215, 216, 219, 220,
        # 221, 223, 229, 230, 233, 234, 235, 239, 241, 242, 245, 246, 248, 249,
        # 252, 256, 260, 263, 264, 271, 272, 276, 279, 285, 286, 291, 294, 295,
        # 298, 299, 300, 302, 303, 305, 306, 310, 313, 316, 317, 318, 319, 321,
        # 322, 327, 328, 329, 330, 331, 336, 337, 340, 343, 344, 345, 346, 349,
        # 350, 351, 352, 354, 356, 357, 358, 359, 361, 362, 363, 364, 367, 368,
        # 369, 370, 372, 378, 379, 380, 386, 388, 391, 392, 393, 394, 395, 397,
        # 398, 402, 404, 405, 408, 410, 412, 413, 415, 418, 420, 421, 424, 425,
        # 427, 430, 431, 435, 436, 437, 438, 440, 441, 442, 443, 445, 446, 449,
        # 451, 453, 454, 458, 459, 460, 463, 466, 467, 469, 474, 476, 478, 479,
        # 480, 484, 485, 486, 487, 488, 491, 493, 494, 496, 497, 499, 501, 502,
        # 503, 504, 507, 508])
        self.keep_ind = None
        
        if self.keep_ind == None:
            ds = FairFace(iat_type=attribute, mode="train_val", equal_split=False, transforms=self.preprocess)
            dl = DataLoader(ds, batch_size=10, num_workers=6)
            x = defaultdict(lambda : [])
            y = []
            for batch in tqdm(dl):
                # print(batch['img'])
                with torch.no_grad():
                    image_embeddings = self.clip.encode_image(batch['img'].to('cuda'))
                for dim in range(hidden_dim):
                    x[dim] += image_embeddings[:, dim].tolist()
                y += batch["iat_label"].tolist()
                # break
                # print(Mixed_KSG(np.array(x[0]), np.array(y)))
                # print(Mixed_KSG(np.array(x[1]), np.array(y)))
                # print(x)
                # print(y)
            scores = []
            for dim in range(hidden_dim): # change for ViT-L/14 which has hidden size of 768
                scores.append(Mixed_KSG(np.array(x[dim]), np.array(y)))
            scores = torch.FloatTensor(scores)
            remaining_inds = torch.topk(scores, m, largest=False).indices
            remaining_inds = remaining_inds.sort().values
            self.keep_ind = remaining_inds
        print(self.keep_ind)

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