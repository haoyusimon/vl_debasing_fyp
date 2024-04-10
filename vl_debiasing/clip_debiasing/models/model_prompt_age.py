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
from sklearn.decomposition import TruncatedSVD

from collections import defaultdict
from tqdm import tqdm
from itertools import permutations

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
    