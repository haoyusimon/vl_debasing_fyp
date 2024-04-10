import torch
import clip
from .clip_models import DebiasedCLIP
from .clip_models import CLIP_prompt
from .clip_models import CLIP_clipped
from .clip_models import CLIP_prompt_age

def load_model(version="ViT-B/16"):
    ''' Load BERT model and corresponding tokenizer '''
    # version="RN50"
    # version="ViT-L/14"
    version="ViT-B/16"
    '''
    ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
    '''
    # model, _ = clip.load(version, device='cpu')

    # model = DebiasedCLIP("ViT-B/16", device='cpu')
    # model.load_state_dict(torch.load("../vl_debiasing/exp/fyp_19/debiased_CLIP_weights_fyp_19_10epochs.pth"))

    # model = CLIP_clipped("ViT-B/16", device='cpu', hidden_dim=512, m=256)

    model = CLIP_prompt_age("ViT-B/16", device='cpu')

    return model


def encode(model, texts, debiaser=None):
    ''' Use tokenizer and model to encode texts '''
    encs = {}
    if debiaser == None:
        for text in texts:
            tokenized = clip.tokenize(text, truncate = True)
            # indexed = tokenizer.convert_tokens_to_ids(tokenized)
            # segment_idxs = [0] * len(tokenized)
            # tokens_tensor = torch.tensor([indexed])
            # segments_tensor = torch.tensor([segment_idxs])
            # enc, _ = model(tokens_tensor, segments_tensor, output_all_encoded_layers=False)

            # enc = enc[:, 0, :]  # extract the last rep of the first input
            enc = model.encode_text(tokenized)
            encs[text] = enc.detach().view(-1).numpy()
    else:
        for text in texts:
            tokenized = clip.tokenize(text, truncate = True)
            enc = model.encode_text(tokenized) + debiaser(model.encode_text(tokenized))
            encs[text] = enc.detach().view(-1).numpy() 
    return encs
