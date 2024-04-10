import clip_debiasing
from clip_debiasing.models.model_contrastive_queue_pareto import DebiasedCLIP
import torch
import clip
import numpy as np
from tqdm import tqdm
from pkg_resources import packaging
import torch.nn.functional as F
import torch.nn as nn
import debias_clip
import utils
import time
import datetime
from clip_debiasing.models.model_clipped import CLIP_clipped
from clip_debiasing.models.model_prompt import CLIP_prompt
from clip_debiasing.models.model_prompt_age import CLIP_prompt_age

from torch.utils.data import DataLoader
from torchvision import transforms
from clip_debiasing.datasets import re_eval_dataset

if __name__ == '__main__':
    device = torch.device('cuda')

                                    
    def create_dataset(preprocess):        
        test_dataset = re_eval_dataset('/temp/haoyu/json_downstream/retrieval/flickr30k_test.json', preprocess, '/temp/haoyu/common-images/flickr30k/')                
        return test_dataset   
    
    def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
        loaders = []
        for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
            if is_train:
                shuffle = (sampler is None)
                drop_last = True
            else:
                shuffle = False
                drop_last = False
            loader = DataLoader(
                dataset,
                batch_size=bs,
                num_workers=n_worker,
                pin_memory=True,
                sampler=sampler,
                shuffle=shuffle,
                collate_fn=collate_fn,
                drop_last=drop_last,
            )              
            loaders.append(loader)
        return loaders  
      
    @torch.no_grad()
    def evaluation(model, data_loader, tokenizer, device):
        # test
        model.eval() 
        
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Evaluation:'    
        
        print('Computing features for evaluation...')
        start_time = time.time()  

        texts = data_loader.dataset.text   
        num_text = len(texts)
        text_bs = 256

        text_embeds = []  
        for i in range(0, num_text, text_bs):
            text = texts[i: min(num_text, i+text_bs)]
            text_input = tokenizer(text).to(device) 
            text_output = model.encode_text(text_input)  
            text_embed = F.normalize(text_output, dim=-1)
            text_embeds.append(text_embed)   
        text_embeds = torch.cat(text_embeds,dim=0)
        
        image_embeds = []
        for image, img_id in data_loader: 
            image = image.to(device) 
            image_embed = model.encode_image(image)         
            image_embed = F.normalize(image_embed,dim=-1)      
            image_embeds.append(image_embed)
        image_embeds = torch.cat(image_embeds,dim=0)
        
        sims_matrix = image_embeds.float() @ text_embeds.t().float() # bs, bs
        score_matrix_i2t = torch.full((len(data_loader.dataset.image),len(texts)),-100.0).to(device)
        
        num_tasks = utils.get_world_size()
        rank = utils.get_rank() 
        step = sims_matrix.size(0)//num_tasks + 1
        start = rank*step
        end = min(sims_matrix.size(0),start+step)

        for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)): 
            topk_sim, topk_idx = sims.topk(k=128, dim=0)

            score_matrix_i2t[start+i,topk_idx] = topk_sim.float()
            
        sims_matrix = sims_matrix.t()
        score_matrix_t2i = torch.full((len(texts),len(data_loader.dataset.image)),-100.0).to(device)
        
        step = sims_matrix.size(0)//num_tasks + 1
        start = rank*step
        end = min(sims_matrix.size(0),start+step)    
        
        for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)): 
            
            topk_sim, topk_idx = sims.topk(k=128, dim=0)

            score_matrix_t2i[start+i,topk_idx] = topk_sim.float()


        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Evaluation time {}'.format(total_time_str)) 

        return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()

    @torch.no_grad()
    def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
        
        #Images->Text 
        ranks = np.zeros(scores_i2t.shape[0])
        for index,score in enumerate(scores_i2t):
            inds = np.argsort(score)[::-1]
            # Score
            rank = 1e20
            for i in img2txt[index]:
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
            ranks[index] = rank

        # Compute metrics
        tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    
        #Text->Images 
        ranks = np.zeros(scores_t2i.shape[0])
        
        for index,score in enumerate(scores_t2i):
            inds = np.argsort(score)[::-1]
            ranks[index] = np.where(inds == txt2img[index])[0][0]

        # Compute metrics
        ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)        

        tr_mean = (tr1 + tr5 + tr10) / 3
        ir_mean = (ir1 + ir5 + ir10) / 3
        r_mean = (tr_mean + ir_mean) / 2

        eval_result =  {'txt_r1': tr1,
                        'txt_r5': tr5,
                        'txt_r10': tr10,
                        'txt_r_mean': tr_mean,
                        'img_r1': ir1,
                        'img_r5': ir5,
                        'img_r10': ir10,
                        'img_r_mean': ir_mean,
                        'r_mean': r_mean}
        return eval_result

    ### Original CLIP
    # model, preprocess = clip.load("ViT-B/16", device=device)
    # input_resolution = model.visual.input_resolution
    # context_length = model.context_length
    # vocab_size = model.vocab_size

    ### Our method
    # model = DebiasedCLIP("ViT-B/16", device=device)
    # preprocess = model.preprocess
    # model.load_state_dict(torch.load("./exp/fyp_15/debiased_CLIP_weights_fyp_15_70epochs.pth"))
    # # model.load_state_dict(torch.load("./exp/v6.17/debiased_CLIP_weights_v6.17_70epochs.pth"))

    # input_resolution = model.clip.visual.input_resolution
    # context_length = model.clip.context_length
    # vocab_size = model.clip.vocab_size

    ### prompt-array debiased CLIP:
    # model, preprocess = debias_clip.load("ViT-B/16-gender", device=device)
    # input_resolution = model.clip.visual.input_resolution
    # context_length = model.clip.context_length
    # vocab_size = model.clip.vocab_size

    # ViT-B/16, ViT-B/32, RN50, ViT-L/14 
    model = CLIP_clipped("ViT-B/16", device='cuda', hidden_dim=512, m=490, attribute='age')
    preprocess = model.preprocess


    # ViT-B/16, ViT-B/32, RN50, ViT-L/14 
    # model = CLIP_prompt("ViT-B/16", device='cuda')
    # preprocess = model.preprocess

    test_dataset = create_dataset(preprocess=preprocess)  
    test_loader = create_loader([test_dataset], batch_size=[8], num_workers=[4], is_trains=[False], collate_fns=[None], samplers=[None])[0]

    score_test_i2t, score_test_t2i = evaluation(model, test_loader, clip.tokenize, device)

    
    # torch.save(model.debiaser.state_dict(), "debiaser.pth")
    # torch.save(model.modality_adapter.state_dict(), "adapter.pth")
    test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt)    
    print(test_result)

