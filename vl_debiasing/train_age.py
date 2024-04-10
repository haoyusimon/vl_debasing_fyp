import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from clip_debiasing.datasets import FairFaceDebiasing_Age
import clip
from clip_debiasing.models.model_age import DebiasedCLIP
import utils
import eval_train
import argparse
import os
import sys
import shutil

# def redirect_to_file(text):
#     original = sys.stdout
#     sys.stdout = open('/path/to/redirect.txt', 'w')
#     print('This is your redirected text:')
#     print(text)
#     sys.stdout = original

#     print('This string goes to stdout, NOT the file!')


def train(model, data_loader, optimizer, epoch=None, warmup_steps=None, device=None, scheduler=None, config=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 10
    # with torch.autograd.set_detect_anomaly(True):
    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        optimizer.zero_grad()
        image = batch["img"].to(device)
        text1, text2, text3 = torch.squeeze(batch["text1"].to(device)), torch.squeeze(batch["text2"].to(device)), torch.squeeze(batch["text3"].to(device))
        # word1, word2 = torch.squeeze(batch["word1"].to(device)), torch.squeeze(batch["word2"].to(device))
        loss_ita, text_diff_loss = model(image, text1, text2, text3, epoch)

        # combine losses
        loss = loss_ita * 0.5 + text_diff_loss * 0.5
        # loss = loss_ita
        # loss = loss_ita * 0.7 + text_sim_loss * 0.01 + text_diff_loss * 0.29
        # loss = loss_ita + text_sim_loss / ((text_sim_loss + 0.1) / loss_ita).detach() + text_diff_loss / ((text_diff_loss + 0.1) / loss_ita).detach()

        loss.backward()

        # for name, param in list(model.modality_adapter.named_parameters()) + list(model.debiaser.named_parameters()):
        #     if param.grad is not None and torch.isnan(param.grad).any():
        #         print("nan gradient found")
        #         print("name: ", name)
        #         print("param:", param.grad)
        #         raise SystemExit
            
        # torch.nn.utils.clip_grad_norm_(list(model.modality_adapter.parameters()) + list(model.debiaser.parameters()), max_norm=0.1)

        optimizer.step()

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])   
        sys.stdout.flush() # timely update the logs
        sys.stderr.flush()

def main(args=None, config=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="latest", type=str)
    args = parser.parse_args()
    # args.version

    # create version directory and copy model file 
    work_dir = os.path.join('.',f'exp/{args.version}')
    os.makedirs(work_dir)

    shutil.copy("./clip_debiasing/models/model_age.py", os.path.join(work_dir, 'model_details.py'))
    shutil.copy("./train_age.py", os.path.join(work_dir, 'train_details.py'))

    sys.stdout = open(os.path.join(work_dir, 'stdout.log'), 'w')
    sys.stderr = open(os.path.join(work_dir, 'stderr.log'), 'w')

    print(f"Experiment version: {args.version}")
            
    device = torch.device('cuda')
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    epochs = 150
    model = DebiasedCLIP("ViT-B/16", device=device).to(device)
    dataset = FairFaceDebiasing_Age(tokenizer=clip.tokenize, transforms=model.preprocess)
    data_loader = DataLoader(dataset, batch_size=512, num_workers=4)
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    optimizer = torch.optim.Adam(list(model.modality_adapter.parameters()) + list(model.debiaser.parameters()), 
                                 lr=1e-6)
    # optimizer = torch.optim.Adam(model.debiaser.parameters(), lr=1e-5)


    for epoch in range(epochs):
        train(model, data_loader, optimizer, device=device, epoch=epoch)

        if epoch % 2 == 0 or (epoch + 1) % 10 == 0:
            print(f"Evaluation for Epoch {epoch}...")
            eval_train.run_eval_train(model, model.preprocess, attribute='age')
            sys.stdout.flush()  
            sys.stderr.flush()

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(work_dir, f"debiased_CLIP_weights_{args.version}_{epoch + 1}epochs.pth"))
            print("Checkpoint saved.")

if __name__ == '__main__':
    main()
