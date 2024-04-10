import os
import subprocess
from abc import ABC
from typing import Callable, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from clip_debiasing import Dotdict, FAIRFACE_DATA_PATH, FACET_DATA_PATH
import json
import re

class IATDataset(Dataset, ABC):
    GENDER_ENCODING = {"Female": 1, "Male": 0}
    AGE_ENCODING = {"0-2": 0, "3-9": 0, "10-19": 0, "20-29": 0, "30-39": 1,
                    "40-49": 1, "50-59": 1, "60-69": 2, "more than 70": 2}

    def __init__(self):
        self.image_embeddings: torch.Tensor = None
        self.iat_labels: np.ndarray = None
        self._img_fnames = None
        self._transforms = None
        self.use_cache = None
        self.iat_type = None
        self.n_iat_classes = None

    def gen_labels(self, iat_type: str, label_encoding: object = None):
        # WARNING: iat_type == "pairwise_adjective" is no longer supported
        if iat_type in ("gender_science", "test_weat", "gender"):
            labels_list = self.labels["gender"]
            label_encoding = IATDataset.GENDER_ENCODING if label_encoding is None else label_encoding
        elif iat_type == "race":
            labels_list = self.labels["race"]
            label_encoding = self.RACE_ENCODING if label_encoding is None else label_encoding
        elif iat_type == "age":
            labels_list = self.labels["age"]
            label_encoding = IATDataset.AGE_ENCODING if label_encoding is None else label_encoding
        else:
            raise NotImplementedError
        # assert set(labels_list.unique()) == set(label_encoding.keys()), "There is a missing label, invalid for WEAT"
        labels_list = np.array(labels_list.apply(lambda x: label_encoding[x]), dtype=int)
        # assert labels_list.sum() != 0 and (1 - labels_list).sum() != 0, "Labels are all equal, invalid for Weat"
        return labels_list, len(label_encoding)


class FairFace(IATDataset):
    RACE_ENCODING = {"White": 0, "Southeast Asian": 1, "Middle Eastern": 2,
                     "Black": 3, "Indian": 4, "Latino_Hispanic": 5, "East Asian": 6}

    def __init__(self, iat_type: str = None, lazy: bool = True, mode: str = "none",
                 _n_samples: Union[float, int] = None, transforms: Callable = None, equal_split: bool = True, ):
        self.DATA_PATH = str(FAIRFACE_DATA_PATH)
        self.mode = mode
        self._transforms = (lambda x: x) if transforms is None else transforms
        partition = mode.split("_")[0]
        self.labels = pd.read_csv(os.path.join(self.DATA_PATH, "labels", partition, f"{partition}_labels.csv"))
        self.labels.sort_values("file", inplace=True)

        if mode == "train_val":
            val_labels = self.labels.sample(frac=0.125, random_state=1)
            self.labels = val_labels
        elif mode == "train_train":
            val_labels = self.labels.sample(frac=0.125, random_state=1)
            # print(val_labels.index)
            self.labels = self.labels.iloc[self.labels.index.difference(val_labels.index)]
            # print(self.labels.index)

        if _n_samples is not None:
            if isinstance(_n_samples, float):
                _n_samples = int(len(self.labels) * _n_samples)

            self.labels = self.labels[:_n_samples]
        if equal_split:
            labels_male = self.labels.loc[self.labels['gender'] == 'Male']
            labels_female = self.labels.loc[self.labels['gender'] == 'Female']

            num_females = labels_female.count()[0]
            num_males = labels_male.count()[0]

            sample_num = min(num_males, num_females)

            labels_male = labels_male.sample(n=sample_num, random_state=1)
            labels_female = labels_female.sample(n=sample_num, random_state=1)

            self.labels = labels_male.append(labels_female, ignore_index=True)
        


        self._img_fnames = [os.path.join(self.DATA_PATH, "imgs", "train_val", x) for x in self.labels["file"]]
        self._fname_to_inx = {fname: inx for inx, fname in enumerate(self._img_fnames)}

        self.images_list = None
        if not lazy:
            self.images_list = list(map(self.__getitem__, range(len(self.labels))))
        
        self.iat_labels = self.gen_labels(iat_type=iat_type)[0]

    def _load_fairface_sample(self, sample_labels) -> dict:
        res = Dotdict(dict(sample_labels))
        img_fname = os.path.join(self.DATA_PATH, "imgs", "train_val", res.file)
        res.img = self._transforms(Image.open(img_fname))
        return res

    def __getitem__(self, index: int):
        if self.images_list is not None:
            return self.images_list[index]

        ff_sample = self._load_fairface_sample(self.labels.iloc[index])
        ff_sample.iat_label = self.iat_labels[index]
        return ff_sample

    def __len__(self):
        return len(self.labels)


class AugmentedDataset(Dataset, ABC):
    def __init__(self):
        self.image_embeddings: torch.Tensor = None
        self.text_embeddings: np.ndarray = None
        self._img_fnames = None
        self._transforms = None
        self.use_cache = None
        self.iat_type = None
        self.n_iat_classes = None
        self._tokenizer = None


class FairFaceDebiasing(AugmentedDataset):
    # RACE_ENCODING = {"White": 0, "Southeast Asian": 1, "Middle Eastern": 2,
    #                  "Black": 3, "Indian": 4, "Latino_Hispanic": 5, "East Asian": 6}

    def __init__(self, iat_type: str = None, lazy: bool = True, mode: str = "train_train",
                 _n_samples: Union[float, int] = None, transforms: Callable = None, equal_split: bool = True, tokenizer: Callable = None,):
        print("Mode:", mode)
        self.DATA_PATH = str(FAIRFACE_DATA_PATH)
        self.mode = mode
        self._transforms = (lambda x: x) if transforms is None else transforms
        
        partition = mode.split("_")[0]
        self.labels = pd.read_csv(os.path.join(self.DATA_PATH, "labels", partition, f"{partition}_labels.csv"))
        # self.labels = pd.read_csv(os.path.join(self.DATA_PATH, "labels", mode, f"{mode}_labels.csv"))
        self.labels.sort_values("file", inplace=True)
        
        if mode == "train_val":
            val_labels = self.labels.sample(frac=0.125, random_state=1)
            self.labels = val_labels
        elif mode == "train_train":
            val_labels = self.labels.sample(frac=0.125, random_state=1)
            # print(val_labels.index)
            self.labels = self.labels.iloc[self.labels.index.difference(val_labels.index)]
            # print(self.labels.index)

        if _n_samples is not None:
            if isinstance(_n_samples, float):
                _n_samples = int(len(self.labels) * _n_samples)

            self.labels = self.labels[:_n_samples]
        if equal_split:
            labels_male = self.labels.loc[self.labels['gender'] == 'Male']
            labels_female = self.labels.loc[self.labels['gender'] == 'Female']

            num_females = labels_female.count()[0]
            num_males = labels_male.count()[0]

            sample_num = min(num_males, num_females)

            labels_male = labels_male.sample(n=sample_num, random_state=1)
            labels_female = labels_female.sample(n=sample_num, random_state=1)

            self.labels = labels_male.append(labels_female, ignore_index=True)

        self._img_fnames = [os.path.join(self.DATA_PATH, "imgs", "train_val", x) for x in self.labels["file"]]
        self._fname_to_inx = {fname: inx for inx, fname in enumerate(self._img_fnames)}

        self.images_list = None
        if not lazy:
            self.images_list = list(map(self.__getitem__, range(len(self.labels))))

        self._tokenizer = tokenizer

    def _load_fairface_sample(self, sample_labels) -> dict:
        opposite_dict = {"Male": "Female", "Female": "Male"}
        res = Dotdict(dict(sample_labels))
        img_fname = os.path.join(self.DATA_PATH, "imgs", "train_val", res.file)
        res.img = self._transforms(Image.open(img_fname))
        # text1 = f"This is the photo of a {res.gender.lower()} person with the race {res.race} and the age range {res.age}."
        # text2 = f"This is the photo of a {opposite_dict[res.gender].lower()} person with the race {res.race} and the age range {res.age}."
        text1 = f"This is the photo of a {res.race} {res.gender.lower()} aged {res.age}."
        text2 = f"This is the photo of a {res.race} {opposite_dict[res.gender].lower()} aged {res.age}."
        res.text1 = self._tokenizer(text1)
        res.text2 = self._tokenizer(text2)
        res.word1 = self._tokenizer(f"The concept of {res.gender.lower()}.")
        res.word2 = self._tokenizer(f"The concept of {opposite_dict[res.gender].lower()}.")
        return res

    def __getitem__(self, index: int):
        if self.images_list is not None:
            return self.images_list[index]

        ff_sample = self._load_fairface_sample(self.labels.iloc[index])
        # ff_sample.iat_label = self.iat_labels[index]
        return ff_sample

    def __len__(self):
        return len(self.labels)

class FairFaceDebiasing_Age(AugmentedDataset):
    # RACE_ENCODING = {"White": 0, "Southeast Asian": 1, "Middle Eastern": 2,
    #                  "Black": 3, "Indian": 4, "Latino_Hispanic": 5, "East Asian": 6}

    def __init__(self, iat_type: str = None, lazy: bool = True, mode: str = "train_train",
                 _n_samples: Union[float, int] = None, transforms: Callable = None, equal_split: bool = True, tokenizer: Callable = None,):
        print("Mode:", mode)
        self.DATA_PATH = str(FAIRFACE_DATA_PATH)
        self.mode = mode
        self._transforms = (lambda x: x) if transforms is None else transforms
        
        partition = mode.split("_")[0]
        self.labels = pd.read_csv(os.path.join(self.DATA_PATH, "labels", partition, f"{partition}_labels.csv"))
        # self.labels = pd.read_csv(os.path.join(self.DATA_PATH, "labels", mode, f"{mode}_labels.csv"))
        self.labels.sort_values("file", inplace=True)
        
        if mode == "train_val":
            val_labels = self.labels.sample(frac=0.125, random_state=1)
            self.labels = val_labels
        elif mode == "train_train":
            val_labels = self.labels.sample(frac=0.125, random_state=1)
            # print(val_labels.index)
            self.labels = self.labels.iloc[self.labels.index.difference(val_labels.index)]
            # print(self.labels.index)

        if _n_samples is not None:
            if isinstance(_n_samples, float):
                _n_samples = int(len(self.labels) * _n_samples)

            self.labels = self.labels[:_n_samples]
        if equal_split:
            labels_male = self.labels.loc[self.labels['gender'] == 'Male']
            labels_female = self.labels.loc[self.labels['gender'] == 'Female']

            num_females = labels_female.count()[0]
            num_males = labels_male.count()[0]

            sample_num = min(num_males, num_females)

            labels_male = labels_male.sample(n=sample_num, random_state=1)
            labels_female = labels_female.sample(n=sample_num, random_state=1)

            self.labels = labels_male.append(labels_female, ignore_index=True)

        self._img_fnames = [os.path.join(self.DATA_PATH, "imgs", "train_val", x) for x in self.labels["file"]]
        self._fname_to_inx = {fname: inx for inx, fname in enumerate(self._img_fnames)}

        self.images_list = None
        if not lazy:
            self.images_list = list(map(self.__getitem__, range(len(self.labels))))

        self._tokenizer = tokenizer

    def _load_fairface_sample(self, sample_labels) -> dict:
        age_conversion = {"0-2": "young", "3-9": "young", "10-19": "young", "20-29": "young", "30-39": "middle-aged",
                "40-49": "middle-aged", "50-59": "middle-aged", "60-69": "old", "more than 70": "old"}
        opposite_dict = {"young": ["middle-aged", "old"], "middle-aged": ["young", "old"], "old": ["young", "middle-aged"]}
        res = Dotdict(dict(sample_labels))
        img_fname = os.path.join(self.DATA_PATH, "imgs", "train_val", res.file)
        res.img = self._transforms(Image.open(img_fname))
        # text1 = f"This is the photo of a {res.gender.lower()} person with the race {res.race} and the age range {res.age}."
        # text2 = f"This is the photo of a {opposite_dict[res.gender].lower()} person with the race {res.race} and the age range {res.age}."
        age = age_conversion[res.age]
        text1 = f"This is the photo of a {age} {res.race} {res.gender.lower()}."
        text2 = f"This is the photo of a {opposite_dict[age][0]} {res.race} {res.gender.lower()}."
        text3 = f"This is the photo of a {opposite_dict[age][1]} {res.race} {res.gender.lower()}."
        res.text1 = self._tokenizer(text1)
        res.text2 = self._tokenizer(text2)
        res.text3 = self._tokenizer(text3)
        # res.word1 = self._tokenizer(f"The concept of {res.gender.lower()}.")
        # res.word2 = self._tokenizer(f"The concept of {opposite_dict[res.gender].lower()}.")
        return res

    def __getitem__(self, index: int):
        if self.images_list is not None:
            return self.images_list[index]

        ff_sample = self._load_fairface_sample(self.labels.iloc[index])
        # ff_sample.iat_label = self.iat_labels[index]
        return ff_sample

    def __len__(self):
        return len(self.labels)

# class IATDataset_FACET(Dataset, ABC):
#     GENDER_ENCODING = {"Female": 1, "Male": 0}
#     AGE_ENCODING = {"0-2": 0, "3-9": 1, "10-19": 2, "20-29": 3, "30-39": 4,
#                     "40-49": 5, "50-59": 6, "60-69": 7, "more than 70": 8}

#     def __init__(self):
#         self.image_embeddings: torch.Tensor = None
#         self.iat_labels: np.ndarray = None
#         self._img_fnames = None
#         self._transforms = None
#         self.use_cache = None
#         self.iat_type = None
#         self.n_iat_classes = None

#     def gen_labels(self, iat_type: str, label_encoding: object = None):
#         # WARNING: iat_type == "pairwise_adjective" is no longer supported
#         if iat_type in ("gender_science", "test_weat", "gender"):
#             labels_list = self.labels["gender"]
#             label_encoding = IATDataset_FACET.GENDER_ENCODING if label_encoding is None else label_encoding
#         elif iat_type == "race":
#             labels_list = self.labels["race"]
#             label_encoding = self.RACE_ENCODING if label_encoding is None else label_encoding
#         elif iat_type == "age":
#             labels_list = self.labels["age"]
#             label_encoding = IATDataset_FACET.AGE_ENCODING if label_encoding is None else label_encoding
#         else:
#             raise NotImplementedError
#         assert set(labels_list.unique()) == set(label_encoding.keys()), "There is a missing label, invalid for WEAT"
#         labels_list = np.array(labels_list.apply(lambda x: label_encoding[x]), dtype=int)

#         # assert labels_list.sum() != 0 and (1 - labels_list).sum() != 0, "Labels are all equal, invalid for Weat"
#         return labels_list, len(label_encoding)

class FACET(IATDataset):
    RACE_ENCODING = {"White": 0, "Southeast Asian": 1, "Middle Eastern": 2,
                     "Black": 3, "Indian": 4, "Latino_Hispanic": 5, "East Asian": 6}

    def __init__(self, iat_type: str = None, lazy: bool = True, mode: str = "test",
                 _n_samples: Union[float, int] = None, transforms: Callable = None, equal_split: bool = True, ):
        self.DATA_PATH = str(FACET_DATA_PATH)
        self.mode = mode
        self._transforms = (lambda x: x) if transforms is None else transforms
        self.labels = pd.read_csv(os.path.join(self.DATA_PATH, "annotations", "annotations.csv"))
        # if iat_type == 'gender':
        self.labels['gender'] = self.labels.apply(self._label_gender, axis=1)
        # if iat_type == 'age':
        self.labels['age'] = self.labels.apply(self._label_age, axis=1)
        labels_young = self.labels.loc[self.labels['age'] == '3-9']
        labels_middle = self.labels.loc[self.labels['age'] == '30-39']
        labels_old = self.labels.loc[self.labels['age'] == '60-69']
        self.labels = labels_young.append(labels_middle, ignore_index=True)
        self.labels = self.labels.append(labels_old, ignore_index=True)

        self.labels.sort_values("filename", inplace=True)


        if mode == "test":
            test_labels = self.labels.sample(frac=0.1, random_state=1)
            self.labels = test_labels
        elif mode == "val":
            test_labels = self.labels.sample(frac=0.1, random_state=1)
            # print(val_labels.index)
            train_labels = self.labels.iloc[self.labels.index.difference(test_labels.index)]
            train_val_labels = train_labels.sample(frac=0.125, random_state=1)
            self.labels = train_val_labels
        else:
            test_labels = self.labels.sample(frac=0.1, random_state=1)
            # print(val_labels.index)
            train_labels = self.labels.iloc[self.labels.index.difference(test_labels.index)]
            train_val_labels = train_labels.sample(frac=0.125, random_state=1)

            train_train_labels = train_labels.iloc[train_labels.index.difference(train_val_labels.index)] 
            self.labels = train_train_labels
            # print(self.labels.index)


        if _n_samples is not None:
            if isinstance(_n_samples, float):
                _n_samples = int(len(self.labels) * _n_samples)

            self.labels = self.labels[:_n_samples]
        if equal_split:
            labels_male = self.labels.loc[self.labels['gender'] == 'Male']
            labels_female = self.labels.loc[self.labels['gender'] == 'Female']

            num_females = labels_female.count()[0]
            num_males = labels_male.count()[0]

            sample_num = min(num_males, num_females)

            labels_male = labels_male.sample(n=sample_num, random_state=1)
            labels_female = labels_female.sample(n=sample_num, random_state=1)

            self.labels = labels_male.append(labels_female, ignore_index=True)

        # self._img_fnames = [os.path.join(self.DATA_PATH, "imgs", "train_val", x) for x in self.labels["file"]]
        # self._fname_to_inx = {fname: inx for inx, fname in enumerate(self._img_fnames)}

        self.images_list = None
        if not lazy:
            self.images_list = list(map(self.__getitem__, range(len(self.labels))))
        
        self.iat_labels = self.gen_labels(iat_type=iat_type)[0]

    def _label_gender(self, row):
        if row['gender_presentation_masc'] == 1:
            return 'Male'
        elif row['gender_presentation_fem'] == 1:
            return 'Female'
        else:
            return 'Non-binary'

    def _label_age(self, row):
        if row['age_presentation_young'] == 1:
            return '3-9'
        elif row['age_presentation_middle'] == 1:
            return '30-39'
        elif row['age_presentation_older'] == 1:
            return '60-69'
        else:
            return 'No-age'
        
    def _load_facet_sample(self, sample_labels) -> dict:
        # res = Dotdict(dict(sample_labels))
        res = Dotdict({'filename': sample_labels['filename']})

        img_fname = self._search_dir(res.filename)

        # crop image to the region containing the specific person
        # print(img_fname)
        assert img_fname != None
        bbox = json.loads(sample_labels['bounding_box'])
        left = int(bbox["x"])
        top = int(bbox["y"])
        right = int(bbox["x"] + bbox["width"])
        bottom = int(bbox["y"] + bbox["height"])
        img = Image.open(img_fname)
        img_cropped = img.crop((left, top, right, bottom))

        res.img = self._transforms(img_cropped)
        # print(res.iat_label)
        return res

    def __getitem__(self, index: int):
        if self.images_list is not None:
            return self.images_list[index]

        ff_sample = self._load_facet_sample(self.labels.iloc[index])
        ff_sample.iat_label = self.iat_labels[index]
        # print(ff_sample.img)
        # print(ff_sample.iat_label)
        return ff_sample
    
    def _search_dir(self, img_name):
        possible_dirs = []
        for idx in range(1, 4):
            possible_dirs.append(os.path.join(self.DATA_PATH, f"imgs_{idx}", f"{img_name}"))

        for possible_dir in possible_dirs:
            # print(possible_dir)
            if os.path.isfile(possible_dir):
                return possible_dir
        return None
    
    def __len__(self):
        return len(self.labels)

def pre_caption(caption,max_words):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption

class re_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        self.ann = json.load(open(ann_file,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words 
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption,self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, index):    
        
        image_path = os.path.join(self.image_root, self.ann[index]['image'])        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  

        return image, index