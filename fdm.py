import torch
from torch import nn, optim
from torch.nn import functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import textwrap
from PIL import Image
import models.sentence_transformers_helper as st_helper
from transformers import AutoTokenizer, AutoModel
import json
import random
from level_dataset import visualize_samples



class imageDataSet(Dataset):

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self,idx):
        return self.data[0][idx], self.data[1][idx], self.data[2][idx]

class ResBlock(nn.Module):
    def __init__(self, kern_size=7, filter_count=128, upsampling=False):
        super().__init__()
        self.upsampling = upsampling
        self.kern_size = kern_size
        self.filter_count = filter_count
        self.layers = nn.Sequential(
            nn.Conv2d(self.filter_count, self.filter_count, kernel_size=self.kern_size, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(self.filter_count),
            nn.Conv2d(self.filter_count, self.filter_count, kernel_size=self.kern_size, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(self.filter_count),
        )


    def forward(self, x):
        if self.upsampling:
            x = nn.Upsample(scale_factor=2, mode='nearest')(x)
        x1 = self.layers(x)
        return x1 + x

class Gen(nn.Module):
    def __init__(self, model_name, num_tiles=13, batch_size=256, embedding_dim=384, z_dim=5, kern_size=7, filter_count=128, num_res_blocks=3):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.model_name = model_name
        self.z_dim = z_dim
        self.filter_count = filter_count
        self.kern_size = kern_size
        self.num_res_blocks = num_res_blocks
        self.num_tiles=num_tiles
        self.batch_size=batch_size

        #new args
        self.sample_path = 'dollarmodel_out/' + self.model_name + "/samples/"



        self.lin1 = nn.Linear(self.embedding_dim + self.z_dim, self.filter_count * 4 * 4)

        self.res_blocks = nn.Sequential()
        for i in range(self.num_res_blocks):
            self.res_blocks.append(ResBlock(self.kern_size, self.filter_count, i < 2))

        self.padding = nn.ZeroPad2d(1)
        self.last_conv = nn.Conv2d(in_channels=self.filter_count, out_channels=16, kernel_size=3)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, embedding, z_dim):
        enc_in_concat = torch.cat((embedding, z_dim), 1)
        x = self.lin1(enc_in_concat)
        x = x.view(-1, self.filter_count, 4, 4)
        # x = torch.reshape(x, (4,4,self.filter_count))
        x = self.res_blocks(x)
        x = self.padding(x)
        x = self.last_conv(x)
        return self.softmax(x)



    def load_data(self, num_tiles=13, scaling_factor=6):

            tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
            model = AutoModel.from_pretrained("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")


            json_path = self.data_path
            print(f"Loading data from {json_path}...")
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            one_hot_scenes = []
            captions = []
            

            for sample in data:
                scene_tensor = torch.tensor(sample["scene"], dtype=torch.long)  # Convert scene to tensor
                one_hot_scene = F.one_hot(scene_tensor, num_classes=self.num_tiles).float()

                augmented_caption = self._augment_caption(sample["caption"])

                one_hot_scenes.append(np.array(one_hot_scene))
                captions.append(augmented_caption)
            
            encoded_captions = st_helper.encode(captions, tokenizer=tokenizer, model=model)


            images=np.array(one_hot_scenes)
            labels=np.array(captions)
            embeddings=np.array(encoded_captions)

            
            embeddings = embeddings * scaling_factor
            

            images, images_test, labels, labels_test, embeddings, embeddings_test = train_test_split(
            images, labels, embeddings, test_size=24, random_state=336)

            train_dataset = [embeddings, images, labels]
            test_dataset = [embeddings_test, images_test, labels_test]

            self.train_set = DataLoader(imageDataSet(train_dataset),
                            batch_size=self.batch_size,
                            shuffle=True,
                            num_workers= 4,
                            persistent_workers=True) 

            self.test_set = DataLoader(imageDataSet(test_dataset),
                            batch_size=self.batch_size,
                            shuffle=True,
                            num_workers= 4,
                            persistent_workers=True) 



    def _augment_caption(self, caption):
        """Shuffles period-separated phrases in the caption."""
        phrases = caption[:-1].split(". ") # [:-1] removes the last period
        random.shuffle(phrases)  # Shuffle phrases
        return ". ".join(phrases) + "."



if __name__ == "__main__":
    seed = 7499629

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"**--Using {device}--**")
    model=Gen(
        model_name="test-titles-2",
        data_path="datasets\SMB1_LevelsAndCaptions-regular.json"
    )
    train(model, 100)