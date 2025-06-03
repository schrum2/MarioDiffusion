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
from level_dataset import samples_to_scenes
from create_ascii_captions import save_level_data



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
    def __init__(self, model_name, data_path, num_tiles=13, batch_size=256, embedding_dim=384, z_dim=5, kern_size=7, filter_count=128, num_res_blocks=3,):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.model_name = model_name
        self.z_dim = z_dim
        self.filter_count = filter_count
        self.kern_size = kern_size
        self.num_res_blocks = num_res_blocks
        self.data_path=data_path
        self.num_tiles=num_tiles
        self.batch_size=batch_size

        #new args
        self.sample_path = 'dollarmodel_out/' + self.model_name + "/samples/"
        #self.gen_to_image = self.map_to_image
        #self.tiles = self.mario_tiles()




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
    
    # Use tiles to construct image of map
    def map_to_image(self, ascii_map, tile_size=16):
        
        tiles = self.tiles
        rows, cols = ascii_map.shape
        image = Image.new('RGB', (cols * tile_size, rows * tile_size))

        for row in range(rows):
            for col in range(cols):
                tile_index = ascii_map[row, col]
                tile = tiles[tile_index]
                image.paste(tile, (col * tile_size, row * tile_size))

        return image
    

    def mario_tiles(self):
        """
        Maps integers 0-15 to 16x16 pixel sprites from mapsheet.png.

        Returns:
            A list of 16x16 pixel tile images for Mario.
        """

        # DEBUGGING
        #raise ValueError("Why is this being called!")

        _sprite_sheet = Image.open("map_tileset//mapsheet.png")

        # Hardcoded coordinates for the first 16 tiles (row, col)
        tile_coordinates = [
            (2,5),    # 0 = Sky
            (2,2),    # 1 = left upper lip of pipe
            (3,2),    # 2 = right upper lip of pipe
            (0,1),    # 3 = question block with power up
            (3,0),    # 4 = Cannon head
            (7,4),    # 5 = enemy
            (2,1),    # 6 = question block with coin
            (2,6),    # 7 = breakable brick block
            (1,0),    # 8 = solid block/floor
            (4,2),    # 9 = left edge of pipe body
            (5,2),    # 10 = right edge of pipe body
            (4,0),    # 11 = Cannon support (should be 5,0 sometimes?)
            (7,1),    # 12 = coin
            # Tile right below decides what the padded tile is (sky currently)
            (2,5),    # 13 = Padding (sky)
            (0,6),    # 14 = Nothing
            (1,6),    # 15 = Nothing (extra just in case)
        ]

        # Extract each tile as a 16x16 image
        tile_images = []
        for col, row in tile_coordinates:
            left = col * 16
            upper = row * 16
            right = left + 16
            lower = upper + 16
            tile = _sprite_sheet.crop((left, upper, right, lower))
            tile_images.append(tile)

        # Add a blank tile for the extra tile (padding)
        blank_tile = Image.new('RGB', (16, 16), color=(128, 128, 128))  # Gray or any color
        tile_images.append(blank_tile)

        # Save each tile image as tile_X.png for inspection
        #for idx, tile_img in enumerate(tile_images):
        #    tile_img.save(f"tile_{idx}.png")

        return tile_images
    

    """def render_images(self, images, labels, title, save_path, embeddings=None, correct_images=None):
        num_images = len(images)
        num_rows = (num_images // 8) + ((num_images % 8) > 0)
        num_subplots_per_image = 1 + (embeddings is not None) + (correct_images is not None)
        
        # add an extra row for spacing after every 8 images
        fig, axs = plt.subplots((num_rows + num_rows // 8) * num_subplots_per_image, 8, figsize=(8*4, (num_rows + num_rows // 8)*4*num_subplots_per_image))

        # If there is only one row, axs will be a 1-dimensional array.
        if isinstance(axs, plt.Axes):
            axs = np.array([[axs]])

        for i, (image, label) in enumerate(zip(images, labels)):
            row = ((i // 8) + (i // 64)) * num_subplots_per_image
            col = i % 8

            pil_image = self.gen_to_image(image)  # assuming gen_to_image is defined elsewhere
            axs[row, col].imshow(pil_image)
            axs[row, col].set_title("\n".join(textwrap.wrap("GEN: " + label, width=30)), fontsize=8)
            axs[row, col].axis('off')

            if correct_images is not None:
                correct_pil_image = self.gen_to_image(correct_images[i])
                axs[row + 1, col].imshow(correct_pil_image)
                axs[row + 1, col].set_title("\n".join(textwrap.wrap("ORIG: " + label, width=30)), fontsize=8)
                axs[row + 1, col].axis('off')

            if embeddings is not None:
                embed_heatmap = axs[row + num_subplots_per_image - 1, col].imshow(np.reshape(embeddings[i], (16, 24)), cmap='hot', interpolation='nearest', vmin=-1, vmax=1)
                fig.colorbar(embed_heatmap, ax=axs[row + num_subplots_per_image - 1, col])
                axs[row + num_subplots_per_image - 1, col].axis('off')

        plt.suptitle(title, fontsize=16)
        plt.subplots_adjust(hspace=0.5)  # adjust space between rows
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)"""
    
    def render_images(self, images, labels, title, save_path, embeddings=None, correct_images=None):
        #print(images)
        #scenes = samples_to_scenes(images)
        #print(scenes)
        save_level_data(images, '..\TheVGLC\Super Mario Bros\smb.json', 
                        os.path.join(save_path, "all_levels.json"), False, 
                        False, exclude_broken=False)



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
            images, labels, embeddings, test_size=24, random_state=seed)

            train_dataset = [embeddings, images, labels]
            test_dataset = [embeddings_test, images_test, labels_test]

            self.train_set = DataLoader(imageDataSet(train_dataset),
                            batch_size=self.batch_size,
                            shuffle=True,
                            num_workers= 8 if device == 'cuda' else 1,
                            pin_memory=(device=="cuda"),
                            persistent_workers=True) # Makes transfer from the CPU to GPU faster

            self.test_set = DataLoader(imageDataSet(test_dataset),
                            batch_size=self.batch_size,
                            shuffle=True,
                            num_workers= 8 if device == 'cuda' else 1,
                            pin_memory=(device=="cuda"),
                            persistent_workers=True) # Makes transfer from the CPU to GPU faster



    def _augment_caption(self, caption):
        """Shuffles period-separated phrases in the caption."""
        phrases = caption[:-1].split(". ") # [:-1] removes the last period
        random.shuffle(phrases)  # Shuffle phrases
        return ". ".join(phrases) + "."




"""def load_data(path, scaling_factor=6, batch_size=256):
    data = np.load(path, allow_pickle=True).item()
    images = np.array(data['images'])
    labels = data['labels']

    embeddings = data['embeddings']
    if isinstance(embeddings, list):
        embeddings = np.array(embeddings)
    embeddings = embeddings * scaling_factor

    images, images_test, labels, labels_test, embeddings, embeddings_test = train_test_split(
    images, labels, embeddings, test_size=24, random_state=seed)

    train_dataset = [embeddings, images, labels]
    test_dataset = [embeddings_test, images_test, labels_test]

    train_set = DataLoader(imageDataSet(train_dataset),
                       batch_size=batch_size,
                       shuffle=True,
                       num_workers= 8 if device == 'cuda' else 1,
                       pin_memory=(device=="cuda")) # Makes transfer from the CPU to GPU faster

    test_set = DataLoader(imageDataSet(test_dataset),
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers= 8 if device == 'cuda' else 1,
                      pin_memory=(device=="cuda")) # Makes transfer from the CPU to GPU faster

    return train_set, test_set"""


def test_set_gen(ep, model, epoch_dir, test_set):
    title = model.model_name + ' Test set samples, epoch' + str(ep)
    file_name = os.path.join(epoch_dir, 'test_set_samples.png')


    for idx, batch in enumerate(test_set):
        embeddings = batch[0]
        labels = batch[2]
        correct_images = batch[1]
        
        predictions = model(embeddings.to(device), torch.rand(len(embeddings), 5).to(device))
        
        argmaxed_gens = predictions.argmax(dim=1)
        

        model.render_images(images=argmaxed_gens, labels=labels, correct_images=np.argmax(correct_images, axis=-1), title=title, save_path=file_name)





def do_renders(model, ep, test_set):
    epoch_dir = os.path.join(model.sample_path, 'epoch_' + str(ep))
    os.makedirs(epoch_dir, exist_ok=True)
    test_set_gen(ep, model, epoch_dir, test_set)


def train(model, EPOCHS):
    model.load_data()

    train_set, test_set = model.train_set, model.test_set

    loss_metric_train = torch.zeros(EPOCHS).to(device)

    model.to(device)

    optimizer = optim.Adam(model.parameters())
    
    for epoch in range(EPOCHS):
        for embeddings, ytrue, _ in train_set:
            optimizer.zero_grad()
            outputs = model(embeddings.to(device), torch.rand(len(embeddings), 5).to(device))
            loss = nn.NLLLoss()(torch.log(outputs), ytrue.argmax(dim=3).to(device))

            loss_metric_train[epoch] += loss

            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}: Loss = {loss_metric_train[epoch]}")
        if epoch%10==0:
            do_renders(model, epoch, test_set)
    
    do_renders(model, epoch, test_set)


if __name__ == "__main__":
    seed = 7499629
    
    input_shape = (10, 10, 16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"**--Using {device}--**")
    model=Gen(
        model_name="temp",
        data_path="datasets\SMB1_LevelsAndCaptions-regular.json"
    )
    train(model, 100)