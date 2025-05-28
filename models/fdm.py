import torch
from torch import nn
from torch.utils.data import Dataset



class imageDataSet(Dataset):

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self,idx):
        return self.data[0][idx], self.data[1][idx]

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
    def __init__(self, model_name, embedding_dim=384, z_dim=5, kern_size=7, filter_count=128, num_res_blocks=3, out_channels=16):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.model_name = model_name
        self.z_dim = z_dim
        self.filter_count = filter_count
        self.kern_size = kern_size
        self.num_res_blocks = num_res_blocks
        self.out_channels = out_channels

        self.lin1 = nn.Linear(self.embedding_dim + self.z_dim, self.filter_count * 4 * 4)

        self.res_blocks = nn.Sequential()
        for i in range(self.num_res_blocks):
            self.res_blocks.append(ResBlock(self.kern_size, self.filter_count, i < 2))

        self.padding = nn.ZeroPad2d(1)
        self.last_conv = nn.Conv2d(in_channels=self.filter_count, out_channels=self.out_channels, kernel_size=9, padding=3)
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
