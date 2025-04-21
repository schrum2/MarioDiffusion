import torch
import torch.nn as nn

class WGAN_Discriminator(nn.Module):
    def __init__(self, isize, nc, ndf, ngpu=1, n_extra_layers=0):
        """
        WGAN Discriminator (Critic) model
        
        Args:
            isize: Input image size (width/height), must be a multiple of 16
            nc: Number of input channels (tile types)
            ndf: Size of feature maps in discriminator
            ngpu: Number of GPUs
            n_extra_layers: Number of extra conv layers
        """
        super(WGAN_Discriminator, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # Input is nc (tile types) x isize x isize
        main.add_module('initial-conv',
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial-lrelu',
                        nn.LeakyReLU(0.2, inplace=True))
        
        csize, cndf = isize // 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module(f'extra-{t}-conv',
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module(f'extra-{t}-batchnorm',
                            nn.BatchNorm2d(cndf))
            main.add_module(f'extra-{t}-lrelu',
                            nn.LeakyReLU(0.2, inplace=True))

        # Downsampling layers
        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module(f'down-{in_feat}-{out_feat}-conv',
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module(f'down-{out_feat}-batchnorm',
                            nn.BatchNorm2d(out_feat))
            main.add_module(f'down-{out_feat}-lrelu',
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = out_feat
            csize = csize // 2

        # Final layer to single value output
        main.add_module('final-conv',
                        nn.Conv2d(cndf, 1, 4, 1, 0, bias=False))
        
        self.main = main

    def forward(self, input):
        if self.ngpu > 1 and isinstance(input.data, torch.cuda.FloatTensor):
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else: 
            output = self.main(input)
        
        # Average across the batch dimension
        output = output.mean(0)
        return output.view(1)


class WGAN_Generator(nn.Module):
    def __init__(self, isize, nz, nc, ngf, ngpu=1, n_extra_layers=0):
        """
        WGAN Generator model
        
        Args:
            isize: Output image size (width/height), must be a multiple of 16
            nz: Size of latent vector
            nc: Number of output channels (tile types)
            ngf: Size of feature maps in generator
            ngpu: Number of GPUs
            n_extra_layers: Number of extra conv layers
        """
        super(WGAN_Generator, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        # Calculate initial feature map size
        cngf, tisize = ngf//2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        # Input is latent vector Z
        main.add_module('initial-convt',
                        nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        main.add_module('initial-batchnorm',
                        nn.BatchNorm2d(cngf))
        main.add_module('initial-relu',
                        nn.ReLU(True))

        csize, cndf = 4, cngf
        # Upsampling layers
        while csize < isize//2:
            main.add_module(f'up-{cngf}-{cngf//2}-convt',
                            nn.ConvTranspose2d(cngf, cngf//2, 4, 2, 1, bias=False))
            main.add_module(f'up-{cngf//2}-batchnorm',
                            nn.BatchNorm2d(cngf//2))
            main.add_module(f'up-{cngf//2}-relu',
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module(f'extra-{t}-conv',
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module(f'extra-{t}-batchnorm',
                            nn.BatchNorm2d(cngf))
            main.add_module(f'extra-{t}-relu',
                            nn.ReLU(True))

        # Final layer to output
        main.add_module('final-convt',
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        
        # We use softmax for one-hot encoding output
        main.add_module('final-activation',
                        nn.Softmax(dim=1))
        
        self.main = main

    def forward(self, input):
        if self.ngpu > 1 and isinstance(input.data, torch.cuda.FloatTensor):
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else: 
            output = self.main(input)
        return output