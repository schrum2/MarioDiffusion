[![arXiv](https://img.shields.io/badge/arXiv-2507.00184-b31b1b.svg)](https://arxiv.org/abs/2507.00184)

# Mario Diffusion

Generate Mario level scenes with a diffusion model conditioned on text input. The initial work on this topic was published in AIIDE 2025 in the publication listed below. However, further research has been conducted with this repo to provide more complex features for more games using more data.

## Citation

If you use this code, please cite our paper:  
[Text-to-Level Diffusion Models With Various Text Encoders for Super Mario Bros](https://arxiv.org/abs/2507.00184)  

```bibtex
@article{schrum:aiide2025,
  title={Text-to-Level Diffusion Models with Various Text Encoders for Super Mario Bros},
  volume={21},
  url={https://ojs.aaai.org/index.php/AIIDE/article/view/36815},
  DOI={10.1609/aiide.v21i1.36815},
  number={1},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment},
  author={Schrum, Jacob and Kilday, Olivia and Salas, Emilio and Hagan, Bess and Williams, Reid},
  year={2025},
  month={Nov.},
  pages={110-120}
}
```

More content related to this research is also available at this website:
[https://people.southwestern.edu/~schrum2/mario.html](https://people.southwestern.edu/~schrum2/mario.html)

## Set up the repository

The following instructions apply no matter what game you are interested in training models for.

**Note:** We developed this code using Python 3.10, but we believe it will work fine with more recent versions. We also used [Anaconda](https://www.anaconda.com/) to create a Python environment for the code, though this is not strictly required.

This repository can be checked out with this command:
```
git clone https://github.com/schrum2/MarioDiffusion.git
```
Next, enter the `MarioDiffusion` repository.
```
cd MarioDiffusion
```
Before running any code, install all requirements with pip:
```
pip install -r requirements.txt
```
**NOTE:** Our code was developed on Windows machines using NVIDIA GPUs with CUDA support, and this requirements file will try to install PyTorch with CUDA 12.6 support. If this does not work, then you can install [PyTorch](https://pytorch.org/) on your own. Although it will be slower, we suspect that inference using pre-trained models will work even without CUDA support, though training models will likely be too slow to be feasible. 

## Details on different games

After setting up the repo with the instructions above, follow one of the links below for more details about whatever game you are interested in.

- [Mario 1 and Mario 2 using VGLC Data](Game_Mario/README.md) (Original work from AIIDE 2025 paper)
- [Mega Man using VGLC Data](Game_MM/README.md) (shared workflow for MM-Simple and MM-Full)
- [Enhanced Mario Levels using Mario Maker 2 Data](Game_MM2/README.md)
- [Enhanced Mega Man Levels using Mega Man Maker Data](Game_MMLV/README.md)
- [Lode Runner using VGLC Data](Game_LR/README.md)













REMOVE BELOW

For more information regarding Mega Man, go to the file named `MM_README.md` 
within the Mario Diffusion directory.

[View MM_README.md](MM_README.md)
