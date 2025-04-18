# MSMCDA

MSMCDA (Multi-view Shared Units and Multi-channel Attention for circRNA–Disease Association prediction) is a computational framework for predicting circRNA–disease associations. It integrates multi-view similarity networks and contrastive learning to effectively capture and enhance the latent relationships between circRNAs and diseases. The model consists of the following key steps:

1. Constructs similarity and meta-path networks for circRNAs and diseases using shared units to enable interactive feature learning.
2. Applies multi-channel attention mechanisms to learn the importance of different similarity networks.
3. Employs contrastive learning to enhance the learned similarity representations.

# Requirements
  * Ensure the following dependencies are installed (from `requirements.txt`):
  * Python 3.7 ≤ 3.11
  * PyTorch 2.6.0 or higher
  * torch-geometric 2.6.1
  * GPU (default)

Other required packages include:

- numpy
- pandas
- matplotlib
- scikit-learn
- scipy
- networkx
- tqdm
- requests
- aiohttp
- dgl
- joblib
- jinja2
- pillow
- and others (full list in `requirements.txt`)
  
Install all requirements using:

```bash
pip install -r requirements.txt
```
# Data
  * Place all associated datasets and similarity data inside the ```datasets/``` folder.
  * Details regarding multiple similarity network construction are provided in the supplementary material of the original paper.

# Running  the Code
  * To execute the model:
```bash
  python main.py
```
  * Parameter ```state='valid'```. Start the 5-fold cross validation training model.
  * Parameter ```state='test'```. Start the independent testing.
  * You can modify the number of training epochs in ```main.py``` via ```Config.epoch```. The default is 128, but you can reduce it for quick testing.

# Note
  * ```torch-geometric``` has strict version compatibility. Please ensure proper installation based on your system and CUDA version.
  * Trained models will be saved in the ```cross valid/``` directory.
  * Outputs and results will be generated upon completion and can be reused for testing or analysis.
