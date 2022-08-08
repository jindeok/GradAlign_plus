# Grad-Align+
Source code for the papers
- Jin-Duk et al. "Grad-Align+: Empowering Gradual Network Alignment Using Attribute Augmentation" CIKM-22


Pytorch_geometric (https://pytorch-geometric.readthedocs.io/en/latest/) package is used for the implementation of graph neural networks (GNNs).

# Dependancy

- Pytorch > 1.8
- torch_geometric and its relevants, whose packages can be downloaded at: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
- numpy
- pands
- sklearn
- tqdm
- matplotlib



# Running

run ``main.py`` script file

or in your prompt, type

``python main.py --graphname 'fb-tt' --k_hop 2 --mode 'not_perturbed' ``  

- --graphname can be either one dataset of the three ('fb-tt' for Facebook vs. Twitter dataset, 'douban' for Douban online vs. offline dataset, 'econ' for Econ perturbed pair.)
- Other description for each arguments are typed in '--help' arguments in main.py argparse.
- for the implemenation of the graphname 'econ', --mode should be changed to 'perturbed' instead of 'not_perturbed'


# etc.
If you need any further information, contact me via e-mail: jindeok6@yonsei.ac.kr
