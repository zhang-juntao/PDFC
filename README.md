# Prerequisite-driven Fair Clustering on Heterogeneous Information Networks

This repository contains a PyTorch implementation of PDFC

### Requirements

The code has been tested running under Python 3.8.8 with the following packages installed (along with their dependencies):

```
numpy==1.20.1
pandas==1.2.4
scipy==1.6.2
torch==1.12.0
```

### Data information
The ``datasets/`` contains three real-world datasets, MOOCCube, DBLP, and Movielens.

MOOCCube: this is an online education dataset, and it was collected and published on XuetangX.

DBLP: this is a citation network dataset, and it was collected and published on AMiner.

Movielens: this is a rating dataset about movies collected by GroupLens.


We describe the contents of each dataset file.

``features/``: This contains the features information of nodes.

``path_info/``: This contains the number of neighbors of nodes in the adjacency matrix.

``prerequisite/``: This contains the prerequisite relations between entities.

``sensitive/``: This contains the sensitive attribute information of nodes.


### Repository Organization
``main.py``: The main running program of PDFC.

``models.py``: PDFC learning node embeddings based on graph model.

``metrics.py``: Metrics for evaluating fair clustering.

``utils.py``: Loading data and data processing.


### Running PDFC
To achieve fair clustering, please run
```bash
python main.py 
```



