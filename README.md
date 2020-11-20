# Analyzing Pooling in Recurrent Architectures

Repository for the paper [Analyzing Pooling in Recurrent Architectures](https://arxiv.org/abs/2005.00159) by [Pratyush Maini](https://pratyush911.github.io), [Kolluru Sai Keshav](https://saikeshav.github.io/), [Danish Pruthi](https://www.cs.cmu.edu/~ddanish/) and [Mausam](http://www.cse.iitd.ac.in/~mausam/)


## Dependencies
The code requires the following dependencies to run can be installed using the `conda` environment file provided:
```
conda env create --file environment.yaml
```

## Running gradients experiments

### Evaluate the Initial gradient distribution
```
python train.py --pool att_max --data_size 20K --gpu_id 0 --mode train --batch_size 1 --task IMDB_LONG --wiki none --epochs 5 --gradients 1 --initial 1 --log 1 --customlstm 1
```
Results are at model_dir/initial_gradients.txt

### Vanishing Ratios
```
python train.py --pool att_max --data_size 20K --gpu_id 0 --mode train --batch_size 32 --task IMDB_LONG --wiki none --epochs 5 --gradients 1 --ratios 1 --log 1 --customlstm 1
```
Results are at model_dir/ratios.txt

## Train a model in the Wiki setting
```
python train.py --pool att_max --data_size 20K --gpu_id 0 --mode train --batch_size 32 --task IMDB_LONG --wiki mid --epochs 20 --log 1 --customlstm 0
```
Logs are at `model_dir/logs.txt`

## Bias Evaluation

### Change the test time distribution
```
python test.py --pool att_max --data_size 20K --gpu_id 0 --mode test --batch_size 32 --task IMDB_LONG --wiki none --vec 3 --customlstm 0
```

### Get the NWI scores
```
python test.py --pool att_max --data_size 20K --gpu_id 0 --mode test --batch_size 32 --task IMDB_LONG --wiki none --NWI 1 --customlstm 0
```

## How can I cite this work?
```
@inproceedings{maini2020pool,
    title = "Why and when should you pool? Analyzing Pooling in Recurrent Architectures",
    author = "Maini, Pratyush and Kolluru, Keshav and Pruthi, Danish and {Mausam}",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.findings-emnlp.410",
}
```
