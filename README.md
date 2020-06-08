# Master's thesis- Use of autoencoders in classification

## How to run
Python version 3.6 or higher
```
virtualenv venv
source venv/bin/activate
pip install -r requirement.txt
jupyter notebook
open jupyter notebook in browser and open file 'for_training local.ipynb'
``` 

## Documentation


#### callbacks.py 
Custom callback for early stopping during training 

#### config.py 
Config for training- GPU enabled during training, GPU memory usage, 

#### constants.py
Defines all constants in program. 
Defines:
* paths to experiments results, via this paths, experiments results can be loaded
* dataset names
* metrics names

#### datasets.py
Defines interface for dataset class. All datasets are defined here. 
Dataset loads, preprocess its data for training - prepsocess images, changing shape, preprocess labels
Some datasets are defined as singleton because of high memory usage during training.    
Defined datasets:
* Cifar 10
* Fashion Mnist
* Mnist
* FashionMnistDatasetFiveClasses
* FashionMnistDatasetOtherFiveClasses
* Cifar Gray
* Cifar10GrayFiveDataset
* Cifar10GrayFiveOtherDataset
* DerivatedDataset - derived from existing datasets
* MergedDataset - derived from existing datasets

#### excel_logger.py
Logger for writing into excel file

#### experiments.py
Experiments are defined here. Train history and trained model can be saved and loaded again later.
   
Experiments:
* ExperimentAutoencoder
* ExperimentClassifier
* ExperimentAutoencoderAndClassifier
* ExperimentAutoClassifier

#### for_training colab.ipynb
Jupyter notebook for training on Google Colaboratory. 

#### for_training local.ipynb
Jupyter notebook for training and result showing on local machine. With an example of usage.


#### global_functions.py
Global functions, annotations. onStart defines what should happen before start of the training

#### models.py
ModelBuilder, which specifies the architecture for encoder, decoder and fully connected layer. 
ModelBuilder builds the Model of autoencoder, classifier and autoclassifier. Keras functional API used for creating models.    

#### predictors.py
Implementation of predicators used for postprocessing from classifier. Evaluetion and reporting to excel 

#### reporter.py
Reports results from training for each kind of experiment into excel.

#### training.py
Defines training cases with concrete datasets, model builders and train parameters.

#### training_data.py
Defines data for training, prepares inputs for experiments. 

#### vizualizer.py
Vizualize the training process. Shows autoencoder images.