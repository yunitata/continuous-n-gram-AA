# Continuous-N-gram-AA
The repository contains the source code to reproduce experiments from the EACL 2017 paper Continuous N-gram representation for Authorship Attribution.

Dependencies
------------
1. Python 2.7
2. Scikit Learn 0.18
3. Keras 1.1.1 (with Theano backend). By default, Keras will use TensorFlow as its tensor manipulation library. Please refers to the [Keras website] (https://keras.io/) [1] to configure the Keras backend.
4. Pandas
5. NLTK 3.0.4

You can install all of these by running:

```pip install -r requirements.txt```

Cloning the repository
----------------------
```git clone https://github.com/yunitata/continuous-n-gram-AA```


Preparing Data
--------------
1. All the dataset need to be requested directly from the author. Please refer the CCAT10 and CCAT50 to this [paper] (http://www.sciencedirect.com/science/article/pii/S0306457307001197) [2] while Judgment and IMDb62 to this [paper](http://www.mitpressjournals.org/doi/pdf/10.1162/COLI_a_00173) [3]. Please note that there are two version of IMDb62 datasets. In this experiment, we used the version which contains 62,000 movie reviews and 17,550 message board posts.
2. CCAT10, CCAT50 and IMDb62 datasets comes in the form of list of files per author. To make things easier, we merge all the documents from each of the author (for each of the dataset) into one csv file. It can be done with this following command:

  ```python data_prep.py folder_path csv_path "data_code"```

  CCAT10 and CCAT50 each comes with train and test folders, thus it will have separate train and test csv file.
  For example to prepare train and test data for CCAT10 data

  ```python data_prep.py "/home/C10train" "/home/C10_train.csv" "ccat"```

  ```python data_prep.py "/home/C10test" "/home/C10_test.csv" "ccat"```
  
  For IMDb62 dataset, it does not come with separate train/test set, to create the csv file: (fix this part)
  
  ```test ``` <br />
  
  Lastly, for Judgment dataset, it already comes in one .txt file, so no data preparation is needed.
  
  
Running Experiment
------------------
To run the experiment, simply run this following command:
  
```THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python training_testing.py datasetname mode trainfilepath testfilepath```
 
  **datasetname** refers to the code-name of the data, it can be ccat10, ccat50, judgment or imdb <br />
  **mode** refers to the model applied, which are *word*, *char*, or *wordchar* <br />
  
The codes are meant to be run in gpu machine. However, it can be run in cpu by changing the device=cpu, although the runtime will be significantly longer.<br />
For example if you want to run the experiment for CCAT10 data with *char* model, then the command will be:

```KERAS_BACKEND=theano THEANO_FLAGS=device=gpu,floatX=float32 python training_testing.py "ccat10" "char" "/home/C10_train.csv" "/home/C10_test.csv"```

For IMDb62 and Judgment dataset, you just need to point to the .txt or .csv file as follows:

```KERAS_BACKEND=theano THEANO_FLAGS=device=gpu,floatX=float32 python training_testing.py "imdb" "char" "/home/imdb.csv"```

Please refer to the paper for the detail of the experiments.

Example outputs
---------------
```Epoch 1/150``` <br />
```450/450 [========================] - 2s - loss: 2.2917 - acc: 0.2689 - val_loss: 2.2667 - val_acc: 0.5000``` <br />
```Epoch 2/150``` <br />
```450/450 [========================] - 2s - loss: 2.2621 - acc: 0.4644 - val_loss: 2.2241 - val_acc: 0.5200```

References
----------
[1] https://keras.io/ <br />
[2] http://www.sciencedirect.com/science/article/pii/S0306457307001197 <br />
[3] http://www.mitpressjournals.org/doi/pdf/10.1162/COLI_a_00173 <br />
