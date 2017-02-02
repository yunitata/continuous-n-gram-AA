# Continuous-N-gram-AA
The repository contains the source code to reproduce experiments from the EACL 2017 paper Continouos N-gram representation for Authorship Attribution (link to paper)

Dependencies
------------
1. Python 2.7
2. Scikit Learn
3. Keras (with Theano backend)

Cloning the repository
----------------------
```git clone https://github.com/yunitata/continuous-n-gram-AA```
----------------------------------------------------------------------

Preparing Data
--------------
1. All the dataset need to be requested directly from the author. Please refer the CCAT10 and CCAT50 to this paper (link paper) while Judgment and IMDb62 to this paper (). Please note that there are two version of IMDb62 datasets. In this experiment, we used the version which contains 62,000 movie reviews and 17,550 message board posts.
2. CCAT10, CCAT50 and IMDb62 datasets comes in the form of list of files per author. To make things easier, we merge all the documents from each of the author (for each of the dataset) into one csv file. It can be done with this following command:

  ```python data_prep.py folder_path csv_path "data_code"```

  CCAT10 and CCAT50 each comes with train and test folders, thus it will have separate train and test csv file.
  For example to prepare train and test data for CCAT10 data

  ```python data_prep.py "/home/C10train" "/home/C10_train.csv" "ccat"```

  ```python data_prep.py "/home/C10test" "/home/C10_test.csv" "ccat"```
  
  For imdb dataset, it does not comes with separate train/test set, to create the csv file:
  ``` ```
  Lastly, for Judgment dataset, it already comes in one .txt file.
  ---------------------------------------------------------------------
  Running Experiment
  ------------------
  To run the experiment, simply run this following command:
  
  ```THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python fastText.py -datasetname -mode -trainfilepath -testfilepath```
  
  **datasetname refers to the code-name of the data, it can be ccat10, ccat50, judgment or imdb
