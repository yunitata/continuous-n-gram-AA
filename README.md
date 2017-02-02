# Continuous-N-gram-AA
The repository contains the source code to reproduce experiments from the EACL 2017 paper Continouos N-gram representation for Authorship Attribution (link to paper)

Dependencies
------------
1. Python 2.7
2. Scikit Learn
3. Keras (with Theano backend)

Cloning the repository
----------------------
<pre>git clone https://github.com/yunitata/continuous-n-gram-AA><code>
----------------------------------------------------------------------

Preparing Data
--------------
1. All the dataset need to be requested directly from the author. Please refer the CCAT10 and CCAT50 to this paper (link paper) while Judgment and IMDb62 to this paper (). Please note that there are two version of IMDb62 datasets. In this experiment, we used the version which contains 62,000 movie reviews and 17,550 message board posts.
2. CCAT10, CCAT50 and IMDb62 datasets comes in the form of list of files per author. To make things easier, we merge all the documents from each of the author (for each of the dataset) into one csv file. It can be done with this following command:
<pre>python data_prep.py folder_path csv_path "data_code"<code>
CCAT10 and CCAT50 each comes with train and test folders, thus it will have separate train and test csv file.
for example to prepare train and test data for CCAT data
<pre>python data_prep.py "/home/C10train" "/home/C10_train.csv" "ccat"<code>
<pre>python data_prep.py "/home/C10test" "/home/C10_test.csv" "ccat"<code>

