## Training Post Retrieval
Code for addressing the training post retrieval problem over constrained search interfaces, which attempts to retrieve a social media post dataset with good balance and diversity for training a text classifier.

## Requirements
* python 3
* pytorch
* torchtext
* numpy
* scikit-learn
* nltk
* gensim

## Files
* main.py - Main file for running experiments.
* text_dataset_labeling_assistant.py - Code for keyword selection algorithms (KSAs).
* data_loading.py - Code for loading data from files.
* simulated_api.py - Simulates a keyword search API using the loaded data.
* cnn_text_classification.py - CNN text classifier from [cnn-text-classification-pytorch](https://github.com/rriva002/cnn-text-classification-pytorch).
* keyword_ranking.py - Double ranking algorithm from [Identifying Search Keywords for Finding Relevant Social Media Posts](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewPaper/12199).
* ds.json - File specifying topics and manually-generated keywords for DailyStrength.
* huffpost.json - File specifying topics (categories) for The Huffington Post.
* reddit.json - File specifying topics (subreddits) and manually-generated keywords for Reddit.

There should also be a data directory with the following files:
* ds_data.csv containing data collected from the topics specified in ds.json.
* reddit_data.csv containing data collected from the subreddits specified in reddit.json.
* News_Category_Dataset_v2.json from [News Category Dataset](https://www.researchgate.net/publication/332141218_News_Category_Dataset) for data from the Huffington Post.

The two .csv files should have two columns (no header):
* Column 1 - The text of the post.
* Column 2 - The topic (DailyStrength) or subreddit (Reddit) of the post.

Note that the data files are not included in this repository.

## Running the Code
Run main.py to perform the experiments on the data. Output to stdout and .csv files will include, for each combination of KSA, data source, topic, and value of m:
* Accuracy of the CNN classifier trained with a dataset retrieved by the KSA.
* Percent positive of the retrieved dataset.
* Kullback-Liebler divergence of the positive and negative portions of the dataset as compared to a random dataset of equal size. Note these are labeled as "Diversity (Positives)" and "Diversity (Negatives)" in the output.
