# ZeroShotRerankingNBL

This is the source code for the paper with the title: Zero-Shot Reranking with Dense Encoder Models for News Background Linking. 

The code is written using both Java (version "14.0.1") and Python (Python 2.7).

Required Data: 
- You need to download first the Washington Post collection file: https://trec.nist.gov/data/wapost/. The dataset is publicly available but you need to sign first an organizational aggrement form. Make Sure you request V3 of this collection.
- For preprocessing the news articles, you need the stop words file provided in this directory.

**1) Indexing the data:**
* Export the archive **ZeroShotDenseNBL.zip** archive.
* Create a maven project using the exported directory. This directory contains the **<Pom.xml>** file that defines the required dependencies for this project.
* Compile your project 
* To create an inverted index for the data, run **<indexer.class>**. The indexer class will first split the dataset file into multiple files for a quick indexing process, then it will call multiple threads to start indexing.

**2) Create a SQL database for the data:**
* Use a local SQL engine such as SQLPro to create a database **"WPostDB"**.
* Import the provided file : **WPostDB.sql** into your created database to create the required tables.
* Run **<Database.class>** from within your maven project to insert the news articles from the dataset into the created database.


**3) Retrieve the Candidate Background Links:**
* Run **<BackgroundLinking.class>**.

**4) Rerank the candidate background links:**
* For each of the Transfromer models that we compare in this paper, you will find a python file that implements the different experiments shown in the paper. 
* The encoding process for each model varies given the model's required input length as shown in the paper. However, there are common methods that is used by all models (such as matching a candidate article's encoding to the query passages encoding). These common used methods are called from the given **<utils.py>** file.
* To run the encoding for queries, you need to have a queries text file that has a line for each query showing the query information like its  topic id, document id and title. We can't share the queries data here as a license is needed to use this dataset according to NIST. However, an example of this file format is given in this repository.
* We preferred to encode the query and candidates first then load the encodings from the encoding files for the matching process for efficiency purposes. This way, one can choose to run the encoding methods either on a CPU or a GPU. For us, we encoded the queries on a CPU as they didn't require much time, and encoded the candidates on a GPU for faster encoding process.
