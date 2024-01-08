# ZeroShotRerankingNBL

This is the source code for the paper with the title: Zero-Shot Reranking with Encoder Models for News Background Linking.

You need to run the following in order to rerank candidate resources for News Background Linking as explained in the paper.

1- Create an inverted index for the news articles collection. You will find Java files for creating the inverted index for the Washignton Post collection in Lucene V8: indexer.java luceneIndex.java thIndexer.java

- In order to create the index, you need to download first the Washington Post V3 collection file from https://trec.nist.gov/
- The indexer class will first split this file into multiple files for quicker indexing process, then it will call multiple threads to start indexing.
- You need to have these dependencies in your created java project: JSOUP, Lucene V8.0 ,JSON
Preprocessing needs a stop words file also (provided in this directory)


2- Next, Use the class BackgroundLinking.java to be able to 

a) Create a SQL database for the articles in the WashingtonPost dataset. This will help you later in retrieving the text of any article splitted into paragraphs. 

b) Retrieve an initial set of background links for each query article. An example on how to call the retrieval process is given in the main method.


3- Now, for each of the encoders that we compare in this paper, you will find a python file that has methods for:

a) Encoding all queries as explained in the paper, and writing an encoding file for each query in a directory. The file for each query contains the encodings of its title and paragraphs. For Bigbird and LongFormer, it includes the encodings of the passages generated as explained in the paper. To run the encoding for queries, you need to have a queries text file that has a line for each query showing the query information like its  topic id, document id and title. We can't share the queries data here as a license is needed to use this dataset according to NIST. However, an example of this file format is given in this repository.

b) Encoding candidates background links for queries.  The file of each candidate contains the encoding of the whole candidate text.

c) Loading the query encoding files and the candidate encoding files, then matching both as explained in the paper.

- We preferred to encode the query and candidates first then load the encodings for the matching process for efficiency purposes. This way, one can choose to run the encoding methods either on a CPU or a GPU. For us, we encoded the queries on a CPU as they didn't require much time, and encoded the candidates on a GPU for faster encoding process.
