import torch
from segtok.segmenter import split_single
from transformers import AutoTokenizer,AutoModel
import numpy as np

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


#methods that attempts to encode the queries paragraphs using LinkBERT


import pymysql

import utils

def encodeParaLBERT(model,tokenizer,text):
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**encoded_input)
    seq_embeddings = mean_pooling(outputs, encoded_input['attention_mask'])
    pEncoding = seq_embeddings[0].detach().cpu().numpy()
    return pEncoding

def encodeQueriesWithLinkBERT(qIDs, model, outputDir):
    tokenizer = AutoTokenizer.from_pretrained('michiyasunaga/LinkBERT-base')
    QueriesEncoding = {}
    for queryid in qIDs.keys():
        print("Processing Query", queryid)
        count = 1
        fwDoc = open(outputDir + "/" + str(queryid), "w")
        statement = "Select title from documents where id like'" + qIDs[queryid] + "'"
        result = utils.accessDatabase(statement)
        title = str(result[0][0]).strip()
        embedding = encodeParaLBERT(model,tokenizer,title)
        fwDoc.write("<Paragraph>")
        for element in embedding:
            fwDoc.write(str(element) + " ")
        statement = "Select data from contents where did like'" + qIDs[queryid] + "' order by pos"
        paragraphs = utils.accessDatabase(statement)
        textToEncode = ""
        for para in paragraphs:
            p = str(para[0]).strip()
            pEncoding= encodeParaLBERT(model, tokenizer, p)
            fwDoc.write("<Paragraph>")
            count = count + 1
            for element in pEncoding:
                fwDoc.write(str(element) + " ")
        print("Added ", count, "Paragraphs")
        fwDoc.close()
    print("Encoded all queries using Sentence Bert")
    return QueriesEncoding

def encodeCandidatesWithLinkBERT(cIDS, model, outputDir):
    tokenizer = AutoTokenizer.from_pretrained('michiyasunaga/LinkBERT-base')
    QueriesEncoding = {}
    for cindex in range(0,len(cIDS)):
        paraEncodings={}
        count = 1
        fwDoc = open(outputDir + "/" + cIDS[cindex], "w")
        statement = "Select title from documents where id like'" + cIDS[cindex] + "'"
        result = utils.accessDatabase(statement)
        title = str(result[0][0]).strip()
        embedding = encodeParaLBERT(model,tokenizer,title)
        paraEncodings[count]=embedding
        count=count+1
        statement = "Select data from contents where did like'" + cIDS[cindex] + "' order by pos"
        paragraphs = utils.accessDatabase(statement)
        textToEncode = ""
        for para in paragraphs:
            p = str(para[0]).strip()
            pEncoding= encodeParaLBERT(model, tokenizer, p)
            paraEncodings[count] = pEncoding
            count = count + 1
        cEncoding=np.mean(list(paraEncodings.values()), axis=0)
        for element in cEncoding:
            fwDoc.write(str(element) + " ")
        fwDoc.close()
    print("Encoded all candidates using Link BERT")
    return QueriesEncoding

if __name__ == '__main__':

    '''
        model = AutoModel.from_pretrained('michiyasunaga/LinkBERT-base')
        queriesIds = utils.getQueriesId("Reranking/Queries2018.txt")
        encodeQueriesWithLinkBERT(queriesIds, model, "LinkBERTEncoding/Queries")
        queriesIds = utils.getQueriesId("Queries2019.txt")
        encodeQueriesWithLinkBERT(queriesIds, model, "LinkBERTEncoding/Queries")
        queriesIds = utils.getQueriesId("Queries2020.txt")
        encodeQueriesWithLinkBERT(queriesIds, model, "LinkBERTEncoding/Queries")
        queriesIds = utils.getQueriesId("Queries2021.txt")
        encodeQueriesWithLinkBERT(queriesIds, model, "LinkBERTEncoding/Queries")
        '''
    # The following is a call for encoding candidates
    '''
    candidateIds = utils.loadCandidateIdsFromBaselineFile("Reranking/Baseline_2018.txt")
    encodeCandidatesWithLinkBERT(candidateIds, "LinkBERTEncoding/Candidates")
    candidateIds = utils.loadCandidateIdsFromBaselineFile("Reranking/Baseline_2019.txt")
    encodeCandidatesWithLinkBERT(candidateIds,  "LinkBERTEncoding/Candidates")
    candidateIds = utils.loadCandidateIdsFromBaselineFile("Reranking/Baseline_2020.txt")
    encodeCandidatesWithLinkBERT(candidateIds,  "LinkBERTEncoding/Candidates")
    candidateIds = utils.loadCandidateIdsFromBaselineFile("Reranking/Baseline_2021.txt")
    encodeCandidatesWithLinkBERT(candidateIds,  "LinkBERTEncoding/Candidates")
    '''

    # The following is a call for the reranking process
    '''
    win=1
    MaxFlag=False
    alpha=1
    utils.rerankDocBasedOnQueryPara_WholeDocSlidedPara("Reranking/Queries2018.txt", "Reranking/Baseline_2018.txt",
                                                     "Reranking/LinkBERT/2018_LinkBERTRunWin_1_"+str(MaxFlag)+".txt",
                                                     "LinkBERTEncoding/Queries", alpha,
                                                     "LinkBERTEncoding/Candidates", MaxFlag, win)
    utils.rerankDocBasedOnQueryPara_WholeDocSlidedPara("Reranking/Queries2019.txt", "Reranking/Baseline_2019.txt",
                                                     "Reranking/LinkBERT/2019_LinkBERTRunWin_1_"+str(MaxFlag)+".txt",
                                                     "LinkBERTEncoding/Queries", alpha,
                                                     "LinkBERTEncoding/Candidates", MaxFlag, win)

    utils.rerankDocBasedOnQueryPara_WholeDocSlidedPara("Reranking/Queries2020.txt", "Reranking/Baseline_2020.txt",
                                                     "Reranking/LinkBERT/2020_LinkBERTRunWin_1_"+str(MaxFlag)+".txt",
                                                     "LinkBERTEncoding/Queries", alpha,
                                                     "LinkBERTEncoding/Candidates", MaxFlag, win)

    utils.rerankDocBasedOnQueryPara_WholeDocSlidedPara("Reranking/Queries2021.txt", "Reranking/Baseline_2021.txt",
                                                     "Reranking/LinkBERT/2021_LinkBERTRunWin_1_"+str(MaxFlag)+".txt",
                                                     "LinkBERTEncoding/Queries", alpha,
                                                     "LinkBERTEncoding/Candidates", MaxFlag, win)


    '''
