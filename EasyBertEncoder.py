import torch
from segtok.segmenter import split_single
from transformers import AutoTokenizer,AutoModel
import numpy as np

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


#methods that attempts to encode the queries paragraphs using LinkBERT

def getQueriesId(queriesfile):
    toggle = True
    queriesIds = {}
    with open(queriesfile, encoding='utf-8') as f:
        for line in f:
            if toggle:
                data = line.split(",")
                queriesIds[data[1]] = data[0]
                toggle = False
            else:
                toggle = True
    return queriesIds

import pymysql
import utils

def encodeTextEaseBERT(model,tokenizer,text):
    sentences = list(split_single(text))
    sentencesEncoding = {}
    sNo = 0
    pooler = lambda last_hidden, att_mask: (last_hidden * att_mask.unsqueeze(-1)).sum(1) / att_mask.sum(-1).unsqueeze(
        -1)
    for s in sentences:
        encoded_input = tokenizer(s, padding=True, truncation=True, return_tensors='pt')
        last_hidden = model(**encoded_input, output_hidden_states=True, return_dict=True).last_hidden_state
        Sentembedding = pooler(last_hidden, encoded_input["attention_mask"])[0]
        sentencesEncoding[sNo] = Sentembedding.detach().cpu().numpy()
        sNo = sNo + 1
    pEncoding = np.mean(list(sentencesEncoding.values()), axis=0)
    return pEncoding

def encodeQueriesWithEaseBERT(qIDs, model, outputDir):
    tokenizer = AutoTokenizer.from_pretrained('sosuke/ease-bert-base-uncased')
    #print("model loaded successfull")
    QueriesEncoding = {}
    for queryid in qIDs.keys():
        print("Processing Query", queryid)
        count = 1
        fwDoc = open(outputDir + "/" + str(queryid), "w")
        statement = "Select title from documents where id like'" + qIDs[queryid] + "'"
        result = utils.accessDatabase(statement)
        title = str(result[0][0]).strip()
        embedding = encodeTextEasyBERT(model,tokenizer,title)
        fwDoc.write("<Paragraph>")
        for element in embedding:
            fwDoc.write(str(element) + " ")
        statement = "Select data from contents where did like'" + qIDs[queryid] + "' order by pos"
        paragraphs = utils.accessDatabase(statement)
        textToEncode = ""
        for para in paragraphs:
            p = str(para[0]).strip()
            pEncoding= encodeTextEasyBERT(model, tokenizer, p)
            fwDoc.write("<Paragraph>")
            count = count + 1
            for element in pEncoding:
                fwDoc.write(str(element) + " ")
        #print("Added ", count, "Paragraphs")
        fwDoc.close()
    print("Encoded all queries using Ease Bert:",outputDir)
    return QueriesEncoding

def encodeCandidatesWithEaseBERT(cIDs, model, outputDir):
    tokenizer = AutoTokenizer.from_pretrained('sosuke/ease-bert-base-uncased')
    #print("model loaded successfull")
    QueriesEncoding = {}
    for cindex in range(0,len(cIDs)):
        paraEncodings={}
        print("Processing Candidate", cIDs[cindex])
        count = 1
        fwDoc = open(outputDir + "/" + str(cIDs[cindex]), "w")
        statement = "Select title from documents where id like'" + cIDs[cindex] + "'"
        result = utils.accessDatabase(statement)
        title = str(result[0][0]).strip()
        embedding = encodeTextEasyBERT(model,tokenizer,title)
        paraEncodings[count]=embedding
        count=count+1
        statement = "Select data from contents where did like'" + cIDs[cindex] + "' order by pos"
        paragraphs = utils.accessDatabase(statement)
        textToEncode = ""
        for para in paragraphs:
            p = str(para[0]).strip()
            pEncoding= encodeTextEasyBERT(model, tokenizer, p)
            paraEncodings[count] = pEncoding
            count = count + 1
        cEncoding=np.mean(list(paraEncodings.values()), axis=0)
        for element in cEncoding:
            fwDoc.write(str(element) + " ")
        fwDoc.close()
    print("Encoded all candidates using Ease Bert:",outputDir)
    return QueriesEncoding


if __name__ == '__main__':
    '''
    model = AutoModel.from_pretrained('sosuke/ease-bert-base-uncased')
    queriesIds = utils.getQueriesId("Reranking/Queries2018.txt")
    encodeQueriesWithEaseBERT(queriesIds, model, "EaseBERTEncoding/Queries")
    queriesIds = utils.getQueriesId("Queries2019.txt")
    encodeQueriesWithEaseBERT(queriesIds, model, "EaseBERTEncoding/Queries")
    queriesIds = utils.getQueriesId("Queries2020.txt")
    encodeQueriesWithEaseBERT(queriesIds, model, "EaseBERTEncoding/Queries")
    queriesIds = utils.getQueriesId("Queries2021.txt")
    encodeQueriesWithEaseBERT(queriesIds, model, "EaseBERTEncoding/Queries")
    '''
    # The following is a call for encoding candidates
    '''
    candidateIds = utils.loadCandidateIdsFromBaselineFile("Reranking/Baseline_2018.txt")
    encodeCandidatesWithEaseBERT(candidateIds, "EaseBERTEncoding/Candidates")
    candidateIds = utils.loadCandidateIdsFromBaselineFile("Reranking/Baseline_2019.txt")
    encodeCandidatesWithEaseBERT(candidateIds,  "EaseBERTEncoding/Candidates")
    candidateIds = utils.loadCandidateIdsFromBaselineFile("Reranking/Baseline_2020.txt")
    encodeCandidatesWithEaseBERT(candidateIds,  "EaseBERTEncoding/Candidates")
    candidateIds = utils.loadCandidateIdsFromBaselineFile("Reranking/Baseline_2021.txt")
    encodeCandidatesWithEaseBERT(candidateIds,  "EaseBERTEncoding/Candidates")
    '''

    # The following is a call for the reranking process
    '''
    win=1
    MaxFlag=False
    alpha=1

    utils.rerankDocBasedOnQueryPara_WholeDocSlidedPara("Reranking/Queries2018.txt", "Reranking/Baseline_2018.txt",
                                                     "Reranking/EaseBERT/2018_EaseBERTRunWin_1_"+str(MaxFlag)+".txt",
                                                     "EaseBERTEncoding/Queries", alpha,
                                                     "EaseBERTEncoding/Candidates", MaxFlag, win)
    utils.rerankDocBasedOnQueryPara_WholeDocSlidedPara("Reranking/Queries2019.txt", "Reranking/Baseline_2019.txt",
                                                     "Reranking/EaseBERT/2019_EaseBERTRunWin_1_"+str(MaxFlag)+".txt",
                                                     "EaseBERTEncoding/Queries", alpha,
                                                     "EaseBERTEncoding/Candidates", MaxFlag, win)

    utils.rerankDocBasedOnQueryPara_WholeDocSlidedPara("Reranking/Queries2020.txt", "Reranking/Baseline_2020.txt",
                                                     "Reranking/EaseBERT/2020_EaseBERTRunWin_1_"+str(MaxFlag)+".txt",
                                                     "EaseBERTEncoding/Queries", alpha,
                                                     "EaseBERTEncoding/Candidates", MaxFlag, win)

    utils.rerankDocBasedOnQueryPara_WholeDocSlidedPara("Reranking/Queries2021.txt", "Reranking/Baseline_2021.txt",
                                                     "Reranking/EaseBERT/2021_EaseBERTRunWin_1_"+str(MaxFlag)+".txt",
                                                     "EaseBERTEncoding/Queries", alpha,
                                                     "EaseBERTEncoding/Candidates", MaxFlag, win)


    '''