
import utils
from transformers import AutoTokenizer,AutoModel
import numpy as np

def encodeQueriesWithErnie(qIDs, model, outputDir):
    tokenizer = AutoTokenizer.from_pretrained('nghuyong/ernie-2.0-base-en')
    QueriesEncoding = {}
    for queryid in qIDs.keys():
        print("Processing Query", queryid)
        count = 1
        fwDoc = open(outputDir + "/" + str(queryid), "w")
        statement = "Select title from documents where id like'" + qIDs[queryid] + "'"
        result = utils.accessDatabase(statement)
        title = str(result[0][0]).strip()
        clsembedding, pooledEmbedding= encodeTextErnie(model,tokenizer,title)
        fwDoc.write("<Paragraph>")
        for element in clsembedding:
            fwDoc.write(str(element) + " ")
        fwDoc.write("<Pooled>")
        for element in pooledEmbedding:
            fwDoc.write(str(element) + " ")
        statement = "Select data from contents where did like'" + qIDs[queryid] + "' order by pos"
        paragraphs = utils.accessDatabase(statement)
        textToEncode = ""
        for para in paragraphs:
            p = str(para[0]).strip()
            clsembedding, pooledEmbedding= encodeTextErnie(model, tokenizer, p)
            count = count + 1
            fwDoc.write("<Paragraph>")
            for element in clsembedding:
                fwDoc.write(str(element) + " ")
            fwDoc.write("<Pooled>")
            for element in pooledEmbedding:
                fwDoc.write(str(element) + " ")
        #print("Added ", count, "Paragraphs")
        fwDoc.close()
    print("Encoded all queries using Sentence Bert")
    return QueriesEncoding

def encodeCandidatesWithErnie(cIDs, model, outputDir):
    tokenizer = AutoTokenizer.from_pretrained('nghuyong/ernie-2.0-base-en')
    QueriesEncoding = {}
    for cindex in range(0,len(cIDs)):
        paraEmbeddings={}
        count = 1
        fwDoc = open(outputDir + "/" + cIDs[cindex], "w")
        statement = "Select title from documents where id like'" + cIDs[cindex] + "'"
        result = utils.accessDatabase(statement)
        title = str(result[0][0]).strip()
        clsembedding, _= encodeTextErnie(model,tokenizer,title)
        paraEmbeddings[count]=clsembedding
        count=count+1
        statement = "Select data from contents where did like'" + cIDs[cindex] + "' order by pos"
        paragraphs = utils.accessDatabase(statement)
        textToEncode = ""
        for para in paragraphs:
            p = str(para[0]).strip()
            clsembedding, _= encodeTextErnie(model, tokenizer, p)
            paraEmbeddings[count] = clsembedding
            count = count + 1
        cEmbedding=np.mean(list(paraEmbeddings.values()), axis=0)
        for element in cEmbedding:
            fwDoc.write(str(element) + " ")

        #print("Added ", count, "Paragraphs")
        fwDoc.close()
    print("Encoded all queries using Sentence Bert")
    return QueriesEncoding

def encodeTextErnie(model,tokenizer,text):
    encoded_input = tokenizer(text, padding=True, max_length=512, truncation=True, return_tensors='pt')
    outputs = model(**encoded_input, output_hidden_states=True, return_dict=True).last_hidden_state
    # print(outputs)
    pooler = lambda last_hidden, att_mask: (last_hidden * att_mask.unsqueeze(-1)).sum(1) / att_mask.sum(-1).unsqueeze(
        -1)
    CLSEmbedding = outputs[0][0].detach().cpu().numpy()
    seq_embeddings = pooler(outputs, encoded_input["attention_mask"])[0].detach().cpu().numpy()
    return CLSEmbedding,seq_embeddings


if __name__ == '__main__':

    # We generate a reranking for each year first for efficiency purposes,
    # then we combine the outputfile before running trec_eval on a qrel file that combines the qrels of all years.


    # The following is a call for encoding the queries
    '''
    model = AutoModel.from_pretrained('nghuyong/ernie-2.0-base-en')
    queriesIds = utils.getQueriesId("Reranking/Queries2018.txt")
    encodeQueriesWithErnie(queriesIds, model, "ErnieEncoding/Queries")
    queriesIds = utils.getQueriesId("Queries2019.txt")
    encodeQueriesWithErnie(queriesIds, model, "ErnieEncoding/Queries")
    queriesIds = utils.getQueriesId("Queries2020.txt")
    encodeQueriesWithErnie(queriesIds, model, "ErnieEncoding/Queries")
    queriesIds = utils.getQueriesId("Queries2021.txt")
    encodeQueriesWithErnie(queriesIds, model, "ErnieEncoding/Queries")
    '''
    # The following is a call for encoding candidates
    '''
    candidateIds = utils.loadCandidateIdsFromBaselineFile("Reranking/Baseline_2018.txt")
    encodeCandidatesWithErnie(candidateIds, "ErnieEncoding/Candidates")
    candidateIds = utils.loadCandidateIdsFromBaselineFile("Reranking/Baseline_2019.txt")
    encodeCandidatesWithErnie(candidateIds,  "ErnieEncoding/Candidates")
    candidateIds = utils.loadCandidateIdsFromBaselineFile("Reranking/Baseline_2020.txt")
    encodeCandidatesWithErnie(candidateIds,  "ErnieEncoding/Candidates")
    candidateIds = utils.loadCandidateIdsFromBaselineFile("Reranking/Baseline_2021.txt")
    encodeCandidatesWithErnie(candidateIds,  "ErnieEncoding/Candidates")
    '''

    # The following is a call for the reranking process
    '''
    win=1
    MaxFlag=False
    alpha=1

    utils.rerankDocBasedOnQueryPara_WholeDocSlidedPara("Reranking/Queries2018.txt", "Reranking/Baseline_2018.txt",
                                                     "Reranking/Ernie/2018_ErnieRunWin_1_"+str(MaxFlag)+".txt",
                                                     "ErnieEncoding/Queries", alpha,
                                                     "ErnieEncoding/Candidates", MaxFlag, win)
    utils.rerankDocBasedOnQueryPara_WholeDocSlidedPara("Reranking/Queries2019.txt", "Reranking/Baseline_2019.txt",
                                                     "Reranking/Ernie/2019_ErnieRunWin_1_"+str(MaxFlag)+".txt",
                                                     "ErnieEncoding/Queries", alpha,
                                                     "ErnieEncoding/Candidates", MaxFlag, win)

    utils.rerankDocBasedOnQueryPara_WholeDocSlidedPara("Reranking/Queries2020.txt", "Reranking/Baseline_2020.txt",
                                                     "Reranking/Ernie/2020_ErnieRunWin_1_"+str(MaxFlag)+".txt",
                                                     "ErnieEncoding/Queries", alpha,
                                                     "ErnieEncoding/Candidates", MaxFlag, win)

    utils.rerankDocBasedOnQueryPara_WholeDocSlidedPara("Reranking/Queries2021.txt", "Reranking/Baseline_2021.txt",
                                                     "Reranking/Ernie/2021_ErnieRunWin_1_"+str(MaxFlag)+".txt",
                                                     "ErnieEncoding/Queries", alpha,
                                                     "ErnieEncoding/Candidates", MaxFlag, win)




    '''

