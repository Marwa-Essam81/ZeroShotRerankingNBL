from transformers import LongformerModel, LongformerTokenizer,LongformerConfig

import pymysql
import numpy as np

def encodeTextLongFormer(model,tokenizer,text):
    pooler = lambda last_hidden, att_mask: (last_hidden * att_mask.unsqueeze(-1)).sum(1) / att_mask.sum(-1).unsqueeze(
        -1)
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True, max_length=4096)
    global_attention_mask = [1].extend([0] * encoded_input["input_ids"].shape[-1])
    encoded_input["global_attention_mask"] = global_attention_mask
    last_hidden = model(**encoded_input, output_hidden_states=True, return_dict=True).last_hidden_state
    CLSEmbedding = last_hidden[0][0].detach().cpu().numpy()
    PooledEmbedding = pooler(last_hidden, encoded_input["attention_mask"])[0].detach().cpu().numpy()

    return CLSEmbedding,PooledEmbedding

def encodeCandidatesWithLongFormer(cIDs, model, outputDir):

    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    tokenizer.model_max_length = 4096
    #tokenizer = tokenizer.from_pretrained('allenai/longformer-base-4096')
    #print("model loaded successfull")
    QueriesEncoding = {}
    for cindex in range(0,len(cIDs)):
        count = 1
        fwDoc = open(outputDir + "/" + cIDs[cindex], "w")
        statement = "Select title from documents where id like'" + cIDs[cindex] + "'"
        result = utils.accessDatabase(statement)
        title = str(result[0][0]).strip()
        textToEncode = title
        statement = "Select data from contents where did like'" + cIDs[cindex] + "' order by pos"
        paragraphs = utils.accessDatabase(statement)

        for para in paragraphs:
            p = str(para[0]).strip()
            textToEncode=textToEncode+" "+p
        clsembedding, pooledEmbedding = encodeTextLongFormer(model, tokenizer, textToEncode)
        for element in clsembedding:
            fwDoc.write(str(element) + " ")
        fwDoc.write("<pooled>")
        for element in pooledEmbedding:
            fwDoc.write(str(element) + " ")
        fwDoc.close()
    return QueriesEncoding

def loadQueryParagraphEncodings(qIds, QueryDir,CLS=True):
    QParaEncodings = {}
    for query in qIds.keys():
        QPNO = 1
        QParagrapgs = {}
        fq = open(QueryDir + "/" + query, encoding='utf-8')
        content = ""
        for line in fq:
            content = content + line
        paragraphs = content.split("<Passage>")
        for p in paragraphs:
            if (len(p.strip()) == 0):
                continue
            pcontent=p.replace("<pooled>","<Pooled>").split("<Pooled>")
            #print(p)
            if(CLS):
                p=pcontent[0].strip()
            else:
                p = pcontent[1].strip()
            qEncoding = []
            data = p.strip().split(" ")
            for d in data:
                qEncoding.append(np.float32(d))
            QParagrapgs[QPNO] = qEncoding
            QPNO = QPNO + 1
        QParaEncodings[query] = QParagrapgs
    print("loaded paragraph vectors for : ", len(QParaEncodings.keys()), " Queries")
    return QParaEncodings

def getDocumentScoreFromQueryParaSlided(QParagrapgs, DocEncoding, maxFlag, window):
    MaxScore = 0.0
    SumScore = 0.0
    content = ""
    paraScores = {}
    count = 1
    toggle = False
    chunk=0
    while (toggle == False):
        start = count
        end = count + window
        if (end >= len(QParagrapgs.keys())):
            end = len(QParagrapgs.keys())
            toggle = True
        chunk=chunk+1
        paraEncodings = {}
        for i in range(start, end + 1):
            paraEncodings[i] = QParagrapgs[i]
        #print("computing the score for ",len(paraEncodings),"paragraphs of the query")
        qEncoding = np.mean(list(paraEncodings.values()), axis=0)
        dot = np.dot(DocEncoding, qEncoding)
        normd = np.linalg.norm(DocEncoding)
        normq = np.linalg.norm(qEncoding)
        paraScore = dot / (normd * normq)
        paraScores[count] = paraScore
        count = count + 1
        if (paraScore > MaxScore):
            MaxScore = paraScore
        SumScore = SumScore + paraScore
        if (toggle):
            break
    SumScore = SumScore / chunk
    if maxFlag == True:
        return MaxScore
    else:
        # print("computing avg score")
        return SumScore

def loadDocEncodingFromCLSPooledEncodings(DocFile,CLS=True):
    fq = open(DocFile, encoding='utf-8')
    content = ""
    for line in fq:
        content = content + line
    clspooled = content.split("<Pooled>")
    if (CLS):
        content = clspooled[0].strip()
    else:
        content = clspooled[1].strip()
    qEncoding = []
    data = content.strip().split(" ")
    for d in data:
        qEncoding.append(np.float32(d))
    return qEncoding

def rerankDocBasedOnQueryPara_WholeDocSlidedPara(queriesfile,CLS, runfile, resultfile, QueryDir, alpha, docEncodingDir, maxFlag,
                                             window):
    queriesIds = utils.getQueriesId(queriesfile)
    # model = SentenceTransformer('all-mpnet-base-v2')
    fw = open(resultfile, "w")
    qParaEncodings = loadQueryParagraphEncodings(queriesIds, QueryDir,CLS)
    Query = "-1"
    queryDocsSemantic = {}
    queryDocsLexical = {}
    queryDocsFinalScore = {}
    docs = {}
    with open(runfile, encoding='utf-8') as f:
        for line in f:
            data = line.split(" ")
            if (data[0] != Query):
                if (Query != "-1"):  # already done with a query, write it to the output file
                    # Aggregating all lexical scores for a query to compute normalized score
                    #print("writing into output file results for query", Query)
                    min = 100000000
                    max = 0
                    for k in queryDocsLexical.keys():
                        if (queryDocsLexical[k] > max):
                            max = queryDocsLexical[k]
                        if (queryDocsLexical[k] < min):
                            min = queryDocsLexical[k]
                    for k in queryDocsSemantic.keys():
                        LexicalScore = (queryDocsLexical[k] - min) / (max - min)
                        queryDocsFinalScore[k] = ((1.0 - alpha) * LexicalScore) + (
                                alpha * queryDocsSemantic[k])  # applying the interpolation
                    queryDocsFinalScore = {k: v for k, v in
                                           sorted(queryDocsFinalScore.items(), key=lambda item: item[1],
                                                  reverse=True)}  # Now Sorting and printing
                    for k in queryDocsFinalScore.keys():
                        fw.write(Query + " Q0 " + k + " 0 " + str(queryDocsFinalScore[k]) + " DenseRunCS\n")
                    queryDocsSemantic = {}
                    queryDocsLexical = {}
                    queryDocsFinalScore = {}

                Query = data[0]
            docID = data[2]
            docs[docID] = 0
            # try:
            docVector = loadDocEncodingFromCLSPooledEncodings(
                docEncodingDir + "/" + docID,CLS)  # loading the encoding of the document from the encoding dir
            if len(docVector) == 0:
                print("faulty document None", docID)
                queryDocsSemantic[data[2]] = 0.0
                queryDocsLexical[data[2]] = float(data[4])
                continue
            queryDocsSemantic[data[2]] = getDocumentScoreFromQueryParaSlided(qParaEncodings[Query], docVector, maxFlag,
                                                                             window)
            queryDocsLexical[data[2]] = float(data[4])

    min = 100000000
    max = 0
    for k in queryDocsLexical.keys():
        if (queryDocsLexical[k] > max):
            max = queryDocsLexical[k]
        if (queryDocsLexical[k] < min):
            min = queryDocsLexical[k]

    for k in queryDocsSemantic.keys():
        LexicalScore = (queryDocsLexical[k] - min) / (max - min)
        queryDocsFinalScore[k] = ((1.0 - alpha) * LexicalScore) + (
                    alpha * queryDocsSemantic[k])  # applying the interpolation

    queryDocsFinalScore = {k: v for k, v in sorted(queryDocsFinalScore.items(), key=lambda item: item[1],
                                                   reverse=True)}  # Now Sorting and printing
    for k in queryDocsFinalScore.keys():
        fw.write(Query + " Q0 " + k + " 0 " + str(queryDocsFinalScore[k]) + " DenseRunCS\n")
    fw.close()
    print("Processed", len(docs), " Unique documents in File")

# In the following method, the sliding window method is used over the title and paragraph list of the query article to create passages
# as show in the paper , then passages are encoded directly using the model
def encodeQueriesWithLongformerSlided(qIDs, model, outputDir):
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    tokenizer.model_max_length = 4096
    #print("model loaded successfull")
    QueriesEncoding = {}
    for queryid in qIDs.keys():
        ParasList = []
        NoOfChunks = 0
        print("Processing Query", queryid)
        count = 1
        fwDoc = open(outputDir + "/" + str(queryid), "w")
        statement = "Select title from documents where id like'" + qIDs[queryid] + "'"
        result = accessDatabase(statement)
        title = str(result[0][0]).strip()
        ParasList.append(title)
        statement = "Select data from contents where did like'" + qIDs[queryid] + "' order by pos"
        paragraphs = accessDatabase(statement)
        textToEncode = ""
        for para in paragraphs:
            ParasList.append(para[0])
        toggle = False
        chunckcount = 0
        while (toggle == False):
            start = chunckcount
            end = chunckcount + 1
            if (end >= (len(ParasList)-1)):
                end = len(ParasList)-1
                toggle = True
            ChunkToEncode=""
            for i in range(start, end + 1):
                ChunkToEncode=ChunkToEncode+ ParasList[i]+" "
            NoOfChunks=NoOfChunks+1
            clsembedding, pooledEmbedding = encodeTextLongFormer(model, tokenizer, ChunkToEncode)
            fwDoc.write("<Passage>")
            for element in clsembedding:
                fwDoc.write(str(element) + " ")
            fwDoc.write("<pooled>")
            for element in pooledEmbedding:
                fwDoc.write(str(element) + " ")
            chunckcount=chunckcount+1
        print(queryid,len(ParasList),chunckcount)
        fwDoc.close()
    print("Encoded all queries using BigBird ",outputDir)
    return QueriesEncoding

import utils
if __name__ == '__main__':

    # First Encoding the query passages
    '''queriesIds = getQueriesId("Queries2018.txt")
    config = LongformerConfig.from_pretrained('allenai/longformer-base-4096')
    config.attention_mode = 'sliding_chunks'
    model = LongformerModel.from_pretrained('allenai/longformer-base-4096', config=config)
    encodeQueriesWithLongformerSlided(queriesIds, model, "LongFormerEncoding/Queries")
    queriesIds = getQueriesId("Queries2019.txt")
    encodeQueriesWithLongformerSlided(queriesIds, model, "LongFormerEncoding/Queries")
    queriesIds = getQueriesId("Queries2020.txt")
    encodeQueriesWithLongformerSlided(queriesIds, model, "LongFormerEncoding/Queries")
    queriesIds = getQueriesId("Queries2021.txt")
    encodeQueriesWithLongformerSlided(queriesIds, model, "LongFormerEncoding/Queries")'''

    #Next Encoding the candidates
    '''
     candidateIds = utils.loadCandidateIdsFromBaselineFile("Reranking/Baseline_2018.txt")
     encodeCandidatesWithLongFormer(candidateIds, "LongFormerEncoding/Candidates")
     candidateIds = utils.loadCandidateIdsFromBaselineFile("Reranking/Baseline_2019.txt")
     encodeCandidatesWithLongFormer(candidateIds,  "LongFormerEncoding/Candidates")
     candidateIds = utils.loadCandidateIdsFromBaselineFile("Reranking/Baseline_2020.txt")
     encodeCandidatesWithLongFormer(candidateIds,  "LongFormerEncoding/Candidates")
     candidateIds = utils.loadCandidateIdsFromBaselineFile("Reranking/Baseline_2021.txt")
     encodeCandidatesWithLongFormer(candidateIds,  "LongFormerEncoding/Candidates")
     '''

    # Finally rerank candidates
    '''
    MaxFlag=True
    win=0 The window is set to 0 here as the passages are already formed using the sliding window while encoding the queries
    
    rerankDocBasedOnQueryPara_WholeDocSlidedPara("Reranking/Queries2018.txt", "Reranking/Baseline_2018.txt",
                                                 "Reranking/LongFormer/2018_LFormerRunWin_1_" + str(MaxFlag) + ".txt",
                                                 "LongFormerEncoding/Queries", 1,
                                                 "LongFormerEncoding/Candidates", MaxFlag, win)
    
    rerankDocBasedOnQueryPara_WholeDocSlidedPara("Reranking/Queries2019.txt", "Reranking/Baseline_2019.txt",
                                                 "Reranking/LongFormer/2019_LFormerRunWin_1_" + str(MaxFlag) + ".txt",
                                                 "LongFormerEncoding/Queries", 1,
                                                 "LongFormerEncoding/Candidates", MaxFlag, win)
    
    rerankDocBasedOnQueryPara_WholeDocSlidedPara("Reranking/Queries2020.txt", "Reranking/Baseline_2020.txt",
                                                 "Reranking/LongFormer/2020_LFormerRunWin_1_" + str(MaxFlag) + ".txt",
                                                 "LongFormerEncoding/Queries", 1,
                                                 "LongFormerEncoding/Candidates", MaxFlag, win)
    
    rerankDocBasedOnQueryPara_WholeDocSlidedPara("Reranking/Queries2021.txt", "Reranking/Baseline_2021.txt",
                                                 "Reranking/LongFormer/2021_LFormerRunWin_1_" + str(MaxFlag) + ".txt",
                                                 "LongFormerEncoding/Queries", 1,
                                                 "LongFormerEncoding/Candidates", MaxFlag, win)
    '''

