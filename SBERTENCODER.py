import utils
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from segtok.segmenter import split_single


#The following method encodes each paragraph of the query using SBERT to prepare for the matching process and stores each query encoding in the output dir.
def encodeQueriesSentenceBert_Paragraphs(qIDs, model, outputDir):
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    QueriesEncoding = {}
    for queryid in qIDs.keys():
        print("Processing Query", queryid)
        count = 1
        fwDoc = open(outputDir + "/" + str(queryid), "w")
        statement = "Select title from documents where id like'" + qIDs[queryid] + "'"
        result = utils.accessDatabase(statement)
        title = str(result[0][0]).strip()
        embedding = model.encode(title)
        fwDoc.write("<Paragraph>")
        for element in embedding:
            fwDoc.write(str(element) + " ")
        statement = "Select data from contents where did like'" + qIDs[queryid] + "' order by pos"
        paragraphs = utils.accessDatabase(statement)
        textToEncode = ""
        for para in paragraphs:
            p = str(para[0]).strip()
            pEncoding = []
            encoded_input = tokenizer(p, padding=False, truncation=False, return_tensors='pt')
            if (len(encoded_input[0]) > 128):
                # Split then encode then get mean of embeddings:
                sentences = list(split_single(p))
                sentencesEncoding = {}
                sNo = 0
                for s in sentences:
                    # print(len(s.split(" ")),flush=True)
                    embedding = model.encode(s)
                    sentencesEncoding[sNo] = embedding
                    sNo = sNo + 1
                pEncoding = np.mean(list(sentencesEncoding.values()), axis=0)
            else:
                pEncoding = model.encode(p)
            fwDoc.write("<Paragraph>")
            count = count + 1
            for element in pEncoding:
                fwDoc.write(str(element) + " ")
        print("Added ", count, "Paragraphs")
        fwDoc.close()
        # cont = input("continue?")
        # if(cont!=0):
        #    return
    print("Encoded all queries using Sentence Bert")
    return QueriesEncoding


def encodeCandidatesSentenceBert(cIDs, model, outputDir):
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    QueriesEncoding = {}
    for cindex in range(0,len(cIDs)):
        print("Processing Candidate", cIDs[cindex])
        paraEncodings={}
        paraindex=0
        fwDoc = open(outputDir + "/" + cIDs[cindex], "w")
        statement = "Select title from documents where id like'" + cIDs[cindex] + "'"
        result = utils.accessDatabase(statement)
        title = str(result[0][0]).strip()
        embedding = model.encode(title)
        paraEncodings[paraindex]=embedding
        paraindex=1
        statement = "Select data from contents where did like'" + cIDs[cindex] + "' order by pos"
        paragraphs = utils.accessDatabase(statement)
        textToEncode = ""
        for para in paragraphs:
            p = str(para[0]).strip()
            encoded_input = tokenizer(p, padding=False, truncation=False, return_tensors='pt')
            if (len(encoded_input[0]) > 128):
                # Split then encode then get mean of embeddings:
                sentences = list(split_single(p))
                sentencesEncoding = {}
                sNo = 0
                for s in sentences:
                    # print(len(s.split(" ")),flush=True)
                    embedding = model.encode(s)
                    sentencesEncoding[sNo] = embedding
                    sNo = sNo + 1
                pEncoding = np.mean(list(sentencesEncoding.values()), axis=0)
            else:
                pEncoding = model.encode(p)
            paraEncodings[paraindex] = embedding
            paraindex = paraindex+1
        candidateEncoding=np.mean(list(paraEncodings.values()), axis=0)
        for element in candidateEncoding:
            fwDoc.write(str(element) + " ")
        fwDoc.close()
    print("Encoded all candidates using Sentence Bert")
    return QueriesEncoding

def OutputScores_SBERT(queriesfile, runfile, resultfile, QueryDir,docEncodingDir, maxFlag,window):
    queriesIds = utils.getQueriesId(queriesfile)
    fw = open(resultfile, "w")
    qParaEncodings = {}
    qParaEncodings = loadQueryParagraphEncodings(queriesIds, QueryDir)

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
                    min = 100000000
                    max = 0
                    for k in queryDocsLexical.keys():
                        if (queryDocsLexical[k] > max):
                            max = queryDocsLexical[k]
                        if (queryDocsLexical[k] < min):
                            min = queryDocsLexical[k]
                    #print("lexical",queryDocsLexical)

                    NormLexical={}
                    for k in queryDocsSemantic.keys():
                        LexicalScore = (queryDocsLexical[k] - min) / (max - min)
                        NormLexical[k]=LexicalScore

                    for k in queryDocsSemantic.keys():
                        fw.write(Query + " " + k + " " + str(queryDocsLexical[k]) + " " + str(queryDocsSemantic[k]) + "\n")
                    queryDocsSemantic = {}
                    queryDocsLexical = {}
                    queryDocsFinalScore = {}

                Query = data[0]
            docID = data[2]
            docs[docID] = 0
            # try:
            docVector = utils.LoadDocumentEncodingFromFile(docEncodingDir + "/" + docID)
            if len(docVector) == 0:
                print("faulty document None", docID)
                queryDocsSemantic[data[2]] = 0.0
                queryDocsLexical[data[2]] = float(data[4])
                continue
            queryDocsSemantic[data[2]] = utils.getDocumentScoreFromQueryParaSlided(qParaEncodings[Query], docVector, maxFlag,
                                                                             window)
            queryDocsLexical[data[2]] = float(data[4])
    min = 100000000
    max = 0
    for k in queryDocsLexical.keys():
        if (queryDocsLexical[k] > max):
            max = queryDocsLexical[k]
        if (queryDocsLexical[k] < min):
            min = queryDocsLexical[k]

    NormLexical = {}
    for k in queryDocsSemantic.keys():
        LexicalScore = (queryDocsLexical[k] - min) / (max - min)
        NormLexical[k] = LexicalScore
    for k in queryDocsSemantic.keys():
        fw.write(Query + " " + k + " " + str(queryDocsLexical[k]) + " " + str(queryDocsSemantic[k]) + "\n")

#        fw.write(Query + " " + k + " " + str(NormLexical[k]) + " " + str(queryDocsSemantic[k]) + "\n")
    fw.close()
    print("Processed", len(docs), " Unique documents in File")


if __name__ == '__main__':

    # We generate a reranking for each year first for efficiency purposes,
    # then we combine the outputfile before running trec_eval on a qrel file that combines the qrels of all years.


    # The following is a call for encoding the queries
    '''
    model = SentenceTransformer('all-mpnet-base-v2')
    queriesIds = utils.getQueriesId("Reranking/Queries2018.txt")
    encodeQueriesSentenceBert_Paragraphs(queriesIds, model, "SBERTEncoding/Queries")
    queriesIds = utils.getQueriesId("Queries2019.txt")
    encodeQueriesSentenceBert_Paragraphs(queriesIds, model, "SBERTEncoding/Queries")
    queriesIds = utils.getQueriesId("Queries2020.txt")
    encodeQueriesSentenceBert_Paragraphs(queriesIds, model, "SBERTEncoding/Queries")
    queriesIds = utils.getQueriesId("Queries2021.txt")
    encodeQueriesSentenceBert_Paragraphs(queriesIds, model, "SBERTEncoding/Queries")
    '''
    # The following is a call for encoding candidates
    '''
    candidateIds = utils.loadCandidateIdsFromBaselineFile("Reranking/Baseline_2018.txt")
    encodeCandidatesSentenceBert(candidateIds, "SBERTEncoding/Candidates")
    candidateIds = utils.loadCandidateIdsFromBaselineFile("Reranking/Baseline_2019.txt")
    encodeCandidatesSentenceBert(candidateIds,  "SBERTEncoding/Candidates")
    candidateIds = utils.loadCandidateIdsFromBaselineFile("Reranking/Baseline_2020.txt")
    encodeCandidatesSentenceBert(candidateIds,  "SBERTEncoding/Candidates")
    candidateIds = utils.loadCandidateIdsFromBaselineFile("Reranking/Baseline_2021.txt")
    encodeCandidatesSentenceBert(candidateIds,  "SBERTEncoding/Candidates")
    '''

    # The following is a call for the reranking process
    '''
    win=1
    MaxFlag=False

    utils.rerankDocBasedOnQueryPara_WholeDocSlidedPara("Reranking/Queries2018.txt", "Reranking/Baseline_2018.txt",
                                                     "Reranking/SBert/2018_SBERTRunWin_1_"+str(MaxFlag)+".txt",
                                                     "SBERTEncoding/Queries", 1,
                                                     "SBERTEncoding/Candidates", MaxFlag, win)
    utils.rerankDocBasedOnQueryPara_WholeDocSlidedPara("Reranking/Queries2019.txt", "Reranking/Baseline_2019.txt",
                                                     "Reranking/SBert/2019_SBERTRunWin_1_"+str(MaxFlag)+".txt",
                                                     "SBERTEncoding/Queries", 1.0,
                                                     "SBERTEncoding/Candidates", MaxFlag, win)

    utils.rerankDocBasedOnQueryPara_WholeDocSlidedPara("Reranking/Queries2020.txt", "Reranking/Baseline_2020.txt",
                                                     "Reranking/SBert/2020_SBERTRunWin_1_"+str(MaxFlag)+".txt",
                                                     "SBERTEncoding/Queries", 1.0,
                                                     "SBERTEncoding/Candidates", MaxFlag, win)

    utils.rerankDocBasedOnQueryPara_WholeDocSlidedPara("Reranking/Queries2021.txt", "Reranking/Baseline_2021.txt",
                                                     "Reranking/SBert/2021_SBERTRunWin_1_"+str(MaxFlag)+".txt",
                                                     "SBERTEncoding/Queries", 1.0,
                                                     "SBERTEncoding/Candidates", MaxFlag, win)




    '''


    # For SBert, we store the semantic and lexical scores for further aggregation options as shown in the paper
    '''
	OutputScores_SBERT("../Reranking/Queries2018.txt", "../Reranking/Baseline_2018_1000.txt",
                                                 "../PaperExp/SBERT/2018Data.txt", "../SentPruning/Queries","../SentPruning/2018", MaxFlag, Win)
    print("Done 2018")

    OutputScores_SBERT("../Reranking/Queries2019.txt", "../Reranking/Baseline_2019_1000.txt","../PaperExp/SBERT/2019Data.txt",
                       "../SentPruning/Queries","../SentPruning/2019", MaxFlag, Win)
    print("Done 2019")

    OutputScores_SBERT("../Reranking/Queries2020.txt", "../Reranking/Baseline_2020_1000.txt",
                       "../PaperExp/SBERT/2020Data.txt", "../SentPruning/Queries", "../SentPruning/2020", MaxFlag, Win)
    print("Done 2020")

    OutputScores_SBERT("../Reranking/Queries2021.txt", "../Reranking/Baseline_2021_1000.txt",
                       "../PaperExp/SBERT/2021Data.txt", "../SentPruning/Queries", "../SentPruning/2021", MaxFlag, Win)
    print("Done 2021")
    '''

