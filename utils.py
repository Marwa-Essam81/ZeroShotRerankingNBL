import numpy as np
from segtok.segmenter import split_single
import pymysql
from sklearn.metrics.pairwise import cosine_similarity


def loadQueryParagraphEncodings(qIds, QueryDir,CLS=True):
    QParaEncodings = {}
    for query in qIds.keys():
        QPNO = 1
        QParagrapgs = {}
        fq = open(QueryDir + "/" + query, encoding='utf-8')
        content = ""
        for line in fq:
            content = content + line
        paragraphs = content.split("<Paragraph>")
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

# the following method scores candidate articles against the query passages (generated from sliding a window over paragraphs)
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

# the following method loads candidate article encoding from a file that has the encoding either from the CLS vector or from the average pooling of the article tokens
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

def accessDatabase(statement):
    # print("Executing statement",statement)
    mydb = pymysql.connect(
        host="localhost",
        user="root",
        password="123456789",
        db="WPostDBV4"
    )

    mycursor = mydb.cursor()

    mycursor.execute(statement)

    myresult = mycursor.fetchall()

    return myresult

def getQueriesId(queriesfile):
    queriesIds = {}
    with open(queriesfile, encoding='utf-8') as f:
        for line in f:
            data = line.split(",")
            queriesIds[data[1]] = data[0]
    return queriesIds

def loadCandidateIdsFromBaselineFile(baselineFile):
    CandidateIds=[]
    with open(baselineFile, encoding='utf-8') as f:
        for line in f:
            CandidateIds.append(line.split(" ")[2])

def LoadDocumentEncodingFromFile(docFile):
    dEncoding = []
    try:
        with open(docFile, encoding='utf-8') as f:
            for line in f:
                data = line.strip().split(" ")
                for d in data:
                    dEncoding.append(np.float32(d))
    except Exception as e:
        print(e)
        print("no file found for document", docFile)
        return dEncoding
    return dEncoding

#The reranking process of candidates taking the queries encoding file and the document encoding file.
def rerankDocBasedOnQueryPara_WholeDocSlidedPara(queriesfile, runfile, resultfile, QueryDir, alpha, docEncodingDir, maxFlag,
                                             window):
    queriesIds = getQueriesId(queriesfile)
    # model = SentenceTransformer('all-mpnet-base-v2')
    fw = open(resultfile, "w")
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
                    #print("writing into output file results for query", Query)
                    min = 100000000
                    max = 0
                    for k in queryDocsLexical.keys():
                        if (queryDocsLexical[k] > max):
                            max = queryDocsLexical[k]
                        if (queryDocsLexical[k] < min):
                            min = queryDocsLexical[k]
                    for k in queryDocsSemantic.keys():
                        #print(Query,queryDocsLexical[k],queryDocsSemantic[k])
                        LexicalScore = (queryDocsLexical[k] - min) / (max - min)
                        queryDocsFinalScore[k] = ((1.0 - alpha) * LexicalScore) + (
                                alpha * queryDocsSemantic[k])  # applying the interpolation
                    queryDocsFinalScore = {k: v for k, v in
                                           sorted(queryDocsFinalScore.items(), key=lambda item: item[1],
                                                  reverse=True)}  # Now Sorting and printing
                    print("-------------------------------------------")
                    for k in queryDocsFinalScore.keys():
                        fw.write(Query + " Q0 " + k + " 0 " + str(queryDocsFinalScore[k]) + " DenseRunCS\n")
                    queryDocsSemantic = {}
                    queryDocsLexical = {}
                    queryDocsFinalScore = {}

                Query = data[0]
            docID = data[2]
            docs[docID] = 0
            # try:
            docVector = LoadDocumentEncodingFromFile(
                docEncodingDir + "/" + docID)  # loading the encoding of the document from the encoding dir
            #print(Query,docVector)
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



# The following method calculates an integration between the lexical and the semantic relevance signals.
# The infile is a file that has the lexical and the semantic scores calculated for each candidate article formated as
# <query> space <doc> space <lexical_score> <semantic_score>

def interpolateRankings(baselinefile,secondrunfile,interpolatedfile,alpha):
    fw = open(interpolatedfile, "w")
    #1- loading document scores from second run file
    queriesDocsScore2nd={}
    f = open(secondrunfile, encoding='utf-8')
    for line in f.readlines():
        data=line.split(" ")
        query=data[0]
        doc = data[2]
        score=float(data[4])
        if queriesDocsScore2nd.keys().__contains__(query)!=True:
            queriesDocsScore2nd[query] = {}
        queriesDocsScore2nd[query][doc]=score
    #2- Reading the baseline file, normalize using 0,1 then interpolated
    #########
    Query = "-1"
    queryDocsLexical = {}
    queryDocsFinalScore = {}
    docs = {}
    with open(baselinefile, encoding='utf-8') as f:
        for line in f:
            data = line.split(" ")
            if (data[0] != Query):
                if (Query != "-1"):  # already done with a query, write it to the output file
                    # Aggregating all lexical scores for a query to compute normalized score
                    print("writing into output file results for query", Query)
                    min = 100000000
                    max = 0
                    for k in queryDocsLexical.keys():
                        if (queryDocsLexical[k] > max):
                            max = queryDocsLexical[k]
                        if (queryDocsLexical[k] < min):
                            min = queryDocsLexical[k]
                    for k in queriesDocsScore2nd[Query].keys():
                        LexicalScore = (queryDocsLexical[k] - min) / (max - min)
                        queryDocsFinalScore[k] = ((1.0 - alpha) * LexicalScore) + (
                                alpha * queriesDocsScore2nd[Query][k])  # applying the interpolation
                    queryDocsFinalScore = {k: v for k, v in
                                           sorted(queryDocsFinalScore.items(), key=lambda item: item[1],
                                                  reverse=True)}  # Now Sorting and printing
                    for k in queryDocsFinalScore.keys():
                        fw.write(Query + " Q0 " + k + " 0 " + str(queryDocsFinalScore[k]) + " DenseRunCS\n")
                    queryDocsLexical = {}
                    queryDocsFinalScore = {}

                Query = data[0]
            docID = data[2]
            queryDocsLexical[docID] = float(data[4])

    min = 100000000
    max = 0
    for k in queryDocsLexical.keys():
        if (queryDocsLexical[k] > max):
            max = queryDocsLexical[k]
        if (queryDocsLexical[k] < min):
            min = queryDocsLexical[k]

    for k in queriesDocsScore2nd[Query].keys():
        LexicalScore = (queryDocsLexical[k] - min) / (max - min)
        queryDocsFinalScore[k] = ((1.0 - alpha) * LexicalScore) + (
                alpha * queriesDocsScore2nd[Query][k])  # applying the interpolation

    queryDocsFinalScore = {k: v for k, v in sorted(queryDocsFinalScore.items(), key=lambda item: item[1],
                                                   reverse=True)}  # Now Sorting and printing
    for k in queryDocsFinalScore.keys():
        fw.write(Query + " Q0 " + k + " 0 " + str(queryDocsFinalScore[k]) + " DenseRunCS\n")
    fw.close()

def ReadAndInterpolate(infile,outfile):
    QueryLexicalScores={}
    QuerySemanticScores={}
    f=open(infile, encoding='utf-8')
    for line in f.readlines():
        try:
            data=line.split(" ")
            query=data[0]
            doc=data[1]
            lexicalScore=float(data[2])
            semanticScore=float(data[3])
            if(QueryLexicalScores.keys().__contains__(query)):
                QueryLexicalScores[query][doc]=lexicalScore
                QuerySemanticScores[query][doc] = semanticScore
            else:
                QueryLexicalScores[query]={}
                QuerySemanticScores[query]={}
                QueryLexicalScores[query][doc] = lexicalScore
                QuerySemanticScores[query][doc] = semanticScore
        except:
            print(line)
    alpha=0.0
    filename=0
    while(alpha<=1.0):
        fw = open(outfile+"_"+str(filename)+".txt", "w")
        for query in QueryLexicalScores.keys():
            queryfinalscores = {}
            for doc in QueryLexicalScores[query].keys():
                queryfinalscores[doc]= ((1.0 - alpha) * QueryLexicalScores[query][doc]) + (
                        alpha * QuerySemanticScores[query][doc])  # applying the interpolation
            queryfinalscores = {k: v for k, v in
                                       sorted(queryfinalscores.items(), key=lambda item: item[1],
                                              reverse=True)}  # Now Sorting and printing
            for k in queryfinalscores.keys():
                    fw.write(query + " Q0 " + k + " 0 " + str(queryfinalscores[k]) + " DenseRunCS\n")
        fw.close()
        alpha=round(alpha+0.1,1)
        filename=filename+10





    print("done")

def ReadAndInterpolateNorm(infile,outfile):
    QueryLexicalScores={}
    QuerySemanticScores={}
    f=open(infile, encoding='utf-8')
    for line in f.readlines():
        try:
            data=line.split(" ")
            query=data[0]
            doc=data[1]
            lexicalScore=float(data[2])
            semanticScore=float(data[3])
            if(QueryLexicalScores.keys().__contains__(query)):
                QueryLexicalScores[query][doc]=lexicalScore
                QuerySemanticScores[query][doc] = semanticScore
            else:
                QueryLexicalScores[query]={}
                QuerySemanticScores[query]={}
                QueryLexicalScores[query][doc] = lexicalScore
                QuerySemanticScores[query][doc] = semanticScore
        except:
            print(line)
    alpha=0.0
    filename=0
    while(alpha<=1.0):
        fw = open(outfile+"_"+str(filename)+".txt", "w")

        for query in QueryLexicalScores.keys():
            maximum=0
            min=10000
            for doc in QuerySemanticScores[query].keys():
                if(QuerySemanticScores[query][doc]>maximum):
                    maximum=QuerySemanticScores[query][doc]
                if(QuerySemanticScores[query][doc]<min):
                    min=QuerySemanticScores[query][doc]


            queryfinalscores = {}
            for doc in QueryLexicalScores[query].keys():
                normsemanticScore=(QuerySemanticScores[query][doc]-min)/(maximum-min)
                queryfinalscores[doc]= ((1.0 - alpha) * QueryLexicalScores[query][doc]) + (
                        alpha * normsemanticScore)  # applying the interpolation
            queryfinalscores = {k: v for k, v in
                                       sorted(queryfinalscores.items(), key=lambda item: item[1],
                                              reverse=True)}  # Now Sorting and printing
            #if (query.__eq__("321")):
            #    print(query, queryfinalscores)
            #    print(maximum, min)
            for k in queryfinalscores.keys():
                    fw.write(query + " Q0 " + k + " 0 " + str(queryfinalscores[k]) + " DenseRunCS\n")
        fw.close()
        alpha=round(alpha+0.1,1)
        filename=filename+10





    print("done")

def ReadAndInterpolateNormProduct(infile,outfile):
    QueryLexicalScores={}
    QuerySemanticScores={}
    f=open(infile, encoding='utf-8')
    for line in f.readlines():
        try:
            data=line.split(" ")
            query=data[0]
            doc=data[1]
            lexicalScore=float(data[2])
            semanticScore=float(data[3])
            if(QueryLexicalScores.keys().__contains__(query)):
                QueryLexicalScores[query][doc]=lexicalScore
                QuerySemanticScores[query][doc] = semanticScore
            else:
                QueryLexicalScores[query]={}
                QuerySemanticScores[query]={}
                QueryLexicalScores[query][doc] = lexicalScore
                QuerySemanticScores[query][doc] = semanticScore
        except:
            print(line)
    alpha=1.0
    filename=0
    while(alpha<=1.0):
        fw = open(outfile+"_"+str(filename)+".txt", "w")

        for query in QueryLexicalScores.keys():
            maximum=0
            min=10000
            for doc in QuerySemanticScores[query].keys():
                if(QuerySemanticScores[query][doc]>maximum):
                    maximum=QuerySemanticScores[query][doc]
                if(QuerySemanticScores[query][doc]<min):
                    min=QuerySemanticScores[query][doc]

            queryfinalscores = {}
            for doc in QueryLexicalScores[query].keys():
                normsemanticScore=(QuerySemanticScores[query][doc]-min)/(maximum-min)
                queryfinalscores[doc]=(normsemanticScore*QueryLexicalScores[query][doc])
                #queryfinalscores[doc]= ((1.0 - alpha) * QueryLexicalScores[query][doc]) + (
                #        alpha * normsemanticScore)  # applying the interpolation
            queryfinalscores = {k: v for k, v in
                                       sorted(queryfinalscores.items(), key=lambda item: item[1],
                                              reverse=True)}  # Now Sorting and printing
            #if (query.__eq__("321")):
            #    print(query, queryfinalscores)
            #    print(maximum, min)
            for k in queryfinalscores.keys():
                    fw.write(query + " Q0 " + k + " 0 " + str(queryfinalscores[k]) + " DenseRunCS\n")
        fw.close()
        alpha=round(alpha+0.1,1)
        filename=filename+10





    print("done")

def ReadAndInterpolateWeightedRanks(infile,outfile):
    QueryLexicalScores={}
    QuerySemanticScores={}
    f=open(infile, encoding='utf-8')
    for line in f.readlines():
        try:
            data=line.split(" ")
            query=data[0]
            doc=data[1]
            lexicalScore=float(data[2])
            semanticScore=float(data[3])
            if(QueryLexicalScores.keys().__contains__(query)):
                QueryLexicalScores[query][doc]=lexicalScore
                QuerySemanticScores[query][doc] = semanticScore
            else:
                QueryLexicalScores[query]={}
                QuerySemanticScores[query]={}
                QueryLexicalScores[query][doc] = lexicalScore
                QuerySemanticScores[query][doc] = semanticScore
        except:
            print(line)
    alpha=0.0
    filename=0
    while(alpha<=1.0):
        fw = open(outfile+"_"+str(filename)+".txt", "w")

        for query in QueryLexicalScores.keys():

            SemanticDocsSorted = {k: v for k, v in
                                sorted(QuerySemanticScores[query].items(), key=lambda item: item[1],
                                       reverse=True)}  # Now Sorting and printing
            SemanticDocsRank={}
            rank=len(SemanticDocsSorted.keys())
            for k in SemanticDocsSorted.keys():
                SemanticDocsRank[k]=rank
                rank=rank-1

            LexicalDocsSorted = {k: v for k, v in
                                sorted(QueryLexicalScores[query].items(), key=lambda item: item[1],
                                       reverse=True)}  # Now Sorting and printing
            LexicalDocsRank = {}
            rank = len(LexicalDocsSorted.keys())
            for k in LexicalDocsSorted.keys():
                LexicalDocsRank[k] = rank
                rank = rank - 1

            queryfinalscores = {}
            for doc in QueryLexicalScores[query].keys():

                queryfinalscores[doc]= ((1.0 - alpha) * LexicalDocsRank[doc]) + (
                        alpha * SemanticDocsRank[doc])  # applying the interpolation
            queryfinalscores = {k: v for k, v in
                                       sorted(queryfinalscores.items(), key=lambda item: item[1],
                                              reverse=True)}  # Now Sorting and printing
            #if (query.__eq__("321")):
            #    print(query, queryfinalscores)
            #    print(maximum, min)
            for k in queryfinalscores.keys():
                    fw.write(query + " Q0 " + k + " 0 " + str(queryfinalscores[k]) + " DenseRunCS\n")
        fw.close()
        alpha=round(alpha+0.1,1)
        filename=filename+10





    print("done")

def ReadAndInterpolateBordaRanks(infile,outfile):
    QueryLexicalScores={}
    QuerySemanticScores={}
    f=open(infile, encoding='utf-8')
    for line in f.readlines():
        try:
            data=line.split(" ")
            query=data[0]
            doc=data[1]
            lexicalScore=float(data[2])
            semanticScore=float(data[3])
            if(QueryLexicalScores.keys().__contains__(query)):
                QueryLexicalScores[query][doc]=lexicalScore
                QuerySemanticScores[query][doc] = semanticScore
            else:
                QueryLexicalScores[query]={}
                QuerySemanticScores[query]={}
                QueryLexicalScores[query][doc] = lexicalScore
                QuerySemanticScores[query][doc] = semanticScore
        except:
            print(line)
    alpha=0.0
    filename=0
    while(alpha<=1.0):
        fw = open(outfile+"_"+str(filename)+".txt", "w")

        for query in QueryLexicalScores.keys():

            SemanticDocsSorted = {k: v for k, v in
                                sorted(QuerySemanticScores[query].items(), key=lambda item: item[1],
                                       reverse=True)}  # Now Sorting and printing
            SemanticDocsRank={}
            rank=1
            for k in SemanticDocsSorted.keys():
                SemanticDocsRank[k]=rank
                rank=rank+1
            for k in SemanticDocsRank.keys():
                SemanticDocsRank[k]=1-((SemanticDocsRank[k]-1)/100)

            LexicalDocsSorted = {k: v for k, v in
                                sorted(QueryLexicalScores[query].items(), key=lambda item: item[1],
                                       reverse=True)}  # Now Sorting and printing
            LexicalDocsRank = {}
            rank = 1
            for k in LexicalDocsSorted.keys():
                LexicalDocsRank[k] = rank
                rank = rank + 1
            for k in LexicalDocsSorted.keys():
                LexicalDocsRank[k]=1-((LexicalDocsRank[k]-1)/100)

            queryfinalscores = {}
            for doc in QueryLexicalScores[query].keys():
                NormSemanticRank=SemanticDocsRank[doc]
                NormLexicalRank=LexicalDocsRank[doc]
                queryfinalscores[doc]= ((1.0 - alpha) * NormLexicalRank) + (
                        alpha * NormSemanticRank)  # applying the interpolation
            queryfinalscores = {k: v for k, v in
                                       sorted(queryfinalscores.items(), key=lambda item: item[1],
                                              reverse=True)}  # Now Sorting and printing
            #if (query.__eq__("321")):
            #    print(query, queryfinalscores)
            #    print(maximum, min)
            for k in queryfinalscores.keys():
                    fw.write(query + " Q0 " + k + " 0 " + str(queryfinalscores[k]) + " DenseRunCS\n")
        fw.close()
        alpha=round(alpha+0.1,1)
        filename=filename+10





    print("done")

def ReadAndInterpolateDowdallRanks(infile,outfile):
    QueryLexicalScores={}
    QuerySemanticScores={}
    f=open(infile, encoding='utf-8')
    for line in f.readlines():
        try:
            data=line.split(" ")
            query=data[0]
            doc=data[1]
            lexicalScore=float(data[2])
            semanticScore=float(data[3])
            if(QueryLexicalScores.keys().__contains__(query)):
                QueryLexicalScores[query][doc]=lexicalScore
                QuerySemanticScores[query][doc] = semanticScore
            else:
                QueryLexicalScores[query]={}
                QuerySemanticScores[query]={}
                QueryLexicalScores[query][doc] = lexicalScore
                QuerySemanticScores[query][doc] = semanticScore
        except:
            print(line)
    alpha=0.0
    filename=0
    while(alpha<=1.0):
        fw = open(outfile+"_"+str(filename)+".txt", "w")

        for query in QueryLexicalScores.keys():

            SemanticDocsSorted = {k: v for k, v in
                                sorted(QuerySemanticScores[query].items(), key=lambda item: item[1],
                                       reverse=True)}  # Now Sorting and printing
            SemanticDocsRank={}
            rank=1
            for k in SemanticDocsSorted.keys():
                SemanticDocsRank[k]=rank
                rank=rank+1

            LexicalDocsSorted = {k: v for k, v in
                                sorted(QueryLexicalScores[query].items(), key=lambda item: item[1],
                                       reverse=True)}  # Now Sorting and printing
            LexicalDocsRank = {}
            rank = 1
            for k in LexicalDocsSorted.keys():
                LexicalDocsRank[k] = rank
                rank = rank + 1

            queryfinalscores = {}
            for doc in QueryLexicalScores[query].keys():
                NormSemanticRank=1/SemanticDocsRank[doc]
                NormLexicalRank=1/LexicalDocsRank[doc]
                queryfinalscores[doc]= ((1.0 - alpha) * NormLexicalRank) + (
                        alpha * NormSemanticRank)  # applying the interpolation
            queryfinalscores = {k: v for k, v in
                                       sorted(queryfinalscores.items(), key=lambda item: item[1],
                                              reverse=True)}  # Now Sorting and printing
            #if (query.__eq__("321")):
            #    print(query, queryfinalscores)
            #    print(maximum, min)
            for k in queryfinalscores.keys():
                    fw.write(query + " Q0 " + k + " 0 " + str(queryfinalscores[k]) + " DenseRunCS\n")
        fw.close()
        alpha=round(alpha+0.1,1)
        filename=filename+10





    print("done")

def ReadAndInterpolateRankAggregation(infile,outfile):
    QueryLexicalScores={}
    QuerySemanticScores={}
    f=open(infile, encoding='utf-8')
    for line in f.readlines():
        try:
            data=line.split(" ")
            query=data[0]
            doc=data[1]
            lexicalScore=float(data[2])
            semanticScore=float(data[3])
            if(QueryLexicalScores.keys().__contains__(query)):
                QueryLexicalScores[query][doc]=lexicalScore
                QuerySemanticScores[query][doc] = semanticScore
            else:
                QueryLexicalScores[query]={}
                QuerySemanticScores[query]={}
                QueryLexicalScores[query][doc] = lexicalScore
                QuerySemanticScores[query][doc] = semanticScore
        except:
            print(line)
    alpha=1.0
    filename=0
    while(alpha<=1.0):
        fw = open(outfile+"_"+str(filename)+".txt", "w")

        for query in QueryLexicalScores.keys():

            SemanticDocsSorted = {k: v for k, v in
                                sorted(QuerySemanticScores[query].items(), key=lambda item: item[1],
                                       reverse=True)}  # Now Sorting and printing
            SemanticDocsRank={}
            rank=len(SemanticDocsSorted.keys())-1
            for k in SemanticDocsSorted.keys():
                SemanticDocsRank[k]=rank
                rank=rank-1

            LexicalDocsSorted = {k: v for k, v in
                                sorted(QueryLexicalScores[query].items(), key=lambda item: item[1],
                                       reverse=True)}  # Now Sorting and printing
            LexicalDocsRank = {}
            rank = len(LexicalDocsSorted.keys())-1
            for k in LexicalDocsSorted.keys():
                LexicalDocsRank[k] = rank
                rank = rank - 1

            queryfinalscores = {}
            for doc in QueryLexicalScores[query].keys():

                queryfinalscores[doc]= (LexicalDocsRank[doc]) + (SemanticDocsRank[doc])  # applying the interpolation
            queryfinalscores = {k: v for k, v in
                                       sorted(queryfinalscores.items(), key=lambda item: item[1],
                                              reverse=True)}  # Now Sorting and printing
            #if (query.__eq__("321")):
            #    print(query, queryfinalscores)
            #    print(maximum, min)
            for k in queryfinalscores.keys():
                    fw.write(query + " Q0 " + k + " 0 " + str(queryfinalscores[k]) + " DenseRunCS\n")
        fw.close()
        alpha=round(alpha+0.1,1)
        filename=filename+10





    print("done")

def ReadAndInterpolateSumNorm(infile,outfile):
    QueryLexicalScores={}
    QuerySemanticScores={}
    f=open(infile, encoding='utf-8')
    for line in f.readlines():
        try:
            data=line.split(" ")
            query=data[0]
            doc=data[1]
            lexicalScore=float(data[2])
            semanticScore=float(data[3])
            if(QueryLexicalScores.keys().__contains__(query)):
                QueryLexicalScores[query][doc]=lexicalScore
                QuerySemanticScores[query][doc] = semanticScore
            else:
                QueryLexicalScores[query]={}
                QuerySemanticScores[query]={}
                QueryLexicalScores[query][doc] = lexicalScore
                QuerySemanticScores[query][doc] = semanticScore
        except:
            print(line)
    alpha=0.0
    filename=0
    while(alpha<=1.0):
        fw = open(outfile+"_"+str(filename)+".txt", "w")

        for query in QueryLexicalScores.keys():
            SumSemantic=0
            SumLexical=0
            for doc in QuerySemanticScores[query].keys():
                SumSemantic=SumSemantic+QuerySemanticScores[query][doc]
                SumLexical=SumLexical+QueryLexicalScores[query][doc]



            queryfinalscores = {}
            for doc in QueryLexicalScores[query].keys():
                normsemanticScore=QuerySemanticScores[query][doc]/SumSemantic
                normlexicalScore = QueryLexicalScores[query][doc] / SumLexical

                queryfinalscores[doc]= ((1.0 - alpha) * normlexicalScore) + (
                        alpha * normsemanticScore)  # applying the interpolation
            queryfinalscores = {k: v for k, v in
                                       sorted(queryfinalscores.items(), key=lambda item: item[1],
                                              reverse=True)}  # Now Sorting and printing
            #if (query.__eq__("321")):
            #    print(query, queryfinalscores)
            #    print(maximum, min)
            for k in queryfinalscores.keys():
                    fw.write(query + " Q0 " + k + " 0 " + str(queryfinalscores[k]) + " DenseRunCS\n")
        fw.close()
        alpha=round(alpha+0.1,1)
        filename=filename+10





    print("done")

def ReadAndInterpolateSumNormThenMax(infile,outfile):
    QueryLexicalScores={}
    QuerySemanticScores={}
    f=open(infile, encoding='utf-8')
    for line in f.readlines():
        try:
            data=line.split(" ")
            query=data[0]
            doc=data[1]
            lexicalScore=float(data[2])
            semanticScore=float(data[3])
            if(QueryLexicalScores.keys().__contains__(query)):
                QueryLexicalScores[query][doc]=lexicalScore
                QuerySemanticScores[query][doc] = semanticScore
            else:
                QueryLexicalScores[query]={}
                QuerySemanticScores[query]={}
                QueryLexicalScores[query][doc] = lexicalScore
                QuerySemanticScores[query][doc] = semanticScore
        except:
            print(line)
    alpha=0.5
    filename=0
    while(alpha<=0.5):
        fw = open(outfile+"_"+str(filename)+".txt", "w")

        for query in QueryLexicalScores.keys():
            SumSemantic=0
            SumLexical=0
            for doc in QuerySemanticScores[query].keys():
                SumSemantic=SumSemantic+QuerySemanticScores[query][doc]
                SumLexical=SumLexical+QueryLexicalScores[query][doc]



            queryfinalscores = {}
            for doc in QueryLexicalScores[query].keys():
                normsemanticScore=QuerySemanticScores[query][doc]/SumSemantic
                normlexicalScore = QueryLexicalScores[query][doc] / SumLexical
                queryfinalscores[doc]=max(normsemanticScore,normlexicalScore)
                #queryfinalscores[doc]= ((1.0 - alpha) * normlexicalScore) + (
                #        alpha * normsemanticScore)  # applying the interpolation
            queryfinalscores = {k: v for k, v in
                                       sorted(queryfinalscores.items(), key=lambda item: item[1],
                                              reverse=True)}  # Now Sorting and printing
            #if (query.__eq__("321")):
            #    print(query, queryfinalscores)
            #    print(maximum, min)
            for k in queryfinalscores.keys():
                    fw.write(query + " Q0 " + k + " 0 " + str(queryfinalscores[k]) + " DenseRunCS\n")
        fw.close()
        alpha=round(alpha+0.1,1)
        filename=filename+10





    print("done")

def ReadAndInterpolateSumNormThenMin(infile,outfile):
    QueryLexicalScores={}
    QuerySemanticScores={}
    f=open(infile, encoding='utf-8')
    for line in f.readlines():
        try:
            data=line.split(" ")
            query=data[0]
            doc=data[1]
            lexicalScore=float(data[2])
            semanticScore=float(data[3])
            if(QueryLexicalScores.keys().__contains__(query)):
                QueryLexicalScores[query][doc]=lexicalScore
                QuerySemanticScores[query][doc] = semanticScore
            else:
                QueryLexicalScores[query]={}
                QuerySemanticScores[query]={}
                QueryLexicalScores[query][doc] = lexicalScore
                QuerySemanticScores[query][doc] = semanticScore
        except:
            print(line)
    alpha=0.5
    filename=0
    while(alpha<=0.5):
        fw = open(outfile+"_"+str(filename)+".txt", "w")

        for query in QueryLexicalScores.keys():
            SumSemantic=0
            SumLexical=0
            for doc in QuerySemanticScores[query].keys():
                SumSemantic=SumSemantic+QuerySemanticScores[query][doc]
                SumLexical=SumLexical+QueryLexicalScores[query][doc]



            queryfinalscores = {}
            for doc in QueryLexicalScores[query].keys():
                normsemanticScore=QuerySemanticScores[query][doc]/SumSemantic
                normlexicalScore = QueryLexicalScores[query][doc] / SumLexical
                queryfinalscores[doc]=min(normsemanticScore,normlexicalScore)
                #queryfinalscores[doc]= ((1.0 - alpha) * normlexicalScore) + (
                #        alpha * normsemanticScore)  # applying the interpolation
            queryfinalscores = {k: v for k, v in
                                       sorted(queryfinalscores.items(), key=lambda item: item[1],
                                              reverse=True)}  # Now Sorting and printing
            #if (query.__eq__("321")):
            #    print(query, queryfinalscores)
            #    print(maximum, min)
            for k in queryfinalscores.keys():
                    fw.write(query + " Q0 " + k + " 0 " + str(queryfinalscores[k]) + " DenseRunCS\n")
        fw.close()
        alpha=round(alpha+0.1,1)
        filename=filename+10





    print("done")

def ReadAndInterpolateSumNormThenMinMax(infile,outfile):
    QueryLexicalScores={}
    QuerySemanticScores={}
    f=open(infile, encoding='utf-8')
    for line in f.readlines():
        try:
            data=line.split(" ")
            query=data[0]
            doc=data[1]
            lexicalScore=float(data[2])
            semanticScore=float(data[3])
            if(QueryLexicalScores.keys().__contains__(query)):
                QueryLexicalScores[query][doc]=lexicalScore
                QuerySemanticScores[query][doc] = semanticScore
            else:
                QueryLexicalScores[query]={}
                QuerySemanticScores[query]={}
                QueryLexicalScores[query][doc] = lexicalScore
                QuerySemanticScores[query][doc] = semanticScore
        except:
            print(line)
    alpha=0.5
    filename=0
    while(alpha<=0.5):
        fw = open(outfile+"_"+str(filename)+".txt", "w")

        for query in QueryLexicalScores.keys():
            SumSemantic=0
            SumLexical=0
            for doc in QuerySemanticScores[query].keys():
                SumSemantic=SumSemantic+QuerySemanticScores[query][doc]
                SumLexical=SumLexical+QueryLexicalScores[query][doc]



            queryfinalscores = {}
            for doc in QueryLexicalScores[query].keys():
                normsemanticScore=QuerySemanticScores[query][doc]/SumSemantic
                normlexicalScore = QueryLexicalScores[query][doc] / SumLexical

                DocMax=max(normsemanticScore,normlexicalScore)
                DocMin = min(normsemanticScore, normlexicalScore)
                queryfinalscores[doc]=DocMax+((DocMin*DocMin)/(DocMax+DocMin))
                #queryfinalscores[doc]= ((1.0 - alpha) * normlexicalScore) + (
                #        alpha * normsemanticScore)  # applying the interpolation
            queryfinalscores = {k: v for k, v in
                                       sorted(queryfinalscores.items(), key=lambda item: item[1],
                                              reverse=True)}  # Now Sorting and printing
            #if (query.__eq__("321")):
            #    print(query, queryfinalscores)
            #    print(maximum, min)
            for k in queryfinalscores.keys():
                    fw.write(query + " Q0 " + k + " 0 " + str(queryfinalscores[k]) + " DenseRunCS\n")
        fw.close()
        alpha=round(alpha+0.1,1)
        filename=filename+10





    print("done")

def ReadAndInterpolateSumNormThenProduct(infile,outfile):
    QueryLexicalScores={}
    QuerySemanticScores={}
    f=open(infile, encoding='utf-8')
    for line in f.readlines():
        try:
            data=line.split(" ")
            query=data[0]
            doc=data[1]
            lexicalScore=float(data[2])
            semanticScore=float(data[3])
            if(QueryLexicalScores.keys().__contains__(query)):
                QueryLexicalScores[query][doc]=lexicalScore
                QuerySemanticScores[query][doc] = semanticScore
            else:
                QueryLexicalScores[query]={}
                QuerySemanticScores[query]={}
                QueryLexicalScores[query][doc] = lexicalScore
                QuerySemanticScores[query][doc] = semanticScore
        except:
            print(line)
    alpha=0.5
    filename=0
    while(alpha<=0.5):
        fw = open(outfile+"_"+str(filename)+".txt", "w")

        for query in QueryLexicalScores.keys():
            SumSemantic=0
            SumLexical=0
            for doc in QuerySemanticScores[query].keys():
                SumSemantic=SumSemantic+QuerySemanticScores[query][doc]
                SumLexical=SumLexical+QueryLexicalScores[query][doc]



            queryfinalscores = {}
            for doc in QueryLexicalScores[query].keys():
                normsemanticScore=QuerySemanticScores[query][doc]/SumSemantic
                normlexicalScore = QueryLexicalScores[query][doc] / SumLexical
                queryfinalscores[doc]=normsemanticScore*normlexicalScore
                #queryfinalscores[doc]= ((1.0 - alpha) * normlexicalScore) + (
                #        alpha * normsemanticScore)  # applying the interpolation
            queryfinalscores = {k: v for k, v in
                                       sorted(queryfinalscores.items(), key=lambda item: item[1],
                                              reverse=True)}  # Now Sorting and printing
            #if (query.__eq__("321")):
            #    print(query, queryfinalscores)
            #    print(maximum, min)
            for k in queryfinalscores.keys():
                    fw.write(query + " Q0 " + k + " 0 " + str(queryfinalscores[k]) + " DenseRunCS\n")
        fw.close()
        alpha=round(alpha+0.1,1)
        filename=filename+10





    print("done")

def getMeanAndVariance(values):
    mean = sum(values) / len(values)
    differences = [(value - mean) ** 2 for value in values]
    sum_of_differences = sum(differences)
    standard_deviation = (sum_of_differences / (len(values) - 1)) ** 0.5
    return mean, standard_deviation

def ReadAndInterpolateZNorm(infile,outfile):
    QueryLexicalScores={}
    QuerySemanticScores={}
    f=open(infile, encoding='utf-8')
    for line in f.readlines():
        try:
            data=line.split(" ")
            query=data[0]
            doc=data[1]
            lexicalScore=float(data[2])
            semanticScore=float(data[3])
            if(QueryLexicalScores.keys().__contains__(query)):
                QueryLexicalScores[query][doc]=lexicalScore
                QuerySemanticScores[query][doc] = semanticScore
            else:
                QueryLexicalScores[query]={}
                QuerySemanticScores[query]={}
                QueryLexicalScores[query][doc] = lexicalScore
                QuerySemanticScores[query][doc] = semanticScore
        except:
            print(line)
    alpha=0.0
    filename=0
    while(alpha<=1.0):
        fw = open(outfile+"_"+str(filename)+".txt", "w")

        for query in QueryLexicalScores.keys():
            SumSemantic=0
            SumLexical=0
            SemanticValues=[]
            for doc in QuerySemanticScores[query].keys():
                SemanticValues.append(QuerySemanticScores[query][doc])
            LexicalValues = []
            for doc in QueryLexicalScores[query].keys():
                LexicalValues.append(QueryLexicalScores[query][doc])


            SemanticMean,SemanticVariance=getMeanAndVariance(SemanticValues)
            LexicalMean, LexicalVariance = getMeanAndVariance(LexicalValues)

            queryfinalscores = {}
            for doc in QueryLexicalScores[query].keys():
                normsemanticScore=(QuerySemanticScores[query][doc]-SemanticMean)/SemanticVariance
                normlexicalScore = (QueryLexicalScores[query][doc]-LexicalMean) / LexicalVariance
                #print(normsemanticScore, normlexicalScore)
                queryfinalscores[doc]= ((1.0 - alpha) * normlexicalScore) + (
                        alpha * normsemanticScore)  # applying the interpolation
            queryfinalscores = {k: v for k, v in
                                       sorted(queryfinalscores.items(), key=lambda item: item[1],
                                              reverse=True)}  # Now Sorting and printing
            #if (query.__eq__("321")):
            #    print(query, queryfinalscores)
            #    print(maximum, min)
            score=len(queryfinalscores.keys())
            for k in queryfinalscores.keys():
                fw.write(query + " Q0 " + k + " 0 " + str(queryfinalscores[k]) + " DenseRunCS\n")
                score=score-1
        fw.close()
        alpha=round(alpha+0.1,1)
        filename=filename+10





    print("done")


if __name__ == '__main__':
    print("done")

    #print(getDocumentEncodingWithSentPruning_Title("SBERTSentEncoding/321",0.5,True))
