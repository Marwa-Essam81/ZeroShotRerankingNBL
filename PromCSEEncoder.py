import torch
from segtok.segmenter import split_single
from transformers import AutoTokenizer
import numpy as np
import pymysql
from ENCODERS.PromCSEModel import RobertaForCL
from transformers import AutoConfig
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

import utils
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

# the text (ex: a paragraph) is split into sentences first before encoding using the PromCSE model
def encodeTextPromCSEBERT(model,tokenizer,text):
    sentences = list(split_single(text))
    sentencesEncoding = {}
    sNo = 0
    for s in sentences:
        encoded_input = tokenizer(s, padding=True, return_tensors='pt')
        outputs = model(**encoded_input, output_hidden_states=True, return_dict=True, sent_emb=True,
                        output_attentions=True)  # .last_hidden_state
        Sentembedding = outputs.pooler_output
        sentencesEncoding[sNo] = Sentembedding[0].detach().cpu().numpy()
        sNo = sNo + 1
    pEncoding = np.mean(list(sentencesEncoding.values()), axis=0)
    return pEncoding


import argparse

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str,
            help="Transformers' model name or path")
    parser.add_argument("--pooler_type", type=str,
            choices=['cls', 'cls_before_pooler', 'avg', 'avg_top2', 'avg_first_last'],
            default='cls',
            help="Which pooler to use")
    parser.add_argument("--temp", type=float,
            default=0.05,
            help="Temperature for softmax.")
    parser.add_argument("--hard_negative_weight", type=float,
            default=0.0,
            help="The **logit** of weight for hard negatives (only effective if hard negatives are used).")
    parser.add_argument("--do_mlm", action='store_true',
            help="Whether to use MLM auxiliary objective.")
    parser.add_argument("--mlm_weight", type=float,
            default=0.1,
            help="Weight for MLM auxiliary objective (only effective if --do_mlm).")
    parser.add_argument("--mlp_only_train", action='store_true',
            help="Use MLP only during training")
    parser.add_argument("--pre_seq_len", type=int,
            default=10,
            help="The length of prompt")
    parser.add_argument("--prefix_projection", action='store_true',
            help="Apply a two-layer MLP head over the prefix embeddings")
    parser.add_argument("--prefix_hidden_size", type=int,
                        default=512,
                        help="The hidden size of the MLP projection head in Prefix Encoder if prefix projection is used")
    parser.add_argument("--do_eh_loss",
                        action='store_true',
                        help="Whether to add Energy-based Hinge loss")
    parser.add_argument("--eh_loss_margin", type=float,
                        default=None,
                        help="The margin of Energy-based Hinge loss")
    parser.add_argument("--eh_loss_weight", type=float,
                        default=None,
                        help="The weight of Energy-based Hinge loss")
    parser.add_argument("--cache_dir", type=str,
                        default=None,
                        help="Where do you want to store the pretrained models downloaded from huggingface.co")
    parser.add_argument("--model_revision", type=str,
                        default="main",
                        help="The specific model version to use (can be a branch name, tag name or commit id).")
    parser.add_argument("--use_auth_token", action='store_true',
                        help="Will use the token generated when running `transformers-cli login` (necessary to use this script "
                             "with private models).")
    parser.add_argument("--mode", type=str,
                        choices=['dev', 'test', 'fasttest'],
                        default='test',
                        help="What evaluation mode to use (dev: fast mode, dev results; test: full mode, test results); fasttest: fast mode, test results")
    parser.add_argument("--task_set", type=str,
                        choices=['sts', 'transfer', 'full', 'na', 'cococxc'],
                        default='sts',
                        help="What set of tasks to evaluate on. If not 'na', this will override '--tasks'")
    parser.add_argument("--tasks", type=str, nargs='+',
                        default=['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                                 'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC',
                                 'SICKRelatedness', 'STSBenchmark'],
                        help="Tasks to evaluate on. If '--task_set' is specified, this will be overridden")
    args = parser.parse_args()
    return args


#This method encodes all queries using PROMCSE as explained in the paper. The output file for each query has the encodings of its title and paragraphs
def encodeQueriesWithPromCSE(qIDs, outputDir):
    tokenizer = AutoTokenizer.from_pretrained('YuxinJiang/sup-promcse-roberta-large')
    config = AutoConfig.from_pretrained('YuxinJiang/sup-promcse-roberta-large')
    # model = AutoModel.from_pretrained('YuxinJiang/sup-promcse-roberta-large',config, add_pooling_layer=False)
    args = {}
    model = RobertaForCL.from_pretrained(
        'YuxinJiang/sup-promcse-roberta-large',
        from_tf=bool(".ckpt" in 'YuxinJiang/sup-promcse-roberta-large'),
        config=config, model_args=getArgs())
    for queryid in qIDs.keys():
        print("Processing Query", queryid)
        count = 1
        fwDoc = open(outputDir + "/" + str(queryid), "w")
        statement = "Select title from documents where id like'" + qIDs[queryid] + "'"
        result = accessDatabase(statement)
        title = str(result[0][0]).strip()
        embedding = encodeTextPromCSEBERT(model,tokenizer,title)
        fwDoc.write("<Paragraph>")
        for element in embedding:
            fwDoc.write(str(element) + " ")
        statement = "Select data from contents where did like'" + qIDs[queryid] + "' order by pos"
        paragraphs = accessDatabase(statement)
        textToEncode = ""
        for para in paragraphs:
            p = str(para[0]).strip()
            pEncoding= encodeTextPromCSEBERT(model, tokenizer, p)
            fwDoc.write("<Paragraph>")
            count = count + 1
            for element in pEncoding:
                fwDoc.write(str(element) + " ")
        fwDoc.close()


def encodeCandidatesWithPromCSE(cIds, outputDir):
    tokenizer = AutoTokenizer.from_pretrained('YuxinJiang/sup-promcse-roberta-large')
    config = AutoConfig.from_pretrained('YuxinJiang/sup-promcse-roberta-large')
    args = {}
    model = RobertaForCL.from_pretrained(
        'YuxinJiang/sup-promcse-roberta-large',
        from_tf=bool(".ckpt" in 'YuxinJiang/sup-promcse-roberta-large'),
        config=config, model_args=getArgs())
    for cIndex in range(0,len(cIds)):
        paraEncodings={}
        print("Processing Query", cIds[cIndex])
        count = 1
        fwDoc = open(outputDir + "/" + cIds[cIndex], "w")
        statement = "Select title from documents where id like'" + cIds[cIndex] + "'"
        result = accessDatabase(statement)
        title = str(result[0][0]).strip()
        embedding = encodeTextPromCSEBERT(model,tokenizer,title)
        statement = "Select data from contents where did like'" + cIds[cIndex] + "' order by pos"
        paragraphs = accessDatabase(statement)
        count = count + 1
        for para in paragraphs:
            p = str(para[0]).strip()
            pEncoding= encodeTextPromCSEBERT(model, tokenizer, p)
            paraEncodings[count] = pEncoding
            count = count + 1
        candiateEncoding=np.mean(list(paraEncodings.values()), axis=0)
        for element in candiateEncoding:
            fwDoc.write(str(element) + " ")
        fwDoc.close()


if __name__ == '__main__':
    # We generate a reranking for each year first for efficiency purposes,
    # then we combine the outputfile before running trec_eval on a qrel file that combines the qrels of all years.

    # The following is a call for encoding the queries
    '''
    queriesIds = utils.getQueriesId("Queries2018.txt")
    encodeQueriesWithPromCSE(queriesIds, "PromCSEEncoding/Queries")
    queriesIds = utils.getQueriesId("Queries2019.txt")
    encodeQueriesWithPromCSE(queriesIds,  "PromCSEEncoding/Queries")
    queriesIds = utils.getQueriesId("Queries2020.txt")
    encodeQueriesWithPromCSE(queriesIds,  "PromCSEEncoding/Queries")
    queriesIds = utils.getQueriesId("Queries2021.txt")
    encodeQueriesWithPromCSE(queriesIds,  "PromCSEEncoding/Queries")
    '''
    # The following is a call for encoding candidates
    '''
    candidateIds = utils.loadCandidateIdsFromBaselineFile("Reranking/Baseline_2018.txt")
    encodeCandidatesWithPromCSE(candidateIds, "PromCSEEncoding/Candidates")
    candidateIds = utils.loadCandidateIdsFromBaselineFile("Reranking/Baseline_2019.txt")
    encodeCandidatesWithPromCSE(candidateIds,  "PromCSEEncoding/Candidates")
    candidateIds = utils.loadCandidateIdsFromBaselineFile("Reranking/Baseline_2020.txt")
    encodeCandidatesWithPromCSE(candidateIds,  "PromCSEEncoding/Candidates")
    candidateIds = utils.loadCandidateIdsFromBaselineFile("Reranking/Baseline_2021.txt")
    encodeCandidatesWithPromCSE(candidateIds,  "PromCSEEncoding/Candidates")
    '''

    #The following is a call for the reranking process
    '''
    win=1
    MaxFlag=False
    alpha=1
    
    utils.rerankDocBasedOnQueryPara_WholeDocSlidedPara("Reranking/Queries2018.txt", "Reranking/Baseline_2018.txt",
                                                     "Reranking/PromCSE/2018_PromCSERunWin_1_"+str(MaxFlag)+".txt",
                                                     "PromCSEEncoding/Queries", 1,
                                                     "PromCSEEncoding/Candidates", MaxFlag, win)
    utils.rerankDocBasedOnQueryPara_WholeDocSlidedPara("Reranking/Queries2019.txt", "Reranking/Baseline_2019.txt",
                                                     "Reranking/PromCSE/2019_PromCSERunWin_1_"+str(MaxFlag)+".txt",
                                                     "PromCSEEncoding/Queries", alpha,
                                                     "PromCSEEncoding/Candidates", MaxFlag, win)

    utils.rerankDocBasedOnQueryPara_WholeDocSlidedPara("Reranking/Queries2020.txt", "Reranking/Baseline_2020.txt",
                                                     "Reranking/PromCSE/2020_PromCSERunWin_1_"+str(MaxFlag)+".txt",
                                                     "PromCSEEncoding/Queries", alpha,
                                                     "PromCSEEncoding/Candidates", MaxFlag, win)

    utils.rerankDocBasedOnQueryPara_WholeDocSlidedPara("Reranking/Queries2021.txt", "Reranking/Baseline_2021.txt",
                                                     "Reranking/PromCSE/2021_PromCSERunWin_1_"+str(MaxFlag)+".txt",
                                                     "PromCSEEncoding/Queries", alpha,
                                                     "PromCSEEncoding/Candidates", MaxFlag, win)
    

    

    '''
