import conllu
import math
import random
from copy import deepcopy


# import treebank function. For multi-word token treebanks, the multi-word tokens are all discarded!!!
def import_treebank(filename, list_to_append):
    with open(filename, "r", encoding="utf-8") as rf:
        all_sents = conllu.parse(rf.read())
        for sent in all_sents:
            sent_to_append = conllu.TokenList()
            for tok in sent:
                if type(tok["id"]) is not tuple:
                    sent_to_append.append(tok)
                    sent_to_append.metadata = sent.metadata
            list_to_append.append(sent_to_append)


# mean dependency distance function
def mdd(sentence):
    list_of_dd = list()
    for tok in sentence:
        if tok["deprel"] not in ["punct", "root"]:
            list_of_dd.append(abs(int(tok["id"]) - int(tok["head"])))

    return sum(list_of_dd)/len(list_of_dd) if len(list_of_dd) > 0 else "n/a"


# normalized dependencies distance function
def ndd(sentence):
    sent_mdd = mdd(sentence)
    if sent_mdd == "n/a":
        return "n/a"

    root_distance = 0
    sentence_length = 0
    for tok in sentence:
        if tok["deprel"] == "root":
            root_distance = int(tok["id"])

        if tok["deprel"] not in ["punct", "root"]:
            sentence_length += 1

    if root_distance < 1 or sentence_length < 1:
        raise Exception(f"ERROR: either root_distance or sentence_length could not be"
                        f"set in {sentence.metadata['sent_id']}")

    return abs(math.log(sent_mdd / math.sqrt(root_distance * sentence_length)))


# maximum tree depth function
def max_tree_depth(sentence):
    depths_list = list()
    for tok in sentence:
        tok_depth = 1
        next_head = tok["head"]
        while next_head != 0:
            tok_depth += 1
            next_head = sentence[int(next_head) - 1]["head"]

        depths_list.append(tok_depth)

    return max(depths_list)


def has_cop_dependent(sentence, tok_id):
    for tok in sentence:
        if tok["deprel"] == "cop" and tok["head"] == tok_id:
            return True

    return False


# number of clauses in sentence function
def clauses_in_sent(sentence):
    clauses_counter = 1
    for tok in sentence:
        if tok["deprel"] in ["csubj", "ccomp", "xcomp", "advcl", "acl", "conj", "parataxis"] and \
                (tok["upos"] == "VERB" or has_cop_dependent(sentence, tok["id"])):
            clauses_counter += 1

    return clauses_counter


# number of T-units in sentence function
def t_units_in_sent(sentence):
    t_unit_counter = 1
    for tok in sentence:
        if tok["deprel"] in ["conj", "parataxis"] and \
                (tok["upos"] == "VERB" or has_cop_dependent(sentence, tok["id"])):
            t_unit_counter += 1

    return t_unit_counter


# calculate results function
def calc_results(sents, results_list):
    for sent in sents:
        cis = clauses_in_sent(sent)
        tis = t_units_in_sent(sent)
        results_list.append((sent.metadata["sent_id"], str(mdd(sent)), str(ndd(sent)), str(max_tree_depth(sent)),
                             str(len(sent)), str(cis), str(tis), str(cis/tis)))


random.seed(9011095)

# first import all datasets
ssj_sents = list()
import_treebank("UD_Slovenian-SSJ-master/sl_ssj-ud-train.conllu", ssj_sents)
import_treebank("UD_Slovenian-SSJ-master/sl_ssj-ud-dev.conllu", ssj_sents)
import_treebank("UD_Slovenian-SSJ-master/sl_ssj-ud-test.conllu", ssj_sents)

sst_sents = list()
import_treebank("UD_Slovenian-SST-master/sl_sst-ud-train.conllu", sst_sents)
import_treebank("UD_Slovenian-SST-master/sl_sst-ud-test.conllu", sst_sents)

print("done importing datasets!")

# shuffle all datasets
random.shuffle(ssj_sents)
random.shuffle(sst_sents)

# take a smaller sample of sentences from ssj at random
ssj_smaller = deepcopy(ssj_sents)
ssj_sample_size = len(sst_sents)
random.shuffle(ssj_smaller)
ssj_smaller = ssj_smaller[:ssj_sample_size]

# define variables to store calculation results
ssj_results = list()
ssj_smaller_results = list()
sst_results = list()

# go through all sentences and calculate results, once for ssj and once for sst
calc_results(ssj_sents, ssj_results)
print("ssj results done!")
calc_results(ssj_smaller, ssj_smaller_results)
print("ssj smaller results done!")
calc_results(sst_sents, sst_results)
print("sst results done!")

# write all the results to a tsv for each treebank
with open("ssj_results.tsv", "w", encoding="utf-8") as wf_ssj:
    wf_ssj.write("\t".join(["SENT_ID", "MDD", "NDD", "MAXIMUM_TREE_DEPTH", "#_OF_TOKENS", "#_OF_CLAUSES",
                            "#_OF_T-UNITS", "CLAUSES_PER_T-UNIT"]) + "\n")
    for ssj_sen_reslt in ssj_results:
        wf_ssj.write("\t".join(ssj_sen_reslt) + "\n")

with open("ssj_smaller_results.tsv", "w", encoding="utf-8") as wf_ssj_smaller:
    wf_ssj_smaller.write("\t".join(["SENT_ID", "MDD", "NDD", "MAXIMUM_TREE_DEPTH", "#_OF_TOKENS", "#_OF_CLAUSES",
                                    "#_OF_T-UNITS", "CLAUSES_PER_T-UNIT"]) + "\n")
    for ssj_smlr_reslt in ssj_smaller_results:
        wf_ssj_smaller.write("\t".join(ssj_smlr_reslt) + "\n")

with open("sst_results.tsv", "w", encoding="utf-8") as wf_sst:
    wf_sst.write("\t".join(["SENT_ID", "MDD", "NDD", "MAXIMUM_TREE_DEPTH", "#_OF_TOKENS", "#_OF_CLAUSES",
                            "#_OF_T-UNITS", "CLAUSES_PER_T-UNIT"]) + "\n")
    for sst_sen_reslt in sst_results:
        wf_sst.write("\t".join(sst_sen_reslt) + "\n")
