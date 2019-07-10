#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals, division, absolute_import
from io import open
import argparse, logging, os
from util import *
from collections import defaultdict
from copy import deepcopy

#######################################
#
# sharding.py
#
# This module uses md5 hashes to split
# a parallel corpus into N shards, which
# can then be used for train/dev/test splitting,
# cross-validation, etc.
#
# The input to this script is a metadata YAML that looks 
# like the following:
#
# data:
#   en:
#     path: <path to English filename>
#   fr:
#     path: <path to French filename>
#
# Paths should be absolute or relative to the YAML file.
#
# Other information can be in the YAML (metadata etc.), so
# long as there's something like the above.  Any additional
# information is just copied as-is into the YAMLs for the 
# shards.
#
# The corpus needs not be limited to two languages;
# multilingual corpora are possible as well.
#
# Since all sentences in the pair/tuple are used to calculate 
# the md5 hash, separately-sharded parallel
# corpora can put one sentence (in a particular language)
# into different shards.  For example, if you're sharding 
# a French-English and German-English bible corpus, it's
# likely that any particular English sentence ends up in 
# different shards, since it was paired with different 
# French and German sentences. 
#
#######################################

def make_shards(dataset, input_dir=".", output_dir=".", num_shards=10):
    ''' Use md5 hashing of sentence tuples to split them into shards '''
    #logging.debug(f"Splitting {paths.values()} into {num_shards} shards")
    
    ids = list(dataset["data"])   # just get the keys
    ids.sort()       # Since Python doesn't hash dictionary keys things the same
                     # way each time, this would lead to different sentence
                     # tuples, and then when we hash *those*, we would get 
                     # different results and then different shards.  So for
                     # reproducability, we sort the keys alphabetically first.
                                  
    sentences_per_id = {}
    for id in dataset["data"]:
        path = dataset["data"][id]["path"]
        if not os.path.isabs(path):
            path = os.path.join(input_dir, path)
        with open(path, "r", encoding="utf-8") as fin:
            sentences_per_id[id] = fin.readlines()
    
    sentences = [ sentences_per_id[id] for id in ids ]     # keep these in the same order
    
    # make sure input files have same number of sentences
    num_sentences = -1
    for ss in sentences:
      if num_sentences != -1 and len(ss) != num_sentences:
        logging.warning("Parallel source files have different number of "
                "sentences in dataset %s" % dataset['metadata']['name'])
    
    shards = { i:defaultdict(list) for i in range(num_shards) }
    
    for sentence_tuple in zip(*sentences):
        h = hash_to_int(sentence_tuple) % num_shards
        for id, s in zip(ids, sentence_tuple):
            shards[h][id].append(s)

    for h in range(num_shards):
        new_dataset = deepcopy(dataset)
        if "metadata" not in new_dataset:
            new_dataset["metadata"] = {}
        new_dataset["metadata"]["sharded"] = True
        new_dataset["metadata"]["shard"] = h
        new_dataset["metadata"]["num_shards"] = num_shards
        for id in ids:
            filename = "shard%s.%s.txt" % (h, id)
            output_path = os.path.join(output_dir, filename)
            new_dataset["data"][id]["path"] = filename
            text = "".join(shards[h][id])
            save_txt(output_path, text)
        filename = "shard%s.yaml" % h
        output_path = os.path.join(output_dir, filename)
        save_yaml(output_path, new_dataset)

def main(input_yaml, output_dir, nshards=10):
    ''' Load a YAML description of an unsharded dataset and make shards and
        YAML descriptions for them '''
    input_dir = os.path.dirname(input_yaml)
    dataset = load_yaml(input_yaml)
    make_shards(dataset, input_dir, output_dir, nshards)

if __name__ == '__main__':
   argparser = argparse.ArgumentParser(description='Create shards from a parallel dataset')
   argparser.add_argument('input_yaml', help='Input YAML describing dataset.')
   argparser.add_argument('output_dir', help='Output directory')
   argparser.add_argument('--nshards', type=int, default=10, help='Number of shards to make')
   args = argparser.parse_args()
   main(args.input_yaml, 
                    args.output_dir, 
                    args.nshards)
