#!/usr/bin/env python3
from tira.third_party_integrations import ensure_pyterrier_is_loaded, get_output_directory
from pathlib import Path
import pandas as pd
import pyterrier as pt
from more_itertools import chunked
from tqdm import tqdm
import random
import argparse

def build_queries_per_document(index, seed=42, allow_list):
    meta = index.getMetaIndex()
    doi = index.getDocumentIndex()
    lex = index.getLexicon()
    di = index.getDirectIndex()
    ret = {}

    for doc_id in tqdm(range(0, doi.getNumberOfDocuments()), 'Extracting Document Representations'):
        if allow_list and doc_id not in allow_list:
            continue
        doc = doi.getDocumentEntry(doc_id)
        term_to_count = {}
        for posting in di.getPostings(doc):
            lee = lex.getLexiconEntry(posting.getId())
            term_to_count[lee.getKey()] = posting.getFrequency() + term_to_count.get(lee.getKey(), 0)
    
        query = list(term_to_count.keys())
        random.Random(seed).shuffle(query)
        query = sorted(query, key=lambda i: term_to_count[i], reverse=True)
        ret[meta.getItem("docno", doc_id)] = ' '.join(query[:top_terms_per_document])

    return ret

def find_neighbours(batch, bm25):
    ret = {i['qid']: [] for i in batch}

    for docno, tmp in dict(iter(bm25([i for i in batch if i['query']]).groupby('qid'))).items():
        ret[docno] = [i for i in tmp['docno'].tolist() if i != docno]

    return [{'docno': qid, 'neighbors': ret[qid]} for qid in ret]

def process_documents(docs, bm25):
    ret = []
    for batch in tqdm(list(chunked(docs, 1000)), 'Processing Batches.'):
        ret += find_neighbours(batch, bm25)

    return pd.DataFrame(ret)

if __name__ == '__main__':
    # The expected output directory, injected via the environment variable TIRA_OUTPUT_DIRECTORY
    output_dir = get_output_directory('.')

    # Document processors persist their results in a file documents.jsonl.gz in the output directory.
    output_file = Path(output_dir) / 'documents.jsonl.gz'

    parser = argparse.ArgumentParser()
    parser.add_argument('index')
    parser.add_argument('--top-terms-per-document', type=int, default=50)
    parser.add_argument('--allow-list-docs', type=str, default=None)
    args = parser.parse_args()

    ensure_pyterrier_is_loaded()
    top_terms_per_document = args.top_terms_per_document
    index = pt.IndexFactory.of(str(Path(args.index).resolve()) + '/index/')
    bm25 = pt.BatchRetrieve(index, wmodel='BM25', num_results=16)

    allow_list = None
    if args.allow_list_docs is not None:
        allow_list = set(list(pd.read_json(args.allow_list_docs, lines=True)['docno'].unique()))
    
    doc_id_to_query = build_queries_per_document(index)
    docs = [{'qid': i, 'query': doc_id_to_query[i]} for i in doc_id_to_query.keys()]

    # process the documents, store results at expected location.
    processed_documents = process_documents(docs, bm25)
    processed_documents.to_json(output_file, lines=True, orient='records')
