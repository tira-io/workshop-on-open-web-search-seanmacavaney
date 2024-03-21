#!/usr/bin/env python3
from tira.third_party_integrations import ensure_pyterrier_is_loaded, get_output_directory
from pathlib import Path
import pandas as pd
import pyterrier as pt
from more_itertools import chunked
from tqdm import tqdm
import random
import argparse

def build_queries_per_document(index, range_start=0, range_end=None, seed=42, top_terms_per_document=50):
    meta = index.getMetaIndex()
    doi = index.getDocumentIndex()
    lex = index.getLexicon()
    di = index.getDirectIndex()
    ret = {}
    range_end = range_end if range_end is not None else doi.getNumberOfDocuments()
    range_end = min(range_end, doi.getNumberOfDocuments())

    for doc_id in tqdm(range(range_start, range_end), 'Extracting Document Representations'):
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
    batch = [i for i in batch if i['query']]
    if len(batch) == 0:
        return []

    for docno, tmp in dict(iter(bm25(batch).groupby('qid'))).items():
        ret[docno] = [i for i in tmp['docno'].tolist() if i != docno]

    return [{'docno': qid, 'neighbors': ret[qid]} for qid in ret]

def process_documents(docs, bm25):
    ret = []
    for batch in tqdm(list(chunked(docs, 50)), 'Processing Batches.'):
        ret += find_neighbours(batch, bm25)

    return pd.DataFrame(ret)

def construct_corpus_graph_for_range(index_dir, top_terms_per_document, range_start=0, range_end=None):
    ensure_pyterrier_is_loaded()
    index = pt.IndexFactory.of(index_dir)
    bm25 = pt.BatchRetrieve(index, wmodel='BM25', num_results=16)
    
    doc_id_to_query = build_queries_per_document(index, range_start, range_end, top_terms_per_document)
    docs = [{'qid': i, 'query': doc_id_to_query[i]} for i in doc_id_to_query.keys()]

    # process the documents, store results at expected location.
    return process_documents(docs, bm25)

if __name__ == '__main__':
    # The expected output directory, injected via the environment variable TIRA_OUTPUT_DIRECTORY
    output_dir = get_output_directory('.')

    # Document processors persist their results in a file documents.jsonl.gz in the output directory.
    output_file = Path(output_dir) / 'documents.jsonl.gz'

    parser = argparse.ArgumentParser()
    parser.add_argument('index')
    parser.add_argument('--top-terms-per-document', type=int, default=50)
    parser.add_argument('--range-start', type=int, default=0)
    parser.add_argument('--range-end', type=int, default=None)
    args = parser.parse_args()

    processed_documents = construct_corpus_graph_for_range(
        str(Path(args.index).resolve()) + '/index/',
        args.top_terms_per_document,
        args.range_start,
        args.range_end
    )
    processed_documents.to_json(output_file, lines=True, orient='records')
