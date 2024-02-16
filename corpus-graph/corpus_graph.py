#!/usr/bin/env python3
# Load a patched ir_datasets that loads the injected data inside the TIRA sandbox
from tira.third_party_integrations import ir_datasets, ensure_pyterrier_is_loaded, get_output_directory
from pathlib import Path
import pandas as pd
import pyterrier as pt
from more_itertools import chunked
from tqdm import tqdm
import argparse
from pathlib import Path

ensure_pyterrier_is_loaded()

tokeniser = pt.autoclass("org.terrier.indexing.tokenisation.Tokeniser").getTokeniser()
def pt_tokenise(text):
    return ' '.join(tokeniser.getTokens(text))


def process_documents(document_iter, retr):
    results = []
    for batch in chunked(tqdm(document_iter, unit='d'), 1000):
        batch = pd.DataFrame(batch).rename(columns={'doc_id': 'qid', 'text': 'query'})
        batch['query'] = batch['query'].apply(pt_tokenise)
        res = retr(batch)
        res = dict(iter(res.groupby('qid')))
        results.extend({
            'docno': d.qid,
            'neighbors': list(res[d.qid]['docno'][res[d.qid]['docno'] != d.qid]),
        } for d, r in zip(batch.itertuples(), res))
    return pd.DataFrame(results)


if __name__ == '__main__':
     # In the TIRA sandbox, this is the injected ir_dataset, injected via the environment variable TIRA_INPUT_DIRECTORY
    dataset = ir_datasets.load('workshop-on-open-web-search/document-processing-20231027-training')

    # The expected output directory, injected via the environment variable TIRA_OUTPUT_DIRECTORY
    output_dir = get_output_directory('.')

    # Document processors persist their results in a file documents.jsonl.gz in the output directory.
    output_file = Path(output_dir) / 'documents.jsonl.gz'

    parser = argparse.ArgumentParser()
    parser.add_argument('index')
    args = parser.parse_args()

    retr = pt.BatchRetrieve(str(Path(args.index).resolve() + '/index/'), wmodel='BM25', num_results=17)

    # You can pass as many additional arguments to your program, e.g., via argparse, to modify the behaviour

    # process the documents, store results at expected location.
    processed_documents = process_documents(dataset.docs_iter(), retr)
    processed_documents.to_json(output_file, lines=True, orient='records')
