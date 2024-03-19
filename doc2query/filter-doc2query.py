#!/usr/bin/env python3
# Load a patched ir_datasets that loads the injected data inside the TIRA sandbox
from tira.third_party_integrations import get_input_directory_and_output_directory, ir_datasets
from pathlib import Path
import json
from glob import glob
import gzip
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='workshop-on-open-web-search/document-processing-20231027-training', required=False)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--input_run', type=str, required=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    ret = {}
    doc_ids = set()

    # In the TIRA sandbox, this is the injected ir_dataset, injected via the environment variable TIRA_INPUT_DIRECTORY
    dataset = ir_datasets.load(args.input)
    for doc in dataset.docs_iter():
        doc_ids.add(str(doc.doc_id))

    # e.g., tira-cli download --dataset  cranfield-20230107-training --approach workshop-on-open-web-search/seanmacavaney/DocT5Query
    for f in glob(args.input_run + '/**/documents.jsonl.gz', recursive=True):
        print('Process', f)
        with gzip.open(f, 'rt') as docs:
            for doc in docs:
                doc = json.loads(doc)
                ret[str(doc['doc_id'])] = doc['querygen']

    # Document processors persist their results in a file documents.jsonl.gz in the output directory.
    output_file = Path(args.output_dir) / 'documents.jsonl.gz'
    ret = [json.dumps({'doc_id': i, 'querygen': ret[i]}) for i in doc_ids]
    
    with gzip.open(output_file, 'wt') as docs:
        for i in ret:
            docs.write(i + '\n')

