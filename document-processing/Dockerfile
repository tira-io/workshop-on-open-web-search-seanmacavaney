#see https://github.com/OpenWebSearch/wows-code/tree/main/ecir24/baselines/dev-container
FROM webis/wows-2024-baseline:0.0.1

ADD baseline_document_processing.py /code/baseline_document_processing.py

RUN pip install git+https://github.com/terrierteam/pyterrier_doc2query.git

ENTRYPOINT [ "python3", "/code/baseline_document_processing.py" ]
