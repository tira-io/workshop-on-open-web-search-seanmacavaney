#see https://github.com/OpenWebSearch/wows-code/tree/main/ecir24/baselines/dev-container
FROM webis/wows-2024-baseline:0.0.1

ADD doc2query.py /code/doc2query.py

RUN pip install git+https://github.com/terrierteam/pyterrier_doc2query.git

RUN python -c "import pyterrier_doc2query ; pyterrier_doc2query.Doc2Query()" # cache the model

ENTRYPOINT [ "python3", "/code/doc2query.py" ]
