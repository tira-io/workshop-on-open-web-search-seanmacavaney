#see https://github.com/OpenWebSearch/wows-code/tree/main/ecir24/baselines/dev-container
FROM webis/wows-2024-baseline:0.0.6

ADD corpus_graph.py /code/corpus_graph.py

ADD ./2023-11-07-11-58-59/output/index/ /code/2023-11-07-11-58-59/output/index/

RUN python3 -c 'from tira.third_party_integrations import ensure_pyterrier_is_loaded; ensure_pyterrier_is_loaded()'

ENTRYPOINT [ "python3", "/code/corpus_graph.py", "$inputRun" ]
