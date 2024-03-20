#see https://github.com/OpenWebSearch/wows-code/tree/main/ecir24/baselines/dev-container
#docker build -t mam10eks/corpus-graph:0.0.1 .
FROM registry.webis.de/code-lib/public-images/webis/spark

USER root
RUN apt-get update && apt-get install -y libz-dev
RUN pip3 install tira python-terrier jupyter

# prevent confusion by removing redundant python installations
RUN rm /bin/python3.10
env PYTERRIER_VERSION=5.8
env PYTERRIER_HELPER_VERSION=0.0.8

RUN python3 -c 'from tira.third_party_integrations import ensure_pyterrier_is_loaded; ensure_pyterrier_is_loaded()'

USER spark

ADD corpus_graph.py /code/corpus_graph.py

ADD corpus_graph_on_reduced_docs.py /code/corpus_graph_on_reduced_docs.py

ADD ./2023-11-07-11-58-59/output/index/ /code/2023-11-07-11-58-59/output/index/

#ENTRYPOINT [ "python3", "/code/corpus_graph_on_reduced_docs.py", "$inputRun" ]
