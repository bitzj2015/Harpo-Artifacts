FROM condaforge/miniforge3

COPY . /app

WORKDIR /app/socket_server

RUN /opt/conda/bin/conda update -n base -c defaults conda

RUN /opt/conda/bin/conda install python=3.8 

RUN /opt/conda/bin/conda install numpy

RUN /opt/conda/bin/conda install scipy

RUN /opt/conda/bin/conda install nltk

RUN /opt/conda/bin/conda install html2text --channel conda-forge

RUN /opt/conda/bin/conda install websockets --channel conda-forge

RUN /opt/conda/bin/conda install grpcio

RUN pip install --upgrade pip

RUN pip install torch==1.9.0

RUN pip install gensim==3.8.0

RUN pip install lcdk

EXPOSE 8765

CMD ["python", "-u", "./socket_server.py"]

