FROM python:3.8.10
WORKDIR /app
COPY ./* /app

EXPOSE 8765

RUN pip install torch
RUN pip install gensim==3.8.0
RUN pip install nltk
RUN pip install lcdk
RUN pip install html2text
RUN pip install numpy
RUN pip install asyncio
RUN pip install websockets

CMD python socket_server.py