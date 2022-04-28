# Harpo (Version 2.0)
Implementation of Harpo as a web extension.

## Installation Procedures
### Back End
#### Docker Environment
1. Launch docker desktop and ensure the the docker engine has started.
2. Navigate to `./socket_server/` directory and add the `.bin` file.
3. Run the `./socket_server/installation.batch` file to build and run the docker image. If the batch file does not respond, please open windows terminal and navigate to this directory `./socket_server/` and run the following commands manually.
```
docker build -t socket_server .
docker run -it --expose 8765 -p 8765:8765 socket_server
```
#### Local Environment
1. Revise your python path in the first line of `./socket_server/socket_server.py`.
2. Install python packages using the following commands. For all backend dependencies, please see the `./requirements.txt` file.
```
pip install torch
pip install gensim==3.8.0
pip install nltk
pip install lcdk
pip install html2text
pip install numpy
pip install asyncio
pip install websockets
```
4. Run the `./socket_server/socket_server.py` program to start the backend server.

5. Or using the following commands:
```
pip install -r requirements.txt
cd socket_server
python socket_server.py
```

### Front End
#### To install extension in Firefox
1. Load the manifest.json file in the Firefox "about:debugging" page

### Download resources
1. Download all models [here](https://drive.google.com/drive/folders/1DnMz4_hBFQ_Cq-fbjysd_Q-dGzw72yvr?usp=sharing)