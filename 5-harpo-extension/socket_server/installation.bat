@echo on
docker build -t socket_server .
docker run -it --expose 8765 -p 8765:8765 socket_server
PAUSE