import os
import docker
from pprint import pprint as pp
from docker.types import LogConfig
import lcdk.lcdk as LeslieChow

__author__ = 'johncook'

class docker_api:
    def __init__(self, **kwargs):
        logsPath = "docker_api.logs"

        if not os.path.exists(logsPath):
            os.system('touch {}'.format(logsPath))
        self.DBG = LeslieChow.lcdk(logsPath=logsPath)


        self.cmd=kwargs.get('cmd')
    
    def run_container(self): 
        cmd = self.cmd
        self.DBG.lt_green('running container')
        self.DBG.lt_green(cmd)

        try: 
            docker_cmd = "docker run -v $PWD/docker-volume:/home/user/Desktop -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --shm-size=4g --cpuset-cpus=7 openwpm:latest {} ".format(cmd)
            os.system(cmd)

        except Exception as e:
            self.DBG.error("exception {} ".format(e))

        self.DBG.lt_green('run complete.')

# d_api = docker_api()
# d_api.run_container()
