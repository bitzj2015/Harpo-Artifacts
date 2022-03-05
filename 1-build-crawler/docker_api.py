import os
import docker
import lcdk.lcdk as LeslieChow
import  irlutils.file.file_utils as fu
from  multiprocessing import Process
__author__ = 'johncook'
cpuset = []
class docker_cfg:
    def __init__(self,**kwargs):
        self.user_desktop = kwargs.get('user_desktop', '/home/user/Desktop/')
        self.export = kwargs.get('e', 'DISPLAY=$DISPLAY')
        self.volumes=kwargs.get('v_docker',{})
        self.v_docker = '$PWD/docker-volume:/home/user/Desktop/'
        self.v_x11 = kwargs.get('v_x11', '/tmp/.X11-unix:/tmp/.X11-unix')
        self.shm_size = kwargs.get('shm-size', '10g')
        self.cpuset_cpus = kwargs.get('cpuset_cpus', '10')
        self.image_name = kwargs.get('image', 'openwpm:latest')

class docker_api:
    def __init__(self, **kwargs):
        logsPath = "docker-volume/logs/docker_api.log"
        self.cmd=kwargs.get('cmd')
        self.cfg=kwargs.get('cfg')
        if kwargs.get('stand_alone', False):
            self.d_cfg = docker_cfg(user_desktop=self.cfg['dump_root'], image="stand_alone_openwpm:latest", cpuset_cpus=self.cfg['cpuset_cpus'])
        else:
            self.d_cfg = docker_cfg(user_desktop=self.cfg['dump_root'])
        self.bids_file = os.path.join(self.d_cfg.user_desktop, 'hb_bids.sqlite')
        self.bid_logs = os.path.join(self.d_cfg.user_desktop, 'hb_server.log')
        fu.touch(logsPath)

        self.DBG = LeslieChow.lcdk(logsPath=logsPath, print_output=False)


    def run_container(self):
        self.DBG.warning(self.cmd)
        openwpm_cmd = self.cmd
        self.DBG.lt_green(openwpm_cmd)


        disable_ipv6 = " --sysctl net.ipv6.conf.all.disable_ipv6=1 --sysctl net.ipv6.conf.default.disable_ipv6=1 --sysctl net.ipv6.conf.lo.disable_ipv6=1 "
        try:
            os.system("yes | sudo docker container prune")
            docker_cmd = "docker run -v {} -e {} -v {} {} --stop-timeout 1200 --shm-size={}  {} {}".format(self.d_cfg.v_docker,
                                                                                                     self.d_cfg.export,
                                                                                                     self.d_cfg.v_x11,
                                                                                                     disable_ipv6,
                                                                                                    #  self.d_cfg.cpuset_cpus,
                                                                                                     self.d_cfg.shm_size,
                                                                                                     self.d_cfg.image_name,
                                                                                                     openwpm_cmd)
            os.system(docker_cmd)

        except Exception as e:
            self.DBG.error("exception {} ".format(e))

        self.DBG.lt_green('run complete.')
