import os
import json
import lcdk.lcdk as LeslieChow
import  irlutils.file.file_utils as fu
import ray

__author__ = 'jiang'

class docker_cfg:
    def __init__(self,**kwargs):
        self.user_desktop = kwargs.get('user_desktop', '/home/user/Desktop/')
        self.export = kwargs.get('e', 'DISPLAY=$DISPLAY')
        self.volumes=kwargs.get('v_docker',{})
        self.v_docker = '$PWD/docker-volume:/home/user/Desktop/'
        # self.v_docker = '/SSD/docker-volume:/home/user/Desktop/'
        self.v_x11 = kwargs.get('v_x11', '/tmp/.X11-unix:/tmp/.X11-unix')
        self.shm_size = kwargs.get('shm-size', '8g')
        self.cpuset_cpus = kwargs.get('cpuset', '1')
        self.image_name = kwargs.get('image', 'openwpm-harpo:latest')


@ray.remote
class docker_api(object):
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

        self.DBG = LeslieChow.lcdk(logsPath=logsPath, print_output=True)


    def run_container(self):
        self.DBG.warning(self.cmd)
        openwpm_cmd = self.cmd
        self.DBG.lt_green(openwpm_cmd)


        disable_ipv6 = " --sysctl net.ipv6.conf.all.disable_ipv6=1 --sysctl net.ipv6.conf.default.disable_ipv6=1 --sysctl net.ipv6.conf.lo.disable_ipv6=1 "
        try:
            os.system("yes | sudo docker container prune")
            docker_cmd = "docker run -v {} -e {} -v {} {} --shm-size={} {} {}".format(self.d_cfg.v_docker,
                                                                                      self.d_cfg.export,
                                                                                      self.d_cfg.v_x11,
                                                                                      disable_ipv6,
                                                                                      self.d_cfg.shm_size,
                                                                                      self.d_cfg.image_name,
                                                                                      openwpm_cmd)
            os.system(docker_cmd)

        except Exception as e:
            self.DBG.error("exception {} ".format(e))
            return 0

        self.DBG.lt_green('run complete.')
        return 1

    def start_container(self, container_name):
        disable_ipv6 = " --sysctl net.ipv6.conf.all.disable_ipv6=1 --sysctl net.ipv6.conf.default.disable_ipv6=1 --sysctl net.ipv6.conf.lo.disable_ipv6=1 "
        try:
            docker_cmd = "docker run -v {} -e {} -v {} {} --stop-timeout 1200 --shm-size={} --name {} -it -d  {} ".format(self.d_cfg.v_docker,
                                                                                                                      self.d_cfg.export,
                                                                                                                      self.d_cfg.v_x11,
                                                                                                                      disable_ipv6,
                                                                                                                      self.d_cfg.shm_size,
                                                                                                                      container_name,
                                                                                                                      self.d_cfg.image_name)
            os.system("yes | sudo docker container stop {}".format(container_name))
            os.system("yes | sudo docker container prune")
            os.system(docker_cmd)
        except Exception as e:
            self.DBG.error("exception {} ".format(e))
            return 0

    def exec_container(self, container_name):
        openwpm_cmd = self.cmd
        try:
            docker_cmd = "docker exec -it {} {}".format(container_name,openwpm_cmd)
            os.system(docker_cmd)

        except Exception as e:
            self.DBG.error("exception {} ".format(e))
            return 0

        self.DBG.lt_green('run complete.')
        return 1


if __name__ == "__main__":
    cmd = "echo hellp world"
    cfg = {"browser_id": "browser_63", 
           "start_time": "202012250237250000", 
           "crawl_type": "user_0.0_63", 
           "dump_root": "/home/user/Desktop/base_crawl_new", 
           "cmd": "", 
           "run_bids_server": True, 
           "load_profile_dir": None, 
           "save_profile_dir": 
           "/home/user/Desktop/base_crawl_new/user_0.0_63/browser_63", 
           "sites": ["https://www.soas.ac.uk/", "https://www.earthclinic.com/", "https://www.kidcyber.com.au/", "http://theaquariumwiki.com/wiki/", "https://www.colorado.gov/", "https://www.india.gov.in/", "https://nifa.usda.gov/", "https://www.drivereasy.com/", "https://www.fraserhealth.ca/", "https://www.suckerpunch.com/", "https://www.navycs.com/", "https://trailrunnermag.com/", "https://www.economist.com/", "https://www.periodni.com/", "https://discovertheforest.org/", "https://www.powerthesaurus.org/", "https://www.ghs.org/", "https://www.naturalchild.org/", "https://www.dci.org/", "https://www.hearth.com/talk/", "http://wwww.speedtest.net", "http://www.kompas.com", "http://www.cnn.com"], 
           "sites_getads": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    cmd = "sudo /usr/bin/python3 /opt/OpenWPM/run_ml_crawl.py --cfg '{}'".format(json.dumps(cfg))
    # cmd = "echo hellp world"
    d_api = docker_api(cmd=cmd, cfg=cfg)
    d_api.start_container("test")
    d_api.exec_container("test")
    d_api.exec_container("test")
