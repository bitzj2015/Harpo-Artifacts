from __future__ import absolute_import

import os
import json
import copy
import random
import sqlite3
import argparse
import requests
import threading
import concurrent
import subprocess
import pandas as pd
import concurrent.futures
from pprint import pprint as pp
from automation import CommandSequence, TaskManager
from urllib.parse import urlparse, urljoin
import irlutils.file.file_utils as fu
import lcdk.lcdk as LeslieChow
from  baseline_crawler import baseline_crawler
import parse_http_content
__author__='johncook'


def get_content(hostname, logs_dir)
    crawl_sqlite = os.path.join(os.path.abspath(logs_dir, 'crawl-sqlite'))
    ldb = os.path.join(os.path.abspath(logs_dir, 'content.ldb'))

    con  = sqlite3.connect(crawl_sqlite)
    cur = con.cursor()
    http_responses = pd.read_sql_query("SELECT * FROM http_responses", con)
    print(http_responses)
    LDB = owpm.get_leveldb(bytes(ldb, "utf-8"))
    content = None

    for i in http_responses.index:
        url = http_responses.at[i, 'url']

        try:
            hashes = [x for x in owpm.get_distinct_content_hashes(cur, url)][0]
            content_hash = bytes(hashes, "utf-8")
            print(content_hash)
            content = owpm.get_content(LDB, hashes)
        except Exception  as e:
            print("Exception: {}".format(e))
            pass
        print("Content: \n\n{}\n\n".format(content))
        print("URL: {}\n----------------------------------".format(url))
        if url == hostname:
            break
    return content

#https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter03/crawling_web_step1.py
def process_link(url):
    logs_dir = logsPath.replace('/openwpm.log',"")
    html_text = get_content(url, logs_dir)
    DBG.lt_cyan('ADN: Extracting links from {}'.format(url))
    parsed_source = urlparse(url)
    result = bs4.BeautifulSoup.

    page = BeautifulSoup(html_text, 'html.parser')

    return get_links(parsed_source, page)


def get_links(parsed_source, page):
    '''Retrieve the links on the page'''
    links = []
    for element in page.find_all('a'):
        link = element.get('href')
        if not link:
            continue

        # Avoid internal, same page links
        if link.startswith('#'):
            continue

        # Always accept local links
        if not link.startswith('http'):
            netloc = parsed_source.netloc
            scheme = parsed_source.scheme
            path = urljoin(parsed_source.path, link)
            link = f'{scheme}://{netloc}{path}'

        # Only parse links in the same domain
        if parsed_source.netloc not in link:
            continue

        links.append(link)

    return links

def adnauseum_get_page_links(base_url, workers=1):
    checked_links = set()
    to_check = [base_url]
    max_checks = 10

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        while to_check:
            futures = [executor.submit(process_link, url)
                       for url in to_check]
            to_check = []
            for data in concurrent.futures.as_completed(futures):
                link, new_links = data.result()

                checked_links.add(link)
                for link in new_links:
                    if link not in checked_links and link not in to_check:
                        to_check.append(link)

                max_checks -= 1
                if not max_checks:
                    return
    return checked_links



p = argparse.ArgumentParser()
p.add_argument('--cfg', help="where all the magic is stored")
args  = p.parse_args()
NUM_BROWSERS = 5
CFG = json.loads(args.cfg)

logsPath = os.path.join(CFG['dump_root'], CFG['crawl_type'], CFG['crawl_id'], 'openwpm', 'openwpm.log')

if not os.path.exists('crawls'):
    os.mkdir('crawls')
if not os.path.exists(os.path.join(CFG['dump_root'], CFG['crawl_type'])):
    os.mkdir(os.path.join(CFG['dump_root'], CFG['crawl_type']))
if not os.path.exists(os.path.join('crawls', CFG['dump_root'], CFG['crawl_type'], CFG['crawl_id'])):
    os.mkdir(os.path.join(CFG['dump_root'], CFG['crawl_type'], CFG['crawl_id']))
if not os.path.exists(os.path.join( CFG['dump_root'], CFG['crawl_type'], CFG['crawl_id'], 'openwpm')):
    os.mkdir(os.path.join(CFG['dump_root'], CFG['crawl_type'], CFG['crawl_id'], "openwpm"))

if not os.path.exists(logsPath):
    os.system('touch {}'.format(logsPath))
DBG = LeslieChow.lcdk(logsPath=logsPath)

class file_manager:
    def __init__(self, **kwargs):
        level0 = kwargs.get('level0', '') # root dump path
        level1 = kwargs.get('level1', '')
        level2 = kwargs.get('level2', '')
        level3 = kwargs.get('level3', '')

        self.dump_path = os.path.join(level0, level1, level2, level3)


        self.mkdir(level0)
        self.mkdir(os.path.join(level0, level1))
        self.mkdir(os.path.join(level0, level1, level2))
        self.mkdir(self.dump_path)



    def mkdir(self,d):
        if not os.path.exists(d):
            os.mkdir(d)
            DBG.lt_cyan("mkdir complete.\npath: {}\n".format(d))

    def compress_path(self, path):
        tmp = fu.tar_packer(tar_dir=path)
        DBG.lt_cyan("compress_path complete.\npath: {}\n".format(path))
        return tmp

    def rm(self, path):
        fu.rmsubtree(location=path)
        DBG.lt_cyan("rm complete.\npath: {}\n".format(path))

    def mv(self,src, dst):
        err = fu.mv(src, dst)
        if not err:
            DBG.lt_cyan("mv complete.\nsrc: {}\ndst:{}\n".format(src, dst))

def load_default_params(mp, bp, num_browsers=1):
        """
        Loads num_browsers copies of the default browser_params dictionary.
        Also loads a single copy of the default TaskManager params dictionary.
        """
        fp = open(os.path.join(os.path.dirname(__file__),
                            'automation', 'default_browser_params.json'))
        preferences = json.load(fp)
        fp.close()
        browser_params = [copy.deepcopy(preferences) for i in range(
            0, num_browsers)]

        fp = open(os.path.join(os.path.dirname(__file__),
                            'automation', 'default_manager_params.json'))
        manager_params = json.load(fp)
        fp.close()
        manager_params['num_browsers'] = num_browsers


        return {'mp': manager_params, 'bp':browser_params}

#initiate config
class config:

    crawl_id =None
    sites = []
    manager_params = None
    browser_params = None
    params = load_default_params(manager_params, browser_params, num_browsers=NUM_BROWSERS)
    manager_params = params['mp']
    browser_params = params['bp']

    crawl_type = None
    start_time = None,
    dump_root = None,
    load_profile = False


cfg = config()
print(cfg.browser_params)
def baseline_config(cfg):
    for i in range(NUM_BROWSERS):
            cfg.browser_params[i]['http_instrument'] = True
            # Record cookie changes
            cfg.browser_params[i]['cookie_instrument'] = True
            # Record Navigations
            cfg.browser_params[i]['navigation_instrument'] = True
            # Record JS Web API calls
            cfg.browser_params[i]['js_instrument'] = True
            # Enable flash for all three browsers
            cfg.browser_params[i]['disable_flash'] = True
            cfg.browser_params[i]['hb_collector'] = True

            cfg.browser_params[i]['headless'] = True
    return cfg
cfg = baseline_config(cfg)
cfg.crawl_id = CFG['crawl_id']
cfg.crawl_type = CFG["crawl_type"]
cfg.dump_root = CFG["dump_root"]
cfg.start_time = CFG['start_time']
#crawl stub logging
userRoot = cfg.dump_root


fm = file_manager(level0=cfg.dump_root, level1=cfg.crawl_type, level2=cfg.crawl_id)

cfg.sites = CFG["sites"]

DBG.lt_cyan("{}".format(CFG))
#initiate bids server
bids_directory = os.path.join(fm.dump_path, 'bids')
fm.mkdir(bids_directory)
cmd = 'python3 hb_collector/server/bid_server.py --DB {}/hb_bids.sqlite'.format(bids_directory)

# cmd = 'python3 hb_collector/server/bid_server.py --DB {}/hb_bids.sqlite --logsPath {}'.format(bids_directory, os.path.join(bids_directory, 'bid_server.logs'))

# p = subprocess.Popen(cmd.split())
# t = threading.Thread(target=os.system, args=(cmd, ), daemon=True)
# t.start()

#openwpm logging and file mgmt
crawl_dir = os.path.join(fm.dump_path, 'openwpm')
cfg.manager_params['data_directory'] = os.path.abspath(crawl_dir)
cfg.manager_params['log_directory'] =  os.path.abspath(crawl_dir)

DBG.lt_cyan("{}".format(cfg.manager_params))
DBG.lt_cyan("{}".format(cfg.browser_params))

# Commands time out by default after 60 seconds
DBG.lt_green("starting crawl")
manager = TaskManager.TaskManager(cfg.manager_params, cfg.browser_params)
# Visits the sites with all browsers simultaneously

s = 0
links = []
for site in cfg.sites:
    DBG.lt_green('visiting site: {}'.format(site))
    command_sequence = CommandSequence.CommandSequence(site)


    # Start by visiting the page
    command_sequence.get(sleep=3, timeout=120)
    # Run commands across the NUMBRWSER browsers (simple parallelization)
    manager.execute_command_sequence(command_sequence)
    # Shuts down the browsers and waits for the data to finish logging
    if cfg.crawl_type == "base_adnauseum":
        if s < 10:
            r = requests.get(site)
            html_content = r.text
            # soup = bs4.BeautifulSoup(html_content, 'lxml')
            # for l in [a.get('href') for a in soup.find_all('a', href=True)]:
            lr= adnauseum_get_page_links(site,workers=1)
            for l in lr:
                links.append(l)

        if s == 11:

            url = links[random.randint(0, len(links)-1)]
            cfg.sites.append(url)
            cfg.sites.apend('http://www.drudgereport.com')
    s+=1

manager.close()

DBG.lt_cyan("shutting down server")
#
