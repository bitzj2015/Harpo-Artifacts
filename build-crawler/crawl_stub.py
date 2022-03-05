from __future__ import absolute_import

import os
import json
import copy
import random
import sqlite3
import argparse
import multiprocessing
import subprocess
import pandas as pd
from automation import CommandSequence, TaskManager
from multiprocessing import Process
import irlutils.file.file_utils as fu
import lcdk.lcdk as LeslieChow
import requests

__author__='johncook'

p = argparse.ArgumentParser()
p.add_argument('--cfg', help="where all the magic is stored")
args  = p.parse_args()
NUM_BROWSERS = 1
CFG = json.loads(args.cfg)

"""
citation: https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter03/crawling_web_step1.py
"""
DBG = LeslieChow.lcdk(print_output=True, logsPath="demo.py.log")
logsPath=os.path.abspath('logs')
logs_dir = logsPath.replace('/openwpm.log',"")
ldb = os.path.join(os.path.abspath(logs_dir), 'content.ldb')

def get_content(hostname, LDB):
    LDB = bytes(LDB)
    DBG.lt_green("{},{}".format(hostname, logs_dir))
    crawl_sqlite = os.path.join(os.path.abspath(logs_dir), 'crawl-data.sqlite')

    con  = sqlite3.connect(crawl_sqlite)
    cur = con.cursor()
    http_responses = pd.read_sql_query("SELECT * FROM http_responses", con)
    print(http_responses)
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
def process_link(url, LDB):
    html_text = get_content(url, LDB)
    DBG.lt_cyan('ADN: Extracting links from {}'.format(url))
    parsed_source = urlparse(url)
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

def adnauseum_get_page_links(base_url, LDB, workers=1):
    checked_links = set()
    to_check = [base_url]
    max_checks = 10

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        while to_check:
            futures = [executor.submit(process_link, url, LDB)
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





"""
Initilize manager and browser configurations
"""
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
# print(cfg.browser_params)
def baseline_config(cfg):
    for i in range(NUM_BROWSERS):
            cfg.browser_params[i]['http_instrument'] = True
            # Record cookie changes
            cfg.browser_params[i]['cookie_instrument'] = True
            # Record Navigations
            cfg.browser_params[i]['navigation_instrument'] = True
            # Record JS Web API calls
            cfg.browser_params[i]['js_instrument'] = True
            # Enable flash for all browsers
            cfg.browser_params[i]['disable_flash'] = True
            cfg.browser_params[i]['hb_collector'] = True
            cfg.browser_params[i]['controlled_scroll'] = True
            cfg.browser_params[i]['headless'] = True
            cfg.browser_params[i]['profile_dir'] = os.path.join(CFG['dump_root'], CFG['crawl_type'], CFG['crawl_id'])
            cfg.browser_params[i]['profile_tar'] = os.path.join(cfg.browser_params[i]['profile_dir'],  'profile_{}.tar.gz'.format(CFG['crawl_type']))

    return cfg


cfg = baseline_config(cfg)
cfg.crawl_id = CFG['crawl_id']
cfg.crawl_type = CFG["crawl_type"]
cfg.dump_root = CFG["dump_root"]
cfg.start_time = CFG['start_time']
#crawl stub logging
fu.mkdir(os.path.join(CFG['dump_root'], CFG['crawl_type'], CFG['crawl_id'], 'bids'))
bids_log = os.path.join(CFG['dump_root'], CFG['crawl_type'], CFG['crawl_id'], 'bids', 'hb_bids.log')
fu.touch(bids_log)
bids_db = os.path.join(CFG['dump_root'], CFG['crawl_type'], CFG['crawl_id'], 'bids', 'hb_bids.sqlite')
fu.touch(bids_db)


"""
launch bid server inside of docker container
"""
cmd = str('/usr/bin/python3 /opt/OpenWPM/hb_server/server/bid_server.py --DB {} --logsPath {} --crawl_id {}'.format(bids_db, bids_log, CFG['crawl_id']))
p = Process(target=os.system, args=(cmd,))
p.start()


logsPath = os.path.join(CFG['dump_root'], CFG['crawl_type'], CFG['crawl_id'], 'openwpm', 'openwpm.log')
fu.mkdir(os.path.join(CFG['dump_root'], CFG['crawl_type'], CFG['crawl_id'], 'openwpm'))
fu.touch(logsPath)
DBG = LeslieChow.lcdk(logsPath=logsPath, print_output=False)

#initiate config
cfg.sites = CFG["sites"]

DBG.lt_cyan("{}".format(CFG))
#initiate bids server
bids_directory = os.path.join(cfg.dump_root, 'bids')
fu.mkdir(bids_directory)
crawl_dir = os.path.join(CFG['dump_root'], CFG['crawl_type'], CFG['crawl_id'], 'openwpm')
cfg.manager_params['data_directory'] = os.path.abspath(crawl_dir)
cfg.manager_params['log_directory'] =  os.path.abspath(crawl_dir)

# DBG.lt_cyan("{}".format(cfg.manager_params))
# DBG.lt_cyan("{}".format(cfg.browser_params))

# Commands time out by default after 60 seconds
DBG.lt_green("starting crawl")
manager = TaskManager.TaskManager(cfg.manager_params, cfg.browser_params)

# Visits the the sites
# wait 3 seconds before issuing a get request

for site in cfg.sites:
    DBG.lt_green('visiting site: {}'.format(site))
    command_sequence = CommandSequence.CommandSequence(site)
    # Start by visiting the page (wait 3 seconds before HTTP GET cmd executed)
    # timeout is max time (sec) to wait for HTTP response
    command_sequence.get(sleep=3, timeout=60)
    command_sequence.save_screenshot()

    manager.execute_command_sequence(command_sequence)

# Shuts down the browsers and waits for the data to finish logging
manager.close()

#get all the links


if CFG['crawl_type'] == "base_adnauseum":
    LDB= os.path.join()
    for url in sites:
        adnauseum_get_page_links(url, )
    try:
        browser_params[0]['profile_tar'] = os.path.join(cfg.browser_params[i]['profile_dir'],  'profile.tar.gz')
        manager = TaskManager.TaskManager(manager_params, browser_params)
        profile_dir = cfg.browser_params[0]['profile_dir']

        for site in test_and_bid_site:
            command_sequence = CommandSequence.CommandSequence(site)

            # Start by visiting the page
            command_sequence.get(sleep=0, timeout=60)

            # dump_profile_cookies/dump_flash_cookies closes the current tab.
            command_sequence.dump_profile_cookies(120)
            command_sequence.dump_profile(profile_dir)
            # index='**' synchronizes visits between the three browsers
            manager.execute_command_sequence(command_sequence)
    except Exception as e:
        DBG.error("ADN: Bid and Test site exception {}".format())
    manager.close()
DBG.lt_cyan("shutting down bids server")

requests.post('http://localhost:5050/kill')
