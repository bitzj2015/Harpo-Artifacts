from __future__ import absolute_import
import http.client
from six.moves import range
from urllib.parse import urlparse, urljoin
from automation import CommandSequence, TaskManager
import os
import bs4
import time
import random
import requests
import sqlite3
import pandas as pd
import concurrent.futures
import lcdk.lcdk as LeslieChow
import irlutils.url.crawl.owpm_helpers as owpm



"""
citation: https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter03/crawling_web_step1.py
"""
CFG = {'crawl_id':'',
                'start_time':int(time.time()),
                'finish_time':'',
                'crawl_type':"base_adnauseum",
                'test_site':'',
                'bid_site':'http://www.drudgereport.com',
                'sites':["http://www.google.com",
            "http://www.youtube.com",
            "http://www.baidu.com",
            "http://www.tmall.com",
            "http://www.qq.com",
            "http://www.sohu.com",
            "http://www.facebook.com",
            "http://www.taobao.com",
            "http://www.wikipedia.org",
            "http://www.yahoo.com",
            "http://www.1688.com"],
                'total':'',
                'dump_root':"test_dump_root",
                'cmd':''
                }

DBG = LeslieChow.lcdk(print_output=True, logsPath="demo.py.log")
logsPath=os.path.abspath('logs')
logs_dir = logsPath.replace('/openwpm.log',"")
ldb = os.path.join(os.path.abspath(logs_dir), 'content.ldb')

def get_content(hostname, LDB):

    DBG.lt_green("{},{}".format(hostname, logs_dir))
    crawl_sqlite = os.path.join(os.path.abspath(logs_dir), 'crawl-data.sqlite')

    con  = sqlite3.connect(crawl_sqlite)
    cur = con.cursor()
    http_responses = pd.read_sql_query("SELECT * FROM http_responses", con)
    print(http_responses)
    content = None
    print('getting content...')
    for i in http_responses.index:
        url = http_responses.at[i, 'url']

        try:
            hashes = [x for x in owpm.get_distinct_content_hashes(cur, url)][0]
            content_hash = bytes(hashes, "utf-8")
            content = owpm.get_content(LDB, hashes)
        except Exception  as e:
            print("Exception: {}".format(e))
            pass

        if url == hostname:
            break
    return content

#https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter03/crawling_web_step1.py
def process_link(url, LDB):
    html_text = get_content(url, LDB)
    DBG.lt_cyan('ADN: Extracting links from {}'.format(url))
    parsed_source = urlparse(url)
    page = bs4.BeautifulSoup(html_text, 'html.parser')

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
    DBG.lt_cyan("to_check: {}".format(to_check))
    max_checks = 10

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        while to_check:
            futures = [executor.submit(process_link, url, LDB)
                       for url in to_check]
            for data in concurrent.futures.as_completed(futures):
                new_links = data.result()


                for link in new_links:
                    checked_links.add(link)
                    if link not in checked_links and link not in to_check:
                        to_check.append(link)

                max_checks -= 1
                if not max_checks:
                     return
    print(checked_links)
    return checked_links





NUM_BROWSERS = 1
sites = ['http://www.cnn.com']

# Loads the manager preference and 3 copies of the default browser dictionaries
manager_params, browser_params = TaskManager.load_default_params(NUM_BROWSERS)

# Update browser configuration (use this for per-browser settings)
for i in range(NUM_BROWSERS):
    # Record HTTP Requests and Responses
    browser_params[i]['http_instrument'] = True
    # Enable flash for all three browsers
    browser_params[i]['disable_flash'] = False
    browser_params[i]['headless'] = True # Launch only browser 0 headless
    browser_params[i]['hb_collector'] = True  # Launch only browser 0 headless
    browser_params[i]['save_all_content'] = True  # Launch only browser 0 headless
    browser_params[i]['profile_dir'] = os.path.join(CFG['dump_root'], CFG['crawl_type'], CFG['crawl_id'])
# Update TaskManager configuration (use this for crawl-wide settings)
manager_params['data_directory'] = 'logs'
manager_params['log_directory'] = 'logs'

# Instantiates the measurement platform
# Commands time out by default after 60 seconds
manager = TaskManager.TaskManager(manager_params, browser_params)
sites_visited=1
# Visits the sites with all browsers simultaneously
if CFG['crawl_type'] == "adn":
    sites = sites[:10]
bid_site = CFG['bid_site']

for site in sites:
    command_sequence = CommandSequence.CommandSequence(site)

    # Start by visiting the page
    command_sequence.get(sleep=0, timeout=60)

    # dump_profile_cookies/dump_flash_cookies closes the current tab.
    command_sequence.dump_profile_cookies(120)
    command_sequence.dump_profile(browser_params[i]['profile_dir'])
    # index='**' synchronizes visits between the three browsers
    manager.execute_command_sequence(command_sequence)


# Shuts down the browsers and waits for the data to finish logging
manager.close()

if CFG['crawl_type'] == "base_adnauseum":
    adn_site = []
    content_links = []
    ldb_path = os.path.join(manager_params['log_directory'],'content.ldb')
    LDB=owpm.get_leveldb(ldb_path)
    DBG.lt_green("sites {}".format(sites))
    for url in sites:
        links = adnauseum_get_page_links(url, LDB)
        for x in links:
            if x not in content_links:
                content_links.append(x)
    # for i in list(range(random.randint(1,5))):
    #     random.shuffle(adn_sites)
    #     time.sleep(random.randint(1,10))
    adn_site = adn_site[random.randint(0, adn_site.__len__() -1)]

    adn_site.append(adn_site)
    adn_site.append(bid_site)

    try:
        browser_params[0]['profile_tar'] = os.path.join(browser_params[i]['profile_dir'],  'profile.tar.gz')
        manager = TaskManager.TaskManager(manager_params, browser_params)
        profile_dir = browser_params[0]['profile_dir']

        for site in adn_site:
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
