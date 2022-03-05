from __future__ import absolute_import

import os
import json
import time
from time import gmtime, strftime
import copy
import argparse
from automation import CommandSequence, TaskManager
from automation.SocketInterface import clientsocket
from multiprocessing import Process
import irlutils.file.file_utils as fu
import lcdk.lcdk as LeslieChow
import requests
import html2text
from adblockparser import AdblockRules


__author__='jiang'

with open('/opt/OpenWPM/easylist.txt', 'rb') as f:
    raw_rules_1 = f.read().decode('utf8').splitlines()
with open('/opt/OpenWPM/easyprivacy.txt', 'rb') as f:
    raw_rules_2 = f.read().decode('utf8').splitlines()
ad_filter = AdblockRules(raw_rules_1 + raw_rules_2)


p = argparse.ArgumentParser()
p.add_argument('--cfg', help="where all the magic is stored")
args  = p.parse_args()
NUM_BROWSERS = 1
CFG = json.loads(args.cfg)


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
    browser_id =None
    sites = []
    manager_params = None
    browser_params = None
    params = load_default_params(manager_params, browser_params, num_browsers=NUM_BROWSERS)
    manager_params = params['mp']
    browser_params = params['bp']

    crawl_type = None
    start_time = None,
    dump_root = None,
    load_profile = True


def collect_links(table_name, get_bids, url, get_ads, url_id, content_dir, bids_path, rules, click, **kwargs):
    driver = kwargs['driver']
    manager_params = kwargs['manager_params']
    text = html2text.html2text(driver.page_source)
    with open(content_dir + "/url_" + str(url_id) + ".json", "w") as json_file:
        json.dump({"url":driver.current_url, "text":text}, json_file)
    if get_bids == 1:
        driver.find_element_by_id('rawdata-tab').click()
        content = driver.find_element_by_tag_name('pre').text
        with open(content_dir + "/oracle_segment.json", "w") as json_file:
            json.dump({"url":url, "text":content}, json_file)
    if get_ads == 1:
        def find_all_iframes(driver):
            ad_list = {}
            iframes = driver.find_elements_by_xpath("//iframe")
            for index, iframe in enumerate(iframes):
                # Your sweet business logic applied to iframe goes here.
                driver.switch_to.frame(index)
                elems = driver.find_elements_by_xpath("//a[@href]")
                for elem in elems:
                    ad_list[elem.get_attribute("href")] = 1
                find_all_iframes(driver)
                driver.switch_to.parent_frame()
            return ad_list
        ad_list = find_all_iframes(driver)
        print("The number of detected ads urls:", len(ad_list))

        current_url = driver.current_url

        sock = clientsocket()
        sock.connect(*manager_params['aggregator_address'])

        query = ("CREATE TABLE IF NOT EXISTS %s ("
                    "top_url TEXT, link TEXT, bids_path TEXT);" % table_name)
        sock.send(("create_table", query))

        for link in list(ad_list.keys()):
            query = (table_name, {
                "top_url": current_url,
                "link": link,
                "bids_path": bids_path
            })
            sock.send(query)
        if click:
            T = time.time()
            link_list = list(ad_list.keys())
            if len(link_list) > 0:
                flag = 0
                for index in range(len(link_list)):
                    visit_ad_url = link_list[index]
                    if rules.should_block(visit_ad_url,{'third-party': True}):
                        flag = 1
                        break
                print(time.time()-T)
                if flag == 1:
                    query = ("CREATE TABLE IF NOT EXISTS %s ("
                                "top_url TEXT, click TEXT, bids_path TEXT);" % "click_ads")
                    sock.send(("create_table", query))
                    query = ("click_ads", {
                            "top_url": current_url,
                            "click": visit_ad_url,
                            "bids_path": bids_path
                    })
                    sock.send(query)
                    sock.close()
                    print("Clicking ad:", visit_ad_url, time.time()-T)
                    driver.get(visit_ad_url)
                    time.sleep(60)
                    print("Clicking ad done", visit_ad_url, time.time()-T)
                    text = html2text.html2text(driver.page_source)
                    with open(content_dir + "/url_ad_" + str(url_id) + ".json", "w") as json_file:
                        json.dump(text, json_file)
                else:
                    sock.close()
                    print("No valid ads!")
        else:
            sock.close()

cfg = config()
# print(cfg.browser_params)
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
    cfg.browser_params[i]['save_all_content'] = False
    if CFG['load_profile_dir'] != None:
        cfg.browser_params[i]['profile_tar'] = CFG['load_profile_dir']
    if CFG['save_profile_dir'] != None:
        cfg.browser_params[i]['profile_archive_dir'] = CFG['save_profile_dir']


cfg.browser_id = CFG['browser_id']
cfg.crawl_type = CFG["crawl_type"]
cfg.dump_root = CFG["dump_root"]
cfg.start_time = CFG['start_time']
#crawl stub logging
fu.mkdir(os.path.join(CFG['dump_root'], CFG['crawl_type'], CFG['browser_id'], 'bids'))
"""
launch bid server inside of docker container
"""
reward = 0.0
avg_bids = 0.0
Time = time.time()
fu.mkdir(os.path.join(CFG['dump_root'], CFG['crawl_type'], CFG['browser_id'], 'bids', \
    '{}'.format(strftime("%Y%m%d%H%M%S", gmtime(Time)))))
bids_log = os.path.join(CFG['dump_root'], CFG['crawl_type'], CFG['browser_id'], 'bids', \
    '{}'.format(strftime("%Y%m%d%H%M%S", gmtime(Time))), 'hb_bids.log')
fu.touch(bids_log)
bids_db = os.path.join(CFG['dump_root'], CFG['crawl_type'], CFG['browser_id'], 'bids', \
    '{}'.format(strftime("%Y%m%d%H%M%S", gmtime(Time))), 'hb_bids.sqlite')
fu.touch(bids_db)
bids_path = os.path.join(CFG['dump_root'], CFG['crawl_type'], CFG['browser_id'], 'bid.json')
fu.touch(bids_path)

bids_content = os.path.join(CFG['dump_root'], CFG['crawl_type'], CFG['browser_id'], 'bids', \
    '{}'.format(strftime("%Y%m%d%H%M%S", gmtime(Time))), 'content')
fu.mkdir(bids_content)
if CFG['run_bids_server'] == True:
    cmd = str('/usr/bin/python3 /opt/OpenWPM/hb_server/server/bid_server.py --DB {} \
    --logsPath {} --crawl_id {} --visit_id {} --visit_url "{}"'.format(bids_db, bids_log, CFG['browser_id'], "1",CFG["sites"]))
    p = Process(target=os.system, args=(cmd,))
    p.start()


logsPath = os.path.join(CFG['dump_root'], CFG['crawl_type'], CFG['browser_id'], 'openwpm', 'openwpm.log')
fu.mkdir(os.path.join(CFG['dump_root'], CFG['crawl_type'], CFG['browser_id'], 'openwpm'))
fu.touch(logsPath)
DBG = LeslieChow.lcdk(logsPath=logsPath, print_output=False)
#initiate config
crawl_dir = os.path.join(CFG['dump_root'], CFG['crawl_type'], CFG['browser_id'], 'openwpm')
cfg.manager_params['data_directory'] = os.path.abspath(crawl_dir)
cfg.manager_params['log_directory'] =  os.path.abspath(crawl_dir)
DBG.lt_green("Starting crawl in {}".format(str(CFG['browser_id'])))
cfg.sites = CFG["sites"]
cfg.sites_getads = CFG["sites_getads"]
manager = TaskManager.TaskManager(cfg.manager_params, cfg.browser_params)
count = 0
for site in cfg.sites:
    DBG.lt_green('Visiting site {} in {}'.format(site, str(CFG['browser_id'])))
    command_sequence = CommandSequence.CommandSequence(site)
    # Start by visiting the page (wait 3 seconds before HTTP GET cmd executed)
    # timeout is max time (sec) to wait for HTTP response
    # command_sequence.get(sleep=60, timeout=90)
    get_bids = 0
    timeout = 90
    command_sequence.get(sleep=1, timeout=90)
    command_sequence.run_custom_function(collect_links, ('page_links', get_bids, site, cfg.sites_getads[count], count+1, \
                                                         bids_content, bids_db, ad_filter, False),timeout=timeout)
#     command_sequence.save_screenshot()
    command_sequence.recursive_dump_page_source()
    manager.execute_command_sequence(command_sequence)
    count += 1
# Shuts down the browsers and waits for the data to finish logging
manager.close()

if CFG['run_bids_server'] == True:
    DBG.lt_cyan("shutting down bids server")
    requests.post('http://localhost:5050/kill')