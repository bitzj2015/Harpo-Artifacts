import os
import json
import time
import sqlite3
import uuid
import random
import argparse
import threading
import docker_api
import pandas as pd
import lcdk.lcdk  as LeslieChow
from time import gmtime, strftime
import irlutils.file.file_utils as fu
from multiprocessing import Pool, Process
import irlutils.url.crawl.database_utils as dbu
from contextlib import contextmanager
from subprocess import check_output


__author__ = 'johncook'
parser = argparse.ArgumentParser()
parser.add_argument('--dump_root')
parser.add_argument('--categories')
parser.add_argument('--cpus')
parser.add_argument('--stand_alone', action='store_true')
parser.add_argument('--experiment_type', help="base_crawl, pbjs_crawl")
args = parser.parse_args()
# A global summary of crawls.
STANDALONE=False
print(args)
if args.stand_alone:
    STANDALONE = args.stand_alone
NUM_CPUS=3
if args.cpus != None:
    NUM_CPUS=args.cpus
NUM_PROCESSES = 0
manifest_df = pd.DataFrame()
crawl_type_summary_df = pd.DataFrame()
base_crawl_summary_df = pd.DataFrame()

CFG = {'crawl_id':'',
                'start_time':'',
                'finish_time':'',
                'crawl_type':'',
                'test_site':'',
                'bid_site':'',
                'sites':[],
                'total':'',
                'dump_root':args.dump_root,
                'cmd':''
                }
DBG = LeslieChow.lcdk()
DOCKER_VOLUME = 'docker-volume'
DB = os.path.join(DOCKER_VOLUME, 'logs', 'manifest_{}.sqlite'.format(args.categories))
SCHEMA = 'manifest_schema.sql'
SCHEMA_SQL = str(open(SCHEMA).read())

path = os.path.join(DOCKER_VOLUME, 'logs')
fu.mkdir(path, exist_ok=True)
fu.touch(DB)
fu.chmod(DOCKER_VOLUME, mode=777,recursive=True)
con = sqlite3.connect(DB, isolation_level=None)
# con.execute('pragma journal_mode=wal')



def _launch_thread(td_idx, d_api, cfg):
    t = threading.Thread(target=d_api.run_container, )
    thead_name_str  = "{}_{}".format(cfg["crawl_id"], td_idx)
    t.setName(thead_name_str)
    t.start()

    DBG.red("started thread: {}".format(t.name))
    if not t.name in thread_pool:
        thread_pool[t.name] = t
        print(thread_pool)
        tp_str = f"{'t_idx':<20}{'thread_name':<60}{'started'}\n---------------------------------------------------------------------------------\n\n"
        cnt=0
        for t_name in thread_pool:
            tp_str += f'{cnt:<20}{t_name:<60}{thread_pool[t_name].is_alive()}\n---------------------------------------------------------------------------------\n'
            cnt+=1
        DBG.lt_cyan("added thread {} to pool\n\n\n\t\t\t----thread pool---\n\n\n\n{}\n\n\n\n\n".format(t.name,tp_str))

    else:
        ValueError()
    clean_up =[]
    DBG_OUT = 30
    elapsed = 0
    while threading.active_count()>3:
        time.sleep(5)
        elapsed+=5
        for t_name in thread_pool:
            td = thread_pool[t_name]
            if elapsed == DBG_OUT:
                DBG.lt_green("t_name: {}\nthread: {}\nthread_count: {}".format(t_name, td, threading.active_count()))
                elapsed = 0
            if not td.is_alive():
                # td.join(timeout=10)
                clean_up.append(t_name)
        for i in clean_up:
            try:
                thread_pool.pop(i)

            except:
                pass
        clean_up =[]




def connect_db():
    cursor = con.cursor()
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA_SQL)
    manifest_df = pd.read_sql_query("select * from manifest", con)
    crawl_type_summary_df = pd.read_sql_query("select * from crawl_type_summary", con)
    base_crawl_summary_df = pd.read_sql_query("select * from base_crawl_summary", con)
    return [con, cursor]

def write_db(cfg):
    manifest_line = {"crawl_id": str(cfg['crawl_id']),
                            "start_time": str(cfg['start_time']),
                            "finish_time" : str(cfg['finish_time']),
                            "crawl_type": str(cfg['crawl_type']),
                            "sites": str(cfg['sites']),
                            "dump_root" :  str(cfg['dump_root']),
                            "cmd": str(cfg['cmd'])}


    crawl_type_summary_line = {"crawl_id":cfg['crawl_id'],
                    "crawl_type": cfg['crawl_type'],
                    "test_site": cfg['test_site'],
                    "bid_site": cfg['bid_site']}

    base_crawl_summary_line = {"crawl_id":cfg['crawl_id'],
                    "crawl_type": cfg['crawl_type'],
                    "total": cfg['total']}

    DBG.lt_purple(manifest_line)
    manifest_df = pd.DataFrame(manifest_df)
    crawl_type_summary_df = pd.DataFrame(crawl_type_summary_df)
    base_crawl_summary_df = pd.DataFrame(base_crawl_summary_df)

    for c in manifest_line:
        manifest_df.at[idx, c] = manifest_line[c]
    for c in crawl_type_summary_line:
        crawl_type_summary_df.at[idx, c] = crawl_type_summary_line[c]
    for c in base_crawl_summary_line:
        base_crawl_summary_df.at[idx, c] = base_crawl_summary_line[c]

    manifest_df.to_sql('manifest',con, if_exists='replace',index=False)
    crawl_type_summary_df.to_sql('crawl_type_summary',con, if_exists='replace',index=False)
    base_crawl_summary_df.to_sql('base_crawl_summary',con, if_exists='replace',index=False)

def round_robbin_shuffle(l):
    round_robbin_l = l[1:len(l)]+[l[0]]
    return round_robbin_l

# make a config object

data = []
sites = []
with open('baseline_sites.json') as f:
    data = json.load(f)
    for d in data:
        for k in d:
            if k == args.categories:

                sites.append(d)
num = 10
start = 0
end = len(sites)
selected_sites_index = []
shuffle_selected_sites_index = []
crawl_seq = {}


with open('crawls_complete.json') as f:
    crawl_seq = json.load(f)
if not args.categories in crawl_seq:
    crawl_seq[args.categories] = []
for j in range(num):

    idx = random.randint(start, end-1)
    if args.categories != "base_alexa_Intent_top50":
        while idx in selected_sites_index and idx in crawl_seq[args.categories]:
            idx = random.randint(start, end-1)
    crawl_seq[args.categories].append(idx)
    selected_sites_index.append(idx)
shuffle_selected_sites_index = selected_sites_index
with open('crawls_complete.json', 'w') as f:
    json.dump(crawl_seq, f, indent=4, separators=(',',':'))


DBG.lt_green('category: {}\nsites index: {}\n'.format(args.categories, selected_sites_index))
thread_pool = {}


def get_pid(name):
    return check_output(["pidof",name])

def dispatch(selected_sites):
    td_idx = 0
    for i in list(range(10)):
        DBG.lt_green('round: {} sites_index: {}\n'.format(i, selected_sites))
        DBG.lt_green('')
        trimmed_sites = []
        for idx in selected_sites:
            trimmed_sites.append(sites[idx])

        idx = 0
    # cpu_set = ['1','2', '3', '4', '5', '6', '7', '8']
    idx = 0
    for t in trimmed_sites:
        site_idx = selected_sites[idx]
        print(site_idx, idx)
        sites_t = t[args.categories]
        base_sites = sites_t[0:9]
        tmp=""
        for s in base_sites:
            tmp+="\n\t{}".format(s)
        test_site = sites_t[10]
        bid_site = sites_t[11]
        DBG.lt_green("profile:{}\nprofile_index: {}\nbase_sites: {}\ntest_site: {}\nbid_site: {}\n".format(args.categories, site_idx, tmp, test_site, bid_site))


        manifest_df = pd.read_sql_query("select * from manifest", con)
        crawl_type_summary_df = pd.read_sql_query("select * from crawl_type_summary", con)
        base_crawl_summary_df = pd.read_sql_query("select * from base_crawl_summary", con)

        cfg = {'crawl_id':'',
            'start_time':'',
            'finish_time':'',
            'crawl_type':'',
            'test_site':'',
            'bid_site':'',
            'sites':[],
            'total':'',
            'cpuset':'',
            'dump_root':args.dump_root,
            'cmd':''
            }
        startTime = time.time()
        cfg["crawl_id"]= "{}.{}".format(int(startTime),uuid.uuid1())
        cfg["startTime"] = "{}".format(strftime("%Y%m%d%H%M%S0000", gmtime(startTime)))
        cfg['sites'] = t[args.categories]
        cfg['crawl_type'] = args.categories

        DBG.lt_green(cfg['cmd'])
        cmd = json.dumps(cfg)

        # cfg['cmd'] = "/usr/bin/python3 /opt/OpenWPM/crawl_stub.py --cfg '{}'".format(cmd)
        # d_api = docker_api.docker_api(cmd=cfg['cmd'], cfg=cfg, stand_alone=False)
        cmd = "python3 OpenWPM/crawl_stub.py --cfg {}".format(cmd)



        manifest_line = {"crawl_id": str(cfg['crawl_id']),
                        "start_time": str(cfg['start_time']),
                        "finish_time" : str(cfg['finish_time']),
                        "crawl_type": str(cfg['crawl_type']),
                        "sites": str(cfg['sites']),
                        "dump_root" :  str(cfg['dump_root']),
                        "cmd": str(cfg['cmd'])}


        crawl_type_summary_line = {"crawl_id":cfg['crawl_id'],
                        "crawl_type": cfg['crawl_type'],
                        "test_site": cfg['test_site'],
                        "bid_site": cfg['bid_site']}

        base_crawl_summary_line = {"crawl_id":cfg['crawl_id'],
                        "crawl_type": cfg['crawl_type'],
                        "total": cfg['total']}
        DBG.lt_purple(manifest_line)
        manifest_df = pd.DataFrame(manifest_df)
        crawl_type_summary_df = pd.DataFrame(crawl_type_summary_df)
        base_crawl_summary_df = pd.DataFrame(base_crawl_summary_df)

        for c in manifest_line:
            manifest_df.at[idx, c] = manifest_line[c]
        for c in crawl_type_summary_line:
            crawl_type_summary_df.at[idx, c] = crawl_type_summary_line[c]
        for c in base_crawl_summary_line:
            base_crawl_summary_df.at[idx, c] = base_crawl_summary_line[c]

        manifest_df.to_sql('manifest',con, if_exists='replace',index=False)
        crawl_type_summary_df.to_sql('crawl_type_summary',con, if_exists='replace',index=False)
        base_crawl_summary_df.to_sql('base_crawl_summary',con, if_exists='replace',index=False)

        con.commit()

        thread_name_str  = "{}_{}".format(cfg["crawl_id"], td_idx)
        p = Process(target=os.system, args=(cmd), name=thread_name_str)
        p.start()
        NUM_PROCESSES+=1



        cfg = {'crawl_id':'',
            'start_time':'',
            'finish_time':'',
            'crawl_type':'',
            'test_site':'',
            'bid_site':'',
            'sites':[],
            'cpuset':'',
            'total':'',
            'dump_root':args.dump_root,
            'cmd':''
            }
        idx+=1
        td_idx+=1
        # selected_sites= round_robbin_shuffle(selected_sites)


pool = Pool(processes=1)
dispatch(shuffle_selected_sites_index)
