import os
import irlutils.url.crawl.owpm_helpers as owpm
import sqlite3
import pandas as pd
import time


def get_content(hostname, crawl_sqlite, ldb):
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