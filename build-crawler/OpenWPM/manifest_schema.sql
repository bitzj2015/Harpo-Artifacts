CREATE TABLE IF NOT EXISTS manifest (
    crawl_id TEXT,
    start_time TEXT, 
    finish_time TEXT, 
    crawl_type TEXT,
    sites TEXT, 
    dump_root TEXT, 
    cmd TEXT);

CREATE TABLE IF NOT EXISTS crawl_type_summary (
    crawl_id TEXT, 
    crawl_type TEXT,
    test_site TEXT, 
    bid_site TEXT);


CREATE TABLE IF NOT EXISTS base_crawl_summary (
    crawl_id TEXT, 
    crawl_type TEXT,
    total TEXT);