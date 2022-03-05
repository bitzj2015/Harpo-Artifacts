CREATE TABLE IF NOT EXISTS http_requests(
        file_id TEXT,
        url TEXT,
        headers TEXT,
        visit_id TEXT,
        crawl_id TEXT,
        top_level_url TEXT,
        volume INTEGER,
        buying_intent TEXT,
        persona_category TEXT,
        time_stamp TEXT

);



CREATE TABLE IF NOT EXISTS pbjs_bids(
    crawl_id TEXT,
    visit_id TEXT,
    visit_url TEXT,
    domain TEXT,
    ad TEXT,
    adId TEXT,
    adUnit TEXT,
    adUnitCode TEXT,
    auctionId BLOB,
    bidder TEXT,
    bidderCode TEXT,
    cpm  REAL,
    creativeId INTEGER,
    currency TEXT,
    mediaType TEXT,
    msg TEXT,
    netRevenue BOOL,
    pbAg REAL,
    timestamp INTEGER,
    responseTimestamp INTEGER,
    width INTEGER,
    height INTEGER,
    size TEXT,
    source TEXT,
    statusMessage TEXT,
    timeToRespond INTEGER,
    ttl INTEGER,
    type TEXT
);


CREATE TABLE IF NOT EXISTS gpt_bids(

    campaingId TEXT,
    advertiserId TEXT,
    size TEXT,
    adUnitCode TEXT,
    adUnitPath TEXT,
    targeting TEXT,
    event TEXT,
    domain TEXT,
    time TEXT,
    type TEXT

);