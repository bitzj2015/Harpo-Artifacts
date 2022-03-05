CREATE TABLE IF NOT EXISTS http_requests(
        
        url TEXT
        

);



CREATE TABLE IF NOT EXISTS pbjs_bids(

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
    pbCg  REAL, 
    pbDg  REAL, 
    pbHg  REAL,
    pbLg  REAL,
    pbMg REAL,
    pbAg REAL,
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