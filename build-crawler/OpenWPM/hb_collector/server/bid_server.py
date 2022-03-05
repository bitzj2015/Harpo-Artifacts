import json
import sqlite3
import argparse

import pandas as pd 
from pprint import pprint as pp
from flask import g, Flask, request
import lcdk.lcdk  as LeslieChow
import requests
from threading import Timer
parser = argparse.ArgumentParser()
parser.add_argument('--DB', help="data base to write bids to")
parser.add_argument('--logsPath', help="data base to write bids to")

args = parser.parse_args()
DBG = LeslieChow.lcdk(logsPath=args.logsPath)
DATABASE = args.DB
app = Flask(__name__)
DBG.red(DATABASE)
class Labels: 
    pbjs = "PBJS"
    gpt = "GPT"
    info = "INFO"
label=Labels()
sep = "\n========\n"

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db

def init_db():
    with app.app_context():
        db = get_db()
        with app.open_resource('schema.sql', mode='r') as f:
            db.cursor().executescript(f.read())
        db.commit()
def query_db(query, args=(), one=False):
    cur = get_db().execute(query, args)
    rv = cur.fetchall()
    cur.close()

    return (rv[0] if rv else None) if one else rv

def make_dicts(cursor, row):
    return dict((cursor.description[idx][0], value)
                for idx, value in enumerate(row))
    
init_db()


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


@app.route('/info', methods=['POST'])
def result():
    l = label.info
    print(l)
    DBG.lt_cyan("{}{}request.form: {}".format(l, sep, request.form)) # should display 'bar'
    return 'Received !' # response to your request.i




@app.route('/gpt', methods=['POST'])
def gpt_result():
    l = label.gpt
    bids = [x for x in request.form.to_dict().keys()][0]

    DBG.lt_cyan("{}{}request.form: {}".format(l, sep, request.form)) # should display 'bar'
    
    DBG.lt_cyan("{}{}bids{}{}".format(l,sep,sep, bids)) # should display 'bar'


    return 'Received !' # response to your request.i



@app.route('/pbjs', methods=['POST'])
def pbjs_result():
    l = label.pbjs
    con = get_db()
    df = pd.read_sql_query('select * from pbjs_bids', con)
    DBG.lt_green("{}{}request.getData: {}".format(l, sep, request.get_data()))
    bids = json.loads(request.get_data().decode('utf-8'))['biddings']
    DBG.lt_green("{}{}bids{}{}".format(l, sep, sep, bids))

    columns  =[     "domain",
                    "ad",
                    "adId",
                    "adUnit",
                    "adUnitCode",
                    "auctionId",
                    "bidder",
                    "bidderCode",
                    "cpm",
                    "creativeId",
                    "currency",
                    "mediaType",
                    "msg",
                    "netRevenue",
                    "pbCg", 
                    "pbDg", 
                    "pbHg",
                    "pbLg",
                    "pbMg",
                    "pbAg",
                    "responseTimestamp",
                    "width",
                    "height",
                    "size",
                    "source",
                    "statusMessage",
                    "timeToRespond",
                    "ttl",
                    "type"]
    item = {}
    for adUnitCode in bids: 
        for bid in bids[adUnitCode]:
            bid['ad']=bid['ad'][0]
            row = {}
            for c in bid: 
                if c in columns: 
                    row.update({c:bid[c]})
                    # DBG.lt_green("{}{}bid{}{}".format(l, sep, sep, bid)) # should display 'bar'
            df = df.append(row, ignore_index=True)

    DBG.lt_green("{}{}df.columns{}{}".format(l, sep, sep, df.columns)) # should display 'bar'
    DBG.lt_green("{}{}df{}{}".format(l, sep, sep, df)) # should display 'bar'
    df.to_sql('pbjs_bids',con, if_exists='replace', index=False)
    con.commit()
    return 'Received !' # response to your request.i



@app.route('/seriouslykill', methods=['POST'])
def seriouslykill():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()

    print("Shutting down...")
    return 'shutting down ...'
@app.route('/kill', methods=['POST'])
def kill():
    def shutdown():
        con = get_db()
        con.commit()
        requests.post('http://localhost:5050/seriouslykill')
 

    Timer(1.0, shutdown).start()  # wait 1 second
    return "Shutting down..."
if __name__ == "main":
    app.run(port=5050,debug=True)

app.run(port=5050,debug=True)
