
import json
import sqlite3
import argparse

import pandas as pd
from pprint import pprint as pp
import irlutils.file.file_utils as fu
from flask import g, Flask, request
import lcdk.lcdk  as LeslieChow
import requests
from threading import Timer
parser = argparse.ArgumentParser()
parser.add_argument('--DB', help="data base to write bids to")
parser.add_argument('--logsPath', help="bid server logsPath")
parser.add_argument('--crawl_id', help="crawl id")
parser.add_argument('--visit_id', help="visit id")
parser.add_argument('--visit_url', help="visit url")


args = parser.parse_args()
logsPath=args.logsPath
fu.touch(logsPath)
DBG = LeslieChow.lcdk(logsPath=logsPath)
DATABASE = args.DB
fu.touch(DATABASE)
fu.touch(logsPath)
app = Flask(__name__)
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
init_db()
def query_db(query, args=(), one=False):
    cur = get_db().execute(query, args)
    rv = cur.fetchall()
    cur.close()

    return (rv[0] if rv else None) if one else rv

def make_dicts(cursor, row):
    return dict((cursor.description[idx][0], value)
                for idx, value in enumerate(row))

def expand_columns(df, column):
    data = df.data
    columns = df.columns
    columns.append(column)
    expanded_df = pd.DataFrame(data, columns=columns)
    return expanded_df



@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


@app.route('/info', methods=['POST'])
def result():
    l = label.info
    DBG.warning("{}{}nrequest".format(l,sep))
    return 'Received !' # response to your request.i




@app.route('/gpt', methods=['POST'])
def gpt_result():
    l = label.gpt
    bids = [x for x in request.form.to_dict().keys()][0]
    # DBG.lt_cyan("{}{}request.form: {}".format(l, sep, request.form))

    # DBG.lt_cyan("{}{}bids{}{}".format(l,sep,sep, bids))

    return 'Received !' # response to your request.i



@app.route('/pbjs', methods=['POST'])
def pbjs_result():
    l = label.pbjs
    con = get_db()
    df = pd.read_sql_query('select * from pbjs_bids', con)
    # #DBG.lt_green("{}{}request.form: {}".format(l, sep, request.form.values())
    bids = json.loads(request.get_data().decode('utf-8'))['biddings']
#     DBG.lt_green("{}{}bids{}".format(l, sep, sep))


    for adUnitCode in bids:
        for bid in bids[adUnitCode]:
            bid['ad']=bid['ad'][0]
            bid['crawl_id'] = args.crawl_id
            bid['visit_id'] = args.visit_id
            bid['visit_url'] = args.visit_url

            #DBG.lt_green("{}{}bid{}{}".format(l, sep, sep, bid))
            df = df.append(bid, ignore_index=True)

    #DBG.lt_green("{}{}df.columns{}{}".format(l, sep, sep, df.columns))
    #DBG.lt_green("{}{}df{}{}".format(l, sep, sep, df))
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

@app.route('/kill', methods=['POST'])
def kill():
    def shutdown():
        requests.post('http://localhost:5050/seriouslykill')


    Timer(2.0, shutdown).start()  # wait 1 second
    return "Shutting down..."
if __name__ == "main":
    app.run(port=5050,debug=True)
app.run(port=5050,debug=True)
