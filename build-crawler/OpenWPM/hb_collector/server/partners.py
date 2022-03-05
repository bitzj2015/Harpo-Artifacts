import pprint
import pymongo
import json
from flask import Flask, request

app = Flask(__name__)
myclient = pymongo.MongoClient('mongodb://localhost:27017/')
mydb = myclient['headerbidding_partners']
mycol = mydb['hb_partners']

@app.route('/', methods=['POST'])
def result():
    reqs = request.form.to_dict()
    bids = {}
    for req in reqs:
        bids = json.loads(req)
        print(bids)
        if 'biddings' in bids:
            if bids['type']=='pbjs':
                bids = bids['biddings'] 
                for adId in bids: 
                    for bid in bids[adId]:
                        # print(bid)
                        mycol.insert_one(bid)
           

                


                # print(bids['biddings'][bid])
    return 'Received !' # response to your request.i

if __name__ == "main":
    app.run(port=5050,debug=True)
app.run(port=5050,debug=True)