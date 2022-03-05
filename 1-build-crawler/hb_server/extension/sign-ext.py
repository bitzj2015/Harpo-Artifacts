import os
import json
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--api')
parser.add_argument('--a')
args = parser.parse_args()
user = ''
key = ''
with open(args.api) as f: 
    data = json.load(f)
    user = data['user']
    key = data['key']

os.system('web-ext sign  -s . -a {} --api-key={} --api-secret={}'.format(args.a, user, key))
