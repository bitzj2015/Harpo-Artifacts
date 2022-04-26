import json
from random import choice

with open("./product_url_100.json") as f:
    obfuscation_url_set = json.load(f)
    obfuscation_url_cats = list(obfuscation_url_set.keys())

def choose_max_on_constraint(max_list, disabled):
    print("choose_max_on_constraint reached")
    for i in max_list:
        cat = obfuscation_url_cats[i]
        if cat in disabled:
            pass
        else:
            # return a randomly chosen obfuscation url from the most relevant category (that is not disabled)
            return cat, choice(list(obfuscation_url_set[cat].keys()))

def load_disabled(file):
    print("load_disabled reached")
    storage=[]
    with open(file, "r") as json_file:
        pref=json.loads(json_file.read())
    for i in pref:
        if pref[i]["checked"]==False:
            storage.append(pref[i]["name"])
    print("load_disabled finished")
    return storage