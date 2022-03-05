import json
import random
import irlutils.url.crawl.domain_utils as du

__author__ = 'johncook'

class site_manager:
    def __init__(self,**kwargs):
        pass

    def trim_sites(self, **kwargs):
        path = (kwargs.get('sites', ''))
         
        with open(path) as f: 
            sites = json.load(f)
        s_trimmed = {}
        used_sequences = {}
        skl = {}
        for s in sites: 
            for k in s: 
                if k not in skl:
                    skl.update({k:1})
                else:
                    skl[k]+=1
        print(skl)
        print(path)
        for s in sites:
            for k in s:
                if k not in s_trimmed: 
                    s_trimmed.update({k:[]})    
                    used_sequences.update({k:[]})
                site_11 = s[k][10]
                site_11 = du.hostname_subparts((site_11), include_ps=True)

                site_11_host = site_11[-2].split('.'+site_11[-1])[0]
        

                if site_11_host not in [s[k][:9]]:
                    idx = s_trimmed[k].__len__()
                    if idx < 10:
                        pickj = [] 
                        while len(pickj) < 50:
                            for i in range(skl[k]):
                                pickj.append(random.randrange(skl[k]))
                        pick = random.sample(pickj, 15)
                        random.choice(pick)
                        if pick not in used_sequences[k]:
                            s_trimmed[k].append({str(idx):s[k]})
                            used_sequences[k].append(pick)
                        # print(used_sequences[k])
        return s_trimmed