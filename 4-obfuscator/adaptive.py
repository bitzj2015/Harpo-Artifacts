import json
import numpy as np
import seaborn as sns
import scipy.io as sio

STAT = 100
alpha = 0.2
metric = "M1"
for index in range(2):
    data = json.load(open("../DATA/rl_result/adap_Intent_{}_{}_{}.json".format(index, alpha, metric),"r"))
    count = np.zeros((STAT,STAT))
    for i in range(STAT):
        for j in range(STAT):
            if j == i:
                continue
            total = 0
            for key in data[j].keys():
                count[i,j] += (data[i][key] - data[j][key]) ** 2
                total += data[j][key]
            #     if data[i][key] > 0 and data[j][key] == 0:
            #         count[i,j] += data[i][key]
            # count[i,j] /= total
            print(total)
            count[i,j] = np.sqrt(count[i,j]) / total

    sio.savemat("../DATA/rl_result/adap_Intent_{}_{}_{}.json".format(index, alpha, metric), {"adap_Intent_{}_M1.json".format(index):count})
    svm = sns.heatmap(count)
    figure = svm.get_figure()    
    figure.savefig('adap_Intent_{}_{}_{}_new.png'.format(index, alpha, metric), dpi=400)


