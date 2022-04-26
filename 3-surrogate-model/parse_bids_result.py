'''
Process bidding result from ./run.sh
'''
import os, csv, json
seg_result = {}
VERSION = "_sep"

from heapq import heappop, heappush
for tag in ["oracle"]:
    h = []
    F_list = []
    with open(f"../DATA/analysis_result/{tag}_result{VERSION}.json", "r") as f:
        name_list = json.load(f)
    seg_result = {}
    for name in name_list.keys():
        try:
            filename = "loss_new_" + name + ".csv"
            if "collect" in filename:
                continue
            seg_result[name] = {}
            seg_result[name].update(name_list[name])
            count_0 = name_list[name]["0"]
            count_1 = name_list[name]["1"]
            max_score = 0
            best_acc_0 = 0
            best_acc_1 = 0
            TP = 0
            TN = 0
            FP = 0
            FN = 0
            precision = 0
            recall = 0
            F1 = 0
            with open("../DATA/csv/" + filename) as csvfile:
                readCSV = csv.reader(csvfile, delimiter=',')
                skip_first_row = 1
                for row in readCSV:
                    if skip_first_row:
                        skip_first_row = 0
                        continue
                    acc_0 = float(row[-2])
                    acc_1 = float(row[-1])
                    if (acc_0 + acc_1) / 2 > max_score:
                        max_score = (acc_0 + acc_1) / 2
                        best_acc_0 = acc_0
                        best_acc_1 = acc_1
                        # Update F1
                        TP = acc_1 * count_1
                        TN = acc_0 * count_0
                        FP = (1 - acc_0) * count_0
                        FN = (1 - acc_1) * count_1
                        precision = TP / (TP + FP)
                        recall = TP / (TP +FN)
                        if precision ==0 or recall == 0:
                            F1 = 0
                        else:
                            F1 = 2 * precision * recall / (precision + recall)
                            F1 = max_score
                    
            seg_result[name].update({"F1":F1, "precision":precision, "recall":recall, \
                                    "acc_0": best_acc_0, "acc_1": best_acc_1})

            heappush(h, (-F1, name, seg_result[name]["acc_0"], seg_result[name]["acc_1"])) 
        except:
            continue
        
    csvFile = open(f"../DATA/analysis_result/end2end_{tag}_result{VERSION}.csv", 'w', newline='')
    writer = csv.writer(csvFile)
    writer.writerow(["time", "ad_location", "bidder", "F1", "precision", "recall", \
                    "class_0_acc", "class_1_acc", "class_0_num", "class_1_num"])

    FP = []
    TP = []
    count = 0
    for i in range(len(h)):
        item = heappop(h)
        print(item)
        name = item[1]
        F1 = seg_result[name]["F1"]
        precision = seg_result[name]["precision"]
        recall = seg_result[name]["recall"]
        class_0_acc = seg_result[name]["acc_0"]
        class_1_acc = seg_result[name]["acc_1"]
        class_0_num = seg_result[name]["0"]
        class_1_num = seg_result[name]["1"]
        # if name.startswith("base_bids"):
        #     collect_time = "no-wait"
        #     bidder = item[1].split("_")[-1]
        #     ad_location = item[1][10:][:-len(bidder)-1]
        #     print(bidder,ad_location, class_0_num, class_1_num, precision, recall)
        # elif name.startswith("collect_bids"):
        #     collect_time = "wait"
        #     bidder = item[1].split("_")[-1]
        #     ad_location = item[1][13:][:-len(bidder)-1]
        #     print(bidder,ad_location, class_0_num, class_1_num, precision, recall)
        print(name, class_0_num, class_1_num, precision, recall, class_0_acc, class_1_acc)
        writer.writerow([name, F1, precision, recall, \
                        class_0_acc, class_1_acc, class_0_num, class_1_num])
        FP.append(1-class_0_acc)
        TP.append(class_1_acc)
        
        with open(f"cp_{tag}{VERSION}.sh", "a") as f:
            cmd = f"cp /SSD/model{VERSION}/simulator_{name}.pkl /SSD/baseline{VERSION}/{tag}/"
            f.write(cmd)
            f.write("\n")
            cmd = f"cp /SSD/dataset{VERSION}/dataset_{name}.h5 /SSD/baseline{VERSION}/data/"
            f.write(cmd)
            f.write("\n")
    csvFile.close()  