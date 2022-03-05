import json
cmds = {"cmds":[]}
with open('cmds.json') as f: 
    for i in f.readlines():
        print(i)
        cmds["cmds"].append(str(i))

with open('cmds_exec.json', 'w') as f: 
    json.dump(cmds, f, indent=4)