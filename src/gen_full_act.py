import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type = str, default = "./data.json")
parser.add_argument('-f', '--full_act', type = str, default = "./full_act.json")
args = parser.parse_args()

with open(args.data) as F:    data = json.loads(F.read())

acts = {}

for key, value in data.items():
	key = key.replace(".json", "")
	act = {}
	for i in range(len(value['log'])):
		if 'dialog_act' in value['log'][i].keys():    act[str(i)] = value['log'][i]['dialog_act']
		else:    print(value['log'][i].keys(), "???????")
	acts[key] = act

with open(args.full_act, 'w') as F:    F.write(json.dumps(acts))