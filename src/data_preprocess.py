import os
import json
import argparse

from delexicalize import delexicalize
from query import dataBase

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dialogues', type = str, default = "./")
parser.add_argument('-s', '--system_act', type = str, default = "./")
parser.add_argument('-db', '--db_path', type = str, default = "./")
parser.add_argument('--gen_new_data', action = "store_true")
args = parser.parse_args()

data = []
db = dataBase(args.db_path)

for file in os.listdir(args.dialogues):
	with open(os.path.join(args.dialogues, file), encoding='utf8') as F:
		dialogues = json.loads(F.read())
	
	for dialogue in dialogues:
		
		d = {}
		history = ""
		d_id = dialogue["dialogue_id"]

		for turn in dialogue['turns']:
			history += str(turn['speaker']) + ": " + str(turn['utterance']) + " "

			if turn['speaker'] == 'SYSTEM':
				if len(data) > 0:
					search_result = db.query(d['Belief State'], data[-1]['Belief State'])
					d['DB State'] = str(len(search_result)) + " match"
				d["Response"] = turn['utterance']
				d['turn_id'] = turn['turn_id']
				data.append(d)
				continue

			d = {}
			d['Dialogue History'] = history
			d['dialogue_id'] = d_id

			belief_state = {}

			for frame in turn['frames']:
				b_state = {}
				for slot in frame['state']['slot_values'].keys():
					slot_value = frame['state']['slot_values'][slot]
					slot = slot.split('-')[-1]
					b_state[slot] = slot_value
				belief_state[frame['service']] = b_state

			d["Belief State"] = belief_state

	print(data[:-1])
	data = delexicalize(args.system_act, data) 
	print(data[:-1])

	if args.gen_new_data:
		for d in data:
			for i in range(len(dialogues)):
				if dialogues[i]['dialogue_id'] == d['dialogue_id']:
					dialogues[i]['turns'][int(d['turn_id'])]['delexical'] = d['Response']

		print(dialogues[0])

		with open(os.path.join(args.dialogues, "new_" + file), 'w', encoding='utf8') as F:
			dialogues = F.write(json.dumps(dialogues))

	break