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
parser.add_argument("-t", "--delexicalized_slot_path", type = str, default = "delexicalized_slot_name.json")
parser.add_argument("-o", "--output_path", type = str, required = True)
args = parser.parse_args()

data = []
db = dataBase(args.db_path)
slot_names = set()
dia_offset = 0

for file in os.listdir(args.dialogues):
	with open(os.path.join(args.dialogues, file), encoding='utf8') as F:
		dialogues = json.loads(F.read())
	
	dia_idx = {}

	for dialogue in dialogues:
		
		d = {}
		history = ""
		d_id = dialogue["dialogue_id"]
		dia_idx[d_id] = len(dia_idx)

		for turn in dialogue['turns']:
			history += str(turn['speaker']) + ": " + str(turn['utterance']) + " "

			if turn['speaker'] == 'SYSTEM':
				if len(data) > 0:
					search_result = db.query(d['Belief State'], data[-1]['Belief State'])
					d['DB State'] = str(len(search_result)) + " match"
				else:
					search_result = db.query(d['Belief State'], {})
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
	data, dia_slot_names = delexicalize(args.system_act, data)
	slot_names |= dia_slot_names
	print(data[:-1])

	if args.gen_new_data:
		print(file)

		for d in data[dia_offset:]:
			print(d)
			dialogues[int(dia_idx[d['dialogue_id']])]['turns'][int(d['turn_id'])]['delexical'] = d['Response']

		print(dialogues[0])

		with open(os.path.join(args.dialogues, "new_" + file), 'w', encoding='utf8') as F:
			json.dump(dialogues, F, indent = 2)

		dia_offset = len(data)

with open(args.output_path, 'w') as f_out:
	json.dump(data, f_out, indent = 2)

with open(args.delexicalized_slot_path, 'w') as f_slot:
	json.dump(list(slot_names), f_slot, indent = 2)
