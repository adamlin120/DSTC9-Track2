import os
import json
import argparse

from delexicalize import delexicalize
from query import dataBase

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dialogues', type = str, default = "./")
parser.add_argument('-s', '--system_act', type = str, default = "./")
parser.add_argument('-db', '--db_path', type = str, default = "./")
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
				data.append(d)
				continue

			d = {}
			d['Dialogue History'] = history
			d['dialogue_id'] = d_id

			belief_state = {}

			for frame in turn['frames']:
				# if not frame['state']['slot_values']:
				# 	break
				b_state = {}
				for slot in frame['state']['slot_values'].keys():
					slot_value = frame['state']['slot_values'][slot]
					slot = slot.split('-')[-1]
					b_state[slot] = slot_value
				belief_state[frame['service']] = b_state

			d["Belief State"] = belief_state

	print(data[:-1])
	print(delexicalize(args.system_act, data)[:-1])
	break
