import os
import json
import re

def delexicalize(system_act_fname, dialogues):
	with open(system_act_fname) as f_sys_act:
		sys_act = json.load(f_sys_act)

	pre_id = ""

	for dia in dialogues:
		dia_id = dia["dialogue_id"].replace(".json", "")
		
		if dia_id != pre_id:
			turn = 1

		print(dia_id)
		if str(turn) in sys_act[dia_id].keys():
			for domain_type, domain_value in sys_act[dia_id][str(turn)].items():
				for token_info in domain_value:
					if token_info[0] != "none":
						token_value = token_info[1]
						
						if token_value[0] == ' ':
							token_value = token_value.replace(' ', "", 1)

						if token_value[-1] == ' ':
							token_value = re.sub(r" $", "", token_value)

						if token_info[0] == "Type":
							if token_value == "guesthouse":
								token_value_test = "guest house"
							elif token_value == "guesthouses":
								token_value_test = "guest houses"

						replace = re.compile(re.escape(token_value), re.IGNORECASE)
						dia["Response"] = replace.sub('[' + domain_type + '-' + token_info[0] + ']', dia["Response"])

						if token_info[0] == "Type" and (token_value == "guesthouse" or token_value == "guesthouses"):
							replace = re.compile(re.escape(token_value_test), re.IGNORECASE)
							dia["Response"] = replace.sub('[' + domain_type + '-' + token_info[0] + ']', dia["Response"])

		turn += 2
		pre_id = dia_id

	return dialogues
