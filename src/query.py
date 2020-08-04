import os
import json

class dataBase:
	def __init__(self, dir):
		self.db = {}

		with open(os.path.join(dir, "attraction_db.json")) as f_att:
			self.db["attraction"] = json.load(f_att)

		with open(os.path.join(dir, "hospital_db.json")) as f_hos:
			self.db["hospital"] = json.load(f_hos)

		with open(os.path.join(dir, "hotel_db.json")) as f_hot:
			self.db["hotel"] = json.load(f_hot)

		with open(os.path.join(dir, "police_db.json")) as f_pol:
			self.db["police"] = json.load(f_pol)

		with open(os.path.join(dir, "restaurant_db.json")) as f_res:
			self.db["restaurant"] = json.load(f_res)

		with open(os.path.join(dir, "train_db.json")) as f_tra:
			self.db["train"] = json.load(f_tra)

	def query(self, b, pre_b):
		results = []

		for dom_key, dom_values in b.items():
			if dom_key in self.db.keys() and (dom_key not in pre_b.keys() or dom_values != pre_b[dom_key]):
				dom_results = self.db[dom_key]

				for state_key, state_values in b[dom_key].items():
					if state_key in self.db[dom_key][0].keys() and "dontcare" not in state_values:
						#dom_results = [x if x[state_key] in state_values for x in dom_results]
						tmp_re = []
						for x in dom_results:
							if x[state_key] in state_values:    tmp_re.append(x)
						dom_results = tmp_re
						
				results += dom_results

		return results
