* 先用 gen_full_act.py 生成完整的 system_acts (default: full_act.json)
```
python gen_full_act.py -d PATH_TO__MultiWOZ_2.1/data.json \
	-f OUTPUT_PATH
```

* 用 data_preprocecss.py 可生成 model input 形式的 json 檔和包含所有 slot 的字典檔，並且可選擇在原 data 中加入 delexicalized response
	* --gen_new_data 可以在原 data 處生成包含 delexicalized response 的新 data，檔名為 new_原檔名
```
python data_preprocess.py -d DIR_OF__DATA \
	-s PATH_TO__full_act.json \
	-db PATH_TO__MultiWOZ_2.1 \
	-o OUTPUT_PATH
	[
	-t PATH_TO__slot.json \
	--gen_new_data
	]
```