# DSTC9-Track2
Multi-Domain Task-Completion Dialog Challenge II

https://github.com/thu-coai/ConvLab-2

Dataset:
https://github.com/thu-coai/ConvLab-2/tree/master/data

Paper List:
https://github.com/budzianowski/multiwoz
Multiwoz 1.0
Multiwoz 2.0
Multiwoz 2.1
Multiwoz 2.2

DST:
[1905.08743] Transferable Multi-Domain State Generator for Task-Oriented Dialogue Systems
[1911.03906] Efficient Dialogue State Tracking by Selectively Overwriting Memory

E2E:
[2005.05298] SOLOIST: Few-shot Task-Oriented Dialog with A Single Pre-trained Auto-regressive Model


Progress:

E2E:

July 20: Reproduce SOLOIST on multiwoz2.2 ( preprocessing )

{
 "Dialogue History": "USER: ... SYSTEM: ... USER: ..."
 "Belief State": [
    "Domain-Slot": "Value",
    "hotel-day": "4",
    ,,,
 ]
 "DB State": [
  ,,,
 ]
 "Response": "Delex response"
}
