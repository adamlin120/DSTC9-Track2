import json
import logging
import os
import re
from argparse import ArgumentParser, Namespace
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Set, List, Tuple, Union
from operator import itemgetter

import numpy as np
from tqdm import tqdm

from utils.multiwoz.data import load_multiwoz
from utils.multiwoz.dbPointer import one_hot_vector, query_result
from utils.multiwoz.delexicalize import delexicalize, prepare_slot_values_independent
from utils.multiwoz.nlp import normalize, normalize_lexical, normalize_beliefstate

np.set_printoptions(precision=3)

np.random.seed(2)

# GLOBAL VARIABLES
DICT_SIZE = 1000000
MAX_LENGTH = 600

DATA_DIR = Path('./resources/')


def fix_delexicalization_with_dialog_actions(
        dialog_id: str,
        dialog: Dict,
        dialog_act: Dict,
        turn_index: int,
        dialog_act_index: int,
):
    """Given system dialogue acts fix automatic delexicalization."""
    try:
        turn = dialog_act[dialog_id.strip('.json')][str(dialog_act_index)]
    except KeyError:
        return dialog

    if not isinstance(turn, bytes) and not isinstance(turn, str):
        for k, act in turn.items():
            if 'Attraction' in k:
                if 'restaurant_' in dialog['log'][turn_index]['text']:
                    dialog['log'][turn_index]['text'] = dialog['log'][turn_index]['text'].replace("restaurant", "attraction")
                if 'hotel_' in dialog['log'][turn_index]['text']:
                    dialog['log'][turn_index]['text'] = dialog['log'][turn_index]['text'].replace("hotel", "attraction")
            if 'Hotel' in k:
                if 'attraction_' in dialog['log'][turn_index]['text']:
                    dialog['log'][turn_index]['text'] = dialog['log'][turn_index]['text'].replace("attraction", "hotel")
                if 'restaurant_' in dialog['log'][turn_index]['text']:
                    dialog['log'][turn_index]['text'] = dialog['log'][turn_index]['text'].replace("restaurant", "hotel")
            if 'Restaurant' in k:
                if 'attraction_' in dialog['log'][turn_index]['text']:
                    dialog['log'][turn_index]['text'] = dialog['log'][turn_index]['text'].replace("attraction", "restaurant")
                if 'hotel_' in dialog['log'][turn_index]['text']:
                    dialog['log'][turn_index]['text'] = dialog['log'][turn_index]['text'].replace("hotel", "restaurant")

    return dialog


def delexicalize_reference_number(sent: str, turn):
    """Based on the belief state, we can find reference number that
    during data gathering was created randomly."""
    domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital']
    if turn['metadata']:
        for domain in domains:
            if turn['metadata'][domain]['book']['booked']:
                for slot in turn['metadata'][domain]['book']['booked'][0]:
                    if slot == 'reference':
                        val = '[' + domain + '_' + slot + ']'
                    else:
                        val = '[' + domain + '_' + slot + ']'
                    key = normalize(turn['metadata'][domain]['book']['booked'][0][slot])
                    sent = (' ' + sent + ' ').replace(' ' + key + ' ', ' ' + val + ' ')

                    # try reference with hashtag
                    key = normalize("#" + turn['metadata'][domain]['book']['booked'][0][slot])
                    sent = (' ' + sent + ' ').replace(' ' + key + ' ', ' ' + val + ' ')

                    # try reference with ref#
                    key = normalize("ref#" + turn['metadata'][domain]['book']['booked'][0][slot])
                    sent = (' ' + sent + ' ').replace(' ' + key + ' ', ' ' + val + ' ')
    return sent


def add_booking_pointer(
        goal: Dict[str, Dict],
        turn_meta: Dict[str, Dict],
        pointer_vector: np.ndarray
) -> np.ndarray:
    """Add information about availability of the booking option."""
    # Booking pointer
    rest_vec = np.array([1, 0])
    if goal['restaurant']:
        if "book" in turn_meta['restaurant']:
            if "booked" in turn_meta['restaurant']['book']:
                if turn_meta['restaurant']['book']["booked"]:
                    if "reference" in turn_meta['restaurant']['book']["booked"][0]:
                        rest_vec = np.array([0, 1])

    hotel_vec = np.array([1, 0])
    if goal['hotel']:
        if "book" in turn_meta['hotel']:
            if "booked" in turn_meta['hotel']['book']:
                if turn_meta['hotel']['book']["booked"]:
                    if "reference" in turn_meta['hotel']['book']["booked"][0]:
                        hotel_vec = np.array([0, 1])

    train_vec = np.array([1, 0])
    if goal['train']:
        if "book" in turn_meta['train']:
            if "booked" in turn_meta['train']['book']:
                if turn_meta['train']['book']["booked"]:
                    if "reference" in turn_meta['train']['book']["booked"][0]:
                        train_vec = np.array([0, 1])

    pointer_vector = np.append(pointer_vector, rest_vec)
    pointer_vector = np.append(pointer_vector, hotel_vec)
    pointer_vector = np.append(pointer_vector, train_vec)

    return pointer_vector


def add_db_pointer(turn):
    """Create database pointer for all related domains."""
    domains = ['restaurant', 'hotel', 'attraction', 'train']
    pointer_vector = np.zeros(6 * len(domains))
    for domain in domains:
        num_entities = query_result(domain, turn['metadata'])
        pointer_vector = one_hot_vector(num_entities, domain, pointer_vector)

    return pointer_vector


def get_summary_belief(belief) -> List[int]:
    """Based on the mturk annotations we form multi-domain belief state"""
    domains = [u'taxi', u'restaurant', u'hospital', u'hotel', u'attraction', u'train', u'police']
    summary_belief: List[int] = []
    for domain in domains:
        domain_active = False

        booking = []
        for slot in sorted(belief[domain]['book'].keys()):
            if slot == 'booked':
                booking.append(1 if belief[domain]['book']['booked'] else 0)
            else:
                booking.append(1 if belief[domain]['book'][slot] != "" else 0)
        if domain == 'train':
            if 'people' not in belief[domain]['book'].keys():
                booking.append(0)
            if 'ticket' not in belief[domain]['book'].keys():
                booking.append(0)
        summary_belief += booking

        for slot in belief[domain]['semi']:
            slot_enc = [0, 0, 0]  # not mentioned, dontcare, filled
            if belief[domain]['semi'][slot] == 'not mentioned':
                slot_enc[0] = 1
            elif belief[domain]['semi'][slot] == 'dont care' or \
                    belief[domain]['semi'][slot] == 'dontcare' or \
                    belief[domain]['semi'][slot] == "don't care":
                slot_enc[1] = 1
            elif belief[domain]['semi'][slot]:
                slot_enc[2] = 1
            if slot_enc != [0, 0, 0]:
                domain_active = True
            summary_belief += slot_enc

        # quasi domain-tracker
        if domain_active:
            summary_belief += [1]
        else:
            summary_belief += [0]

    assert len(summary_belief) == 94
    return summary_belief


def get_belief(belief) -> List[Tuple[str, str, str]]:
    domains = [u'taxi', u'restaurant', u'hospital', u'hotel', u'attraction', u'train', u'police']
    raw_belief: List[Tuple[str, str, str]] = []
    for domain in domains:
        for slot, value in belief[domain]['semi'].items():
            if value:
                raw_belief.append((domain, slot, normalize_beliefstate(value)))
        for slot, value in belief[domain]['book'].items():
            if slot == 'booked':
                continue
            if value:
                raw_belief.append((domain, f'book {slot}', normalize_beliefstate(value)))
    return raw_belief


def process_dialog(
        dialog,
        max_len: int,
) -> Dict[str, Union[List[str], List[str], List[List], List[List], List[List]]]:
    """Cleaning procedure for all kinds of errors in text and annotation."""
    num_turns = len(dialog['log'])
    assert num_turns % 2, f'Expect # of turns to be odd, but got {num_turns=}'

    processed_dialog = {
        'goal': dialog['goal'],
        'usr_log': [],
        'sys_log': [],
    }
    for i, turn in enumerate(dialog['log']):
        utterance: str = turn['text']
        num_tokens = len(utterance.split())
        assert num_tokens <= max_len, f"{num_tokens=} > {max_len=}"
        assert utterance.isascii(), f'Non ascii {utterance=}'

        if i % 2 == 0:  # usr turn
            assert 'db_pointer' in turn, "no db_pointer, probably 2 usr turns in a row, wrong dialogue"
            processed_dialog['usr_log'].append(turn)
        else:  # sys turn
            turn['belief_summary'] = get_summary_belief(turn['metadata'])
            turn['belief_state'] = get_belief(turn['metadata'])
            processed_dialog['sys_log'].append(turn)

    user_logs = list(map(itemgetter('text', processed_dialog['usr_log'])))
    system_logs = list(map(itemgetter('text', processed_dialog['sys_log'])))
    db_pointers = list(map(itemgetter('db_pointer', processed_dialog['usr_log'])))
    belief_summaries = list(map(itemgetter('belief_summary', processed_dialog['sys_log'])))
    belief_states = list(map(itemgetter('belief_state', processed_dialog['sys_log'])))
    assert len(user_logs) == len(system_logs) == len(db_pointers) == \
           len(belief_summaries) == len(belief_states)
    return {
        'usr': user_logs,
        'sys': system_logs,
        'db': db_pointers,
        'bs': belief_summaries,
        'bstate': belief_states,
    }


def createDict(word_freqs):
    words = [k for k in word_freqs.keys()]
    freqs = [v for v in word_freqs.values()]

    sorted_idx = np.argsort(freqs)
    sorted_words = [words[ii] for ii in sorted_idx[::-1]]

    # Extra vocabulary symbols
    _GO = '_GO'
    EOS = '_EOS'
    UNK = '_UNK'
    PAD = '_PAD'
    SEP0 = '_SEP0'
    SEP1 = '_SEP1'
    SEP2 = '_SEP2'
    SEP3 = '_SEP3'
    SEP4 = '_SEP4'
    SEP5 = '_SEP5'
    SEP6 = '_SEP6'
    SEP7 = '_SEP7'
    extra_tokens = [_GO, EOS, UNK, PAD, SEP0, SEP1, SEP2, SEP3, SEP4, SEP5, SEP6, SEP7]
    # extra_tokens = [_GO, EOS, UNK, PAD]

    worddict = OrderedDict()
    for ii, ww in enumerate(extra_tokens):
        worddict[ww] = ii
    for ii, ww in enumerate(sorted_words):
        worddict[ww] = ii  # + len(extra_tokens)

    new_worddict = worddict.copy()
    for key, idx in worddict.items():
        if idx >= DICT_SIZE:
            del new_worddict[key]
    return new_worddict


def create_delexicalized_data() -> Dict[str, Dict]:
    """Main function of the script - loads delexicalized dictionary,
    goes through each dialogue and does:
    1) data normalization
    2) delexicalization
    3) addition of database pointer
    4) saves the delexicalized data
    """
    data, dialog_acts = load_multiwoz()

    # create dictionary of delexicalized values that then we will search against, order matters here!
    slot_values = prepare_slot_values_independent()
    delex_data = {}

    for dialogue_id in tqdm(data):
        dialogue = data[dialogue_id]

        for idx, turn in enumerate(dialogue['log']):
            # normalization, split and delexicalization of the sentence
            utterance = normalize(turn['text'])

            words = utterance.split()
            utterance = delexicalize(' '.join(words), slot_values)

            # parsing reference number GIVEN belief state
            utterance = delexicalize_reference_number(utterance, turn)

            # changes to numbers only here
            digitpat = re.compile('\d+')
            utterance = re.sub(digitpat, '[value_count]', utterance)

            # delexicalized sentence added to the dialogue
            dialogue['log'][idx]['text'] = utterance

            if idx % 2 == 1:  # if it's a system turn
                pointer_vector = add_db_pointer(turn)
                pointer_vector = add_booking_pointer(dialogue['goal'],
                                                     turn['metadata'],
                                                     pointer_vector)

                dialogue['log'][idx - 1]['db_pointer'] = pointer_vector.tolist()

            dialogue = fix_delexicalization_with_dialog_actions(dialogue_id, dialogue, dialog_acts, idx, idx + 1)

        delex_data[dialogue_id] = dialogue

    delex_path = DATA_DIR / 'multi-woz/delex.json'
    delex_path.write_text(json.dumps(delex_data))

    return delex_data


def create_lexical_data() -> Dict[str, Dict]:
    """Main function of the script - loads lexical dictionary,
    goes through each dialogue and does:
    1) data normalization
    2) addition of database pointer
    3) saves the lexicalized data
    """
    data, _ = load_multiwoz()

    lex_data = {}
    for dialogue_name in tqdm(data):
        dialogue = data[dialogue_name]
        for idx, turn in enumerate(dialogue['log']):
            normalized_utterance = normalize_lexical(turn['text'])
            dialogue['log'][idx]['text'] = normalized_utterance
            if idx % 2 == 1:  # system turn
                pointer_vector = add_db_pointer(turn)
                pointer_vector = add_booking_pointer(dialogue, turn, pointer_vector)
                dialogue['log'][idx - 1]['db_pointer'] = pointer_vector.tolist()

        lex_data[dialogue_name] = dialogue

    lex_data_path = DATA_DIR / 'multi-woz/lex.json'
    lex_data_path.write_text(json.dumps(lex_data))

    return lex_data


def get_turn_dialog_action(
        dialog_action: Dict[str, Dict[str, List[List[str, str]]]],
        turn_id: str
) -> Tuple[Union[str, Dict[str, List[List[str, str]]], List],
           List[Tuple[str, str, str]]]:
    if turn_id in dialog_action:
        turn_action = dialog_action[turn_id]
        if isinstance(turn_action, str):
            return turn_action, []

        acts = defaultdict(lambda: defaultdict(list))
        for domain_action, slot_values in turn_action.items():
            domain, act = [w.lower() for w in domain_action.split('-')]
            for slot, value in slot_values:
                slot = ' '.join(slot.lower().strip().split('\t'))
                value = ' '.join(value.lower().strip().split('\t'))
                if domain in acts and \
                        act in acts[domain] and \
                        slot in acts[domain][act]:
                    # already domain-act is considered, skip
                    continue
                acts[domain][act].append((slot, value))

        concat = [
            (domain, act, slot)
            for domain in acts
            for act in acts[domain]
            for slot, value in acts[domain][act]
        ]
        return turn_action, concat
    return [], []


def divide_data(
        dialogs: Dict[str, Dict],
        is_lexicalized: bool = False
) -> Tuple[Counter, Counter, Counter]:
    """Given test and validation sets, divide
    the data for three different sets"""
    multiwoz_dir = DATA_DIR / 'multi-woz/'
    train_list_path = multiwoz_dir / 'trainListFile'
    val_list_path = multiwoz_dir / 'valListFile.json'
    test_list_path = multiwoz_dir / 'testListFile.json'
    dialog_actions_path = multiwoz_dir / 'dialogue_acts.json'

    train_list_file: List[str] = []
    val_list_file: Set[str] = set(val_list_path.read_text().splitlines())
    test_list_file: Set[str] = set(test_list_path.read_text().splitlines())
    dialog_actions: Dict[str, Dict[str, Dict[str, List[List[str, str]]]]] = \
        json.loads(dialog_actions_path.read_text())

    train_dials = {}
    val_dials = {}
    test_dials = {}

    # dictionaries
    user_dict = Counter()
    system_dict = Counter()
    dialog_history_dict = Counter()
    belief_dict = Counter()

    for dialog_id, dialog in tqdm(dialogs.items()):

        dial = process_dialog(dialog)

        dialogue = {
            **dial,
        }
        dialog_action = dialog_actions[dialog_id.rstrip('.json')]
        dialogue['sys_act_raw'], dialogue['sys_act'] = \
            list(map(list,
                     zip(*[
                             get_turn_dialog_action(dialog_action, str(turn_id))
                             for turn_id, _ in enumerate(dialogue['sys'], 1)
                           ]
                         )
                     )
                 )

        if dialog_id in test_list_file:
            test_dials[dialog_id] = dialogue
        elif dialog_id in val_list_file:
            val_dials[dialog_id] = dialogue
        else:
            train_list_file.append(dialog_id)
            train_dials[dialog_id] = dialogue

        user_dict += Counter((line.strip().split(' ')for line in dial['usr']))
        system_dict += Counter((line.strip().split(' ') for line in dial['sys']))
        dialog_history_dict = user_dict + system_dict

        act_words = [word for dial_act in dialogue['sys_act'] for word in dial_act]
        action_dict = Counter(act_words)
        system_dict += action_dict
        dialog_history_dict += action_dict

        belief_words = []
        for dial_bstate in dialogue['bstate']:
            for domain, slot, value in dial_bstate:
                belief_words.extend([domain, slot])
                belief_words.extend(normalize_beliefstate(value).strip().split(' '))
        for w in belief_words:
            if w not in system_dict:
                system_dict[w] = 0
            system_dict[w] += 1
            if w not in dialog_history_dict:
                dialog_history_dict[w] = 0
            dialog_history_dict[w] += 1
            if w not in belief_dict:
                belief_dict[w] = 0
            belief_dict[w] += 1

    train_list_path.write_text('\n'.join(train_list_file))

    train_filename = DATA_DIR / f"train_dials{'_lexicalized' if is_lexicalized else ''}.json"
    val_filename = DATA_DIR / f"val_dials{'_lexicalized' if is_lexicalized else ''}.json"
    test_filename = DATA_DIR / f"test_dials{'_lexicalized' if is_lexicalized else ''}.json"

    train_filename.write_text(json.dumps(train_dials, indent=4))
    val_filename.write_text(json.dumps(val_dials, indent=4))
    test_filename.write_text(json.dumps(test_dials, indent=4))

    return user_dict, system_dict, dialog_history_dict


def build_dictionaries(word_freqs_usr, word_freqs_sys, word_freqs_histoy, is_lexicalized=False):
    """Build dictionaries for both user and system sides.
    You can specify the size of the dictionary through DICT_SIZE variable."""
    dicts = []
    worddict_usr = createDict(word_freqs_usr)
    dicts.append(worddict_usr)
    worddict_sys = createDict(word_freqs_sys)
    dicts.append(worddict_sys)
    worddict_history = createDict(word_freqs_histoy)
    dicts.append(worddict_history)

    # reverse dictionaries
    idx2words = []
    for dictionary in dicts:
        dic = {}
        for k, v in dictionary.items():
            dic[v] = k
        idx2words.append(dic)

    if is_lexicalized:
        input_index2word_filename = os.path.join(DATA_DIR, 'input_lang.index2word_lexicalized.json')
        input_word2index_filename = os.path.join(DATA_DIR, 'input_lang.word2index_lexicalized.json')
        output_index2word_filename = os.path.join(DATA_DIR, 'output_lang.index2word_lexicalized.json')
        output_word2index_filename = os.path.join(DATA_DIR, 'output_lang.word2index_lexicalized.json')
        history_index2word_filename = os.path.join(DATA_DIR, 'history_lang.index2word_lexicalized.json')
        history_word2index_filename = os.path.join(DATA_DIR, 'history_lang.word2index_lexicalized.json')
    else:
        input_index2word_filename = os.path.join(DATA_DIR, 'input_lang.index2word.json')
        input_word2index_filename = os.path.join(DATA_DIR, 'input_lang.word2index.json')
        output_index2word_filename = os.path.join(DATA_DIR, 'output_lang.index2word.json')
        output_word2index_filename = os.path.join(DATA_DIR, 'output_lang.word2index.json')
        history_index2word_filename = os.path.join(DATA_DIR, 'history_lang.index2word.json')
        history_word2index_filename = os.path.join(DATA_DIR, 'history_lang.word2index.json')

    with open(input_index2word_filename, 'w') as f:
        json.dump(idx2words[0], f, indent=2)
    with open(input_word2index_filename, 'w') as f:
        json.dump(dicts[0], f, indent=2)
    with open(output_index2word_filename, 'w') as f:
        json.dump(idx2words[1], f, indent=2)
    with open(output_word2index_filename, 'w') as f:
        json.dump(dicts[1], f, indent=2)
    with open(history_index2word_filename, 'w') as f:
        json.dump(idx2words[2], f, indent=2)
    with open(history_word2index_filename, 'w') as f:
        json.dump(dicts[2], f, indent=2)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('mode', type=str, help='delex or lexical')
    return parser.parse_args()


def main():
    args = parse_args()

    assert args.mode in {'delex', 'lexixal'}, ValueError(f'Unknown mode: {args.mode}')
    logging.info(f'MultiWoz Create {args.mode} dialogues.')

    is_lexicalize = args.mode == 'lexixal'
    if is_lexicalize:
        data_path = DATA_DIR / 'multi-woz/lex.json'
        data = json.loads(data_path.read_text()) if data_path.is_file() else create_lexical_data()
    else:
        data_path = DATA_DIR / 'multi-woz/delex.json'
        data = json.loads(data_path.read_text()) if data_path.is_file() else create_delexicalized_data()

    logging.info('Divide dialogues for separate bits - usr, sys, db, bs')
    word_freqs_usr, word_freqs_sys, word_freqs_history = divide_data(data, is_lexicalize)

    logging.info('Building dictionaries')
    build_dictionaries(word_freqs_usr, word_freqs_sys, word_freqs_history, is_lexicalize)


if __name__ == "__main__":
    main()