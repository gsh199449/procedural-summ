import json, sys, os
from typing import List
from tqdm import tqdm

import nltk

def wc_file(file_path: str) -> int:
    return int(os.popen('wc -l {}'.format(file_path)).read().split()[0])

fi = open(sys.argv[1], encoding='utf8')
fo = open(sys.argv[2], 'w', encoding='utf8')
total = wc_file(sys.argv[1])

for line in tqdm(fi, total=total):
    data = json.loads(line)
    entity = []
    counter = 0
    for se, scb, sca in zip(data['se_e'], data['e_scb'], data['e_sca']):
        entity.append([{'entity': str(e), 'before': str(scb[2][i]), 'after': str(sca[2][i])} for i, e in enumerate(se[1])])
        counter += len(se[1])
        # entity[se[0]] = [{'entity': str(e), 'before': str(scb[2][i]), 'after': str(sca[2][i])} for i, e in enumerate(se[1])]
    if counter == 0:
        continue
    data['sentence_entity'] = entity
    del data['se_e']
    del data['e_scb']
    del data['e_sca']
    gt_entity_list = []  # type: List[str]
    for s in data['summary']:
        e = []
        for word, pos in nltk.pos_tag(nltk.word_tokenize(s)):
            if pos.startswith('NN'):
                e.append(word)
        gt_entity_list.extend(e)
    data['entity_list'] = list(set(gt_entity_list))
    if len(data['entity_list']) == 0:
        continue
    fo.write(f'{json.dumps(data)}\n')

    # for inp_sent, tar_sent in zip(data['text'], data['summary']):
    #     dd = {"content": inp_sent, 'summary': tar_sent}
    #     fo.write(f'{json.dumps(dd)}\n')
fo.close()
fi.close()

