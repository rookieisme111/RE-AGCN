import csv

def parse_sentence(sentence):
    strs = ['<e1>','</e1>','<e2>','</e2>']
    e1_start = sentence.find(strs[0])
    e1_end = sentence.find(strs[1])
    e2_start = sentence.find(strs[2])
    e2_end = sentence.find(strs[3])    
    e1 = sentence[e1_start+4:e1_end]
    e2 = sentence[e2_start+4:e2_end] 
    return (e1,e2)   

raw_data_path = 'SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT'

tsv_dump_data = []
with open(raw_data_path,'r') as f:
    lines = f.readlines()
    len1 = len(lines)
    i = 0
    while(i<len1):
        sentence = lines[i].strip().split('	')[-1]
        # print(sentence[-1])
        while(sentence[-1] != '"'):
            i = i+1
            sentence += lines[i].rstrip()
            # print(sentence)
        i += 1
        relation = lines[i].strip()

        e1 , e2 = parse_sentence(sentence)
        sentence = sentence.strip('"')

        print(e1)
        print(e2)
        print(relation)
        print(sentence)

        while (len(lines[i].strip())!=0):
            i +=1
        while (i<len1 and len(lines[i].strip())==0):
            i +=1

        tsv_dump_data.append([e1,e2,relation,sentence])


with open(r'train.tsv', 'w',newline='') as f:
    tsv_w = csv.writer(f, delimiter='\t')
    tsv_w.writerows(tsv_dump_data)  # 多行写入

