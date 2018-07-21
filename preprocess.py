from pyltp import Segmentor
from collections import Counter
DATA_FILE = 'sogou_news/data_tiny.txt'
CWS_MODEL_FILE = 'ltp_data_v3.4.0/cws.model'

SRC_FILE =      'sogou_news/data_tiny/src.txt'
SRC_TEST_FILE = 'sogou_news/data_tiny/src_test.txt'
TGT_TEST_FILE = 'sogou_news/data_tiny/tgt_test.txt'
TGT_FILE =      'sogou_news/data_tiny/tgt.txt'
VOC_FILE =      'sogou_news/data_tiny/voc.txt'

def xml2csv(fin="sogou_news/news.txt",fo="sogou_news/data.txt"):
    with open(fin,'r') as f:
        with open(fo,'w') as data_file:
            count = 0
            mark1 = '<contenttitle>'
            mark2 = '<content>'
            title_content = []
            for line in f:
                if line.startswith(mark1):
                    assert len(title_content)==0
                    line = line.strip()
                    title_content.append(str(count))
                    title_content.append(line[len(mark1):-(len(mark1)+1)])
                if line.startswith(mark2):
                    assert len(title_content)==2
                    line = line.strip()
                    if len(line)>len(mark2)*2+1:
                        title_content.append(line[len(mark2):-(len(mark2)+1)])
                        data_file.write('\t'.join(title_content)+'\n')
                        count += 1
                    title_content = []


if __name__ == '__main__':
    vocab_counter = Counter()
    segmentor = Segmentor()
    segmentor.load(CWS_MODEL_FILE)
    f = open(DATA_FILE,'r')
    src_f = open(SRC_FILE,'w')
    src_test_f = open(SRC_TEST_FILE,'w')
    tgt_test_f = open(TGT_TEST_FILE,'w')
    tgt_f = open(TGT_FILE,'w')

    for line in f:
        index,title,content = line.strip().split('\t')
        title_tokens = segmentor.segment(title)
        content_tokens = segmentor.segment(content)
        vocab_counter.update(title_tokens)
        vocab_counter.update(content_tokens)

        if (int(index)+1)%10==0:
            src_test_f.write(' '.join([index] + list(content_tokens)) + '\n')
            tgt_test_f.write(' '.join([index] + list(title_tokens)) + '\n')
        else:
            src_f.write(' '.join([index]+list(content_tokens))+'\n')
            tgt_f.write(' '.join([index]+list(title_tokens))+'\n')

        if (int(index)+1)%100==0:
            print (int(index)+1)

    f.close()
    src_f.close()
    tgt_f.close()
    src_test_f.close()
    tgt_test_f.close()

    vocab = ['<UNK>','<SOS>','<EOS>'] + [k for k,v in vocab_counter.most_common()]
    with open(VOC_FILE,'w') as voc_f:
        for word in vocab:
            voc_f.write(word+'\n')

