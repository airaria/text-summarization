SRC_FILE = 'sogou_news/data_small/src.txt'
TGT_FILE = 'sogou_news/data_small/tgt.txt'
VOC_FILE = 'sogou_news/data_small/voc.txt'
SRC_TEST_FILE = 'sogou_news/data_small/src_test.txt'
TGT_TEST_FILE = 'sogou_news/data_small/tgt_test.txt'

UNK_ID = 0
UNK = "<UNK>"
EOS = "<EOS>"
SOS = "<SOS>"
PREFETCH_SIZE = 10000
BATCH_SIZE = 64
MAX_SRC_LEN = 160
MAX_TGT_LEN = 25
MAX_DECODE_STEP = MAX_TGT_LEN
TRAINING_STEPS = 50000
CHECKPOINT_EVERY = 1000
PRINT_EVERY = 100
OUT_DIR = './saved_model'
EXPORT_DIR = './saved_pb'
MAX_TO_KEEP = 10

VOCAB_SIZE = 36000
GEN_VOCAB_SIZE = 18000
LR = 0.0001
LOAD_MODEL_PATH = 'saved_model/my_model_3.002-10938' # 'saved_model/my_model_3.834-43375'
