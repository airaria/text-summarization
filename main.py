import os,sys
import argparse
import tensorflow as tf
import random
from itertools import chain
from bleu import bleu
from rougescore import rouge_1,rouge_2,rouge_l

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.log_device_placement = True

import numpy as np
from config import *
from attention_model import attention_model
from collections import namedtuple

def load_vocab(vocab_file):
  vocab = []
  with open(vocab_file, "r") as f:
    for word in f:
      vocab.append(word.strip())
  return np.array(vocab)

def tf_create_lookup_table(voc_file):
    #automatically add filepath to collection asset_filepath, but ...
    vocab_table = tf.contrib.lookup.index_table_from_file(
        vocabulary_file = voc_file,default_value = UNK_ID, vocab_size=VOCAB_SIZE)
    reverse_vocab_table = tf.contrib.lookup.index_to_string_table_from_file(
        vocabulary_file = voc_file, default_value = UNK, vocab_size=VOCAB_SIZE)
    return vocab_table,reverse_vocab_table

def build_eval_dataset(src_file,tgt_file,vocab_table):
    tgt_sos_id = tf.cast(vocab_table.lookup(tf.constant(SOS)), tf.int32)
    tgt_eos_id = tf.cast(vocab_table.lookup(tf.constant(EOS)), tf.int32)
    src_eos_id = tf.cast(vocab_table.lookup(tf.constant(EOS)), tf.int32)

    src_dataset = tf.data.TextLineDataset(src_file)
    tgt_dataset = tf.data.TextLineDataset(tgt_file)
    src_tgt_dataset = tf.data.Dataset.zip((src_dataset,tgt_dataset))
    src_tgt_dataset = src_tgt_dataset.map(lambda src,tgt:(tf.string_split([src]).values[1:],tf.string_split([tgt]).values[1:]))

    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (tf.cast(vocab_table.lookup(src), tf.int32),
                          tf.cast(vocab_table.lookup(tgt), tf.int32)))

    src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt: (src[:MAX_SRC_LEN],tgt[:MAX_TGT_LEN]))

    src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt: (src,tf.concat(([tgt_sos_id], tgt), 0),tf.concat((tgt, [tgt_eos_id]), 0)))

    src_tgt_dataset = src_tgt_dataset.map(lambda src,tgt_in,tgt_out:
                                          (src,tgt_in,tgt_out, tf.size(src), tf.size(tgt_in))).prefetch(PREFETCH_SIZE)

    src_tgt_dataset = src_tgt_dataset.padded_batch(
            BATCH_SIZE,
            padded_shapes=(
               tf.TensorShape([None]),  # src
               tf.TensorShape([None]),  # tgt_input
               tf.TensorShape([None]),  # tgt_output
               tf.TensorShape([]),  # src_len
               tf.TensorShape([])),  # tgt_len
            # Pad the source and target sequences with eos tokens.
            # (Though notice we don't generally need to do this since
            # later on we will be masking out calculations past the true sequence.
            padding_values=(
               src_eos_id,  # src
               tgt_eos_id,  # tgt_input
               tgt_eos_id,  # tgt_output
               0,  # src_len -- unused
               0)).prefetch(5)  # tgt_len -- unused

    iterator = src_tgt_dataset.make_initializable_iterator()
    src,tgt_in,tgt_out,src_len,tgt_len = iterator.get_next(name='eval_next')
    batchedInput = namedtuple("batchedInput",
                              ('initializer','source',
                               'target_input','target_output',
                               'source_length','target_length'))
    return batchedInput(
        initializer=iterator.initializer,
        source=src,
        target_input = tgt_in,
        target_output = tgt_out,
        source_length = src_len,
        target_length = tgt_len
        )

def buid_train_dataset(src_file,tgt_file,vocab_table):
    tgt_sos_id = tf.cast(vocab_table.lookup(tf.constant(SOS)), tf.int32)
    tgt_eos_id = tf.cast(vocab_table.lookup(tf.constant(EOS)), tf.int32)
    src_eos_id = tf.cast(vocab_table.lookup(tf.constant(EOS)), tf.int32)

    src_dataset = tf.data.TextLineDataset(src_file)
    tgt_dataset = tf.data.TextLineDataset(tgt_file)
    src_tgt_dataset = tf.data.Dataset.zip((src_dataset,tgt_dataset))
    src_tgt_dataset = src_tgt_dataset.map(lambda src,tgt:(tf.string_split([src]).values[1:],tf.string_split([tgt]).values[1:]))

    src_tgt_dataset = src_tgt_dataset.shuffle(buffer_size=PREFETCH_SIZE)
    #src_tgt_dataset = src_tgt_dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=PREFETCH_SIZE))

    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (tf.cast(vocab_table.lookup(src), tf.int32),
                          tf.cast(vocab_table.lookup(tgt), tf.int32)))

    src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt: (src[:MAX_SRC_LEN],tgt[:MAX_TGT_LEN]))

    src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt: (src,tf.concat(([tgt_sos_id], tgt), 0),tf.concat((tgt, [tgt_eos_id]), 0)))

    src_tgt_dataset = src_tgt_dataset.map(lambda src,tgt_in,tgt_out:
                                          (src,tgt_in,tgt_out, tf.size(src), tf.size(tgt_in))).prefetch(PREFETCH_SIZE)

    src_tgt_dataset = src_tgt_dataset.padded_batch(
            BATCH_SIZE,
            padded_shapes=(
               tf.TensorShape([None]),  # src
               tf.TensorShape([None]),  # tgt_input
               tf.TensorShape([None]),  # tgt_output
               tf.TensorShape([]),  # src_len
               tf.TensorShape([])),  # tgt_len
            # Pad the source and target sequences with eos tokens.
            # (Though notice we don't generally need to do this since
            # later on we will be masking out calculations past the true sequence.
            padding_values=(
               src_eos_id,  # src
               tgt_eos_id,  # tgt_input
               tgt_eos_id,  # tgt_output
               0,  # src_len -- unused
               0)).prefetch(5)  # tgt_len -- unused

    iterator = src_tgt_dataset.make_initializable_iterator()
    src,tgt_in,tgt_out,src_len,tgt_len = iterator.get_next(name='train_next')
    batchedInput = namedtuple("batchedInput",
                              ('initializer','source',
                               'target_input','target_output',
                               'source_length','target_length'))
    return batchedInput(
        initializer=iterator.initializer,
        source=src,
        target_input = tgt_in,
        target_output = tgt_out,
        source_length = src_len,
        target_length = tgt_len
        )

def build_infer_dataset(src_file,vocab_table):
    src_eos_id = tf.cast(vocab_table.lookup(tf.constant(EOS)), tf.int32)

    src_dataset = tf.data.TextLineDataset(src_file)
    src_dataset = src_dataset.map(lambda src: tf.string_split([src]).values[1:])
    src_dataset = src_dataset.prefetch(PREFETCH_SIZE)
    src_dataset = src_dataset.map(lambda src: tf.cast(vocab_table.lookup(src), tf.int32))
    src_dataset = src_dataset.map(lambda src: src[:MAX_SRC_LEN])
    src_dataset = src_dataset.map(lambda src: (src, tf.size(src)))

    src_dataset = src_dataset.padded_batch(
        BATCH_SIZE,
            padded_shapes=(
               tf.TensorShape([None]),  # src
               tf.TensorShape([])),  # src_len
            padding_values=(
               src_eos_id,  # src
               0))  # src_len -- unused


    iterator = src_dataset.make_initializable_iterator()
    src,src_len = iterator.get_next(name='infer_next')
    batchedInput = namedtuple("batchedInput",
                              ('initializer','source','source_length'))
    return batchedInput(
        initializer=iterator.initializer,
        source=tf.identity(src,'src'),
        source_length = tf.identity(src_len,'src_len')
        )

def build_infer_placeholder(src_ph,vocab_table):
    src_eos_id = tf.cast(vocab_table.lookup(tf.constant(EOS)), tf.int32)

    inputs = tf.string_split(src_ph)
    inputs = tf.cast(vocab_table.lookup(inputs),tf.int32)
    shape = tf.shape(inputs)
    slice_size =tf.cast(tf.stack([shape[0],MAX_SRC_LEN]),tf.int64)
    slice_start = tf.constant([0,0],dtype=tf.int64)
    inputs = tf.sparse_slice(inputs,start=slice_start,size=slice_size)
    line_number = inputs.indices[:, 0]
    line_position = inputs.indices[:, 1]
    lengths = tf.segment_max(data=line_position,
                             segment_ids=line_number) + 1
    inputs = tf.sparse_tensor_to_dense(inputs,src_eos_id)

    src = inputs
    src_len = lengths

    batchedInput = namedtuple("batchedInput",
                              ('initializer','source','source_length'))
    return batchedInput(
        initializer=None,
        source=tf.identity(src,'src'),
        source_length = tf.identity(src_len,'src_len')
        )

def decode_for_human(sample_words,stop_word,join=True):
    wb = [[w.decode() for w in sentence] for sentence in sample_words]
    wb = [[w for w in sentence if w!=stop_word] for sentence in wb]
    if join:
        return  [''.join(sentence) for sentence in wb]
    else:
        return wb

def get_avg_loss(losses,sizes):
    losses = np.array(losses)
    sizes = np.array(sizes)
    losses *= sizes
    return np.sum(losses)/np.sum(sizes)

def run_eval(sess,inputs,model):
    sess.run(inputs.initializer)
    total_losses = []
    total_sizes = []
    references = []
    candidates = []

    while True:
        try:
            loss,true_word,sample_word = sess.run(
                [model.loss,model.true_word,model.sample_word])
            total_losses.append(loss)
            total_sizes.append(len(sample_word))
            candidates.append(decode_for_human(sample_word,EOS,join=False))
            references.append(decode_for_human(true_word,EOS,join=False))


        except tf.errors.OutOfRangeError:
            avg_loss = get_avg_loss(total_losses,total_sizes)
            print ("Evaluation done. Avg eval loss: %f" % avg_loss)

            candidates = list(chain(*candidates))
            references = list(chain(*references))
            #TODO add bleu and rouge
            print (rouge_1(candidates[1],[references[1]],[0,0.5]))
            rouge_1_recall, rouge_1_F1 = list(np.mean([rouge_1(c,[r],[0,0.5]) for c,r in zip(candidates,references)],axis=0))
            rouge_2_recall, rouge_2_F1 = list(np.mean([rouge_2(c,[r],[0,0.5]) for c,r in zip(candidates,references)],axis=0))
            rouge_L_recall, rouge_L_F1 = list(np.mean([rouge_l(c,[r],[0,0.5]) for c,r in zip(candidates,references)],axis=0))

            print("rouge-1 recall: %.5f \t F1: %.5f" % (rouge_1_recall, rouge_1_F1))
            print("rouge-2 recall: %.5f \t F1: %.5f" % (rouge_2_recall, rouge_2_F1))
            print("rouge-L recall: %.5f \t F1: %.5f" % (rouge_L_recall, rouge_L_F1))

            for c,r in random.sample(list(zip(candidates,references)),20):
                print ('\n'.join(["candidate:"+''.join(c),"reference:"+''.join(r),'------']))

            break

    return avg_loss

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",type=str,required=True,choices=['TRAIN','INFER','EXPORT','DEBUG_EXPORT'])
    parser.add_argument("--loadcheckpoint",action='store_true')
    parser.add_argument("--src_file",type=str,default=SRC_FILE)
    parser.add_argument("--target_file",type=str,default=TGT_FILE)
    parser.add_argument("--voc_file",type=str,default=VOC_FILE)
    parser.add_argument("--src_val_file",type=str,default=SRC_TEST_FILE)
    parser.add_argument("--target_val_file",type=str,default=TGT_TEST_FILE)
    parser.add_argument("--src_test_file",type=str,default=SRC_TEST_FILE)
    parser.add_argument("--model_file",type=str,default=LOAD_MODEL_PATH)

    args = parser.parse_args()

    mode = args.mode
    loadcheckpoint = args.loadcheckpoint
    src_file = args.src_file
    tgt_file = args.target_file
    voc_file = args.voc_file
    src_val_file = args.src_val_file
    tgt_val_file = args.target_val_file
    src_test_file = args.src_test_file
    model_file = args.model_file

    sess = tf.Session(config=config)

    if mode=='DEBUG_EXPORT':
        signature_key = 'predict_signature'
        input_src ='input_src'
        output_word = 'output_word'

        meta_graph_def = tf.saved_model.loader.load(sess,[tf.saved_model.tag_constants.SERVING],EXPORT_DIR)
        signature = meta_graph_def.signature_def
        input_src_tensor_name = signature[signature_key].inputs[input_src].name
        output_word_tensor_name = signature[signature_key].outputs[output_word].name

        input_src_tensor = sess.graph.get_tensor_by_name(input_src_tensor_name)
        output_word_tensor = sess.graph.get_tensor_by_name(output_word_tensor_name)

        sys.exit()


    vocab_table, reverse_vocab_table = tf_create_lookup_table(voc_file)


    if mode == 'EXPORT':
        src_ph = tf.placeholder(tf.string, name='src_placeholder')
        infer_ph_inputs = build_infer_placeholder(src_ph, vocab_table)

        infer_export_model = attention_model(
            inputs=infer_ph_inputs,
            vocab_table=vocab_table,
            reverse_vocab_table=reverse_vocab_table,
            vocab_size=VOCAB_SIZE,
            gen_vocab_size=GEN_VOCAB_SIZE,
            embed_size=200,
            num_units=256,
            lr=LR,
            mode="INFER",
            dropout=0,
            grad_clip=None)
        infer_export_model.saver.restore(sess, model_file)

        builder = tf.saved_model.builder.SavedModelBuilder(EXPORT_DIR)
        inputs = {'input_src': tf.saved_model.utils.build_tensor_info(src_ph)}
        outputs = {'output_word': tf.saved_model.utils.build_tensor_info(infer_export_model.sample_word)}
        signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs=inputs, outputs=outputs, method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

        #table_initializer = tf.tables_initializer()
        #init_op = tf.group(table_initializer)
        builder.add_meta_graph_and_variables(sess,
                                             [tf.saved_model.tag_constants.SERVING],
                                             signature_def_map={'predict_signature': signature},
                                             # assets_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS),
                                             )#main_op=init_op)
        builder.save()
        sys.exit()


    table_initializer = tf.tables_initializer()
    sess.run(table_initializer)


    if mode=='INFER':
        infer_inputs = build_infer_dataset(src_test_file, vocab_table)
        infer_model = attention_model(
            inputs=infer_inputs,
            vocab_table=vocab_table,
            reverse_vocab_table=reverse_vocab_table,
            vocab_size=VOCAB_SIZE,
            gen_vocab_size=GEN_VOCAB_SIZE,
            embed_size=200,
            num_units=256,
            lr=LR,
            mode="INFER",
            dropout=0,
            grad_clip=None)
        sess.run([infer_inputs.initializer])
        sess.run(tf.global_variables_initializer())
        
        infer_model.saver.restore(sess, model_file)
        sample_word = sess.run(infer_model.sample_word)
        print(decode_for_human(sample_word, EOS))

        sys.exit()

    if mode=='TRAIN':

        train_inputs = buid_train_dataset(src_file, tgt_file, vocab_table)
        eval_inputs = build_eval_dataset(src_val_file, tgt_val_file, vocab_table)

        train_model = attention_model(
            inputs=train_inputs,
            vocab_table=vocab_table,
            reverse_vocab_table=reverse_vocab_table,
            vocab_size=VOCAB_SIZE,
            gen_vocab_size=GEN_VOCAB_SIZE,
            embed_size=200,
            num_units=256,
            lr=LR,
            mode="TRAIN",
            dropout = 0.1,
            grad_clip=None)

        eval_model = attention_model(
            inputs=eval_inputs,
            vocab_table=vocab_table,
            reverse_vocab_table=reverse_vocab_table,
            vocab_size=VOCAB_SIZE,
            gen_vocab_size=GEN_VOCAB_SIZE,
            embed_size=200,
            num_units=256,
            lr = LR,
            mode='EVAL',
            dropout=0
        )
        sess.run([train_inputs.initializer, eval_inputs.initializer])
        sess.run(tf.global_variables_initializer())
        #summary_writer = tf.summary.FileWriter('log_simple_stats', sess.graph)
        #summary_writer.close()
        best_eval_score = 1000000 # The smaller the better
        out_dir = OUT_DIR
        finished_epoch = 0
        finished_step_in_epoch = 0
        last_checkpoint_step = 0
        last_print_step = 0
        global_step = 0
        losses = []
        sizes = []

        if loadcheckpoint:
            print ("Loading ",model_file)
            train_model.saver.restore(sess, model_file)
            val_loss = run_eval(sess,eval_inputs,eval_model)
            best_eval_score = val_loss


        while global_step < TRAINING_STEPS:
            if (global_step+1)%10==0 :
                print (global_step+1)
            try:
                loss, tgt_len,_ = sess.run([train_model.loss,train_model.target_length,train_model.train_op])

                finished_step_in_epoch += 1
                global_step += 1

                sizes.append(len(tgt_len))
                losses.append(loss)

            except tf.errors.OutOfRangeError:
                print ("here")
                sess.run(train_inputs.initializer)
                finished_epoch += 1
                print("Finished epoch %d, total %d step."%(finished_epoch,finished_step_in_epoch))

                epoch_losses = losses[-finished_step_in_epoch:]
                epoch_sizes = sizes[-finished_step_in_epoch:]
                train_loss = get_avg_loss(epoch_losses,epoch_sizes)
                print("Avg train loss of the epoch: %f" % train_loss)

                finished_step_in_epoch = 0
                val_loss = run_eval(sess,eval_inputs,eval_model)
                last_checkpoint_step = global_step
                if loss < best_eval_score:
                    best_eval_score = loss
                    train_model.saver.save(sess,os.path.join(out_dir,"my_model_%.3f"%loss),global_step=global_step)

                continue

            if (global_step-last_print_step)== PRINT_EVERY:
                last_print_step = global_step
                print ("Current step %d. Avg train loss of last 30 iterations:"%(global_step,),
                       get_avg_loss(losses[-30:],sizes[-30:]))


            if (global_step-last_checkpoint_step)==CHECKPOINT_EVERY:
                loss = run_eval(sess,eval_inputs,eval_model)
                last_checkpoint_step = global_step
                if loss < best_eval_score:
                    best_eval_score = loss
                    train_model.saver.save(sess,os.path.join(out_dir,"my_model_%.3f"%loss),global_step=global_step)
                    print ("Current step %d. Model saved." % (global_step,))


        # Done training
        print("Done training.")
        loss = run_eval(sess,eval_inputs,eval_model)
        last_checkpoint_step = global_step
        if loss < best_eval_score:
            best_eval_score = loss
            train_model.saver.save(sess, os.path.join(out_dir, "my_model_%.3f" % loss), global_step=global_step)
