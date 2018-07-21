from config import *
import tensorflow as tf
import copynet
from model_helper import _cell_list, _single_cell

class attention_model():
    def __init__(self,inputs,vocab_table,reverse_vocab_table,
                 vocab_size,gen_vocab_size,embed_size,
                 num_units, lr,mode,dropout,grad_clip = None):
        self.mode = mode
        self.source = inputs.source
        self.source_length = inputs.source_length
        self.vocab_table = vocab_table
        self.reverse_vocab_table = reverse_vocab_table
        self.tgt_sos_id = tf.cast(self.vocab_table.lookup(tf.constant(SOS)),tf.int32)
        self.tgt_eos_id = tf.cast(self.vocab_table.lookup(tf.constant(EOS)),tf.int32)
        if self.mode =='TRAIN' or self.mode=='EVAL':
            self.target_input = inputs.target_input
            self.target_output = inputs.target_output
            self.target_length = inputs.target_length
            self.dropout = dropout
        elif self.mode == 'INFER':
            self.dropout = 0
        else:
            raise  NotImplementedError

        self.num_units = num_units

        self.lr = lr
        self.grad_clip = None

        batch_size = tf.shape(self.source)[0]

        with tf.variable_scope("Model",reuse=tf.AUTO_REUSE) as scope:

            with tf.variable_scope("Embedding") as scope:
                self.embedding_matrix = tf.get_variable("shared_embedding_matrix", [vocab_size, embed_size], dtype=tf.float32)

            with tf.variable_scope("Encoder") as scope:
                self.encoder_emb_inp = tf.nn.embedding_lookup(
                    self.embedding_matrix, self.source)
                bi_inputs = self.encoder_emb_inp
                fw_cells = _cell_list(num_units,num_layers=2,dropout=self.dropout)
                bw_cells = _cell_list(num_units,num_layers=2,dropout=self.dropout)
                # bi_outputs: batch_size * [L , 2*num_units]
                # bi_fw_state num_layers * [batch_size , num_units]
                # bi_bw_state num_layers * [batch_size ,num_units]
                bi_outputs, bi_fw_state,bi_bw_state = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                    cells_fw=fw_cells,
                    cells_bw=bw_cells,
                    inputs=bi_inputs,
                    sequence_length=self.source_length,
                    dtype = tf.float32
                )

            with tf.variable_scope("Decoder") as scope:

                self.encoder_final_state = tuple([tf.concat((bi_fw_state[i],bi_bw_state[i]),axis=-1) for i in range(2)])

                self.decoder_cell = tf.contrib.rnn.MultiRNNCell(
                    _cell_list(num_units * 2, num_layers=2, dropout=0)
                )
                self.attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                    num_units=num_units * 2,
                    memory = bi_outputs,
                    memory_sequence_length = self.source_length)
                self.atten_cell = tf.contrib.seq2seq.AttentionWrapper(
                    cell = self.decoder_cell,
                    attention_mechanism = self.attention_mechanism,
                    attention_layer_size=num_units)

                atten_zero_state = self.atten_cell.zero_state(batch_size=batch_size,dtype=tf.float32)
                self.decoder_initial_state =atten_zero_state.clone(cell_state=self.encoder_final_state)

                #CopyNet
                with tf.variable_scope("CopyNet") as scope:

                    self.copy_cell = copynet.CopyNetWrapper(self.atten_cell, bi_outputs, self.source,
                                      vocab_size, gen_vocab_size) #,encoder_state_size=num_units*2)

                    copy_zero_state = self.copy_cell.zero_state(batch_size=batch_size,dtype=tf.float32)
                    self.decoder_initial_state = copy_zero_state.clone(cell_state=self.decoder_initial_state)

                    self.final_cell = self.copy_cell
                    self.output_layer = None
                #CopyNet end

                #self.output_layer = tf.layers.Dense(vocab_size, use_bias=False, name="output_projection")
                #self.final_cell = self.atten_cell

                if self.mode=='TRAIN' or self.mode=='EVAL':
                    self.decoder_emb_inp = tf.nn.embedding_lookup(self.embedding_matrix,self.target_input)
                    helper = tf.contrib.seq2seq.TrainingHelper(
                        inputs=self.decoder_emb_inp,
                        sequence_length=self.target_length)
                    decoder = tf.contrib.seq2seq.BasicDecoder(
                        cell=self.final_cell,
                        helper=helper,
                        initial_state=self.decoder_initial_state,
                        output_layer=self.output_layer
                    )
                    final_outputs, final_state, final_seq_lengths = tf.contrib.seq2seq.dynamic_decode(
                        decoder=decoder,
                        swap_memory=True
                    )


                elif self.mode=='INFER' :

                    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                        embedding=self.embedding_matrix,
                        start_tokens=tf.fill([batch_size], self.tgt_sos_id),
                        end_token=self.tgt_eos_id)
                    decoder = tf.contrib.seq2seq.BasicDecoder(
                        cell=self.final_cell,
                        helper=helper,
                        initial_state=self.decoder_initial_state,
                        output_layer=self.output_layer
                    )
                    final_outputs, final_state, final_seq_lengths = tf.contrib.seq2seq.dynamic_decode(
                        decoder=decoder,
                        swap_memory=True,
                        maximum_iterations = MAX_DECODE_STEP
                    )

                else:
                    raise NotImplementedError
                # fw_cell.zero_state 's shape: num_layers * (batch_size,num_units)


            self.final_state = final_state
            self.logits = final_outputs.rnn_output
            self.sample_id = final_outputs.sample_id

            # build loss
            if self.mode == 'TRAIN' or self.mode=='EVAL':
                with tf.variable_scope("Loss") as scope:
                    max_time = tf.shape(self.target_output)[1]
                    self.crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=self.target_output, logits=self.logits)
                    self.target_weights = tf.sequence_mask(
                        self.target_length, max_time, dtype=self.logits.dtype)
                    self.loss = tf.reduce_mean(
                        tf.reduce_sum(self.crossent*self.target_weights,axis=-1)/tf.to_float(self.target_length)
                    )

                    '''self.cost = tf.losses.sparse_softmax_cross_entropy(
                        labels = self.target_output,
                        logits = self.logits
                    )   # default is reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
                    '''
                self.true_word = self.reverse_vocab_table.lookup(tf.to_int64(self.target_output))

            if self.mode=='TRAIN':

                #build train_op
                optimizer = tf.train.AdamOptimizer(learning_rate=lr)
                if self.grad_clip is None:
                    self.train_op = optimizer.minimize(self.loss)
                else:
                    tvars = tf.trainable_variables()
                    grads,_ = tf.clip_by_global_norm(
                        tf.gradients(self.loss,tvars),tf.constant(self.grad_clip,dtype=tf.float32))
                    self.train_op = optimizer.apply_gradients(zip(grads,tvars))


            self.probs = tf.nn.softmax(self.logits)
            self.predict = self.sample_id

            self.sample_word = self.reverse_vocab_table.lookup(tf.to_int64(self.sample_id))

            self.saver = tf.train.Saver(
                tf.global_variables(), max_to_keep=MAX_TO_KEEP)
