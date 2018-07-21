"""Utility functions for building models."""

import numpy as np
import tensorflow as tf


def _single_cell(num_units,dropout):
    #single_cell = tf.contrib.rnn.GRUCell(num_units)
    single_cell = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(num_units)
    single_cell = tf.contrib.rnn.DropoutWrapper(
      cell = single_cell, input_keep_prob = (1.0 - dropout))
    return single_cell

def _cell_list(num_units, num_layers, dropout):
  cell_list = []
  for i in range(num_layers):
    single_cell = _single_cell(num_units=num_units,dropout=dropout)
    cell_list.append(single_cell)
  return cell_list


def gradient_clip(gradients, max_gradient_norm):
  """Clipping gradients of a model."""
  clipped_gradients, gradient_norm = tf.clip_by_global_norm(
      gradients, max_gradient_norm)
  gradient_norm_summary = [tf.summary.scalar("grad_norm", gradient_norm)]
  gradient_norm_summary.append(
      tf.summary.scalar("clipped_gradient", tf.global_norm(clipped_gradients)))

  return clipped_gradients, gradient_norm_summary, gradient_norm


def create_or_load_model(model, model_dir, session, name):
  """Create translation model and initialize or load parameters in session."""
  latest_ckpt = tf.train.latest_checkpoint(model_dir)
  if latest_ckpt:
    model = load_model(model, latest_ckpt, session, name)
  else:
    start_time = time.time()
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    utils.print_out("  created %s model with fresh parameters, time %.2fs" %
                    (name, time.time() - start_time))

  global_step = model.global_step.eval(session=session)
  return model, global_step

