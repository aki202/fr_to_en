from __future__ import unicode_literals, print_function, division
from io import open
import string
import random

import torch
import torch.nn as nn
from torch import optim
from pprint import pprint as pp
import time
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

from lang import Lang
import util
from util import prepareData
import const
from encoder_rnn import EncoderRNN
from decoder_rnn import DecoderRNN

device = util.device()
teacher_forcing_ratio = 0.5

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=const.MAX_LENGTH):
  encoder_hidden = encoder.initHidden()

  encoder_optimizer.zero_grad()
  decoder_optimizer.zero_grad()

  input_length = input_tensor.size(0)
  target_length = target_tensor.size(0)

  encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

  loss = 0

  for ei in range(input_length):
    encoder_output, encoder_hidden = encoder(
      input_tensor[ei], encoder_hidden)
    encoder_outputs[ei] = encoder_output[0, 0]

  decoder_input = torch.tensor([[const.SOS_token]], device=device)

  decoder_hidden = encoder_hidden

  use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

  if use_teacher_forcing:
    # Teacher forcing: Feed the target as the next input
    for di in range(target_length):
      decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
      loss += criterion(decoder_output, target_tensor[di])
      decoder_input = target_tensor[di]  # Teacher forcing

  else:
    # Without teacher forcing: use its own predictions as the next input
    for di in range(target_length):
      decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
      topv, topi = decoder_output.topk(1)
      decoder_input = topi.squeeze().detach()  # detach from history as input

      loss += criterion(decoder_output, target_tensor[di])
      if decoder_input.item() == const.EOS_token:
        break

  loss.backward()

  encoder_optimizer.step()
  decoder_optimizer.step()

  return loss.item() / target_length


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
  start = time.time()
  plot_losses = []
  print_loss_total = 0 # Reset every print_every
  plot_loss_total = 0 # Reset every plot_every

  encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
  decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
  training_pairs = [util.tensorsFromPair(random.choice(pairs), input_lang, output_lang) \
    for i in range(n_iters)]
  criterion = nn.NLLLoss()

  for iter in range(1, n_iters + 1):
    training_pair = training_pairs[iter - 1]
    input_tensor = training_pair[0]
    target_tensor = training_pair[1]

    loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
    print_loss_total += loss
    plot_loss_total += loss

    if iter % print_every == 0:
      print_loss_avg = print_loss_total / print_every
      print_loss_total = 0
      print('%s (%d %d%%) %.4f' % (util.timeSince(start, iter / n_iters),
        iter, iter / n_iters * 100, print_loss_avg))

    if iter % plot_every == 0:
      plot_loss_avg = plot_loss_total / plot_every
      plot_losses.append(plot_loss_avg)
      plot_loss_total = 0

  showPlot(plot_losses)

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder1 = DecoderRNN(hidden_size, output_lang.n_words).to(device)

trainIters(encoder1, decoder1, 75000, print_every=5000)
#trainIters(encoder1, decoder1, n_iters=100, print_every=10, plot_every=10)

torch.save(encoder1.state_dict(), 'models/encoder1.bin')
torch.save(decoder1.state_dict(), 'models/decoder1.bin')
