from __future__ import unicode_literals, print_function, division
from io import open
import string
import random

import torch
import torch.nn as nn
from pprint import pprint as pp
import time

from lang import Lang
import util
from util import prepareData
from encoder_rnn import EncoderRNN
from attn_decoder_rnn import AttnDecoderRNN
import const

device = util.device()

def evaluate(encoder, decoder, sentence, max_length=const.MAX_LENGTH):
  with torch.no_grad():
    input_tensor = util.tensorFromSentence(input_lang, sentence)
    input_length = input_tensor.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    for ei in range(input_length):
      encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
      encoder_outputs[ei] += encoder_output[0, 0]

    decoder_input = torch.tensor([[const.SOS_token]], device=device) # SOS

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
      decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)

      decoder_attentions[di] = decoder_attention.data
      topv, topi = decoder_output.data.topk(1)
      if topi.item() == const.EOS_token:
        decoded_words.append('<EOS>')
        break
      else:
        decoded_words.append(output_lang.index2word[topi.item()])

      decoder_input = topi.squeeze().detach()

    return decoded_words, decoder_attentions[:di + 1]


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
encoder1.load_state_dict(torch.load('./models/encoder1.bin', map_location=device))
attn_decoder1.load_state_dict(torch.load('./models/attn_decoder1.bin', map_location=device))

evaluateRandomly(encoder1, attn_decoder1)
