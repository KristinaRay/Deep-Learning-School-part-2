import random
import torch
from torch import nn
from torch.nn import functional as F

def softmax(x, temperature=10): # use your temperature
    e_x = torch.exp(x / temperature)
    return e_x / torch.sum(e_x, dim=0)

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, bidirectional):
        super().__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers, dropout=dropout, bidirectional=bidirectional)
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, src):
        
        #src = [src sent len, batch size]
        
        # Compute an embedding from the src data and apply dropout to it
        embedded = self.dropout(self.embedding(src))
        
        #embedded = [src sent len, batch size, emb dim]
        
        # Compute the RNN output values of the encoder RNN. 
        # outputs, hidden and cell should be initialized here. Refer to nn.LSTM docs ;)
        
        outputs, (hidden, cell) = self.rnn(embedded)
        
        #outputs = [src sent len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer
        #if self.bidirectional:
            
        return outputs, hidden, cell


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        
        self.attn_in = (enc_hid_dim * 2) + dec_hid_dim
        self.attn = nn.Linear(self.attn_in, enc_hid_dim)
        self.v = nn.Linear(enc_hid_dim, 1)
        
    def forward(self, decoder_hidden, encoder_outputs):
        
        # encoder_outputs = [src sent len, batch size, enc_hid_dim]
        # hidden = [1, batch size, dec_hid_dim]
        
        # repeat hidden and concatenate it with encoder_outputs
        max_len = encoder_outputs.shape[0]
        decoder_hidden = decoder_hidden[-1] # [batch_size, decoder_hid_dim]
        decoder_hidden=decoder_hidden.unsqueeze(1) # [batch_size, 1, decoder_hid_dim]
        repeated_decoder_hidden = decoder_hidden.repeat(1, max_len, 1) # [batch_size, max_len, decoder_hid_dim]
        
        encoder_outputs = encoder_outputs.permute(1,0,2) # [batch_size, max_len, encoder_hid_dim]
        
        # calculate energy
        concatenated_enc_outputs_hid = self.attn(torch.cat((repeated_decoder_hidden,
            encoder_outputs), dim = 2))
        energy = torch.tanh(concatenated_enc_outputs_hid) # [enc_hid_dim + dec_hid_dim, n layers * n directions, batch_size]

        result = self.v(energy).squeeze(1) # [batch_size, n layers * n directions]
        
        # get attention, use softmax function which is defined, can change temperature
        final_result = F.softmax(result, dim=1)
        final_result = final_result.squeeze(2) # [batch_size, n layers * n directions]

        return final_result
    
    
class DecoderWithAttention(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.GRU((2 * enc_hid_dim) + emb_dim, dec_hid_dim, dropout=dropout) # use GRU
        
        self.out = nn.Linear(self.attention.attn_in + emb_dim, output_dim) # linear layer to get next word
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, decoder_hidden, encoder_outputs):
        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        
        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        
        input = input.unsqueeze(0) # because only one word, no words sequence 
        
        # embedded over input and dropout 
        embedded = self.dropout(self.embedding(input)) #  [1, batch size, decoder_emb_dim]
        
        # get weighted sum of encoder_outputs
        
        a = self.attention(decoder_hidden, encoder_outputs) #  [batch_size, n layers * n directions]
        a = a.unsqueeze(1)
        weights = a.bmm(encoder_outputs.transpose(0,1)) #  [1, batch_size, decoder_hid_dim * 2]
        weights = weights.transpose(0,1) 

        # concatenate weighted sum and embedded, break through the GRU
        rnn_input = torch.cat([embedded, weights], 2) # [(2 * enc_hid_dim) + emb_dim, batch_size]
        
        rnn_output, rnn_hidden = self.rnn(rnn_input, decoder_hidden[-1].unsqueeze(0)) 
        
        concat_tensor = torch.cat([rnn_output, weights, embedded],2)
        concat_tensor = concat_tensor.squeeze(0)

        # get predictions
        
        prediction = self.out(concat_tensor) # [batch size, output dim]
        
        return prediction, rnn_hidden
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.dec_hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        # Again, now batch is the first dimention instead of zero
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        enc_states, hidden, cell = self.encoder(src)
        
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        for t in range(1, trg_len):

            output, hidden = self.decoder(input, hidden, enc_states)

            outputs[t] = output
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            #get the highest predicted token from our predictions
            top1 = output.argmax(-1) 
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1
        
        return outputs
