import torch
import torch.nn as nn
from torch.autograd import Variable
import random 
import string 
import time, math

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        
        self.encoder = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)
    
    def forward(self, input, hidden):
        input = self.encoder(input.view(1, -1))
        output, hidden = self.gru(input.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))

class RNNLyricsGenerator:
    def __init__(self, decoder, chunk_size=500, lr = 0.005):

        self.all_characters = string.printable
        self.decoder = decoder
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.chunk_size = chunk_size
        pass


    def time_since(self, since):
        s = time.time() - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def sample_lyrics(self):
        start = random.randint(0, len(self.lyrics) - self.chunk_size)
        end = start + self.chunk_size + 1
        return self.lyrics[start:end]

    def encode_lyrics(self, lyrics_chunk):
        tensor = torch.zeros(len(lyrics_chunk)).long()
        for c in range(len(lyrics_chunk)):
            tensor[c] = self.all_characters.index(lyrics_chunk[c])
        return Variable(tensor)

    def get_training_set(self):
        chunk = self.sample_lyrics()
        inp = self.encode_lyrics(chunk[:-1])
        target = self.encode_lyrics(chunk[1:])
        return inp, target

    def train_epoch(self):
        inp, target = self.get_training_set()
        hidden = self.decoder.init_hidden()
        self.decoder.zero_grad()
        loss = 0
        for c in range(self.chunk_size):
            output, hidden = self.decoder(inp[c], hidden)
            loss += self.criterion(output, target[c].unsqueeze(0))

        loss.backward()
        self.decoder_optimizer.step()
        return loss.data.item() / self.chunk_size
    
    def train(self, lyrics, n_epochs, print_every=100, plot_every=10):
        self.lyrics = lyrics
        start = time.time()
        all_losses = []
        loss_avg = 0

        for epoch in range(1, n_epochs + 1):
            loss = self.train_epoch()       
            loss_avg += loss

            if epoch % print_every == 0:
                print('[%s (%d %d%%) %.4f]' % (self.time_since(start), epoch, epoch / n_epochs * 100, loss))
                print(self.generate('Wh', 100), '\n')

            if epoch % plot_every == 0:
                all_losses.append(loss_avg / plot_every)
                loss_avg = 0
    

    def generate(self, prime_str='A', predict_len=100, temperature=0.8):
        hidden = self.decoder.init_hidden()
        prime_input = self.encode_lyrics(prime_str)
        predicted = prime_str
        for p in range(len(prime_str) - 1):
            _, hidden = self.decoder(prime_input[p], hidden)
        inp = prime_input[-1]
        
        for p in range(predict_len):
            output, hidden = self.decoder(inp, hidden)
            output_dist = output.data.view(-1).div(temperature).exp()
            top_i = torch.multinomial(output_dist, 1)[0]
            predicted_char = self.all_characters[top_i]
            predicted += predicted_char
            inp = self.encode_lyrics(predicted_char)

        return predicted