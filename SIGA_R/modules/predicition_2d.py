import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Attention(nn.Module):
    def __init__(self, input_size, hidden_size, num_class, num_char_embeddings=256):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_class = num_class
        self.generator = nn.Linear(hidden_size, num_class)
        self.char_embeddings = nn.Embedding(num_class+1, num_char_embeddings)
        self.rnn = nn.LSTMCell(input_size + num_char_embeddings, hidden_size)

    def forward(self, batch_H, text, is_train=True, batch_max_length=25):
        """
        input:
            batch_H : contextual_feature H = hidden state of encoder. [batch_size x num_steps x contextual_feature_channels]
            text : the text-index of each image. [batch_size x (max_length+1)]. +1 for [SOS] token. text[:, 0] = [SOS].
        output: probability distribution at each step [batch_size x num_steps x num_class]
        """
        batch_size = batch_H.size(0)
        num_steps = batch_max_length + 1  # +1 for [EOS] at end of sentence.

        output_hiddens = (torch.FloatTensor(batch_size, num_steps, self.hidden_size).fill_(0).to(device))
        hidden = (
            torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device),
            torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device),
        )

        if is_train:
            for i in range(num_steps):
                char_embeddings = self.char_embeddings(text[:, i])
                hidden = self.rnn(torch.cat((batch_H[:, i, :], char_embeddings), dim=1), hidden)
                output_hiddens[:, i, :] = hidden[0]
            probs = self.generator(output_hiddens)
        else:
            targets = text[0].expand(batch_size)  # should be fill with [SOS] token
            probs = (torch.FloatTensor(batch_size, num_steps, self.num_class).fill_(0).to(device))
            for i in range(num_steps):
                char_embeddings = self.char_embeddings(targets)
                hidden = self.rnn(torch.cat((batch_H[:, i, :], char_embeddings), dim=1), hidden)
                probs_step = self.generator(hidden[0])
                output_hiddens[:, i, :] = hidden[0]
                probs[:, i, :] = probs_step
                _, next_input = probs_step.max(1)
                targets = next_input
        return probs, output_hiddens  # batch_size x num_steps x num_class
