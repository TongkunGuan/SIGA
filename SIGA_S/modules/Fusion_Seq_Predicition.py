import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.Fusion_Package import Cat_fusion
# from modules.MLP_Head import MultiModal_Fusion

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Seq_Fusion(nn.Module):
    def __init__(self, input_size, hidden_size, input_H, input_W, num_class, iterable=1, num_char_embeddings=256):
        super(Seq_Fusion, self).__init__()
        # self.fusion = MultiModal_Fusion(input_size, hidden_size, input_H, input_W, 1, iterable)
        self.fusion = Cat_fusion(hidden_size)
        self.hidden_size = hidden_size
        self.num_class = num_class
        self.generator = nn.Linear(hidden_size, num_class)
        self.char_embeddings = nn.Embedding(num_class, num_char_embeddings)
        self.rnn = nn.LSTMCell(input_size + num_char_embeddings, hidden_size)

    def forward(self, batch_H, Attentive_Sequence, text, is_train=True, batch_max_length=25):
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
        Char = torch.FloatTensor(batch_size, num_steps, self.hidden_size).fill_(0).to(device)
        if is_train:
            for i in range(num_steps):
                char_embeddings = self.char_embeddings(text[:, i])
                I_char = self.fusion(Attentive_Sequence[:, i, :], batch_H[:, i, :])
                # hidden : decoder's hidden s_{t-1}, batch_H : encoder's hidden H, char_embeddings : f(y_{t-1})
                concat_context = torch.cat([I_char, char_embeddings], 1)  # batch_size x (num_channel + num_embedding)
                hidden = self.rnn(concat_context, hidden)
                output_hiddens[:, i, :] = hidden[0]  # LSTM hidden index (0: hidden, 1: Cell)
                Char[:, i, :] = I_char
            probs = self.generator(output_hiddens)

        else:
            targets = text[0].expand(batch_size)  # should be fill with [SOS] token
            probs = (torch.FloatTensor(batch_size, num_steps, self.num_class).fill_(0).to(device))
            for i in range(num_steps):
                char_embeddings = self.char_embeddings(targets)
                I_char = self.fusion(Attentive_Sequence[:, i, :], batch_H[:, i, :])
                concat_context = torch.cat([I_char, char_embeddings], 1)  # batch_size x (num_channel + num_embedding)
                hidden = self.rnn(concat_context, hidden)
                output_hiddens[:, i, :] = hidden[0]  # LSTM hidden index (0: hidden, 1: Cell)
                probs_step = self.generator(hidden[0])
                probs[:, i, :] = probs_step
                Char[:, i, :] = I_char
                _, next_input = probs_step.max(1)
                targets = next_input
        return probs, output_hiddens, Char  # batch_size x num_steps x num_class

