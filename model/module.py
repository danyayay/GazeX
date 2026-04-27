import torch
import torch.nn as nn


class HiddenLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers):
        super(HiddenLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)

    def forward(self, input, h=None, return_h=False):
        lstm_out, h = self.lstm(input, h) # (hidden[0][1] == lstm_out[:,-1,:]).all() = True
        if return_h:
            return lstm_out, h
        else:
            return lstm_out
    

class HiddenGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, device):
        super(HiddenGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device 
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)

    def forward(self, input, hidden=None):
        batch_size = input.size(0)
        if hidden is None:
            hidden = torch.zeros((self.n_layers, batch_size, self.hidden_dim), device=self.device)
        out, hidden = self.gru(input, hidden) # (hidden[0][1] == out[:,-1,:]).all() = True
        return out, hidden


class HiddenGRUProjected(nn.Module):
    def __init__(self, output_dim, hidden_dim, n_layers, device):
        super(HiddenGRUProjected, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device
        self.gru = nn.GRU(output_dim, hidden_dim, n_layers, batch_first=True)
        self.projection_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden=None):
        batch_size = input.size(0)
        if hidden is None:
            hidden = torch.zeros((self.n_layers, batch_size, self.hidden_dim), device=self.device)
        gru_out, hidden = self.gru(input, hidden)
        projected = self.projection_layer(gru_out)
        return projected, hidden
    

class HiddenDense(nn.Module):
    def __init__(
            self, input_dim, hidden_dim, output_dim, dense_n_layers,
            with_reg=False, dropout=0.3):
        super(HiddenDense, self).__init__()
        self.with_reg = with_reg
        self.n_layers = dense_n_layers

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        if self.n_layers == 2: 
            self.fc2 = nn.Linear(hidden_dim, output_dim)

        if self.with_reg:
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.dropout1 = nn.Dropout(dropout)
            if self.n_layers == 2:
                self.bn2 = nn.BatchNorm1d(output_dim)
                self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout1(torch.relu(self.bn1(self.fc1(x)))) if self.with_reg else torch.relu(self.fc1(x))
        if self.n_layers == 2:
            x = self.dropout2(torch.relu(self.bn2(self.fc2(x)))) if self.with_reg else torch.relu(self.fc2(x))
        return x
    

class OutputDense(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(OutputDense, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, input):
        x = self.fc(input)
        return x
    

class CategoricalEmbedding(nn.Module):
    def __init__(
            self, aux_names: list[str], aux_hidden_dims: list[int], 
            aux_num_dict: dict, device: str):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(aux_num_dict[aux], aux_hidden_dim, device=device)
            for aux, aux_hidden_dim in zip(aux_names, aux_hidden_dims)
        ])

    def forward(self, x: torch.Tensor):
        # x shape: [batch_size, num_features] with int values
        out = []
        for i, emb in enumerate(self.embeddings):
            out.append(emb(x[:, i]))
        return torch.cat(out, dim=-1)
    

class AutoregressiveLSTMDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, len_pred, device):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.len_pred = len_pred
        self.device = device

        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, h, labels=None, teacher_forcing_ratio=0.5):
        batch_size = h.size(0)
        input_t = torch.zeros(batch_size, 1, self.output_dim, device=self.device)

        # hidden state initialization from z + cond
        # h = torch.tanh(torch.cat([z, cond], dim=1)).unsqueeze(0)
        h = h.unsqueeze(0)  # (1, batch_size, hidden_dim)

        outputs = []
        for t in range(self.len_pred):
            out, h = self.gru(input_t, h)
            mu = self.fc_out(out.squeeze(1))
            outputs.append(mu.unsqueeze(1))

            if labels is not None and torch.rand(1).item() < teacher_forcing_ratio:
                input_t = labels[:, t:t+1, :]
            else:
                input_t = mu.unsqueeze(1)

        return torch.cat(outputs, dim=1)