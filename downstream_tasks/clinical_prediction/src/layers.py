import math
from turtle import forward
from unicodedata import bidirectional
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.nn.parameter import Parameter


def init_model(args, note_dim=200, ts_dim=104):

    Y = 570 if args.task == 'drg' else 2

    if args.modality == 'text':
        model = TextClassifier(
            Y=Y,
            input_size=note_dim,
            hidden_size=args.hidden_size_txt,
            dropout=args.dropout
        )

    elif args.modality == 'struct':
        model = StructClassifier(
            Y=Y,
            input_size=ts_dim,
            hidden_size=args.hidden_size_ts,
            dropout=args.dropout
        )

    elif args.modality == 'both':
        model = BimodalClassifier(
            Y=Y,
            input_txt_size=note_dim,
            input_ts_size=ts_dim,
            hidden_size_txt=args.hidden_size_txt,
            hidden_size_ts=args.hidden_size_ts,
            dropout=args.dropout
        )

    else:
        raise NotImplementedError

    return model



class FilterLinear(nn.Module):
    def __init__(self, in_features, out_features, identity_matric_size, bias=True):
        super(FilterLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        assert in_features > 1 and out_features > 1, "Passing in nonsense sizes"

        filter_square_matrix = torch.eye(identity_matric_size, requires_grad=False)
        self.register_buffer("filter_square_matrix", filter_square_matrix)

        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias: self.bias = Parameter(torch.Tensor(out_features))
        else:    self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None: self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return F.linear(
            x,
            self.filter_square_matrix.mul(self.weight),
            self.bias
        )

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) + ')'


class GRUD(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, **kwargs):
        super(GRUD, self).__init__()


        self.hidden_size = hidden_size
        self.delta_size = input_size
        self.mask_size = input_size

        self.zl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size) # Wz, Uz are part of the same network. the bias is bz
        self.rl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size) # Wr, Ur are part of the same network. the bias is br
        self.hl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size) # W, U are part of the same network. the bias is b
        
        self.gamma_x_l = FilterLinear(self.delta_size, self.delta_size, input_size)
        self.gamma_h_l = nn.Linear(self.delta_size, self.hidden_size) 

        self.recurrent_dropout1 = nn.Dropout(dropout)
        self.recurrent_dropout2 = nn.Dropout(dropout)
        self.recurrent_dropout3 = nn.Dropout(dropout)

    def step(self, x, x_last_obsv, x_mean, h, mask, delta):

        batch_size = x.size()[0]
        dim_size = x.size()[1]

        self.zeros = torch.zeros(batch_size, self.delta_size).type_as(h)
        self.zeros_h=torch.zeros(batch_size, self.hidden_size).type_as(h)

        # x
        # gamma_x_l_delta = self.gamma_x_l(delta)
        delta_x = torch.exp(-torch.max(self.zeros, self.gamma_x_l(delta))) #exponentiated negative rectifier

        x_mean = x_mean.repeat(batch_size, 1)
        x = mask * x + (1 - mask) * (delta_x * x_last_obsv + (1 - delta_x) * x_mean)

        # h
        # gamma_h_l_delta = self.gamma_h_l(delta)
        delta_h = torch.exp(-torch.max(self.zeros_h, self.gamma_h_l(delta))) #self.zeros became self.zeros_h to accomodate hidden size != input size
        
        h = delta_h * h

        # Basically trying to follow the recurrent dp patterns in https://github.com/PeterChe1990/GRU-D/blob/9e1274a1ad67135137f53159eafc92c7278a931a/nn_utils/grud_layers.py#L270-L293

        comb1 = torch.cat((x, self.recurrent_dropout1(h), mask), 1)
        comb2 = torch.cat((x, self.recurrent_dropout2(h), mask), 1)

        z = torch.sigmoid(self.zl(comb1))
        r = torch.sigmoid(self.rl(comb2))

        comb3 = torch.cat((x, r * self.recurrent_dropout3(h), mask), 1)
        h_tilde = torch.tanh(self.hl(comb3))

        # previous implem w/o dp
        # # comb
        # combined = torch.cat((x, h, mask), 1)
        # z = torch.sigmoid(self.zl(combined)) #sigmoid(W_z*x_t + U_z*h_{t-1} + V_z*m_t + bz)
        # r = torch.sigmoid(self.rl(combined)) #sigmoid(W_r*x_t + U_r*h_{t-1} + V_r*m_t + br)
        # # comb reset
        # combined_r = torch.cat((x, r * h, mask), 1)
        # h_tilde = torch.tanh(self.hl(combined_r)) #tanh(W*x_t +U(r_t*h_{t-1}) + V*m_t) + b


        # gated
        h = (1 - z) * h + z * h_tilde
        
        return h

    def forward(self, X, X_last_obsv, Mask, Delta):
        batch_size, step_size, spatial_size = X.size()

        Hidden_State = torch.zeros(batch_size, self.hidden_size).type_as(X)

        assert self.X_mean.sum != 0, 'init X mean required'

        outputs = None
        for i in range(step_size):
            Hidden_State = self.step(
                    torch.squeeze(X[:,i:i+1,:], 1),
                    torch.squeeze(X_last_obsv[:,i:i+1,:], 1),
                    torch.squeeze(self.X_mean[:,i:i+1,:], 1),
                    Hidden_State,
                    torch.squeeze(Mask[:,i:i+1,:], 1),
                    torch.squeeze(Delta[:,i:i+1,:], 1),
                )

            if outputs is None:
                outputs = Hidden_State.unsqueeze(1)
            else:
                outputs = torch.cat((outputs, Hidden_State.unsqueeze(1)), 1)
                
        return outputs, outputs[:,-1,:]

    def _init_x_mean(self, X_mean):
        X_mean = torch.from_numpy(X_mean.copy()).float()
        self.register_buffer('X_mean', X_mean)



class TextClassifier(nn.Module):
    def __init__(self, Y, input_size, hidden_size, dropout, **kwargs):
        super().__init__()

        self.LSTM = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)

        self.dropout = nn.Dropout(dropout)

        encode_size = hidden_size * 2

        self.fc = nn.Sequential(
            nn.Linear(encode_size, encode_size),
            nn.Tanh(),
            nn.Linear(encode_size, Y)
        )

    def forward(self, x_note):
        _, (h_n, _) = self.LSTM(x_note)
        x_note = torch.cat([h_n[0], h_n[1]], -1)
        logits = self.fc(self.dropout(x_note))
        return logits 



class StructClassifier(nn.Module):
    def __init__(self, Y, input_size, hidden_size, dropout, **kwargs):
        super().__init__()

        self.GRUD = GRUD(input_size, hidden_size, dropout)

        self.dropout = nn.Dropout(dropout)

        encode_size = hidden_size

        self.fc = nn.Sequential(
            nn.Linear(encode_size, encode_size),
            nn.Tanh(),
            nn.Linear(encode_size, Y)
        )

    def forward(self, x_ts):

        x = x_ts.float()

        mask = x[:, :, torch.arange(0, x.size(2), 3)]
        measurement = x[:, :, torch.arange(1, x.size(2), 3)]
        time = x[:, :, torch.arange(2, x.size(2), 3)]

        measurement_last_obsv = measurement # followed mimic extract repo; masking will take care of imputed values
        x_input = (measurement, measurement_last_obsv, mask, time)

        _, h_n = self.GRUD(*x_input)
        logits = self.fc(self.dropout(h_n))
        return logits 
    
    

class BimodalClassifier(nn.Module):
    def __init__(self, Y, input_txt_size, input_ts_size, hidden_size_txt, hidden_size_ts, dropout, **kwargs):
        super().__init__()

        self.LSTM = nn.LSTM(input_size=input_txt_size, hidden_size=hidden_size_txt, batch_first=True, bidirectional=True)
        self.GRUD = GRUD(input_ts_size, hidden_size_ts, dropout)

        self.dropout = nn.Dropout(dropout)

        encode_size = hidden_size_txt * 2 + hidden_size_ts

        self.fc = nn.Sequential(
            nn.Linear(encode_size, encode_size),
            nn.Tanh(),
            nn.Linear(encode_size, Y)
        )

    def forward(self, x_note, x_ts):

        # txt
        _, (h_n, _) = self.LSTM(x_note)
        x_note = torch.cat([h_n[0], h_n[1]], -1)
        x_note = self.dropout(x_note)

        # ts 
        x = x_ts.float()

        mask = x[:, :, torch.arange(0, x.size(2), 3)]
        measurement = x[:, :, torch.arange(1, x.size(2), 3)]
        time = x[:, :, torch.arange(2, x.size(2), 3)]

        measurement_last_obsv = measurement # followed mimic extract repo; masking will take care of imputed values
        x_input = (measurement, measurement_last_obsv, mask, time)

        _, h_n = self.GRUD(*x_input)
        x_ts = self.dropout(h_n)

        # fuse
        x = torch.cat([x_note, x_ts], -1)

        logits = self.fc(x)
        return logits

