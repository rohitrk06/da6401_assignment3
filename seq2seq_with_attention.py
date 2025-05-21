import torch
from torch import nn
import torch.nn.functional as F
import lightning as L

#We are implementing the Encoder Decoder architecture with attention module in this file.
#Let's define the encoder, decoder architecture, 
class Encoder(nn.Module):
    def __init__(self, cell_type, input_embedding_size, embedding_dimension, hidden_layer_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.dropout = dropout

        # Embedding layer
        self.embedding = nn.Embedding(input_embedding_size, embedding_dimension, padding_idx=0)

        # Here while defining the embedding layer, we have set the padding_idx to 0, which is the index of the <PAD> token in our vocab.
        # This means that the embedding layer will ignore the padding tokens when computing the embeddings.
        # The embedding layer will learn the embeddings for the input characters.

        # Encoder RNN cell
        self.rnn_cell = {
            "RNN": nn.RNN,
            "LSTM": nn.LSTM,
            "GRU": nn.GRU
        }.get(cell_type)
        if self.rnn_cell is None:
            raise ValueError("Invalid cell type. Choose 'RNN', 'LSTM' or 'GRU'.")
        self.rnn = self.rnn_cell(embedding_dimension, hidden_layer_size, num_layers, dropout=dropout, batch_first=True)

    def forward(self, x):
        input = self.embedding(x)
        output, hidden = self.rnn(input)
        return output, hidden
    
class AttentionModule(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionModule, self).__init__()
        self.attn = nn.Linear(hidden_size*2,hidden_size)
        self.v = nn.Linear(hidden_size,1,bias=False)

    def forward(self,decoder_hidden,encoder_outputs, mask=None):
        #Here, dimension of decoder_hidden = (batch, 1, hidden)
        # dimension of encoder_outputs: (batch, src_len, hidden)
        src_len = encoder_outputs.size(1)
        decoder_hidden  = decoder_hidden.repeat(1,src_len,1) # this will make the dimension: (batch, src_len, hidden)

        energy = torch.tanh(self.attn(torch.cat((decoder_hidden,encoder_outputs), dim = 2)))
        attention = self.v(energy).squeeze(2)
        if mask is not None:
            attention = attention.masked_fill(mask == 0, torch.finfo(attention.dtype).min)
        return F.softmax(attention,dim=1)
    
class Decoder(nn.Module):
    def __init__(self, cell_type, output_embedding_size, embedding_dimension, hidden_layer_size, num_layers, dropout):
        super(Decoder, self).__init__()
        # Embedding layer
        self.embedding = nn.Embedding(output_embedding_size, embedding_dimension, padding_idx=0)

        rnn_cell = {
            "RNN": nn.RNN,
            "LSTM": nn.LSTM,
            "GRU": nn.GRU
        }.get(cell_type)
        if rnn_cell is None:
            raise ValueError("Invalid cell type. Choose 'RNN', 'LSTM' or 'GRU'.")
        
        self.rnn = rnn_cell(embedding_dimension + hidden_layer_size, hidden_layer_size, num_layers, dropout=dropout, batch_first=True)

        # Output layer
        self.fc = nn.Linear(hidden_layer_size * 2, output_embedding_size)

        #attention module
        self.attention = AttentionModule(hidden_layer_size)
        self.cell_type = cell_type
        
    def forward(self, x, hidden, encoder_outputs, mask = None):
        embedded = self.embedding(x)

        if self.cell_type == "LSTM":
            dec_hidden = hidden[0][-1].unsqueeze(1)
        else:
            dec_hidden = hidden[-1].unsqueeze(1)

        attn_weights = self.attention(dec_hidden, encoder_outputs, mask).unsqueeze(1)

        context = torch.bmm(attn_weights, encoder_outputs)

        rnn_input = torch.cat((embedded, context), dim = 2)

        output, hidden = self.rnn(rnn_input, hidden)
        output = output.squeeze(1)
        context = context.squeeze(1)

        output = self.fc(torch.cat((output,context),dim=1))
                
        return output.unsqueeze(1), hidden, attn_weights.squeeze(1)
    

class Seq2Seq(L.LightningModule):
    def __init__(self, input_embedding_size, output_embedding_size, embedding_dimension, 
                 hidden_layer_size, number_of_layers_encoder, number_of_layers_decoder, 
                 dropout, cell_type, learning_rate, teacher_forcing_ratio=0.5):
        super().__init__()
        self.save_hyperparameters()

        # Initialize encoder and decoder
        self.encoder = Encoder(cell_type, input_embedding_size, embedding_dimension,
                              hidden_layer_size, number_of_layers_encoder, dropout)
        self.decoder = Decoder(cell_type, output_embedding_size, embedding_dimension,
                              hidden_layer_size, number_of_layers_decoder, dropout)
        
        # Loss function ignoring padding
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.learning_rate = learning_rate
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, src, tgt, teacher_forcing_ratio=None):
        batch_size, tgt_len = tgt.size()
        outputs = torch.zeros(batch_size, tgt_len-1, self.hparams.output_embedding_size)

        attn_weights_all = []
        # Encoder forward
        encoder_outputs, hidden = self.encoder(src)
        
        # Initialize decoder hidden state
        if self.hparams.cell_type == 'LSTM':
            decoder_hidden = (hidden[0][:self.hparams.number_of_layers_decoder],
                             hidden[1][:self.hparams.number_of_layers_decoder])
        else:
            decoder_hidden = hidden[:self.hparams.number_of_layers_decoder]
        
        decoder_input = tgt[:, 0].unsqueeze(1)  # Start with SOS token
        mask = (src!=0)
        # Decoder forward
        for t in range(tgt_len-1):
            decoder_output, decoder_hidden, attn_weights = self.decoder(decoder_input, decoder_hidden, encoder_outputs, mask)
            outputs[:, t] = decoder_output.squeeze(1)
            attn_weights_all.append(attn_weights.detach().cpu())
            # Teacher forcing
            tf_ratio = self.teacher_forcing_ratio if teacher_forcing_ratio is None else teacher_forcing_ratio
            teacher_force = torch.rand(1).item() < tf_ratio
            top1 = decoder_output.argmax(2)
            decoder_input = tgt[:, t+1].unsqueeze(1) if teacher_force else top1

        attn_weights_all = torch.stack(attn_weights_all, dim=1)
        return outputs, attn_weights_all

    def __shared_step(self, batch, batch_idx, stage):
        src, tgt = batch
        output,_ = self(src, tgt, teacher_forcing_ratio=0 if stage != 'train' else None)
        tgt = tgt.to(output.device)
        loss = self.criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
        self.log(f"{stage}_loss", loss)
        
        # Calculate metrics
        preds = output.argmax(2)
        non_pad = tgt[:, 1:] != 0
        correct = (preds == tgt[:, 1:]) & non_pad
        
        # Token-level accuracy
        token_acc = correct.sum().float() / non_pad.sum()
        self.log(f"{stage}_token_acc", token_acc, prog_bar=True)
        
        # Sequence-level accuracy
        seq_acc = ((preds == tgt[:, 1:]) | ~non_pad).all(dim=1).float().mean()
        self.log(f"{stage}_seq_acc", seq_acc, prog_bar=True)
        
        return loss

    def training_step(self, batch, batch_idx):
        return self.__shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.__shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.__shared_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx):
        src, _ = batch  # Don't require target during prediction
        output,_ = self(src, torch.zeros_like(src), teacher_forcing_ratio=0)
        return output.argmax(2)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
