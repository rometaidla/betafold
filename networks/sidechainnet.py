"""Example models for training protein sequence to angle coordinates tasks."""

import numpy as np
import torch
import sidechainnet as scn
from sidechainnet.structure.structure import inverse_trig_transform
from sidechainnet.structure.build_info import NUM_ANGLES


class BaseProteinAngleRNN(torch.nn.Module):
    """A simple RNN that consumes protein sequences and produces angles."""

    def __init__(self,
                 size,
                 n_layers=1,
                 d_in=20,
                 n_angles=NUM_ANGLES,
                 bidirectional=True,
                 sincos_output=True,
                 device=torch.device('cpu')):
        super(BaseProteinAngleRNN, self).__init__()
        self.size = size
        self.n_layers = n_layers
        self.sincos_output = sincos_output
        self.d_out = n_angles * 2 if sincos_output else n_angles
        self.lstm = torch.nn.LSTM(d_in,
                                  size,
                                  n_layers,
                                  bidirectional=bidirectional,
                                  batch_first=True)
        self.n_direction = 2 if bidirectional else 1
        self.hidden2out = torch.nn.Linear(self.n_direction * size, self.d_out)
        self.output_activation = torch.nn.Tanh()
        self.device_ = device

    def init_hidden(self, batch_size):
        """Initialize the hidden state vectors at the start of a batch iteration."""
        h, c = (torch.zeros(self.n_layers * self.n_direction, batch_size,
                            self.size).to(self.device_),
                torch.zeros(self.n_layers * self.n_direction, batch_size,
                            self.size).to(self.device_))
        return h, c

    def forward(self, *args, **kwargs):
        """Run one forward step of the model."""
        raise NotImplementedError


class IntegerSequenceProteinRNN(BaseProteinAngleRNN):
    """A protein sequence-to-angle model that consumes integer-coded sequences."""

    def __init__(self,
                 size,
                 n_layers=1,
                 d_in=20,
                 n_angles=NUM_ANGLES,
                 bidirectional=True,
                 device=torch.device('cpu'),
                 sincos_output=True):
        super(IntegerSequenceProteinRNN, self).__init__(size=size,
                                                        n_layers=n_layers,
                                                        d_in=d_in,
                                                        n_angles=n_angles,
                                                        bidirectional=bidirectional,
                                                        device=device,
                                                        sincos_output=sincos_output)

        self.input_embedding = torch.nn.Embedding(21, 20, padding_idx=20)

    def forward(self, sequence):
        """Run one forward step of the model."""
        # First, we compute the length of each sequence to use pack_padded_sequence
        lengths = sequence.shape[-1] - (sequence == 20).sum(axis=1)
        h, c = self.init_hidden(len(lengths))
        # Our inputs are sequences of integers, allowing us to use torch.nn.Embeddings
        sequence = self.input_embedding(sequence)
        sequence = torch.nn.utils.rnn.pack_padded_sequence(sequence,
                                                           lengths.cpu(),
                                                           batch_first=True,
                                                           enforce_sorted=False)
        output, (h, c) = self.lstm(sequence, (h, c))
        output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(output,
                                                                        batch_first=True)
        output = self.hidden2out(output)
        if self.sincos_output:
            output = self.output_activation(output)
            output = output.view(output.shape[0], output.shape[1], int(self.d_out / 2), 2)
        else:
            # We push the output through a tanh layer and multiply by pi to ensure
            # values are within [-pi, pi] for predicting raw angles.
            output = self.output_activation(output) * np.pi
            output = output.view(output.shape[0], output.shape[1], self.d_out)
        return output


class ProteinRNN(torch.nn.Module):
    """A protein sequence-to-angle model that consumes integer-coded sequences."""

    def __init__(self,
                 d_hidden,
                 n_layers=1,
                 d_in=21,
                 d_embedding=32,
                 integer_sequence=True,
                 n_angles=scn.structure.build_info.NUM_ANGLES):
        super(ProteinRNN, self).__init__()
        # Dimensionality of RNN hidden state
        self.d_hidden = d_hidden

        # Number of RNN layers (depth)
        self.n_layers = n_layers

        # Underlying RNN (a Long Short-Term Memory network)
        self.lstm = torch.nn.LSTM(d_embedding,
                                  d_hidden,
                                  n_layers,
                                  bidirectional=False,
                                  batch_first=True)

        # Output vector dimensionality (per amino acid)
        self.d_out = n_angles * 2

        # Output projection layer. (from RNN -> target tensor)
        self.hidden2out = torch.nn.Linear(d_hidden, self.d_out)

        # Activation function for the output values (bounds values to [-1, 1])
        self.output_activation = torch.nn.Tanh()

        # We embed our model's input differently depending on the type of input
        self.integer_sequence = integer_sequence
        if self.integer_sequence:
            self.input_embedding = torch.nn.Embedding(d_in, d_embedding, padding_idx=20)
        else:
            self.input_embedding = torch.nn.Linear(d_in, d_embedding)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def init_hidden(self, batch_size):
        """Initialize the hidden state vectors at the start of an iteration."""
        h = torch.zeros(self.n_layers, batch_size, self.d_hidden).to(self.device)
        c = torch.zeros(self.n_layers, batch_size, self.d_hidden).to(self.device)
        return h, c

    def get_lengths(self, sequence):
        """Compute the lengths of each sequence in the batch."""
        if self.integer_sequence:
            lengths = sequence.shape[-1] - (sequence == 20).sum(axis=1)
        else:
            lengths = sequence.shape[1] - (sequence == 0).all(axis=-1).sum(axis=1)
        return lengths.cpu()

    def forward(self, sequence):
        """Run one forward step of the model."""
        # First, we initialize the hidden state vectors and compute sequence lengths
        h, c = self.init_hidden(sequence.shape[0])
        lengths = self.get_lengths(sequence)

        # Next, we embed our input tensors for input to the RNN
        sequence = self.input_embedding(sequence)

        # Then we pass in our data into the RNN via PyTorch's pack_padded_sequences
        sequence = torch.nn.utils.rnn.pack_padded_sequence(sequence,
                                                           lengths,
                                                           batch_first=True,
                                                           enforce_sorted=False)
        output, (h, c) = self.lstm(sequence, (h, c))
        output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(output,
                                                                        batch_first=True)

        # At this point, output has the same dimentionality as the RNN's hidden
        # state: i.e. (batch, length, d_hidden).

        # We use a linear transformation to transform our output tensor into the
        # correct dimensionality (batch, length, 24)
        output = self.hidden2out(output)

        # Next, we need to bound the output values between [-1, 1]
        output = self.output_activation(output)

        # Finally, reshape the output to be (batch, length, angle, (sin/cos val))
        output = output.view(output.shape[0], output.shape[1], 12, 2)

        return output


class PSSMProteinRNN(BaseProteinAngleRNN):
    """A protein structure model consuming 1-hot sequences, 2-ary structures, & PSSMs."""

    def __init__(self,
                 size,
                 n_layers=1,
                 d_in=49,
                 n_angles=NUM_ANGLES,
                 bidirectional=True,
                 device=torch.device('cpu'),
                 sincos_output=True):
        """Create a PSSMProteinRNN model with input dimensionality 41."""
        super(PSSMProteinRNN, self).__init__(size=size,
                                             n_layers=n_layers,
                                             d_in=d_in,
                                             n_angles=n_angles,
                                             bidirectional=bidirectional,
                                             device=device,
                                             sincos_output=sincos_output)

    def forward(self, sequence):
        """Run one forward step of the model."""
        # First, we compute the length of each sequence to use pack_padded_sequence
        lengths = sequence.shape[1] - (sequence == 0).all(axis=2).sum(axis=1)
        h, c = self.init_hidden(len(lengths))
        sequence = torch.nn.utils.rnn.pack_padded_sequence(sequence,
                                                           lengths.cpu(),
                                                           batch_first=True,
                                                           enforce_sorted=False)
        output, (h, c) = self.lstm(sequence, (h, c))
        output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(output,
                                                                        batch_first=True)
        output = self.hidden2out(output)
        if self.sincos_output:
            output = self.output_activation(output)
            output = output.view(output.shape[0], output.shape[1], int(self.d_out / 2), 2)
        else:
            # We push the output through a tanh layer and multiply by pi to ensure
            # values are within [-pi, pi] for predicting raw angles.
            output = self.output_activation(output) * np.pi
            output = output.view(output.shape[0], output.shape[1], self.d_out)
        return output

def build_visualizable_structures(model, data, device, mode=None):
  """Build visualizable structures for one batch of model's predictions on data."""
  with torch.no_grad():
    for batch in data:
      if mode == "seqs":
        model_input = batch.int_seqs.to(device)
      elif mode == "pssms":
        model_input = batch.seq_evo_sec.to(device)

      # Make predictions for angles, and construct 3D atomic coordinates
      predicted_angles_sincos = model(model_input)
      # Because the model predicts sin/cos values, we use this function to recover the original angles
      predicted_angles = inverse_trig_transform(predicted_angles_sincos)

      # EXAMPLE
      # Use BatchedStructureBuilder to build an entire batch of structures
      sb_pred = scn.BatchedStructureBuilder(batch.int_seqs, predicted_angles.cpu())
      sb_true = scn.BatchedStructureBuilder(batch.int_seqs, batch.crds.cpu())
      break
  return sb_pred, sb_true