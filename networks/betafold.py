
import torch
from torch import nn
import sidechainnet as scn
from transformers import BertModel


class BetaFold(nn.Module):

    def __init__(self):
        super(BetaFold, self).__init__()
        self.bert = BertModel.from_pretrained("Rostlab/prot_bert_bfd")
        d_output = scn.structure.build_info.NUM_ANGLES*2
        self.classifier = nn.Linear(1024, d_output, bias=True)
        self.output_activation = torch.nn.Tanh()

    def forward(self, encoded_input):
        output = self.bert(**encoded_input)
        output = self.classifier(output.last_hidden_state)
        output = self.output_activation(output)
        output = output.view(output.shape[0], output.shape[1], 12, 2)
        return output

