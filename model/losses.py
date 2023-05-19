
from fastNLP import LossBase
import torch.nn.functional as F
from fastNLP import seq_len_to_mask
import torch

def get_one_hot(label, N, off_value, on_value):

    size = list(label.size())
    size.append(N)
    label = label.view(-1)
    ones = torch.sparse.torch.eye(N) * on_value
    
    ones = ones.index_select(0, label.cpu())
    ones += off_value
    ones = ones.view(*size)
    ones = ones.cuda()

    ones[:,:,0:1] = 0
    return ones

def KL(alpha, beta):

    S_alpha = torch.sum(alpha, dim=1, keepdim=False)
    S_beta = torch.sum(beta, dim=1, keepdim=False)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=False)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=False) - torch.lgamma(S_beta)
    dg0 = torch.digamma(torch.sum(alpha, dim=1, keepdim=True))
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=False) + lnB + lnB_uni

    return kl

class Seq2SeqLoss(LossBase):
    def __init__(self):
        super().__init__()

    def get_loss(self, tgt_tokens, tgt_seq_len, pred):
        '''
        :param all_span_rep: shape: (bs, n_span, n_class)
        :param span_label_ltoken:
        :param real_span_mask_ltoken:
        :return:
        '''

        tgt_seq_len = tgt_seq_len - 1
        mask = seq_len_to_mask(tgt_seq_len, max_len=tgt_tokens.size(1) - 1).eq(0)
        tgt_tokens = tgt_tokens[:, 1:].masked_fill(mask, 0)

        alpha = F.softplus(pred) + 1

        S = torch.sum(alpha, dim=1, keepdim=True)
        E = alpha - 1
        bu = E/S
        beta = torch.ones_like(alpha)
        n_class = pred.size(2)
        
        soft_output = get_one_hot(tgt_tokens, n_class, 0, 1)
        
        A = torch.sum((1-bu) * soft_output * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=False) 
        alp = E * (1 - soft_output) + 1
        B = KL(alp, beta)

        loss = (A+B/(n_class))
        loss = loss.sum()

        return loss
