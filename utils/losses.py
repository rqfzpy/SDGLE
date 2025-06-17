# Thanks to rwightman's timm package
# github.com:rwightman/pytorch-image-models
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .metrics import ECELoss, AdaptiveECELoss, ClasswiseECELoss, OELoss, UELoss
class CustomCrossEntropyLoss:
    def __init__(self):
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
    
    def __call__(self, output, target):
        return self.compute_loss(output, target)
    
    def compute_loss(self, output, target):
        return torch.mean(torch.sum(-target * self.logsoftmax(output), dim=1))
def smooth_one_hot(target: torch.Tensor, num_classes: int, smoothing=0.0):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method
    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    true_dist = target.new_zeros(size=(len(target), num_classes)).float()
    true_dist.fill_(smoothing / (num_classes - 1))
    true_dist.scatter_(1, target.data.unsqueeze(1), confidence)
    return true_dist

class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.0):
        super(LabelSmoothingLoss, self).__init__()
        assert 0 <= smoothing < 1
        self.num_classes = num_classes
        self.smoothing = smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        bs = float(pred.size(0))
        pred = pred.log_softmax(dim=1)
        if len(target.shape) == 2:
            true_dist = target
        else:
            true_dist = smooth_one_hot(target, self.num_classes, self.smoothing)
        loss = (-pred * true_dist).sum() / bs
        return loss

class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def _compute_losses(self, x, target):
        log_prob = F.log_softmax(x, dim=-1)
        nll_loss = -log_prob.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_prob.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss

    def forward(self, x, target):
        return self._compute_losses(x, target).mean()

class AbsLoss(object):
    r"""An abstract class for loss functions. 
    """
    def __init__(self):
        self.record = []
        self.bs = []
    
    def compute_loss(self, pred, gt):
        r"""Calculate the loss.
        
        Args:
            pred (torch.Tensor): The prediction tensor.
            gt (torch.Tensor): The ground-truth tensor.

        Return:
            torch.Tensor: The loss.
        """
        pass
    
    def _update_loss(self, pred, gt):
        loss = self.compute_loss(pred, gt)
        self.record.append(loss.item())
        self.bs.append(pred.size()[0])
        return loss
    
    def _average_loss(self):
        record = np.array(self.record)
        bs = np.array(self.bs)
        return (record*bs).sum()/bs.sum()
    
    def _reinit(self):
        self.record = []
        self.bs = []

class KL_DivLoss(AbsLoss):
    r"""The Kullback-Leibler divergence loss function.
    """
    def __init__(self):
        super(KL_DivLoss, self).__init__()
        
        self.loss_fn = nn.KLDivLoss()
        
    def compute_loss(self, pred, gt):
        r"""
        """
        loss = self.loss_fn(pred, gt)
        return loss

class JSloss(nn.Module):
    """  Compute the Jensen-Shannon loss using the torch native kl_div"""
    def __init__(self, reduction='batchmean'):
        super(JSloss, self).__init__()
        self.red = reduction
        
    def forward(self, input, target):
        net = F.softmax(((input + target)/2.),dim=1)
        return 0.5 * (F.kl_div(input.log(), net, reduction=self.red) + 
                    F.kl_div(target.log(), net, reduction=self.red))


##EMD loss
class EMDLoss(nn.Module):
    """EMDLoss class
    """
    def __init__(self):
        super(EMDLoss, self).__init__()

    def forward(self, p_pred: torch.Tensor, p_true: torch.Tensor):
        assert p_true.shape == p_pred.shape, 'Length of the two distribution must be the same'
        cdf_target = torch.cumsum(p_true, dim=1)  # cdf for values [1, 2, ..., 10]
        cdf_estimate = torch.cumsum(p_pred, dim=1)  # cdf for values [1, 2, ..., 10]
        cdf_diff = cdf_estimate - cdf_target
        samplewise_emd = torch.sqrt(torch.mean(torch.pow(torch.abs(cdf_diff), 2), dim=1))
        return samplewise_emd.mean()

def ECE(probs, labels, n_bins=15):
    ece_criterion = ECELoss(n_bins).cuda()
    ece = ece_criterion(probs, labels)
    return ece

def AECE(probs, labels, n_bins=15):
    aece_criterion = AdaptiveECELoss(n_bins).cuda()
    aece = aece_criterion(probs, labels)
    return aece

def dcg_at_k(scores, k):
    """Compute DCG at K"""
    scores = scores[:k]
    gains = 2**scores - 1
    discounts = torch.log2(torch.arange(len(scores), dtype=torch.float32) + 2).cuda()
    return torch.sum(gains / discounts)

def ndcg_at_k(preds, labels, k=256):
    """Compute NDCG at K for a single instance"""
    # Sort by predicted scores
    _, pred_indices = torch.sort(preds, descending=True)
    sorted_labels_by_preds = labels[pred_indices]
    
    # Sort by true scores
    _, ideal_indices = torch.sort(labels, descending=True)
    ideal_labels = labels[ideal_indices]
    
    # Calculate DCG and IDCG
    dcg = dcg_at_k(sorted_labels_by_preds, k)
    idcg = dcg_at_k(ideal_labels, k)
    
    if idcg == 0:
        return 0.0
    else:
        return dcg / idcg

class NDCGLoss(nn.Module):
    def __init__(self, k=256):
        super(NDCGLoss, self).__init__()
        self.k = k

    def forward(self, preds, labels):
        ndcg_score = ndcg_at_k(preds, labels, self.k)
        loss = 1.0 - ndcg_score
        return loss


def dcg(scores, discounts):
    """Compute DCG given scores and discounts."""
    return torch.sum(scores / discounts)

def soft_dcg(preds, discounts, k):
    """Compute the DCG for predicted scores using softmax approximation."""
    pred_probs = torch.softmax(preds, dim=0)
    return dcg(pred_probs, discounts[:k])

def soft_ndcg(preds, labels, discounts, k):
    """Compute NDCG for a single instance using softmax approximation."""
    # dcg_val = dcg(preds[:k], discounts[:k])
    # print(torch.sum(preds))
    
    _, ideal_index = torch.sort(labels, descending=True)
    sorted_presd = torch.gather(preds, 0, ideal_index)

    dcg_val = dcg(sorted_presd[:k], discounts[:k])

    ipreds ,_= torch.sort(preds, descending=True)

    idcg_val = dcg(ipreds[:k], discounts[:k])
    
    if idcg_val == 0:
        return torch.tensor(0.0, device=preds.device)
    else:
        return torch.abs(dcg_val - idcg_val)

class SoftNDCGLoss(nn.Module):
    def __init__(self, k=256):
        super(SoftNDCGLoss, self).__init__()
        self.k = k

    def forward(self, preds, labels):
        discounts = torch.log2(torch.arange(labels.size(0), dtype=torch.float32) + 2).to(preds.device)
     
        ndcg = soft_ndcg(preds, labels, discounts, self.k)

        loss = torch.abs(1.0 - ndcg)
        return loss
    

class BSoftNDCGLoss(nn.Module):
    def __init__(self, k=2):
        super(BSoftNDCGLoss, self).__init__()
        self.k = k

    def forward(self, preds, labels):
        batch_size = preds.size(0)
        discounts = torch.log2(torch.arange(labels.size(1), dtype=torch.float32) + 2).to(preds.device)
        total_ndcg = 0.0
        
        for i in range(batch_size):
            total_ndcg += soft_ndcg(preds[i], labels[i], discounts, self.k)
        
        avg_ndcg = total_ndcg / batch_size

        # loss = torch.abs(1.0 - avg_ndcg)
        # loss = avg_ndcg
        return avg_ndcg