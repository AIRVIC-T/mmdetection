import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS


@MODELS.register_module()
class SupConProxyAnchorLoss(torch.nn.Module):
    def __init__(self, class_num=37, size_contrast=512, stage=1, mrg=0, alpha=32, loss_weight=0.5):
        torch.nn.Module.__init__(self)
        self.nb_classes = class_num
        self.sz_embed = size_contrast
        self.mrg = mrg
        self.alpha = alpha
        self.loss_weight = loss_weight
        self.stage = stage
        self.proxies = torch.nn.Parameter(torch.randn(
            self.nb_classes + 1, self.sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

    @torch.no_grad()
    def init_proxies(self, features, labels, cls_scores):
        assert features.shape[0] == labels.shape[0] == cls_scores.shape[0]
        cls_scores = F.softmax(
            cls_scores, dim=1) if cls_scores is not None else None
        cls_scores = cls_scores.cpu()
        labels = labels.cpu()
        features = features.cpu()
        self.class_label = torch.arange(0, self.nb_classes + 1)  # .cuda()
        if len(labels.shape) == 1:
            labels = labels.reshape(-1, 1)
        label_mask = torch.eq(labels, self.class_label.T).float()
        score_mask = torch.mul(label_mask, cls_scores).T
        score_mask = torch.where(
            score_mask > 0.6, score_mask, torch.zeros_like(score_mask))
        score_mask = score_mask.unsqueeze(dim=-1).repeat([1, 1, self.sz_embed])
        features = features.unsqueeze(dim=0).repeat([self.nb_classes+1, 1, 1])
        proxies = torch.mul(score_mask, features)
        proxies = F.normalize(proxies.sum(dim=1), dim=-1)
        proxies = torch.add(proxies, self.proxies.cpu())
        self.proxies = F.normalize(proxies, dim=-1)
        if self.iter_now < 10001:
            self.iter_now = self.iter_now + 1
        else:
            self.epoch_now = self.epoch_now + 1
            self.iter_now = 0
            torch.save(
                self.proxies, '/home/lwt/work/PETDet/work_dirs/proxy_anchor_init/'+str(self.epoch_now)+'.pt')

    def forward(self, X, labels):
        assert X.shape[0] == labels.shape[0]
        if len(labels.shape) == 1:
            labels = labels.reshape(-1, 1)
        P = F.normalize(self.proxies, dim=-1)
        X = F.normalize(X, dim=-1)
        cos = F.linear(X, P)  # Calcluate cosine similarity
        class_label = torch.arange(0, self.nb_classes + 1).cuda()
        P_one_hot = torch.eq(labels, class_label.T).float().cuda()
        N_one_hot = 1 - P_one_hot

        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))
        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(
            dim=1)  # The set of positive proxies of data in the batch

        # The number of positive proxies
        num_valid_proxies = len(with_pos_proxies)
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp,
                                torch.zeros_like(pos_exp)).sum(dim=0)
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp,
                                torch.zeros_like(neg_exp)).sum(dim=0)

        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term

        return self.loss_weight * loss
