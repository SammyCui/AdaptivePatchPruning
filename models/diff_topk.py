from math import sqrt
import numpy as np
import torch
from einops import rearrange
from torch import nn
import torch.nn.functional as F

from models.perturbations import perturbed


def HardTopK(k, x):
    topk_results = torch.topk(x, k=k, dim=-1, sorted=False)
    indices = topk_results.indices  # b, k
    indices = torch.sort(indices, dim=-1).values
    return indices


class PerturbedTopKFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k: int, num_samples: int = 1000, sigma: float = 0.05):
        b, d = x.shape
        # for Gaussian: noise and gradient are the same.
        noise = torch.normal(mean=0.0, std=1.0, size=(b, num_samples, d)).to(x.device)
        perturbed_x = x[:, None, :] + noise * sigma  # b, nS, d
        topk_results = torch.topk(perturbed_x, k=k, dim=-1, sorted=False)
        indices = topk_results.indices  # b, nS, k
        indices = torch.sort(indices, dim=-1).values  # b, nS, k

        perturbed_output = torch.nn.functional.one_hot(indices, num_classes=d).float()
        indicators = perturbed_output.mean(dim=1)  # b, k, d

        # constants for backward
        ctx.k = k
        ctx.num_samples = num_samples
        ctx.sigma = sigma

        # tensors for backward
        ctx.perturbed_output = perturbed_output
        ctx.noise = noise
        return indicators

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return tuple([None] * 5)

        noise_gradient = ctx.noise
        if ctx.sigma <= 1e-20:
            b, _, k, d = ctx.perturbed_output.size()
            expected_gradient = torch.zeros(b, k, d).to(grad_output.device)
        else:
            expected_gradient = (
                    torch.einsum("bnkd,bnd->bkd", ctx.perturbed_output, noise_gradient)
                    / ctx.num_samples
                    / (ctx.sigma)
            )

        grad_input = torch.einsum("bkd,bkd->bd", grad_output, expected_gradient)

        return (grad_input,) + tuple([None] * 5)


def batched_index_select(input, dim, index):
    for i in range(1, len(input.shape)):
        if i != dim:
            index = index.unsqueeze(i)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)


def extract_patches_from_indices(x, indices):
    batch_size, _, channels = x.shape
    k = indices.shape[-1]
    patches = x
    patches = batched_index_select(patches, 1, indices)
    patches = patches.contiguous().view(batch_size, k, channels)
    return patches


def extract_patches_from_indicators(x, indicators):
    indicators = rearrange(indicators, "b d k -> b k d")
    patches = torch.einsum("b k d, b d c -> b k c",
                           indicators, x)
    return patches


def min_max_norm(x):
    flatten_score_min = x.min(axis=-1, keepdim=True).values
    flatten_score_max = x.max(axis=-1, keepdim=True).values
    norm_flatten_score = (x - flatten_score_min) / (flatten_score_max - flatten_score_min + 1e-5)
    return norm_flatten_score


def diff_topk(x, k, num_samples, sigma, noise):

    def topk(x, k):
        topk_results = torch.topk(x, k=k, dim=-1, sorted=False)
        indices = topk_results.indices  # b, nS, k
        indices = torch.sort(indices, dim=-1).values  # b, nS, k

        perturbed_output = torch.nn.functional.one_hot(indices, num_classes=x.shape[-1]).float()
        return perturbed_output

    pert_topk = perturbed(topk,
                          num_samples=num_samples,
                          sigma=sigma,
                          noise=noise,
                          batched=True,
                          device=x.device)

    return pert_topk(x, k)


def sinkhorn_forward(C, mu, nu, epsilon, max_iter):
    bs, n, k_ = C.size()

    v = torch.ones([bs, 1, k_]) / (k_)
    G = torch.exp(-C / epsilon)
    if torch.cuda.is_available():
        v = v.cuda()

    for i in range(max_iter):
        u = mu / (G * v).sum(-1, keepdim=True)
        v = nu / (G * u).sum(-2, keepdim=True)

    Gamma = u * G * v
    return Gamma


def sinkhorn_forward_stablized(C, mu, nu, epsilon, max_iter):
    bs, n, k_ = C.size()
    k = k_ - 1

    f = torch.zeros([bs, n, 1])
    g = torch.zeros([bs, 1, k + 1])
    if torch.cuda.is_available():
        f = f.cuda()
        g = g.cuda()

    epsilon_log_mu = epsilon * torch.log(mu)
    epsilon_log_nu = epsilon * torch.log(nu)

    def min_epsilon_row(Z, epsilon):
        return -epsilon * torch.logsumexp((-Z) / epsilon, -1, keepdim=True)

    def min_epsilon_col(Z, epsilon):
        return -epsilon * torch.logsumexp((-Z) / epsilon, -2, keepdim=True)

    for i in range(max_iter):
        f = min_epsilon_row(C - g, epsilon) + epsilon_log_mu
        g = min_epsilon_col(C - f, epsilon) + epsilon_log_nu

    Gamma = torch.exp((-C + f + g) / epsilon)
    return Gamma


def sinkhorn_backward(grad_output_Gamma, Gamma, mu, nu, epsilon):
    nu_ = nu[:, :, :-1]
    Gamma_ = Gamma[:, :, :-1]

    bs, n, k_ = Gamma.size()

    inv_mu = 1. / (mu.view([1, -1]))  # [1, n]
    Kappa = torch.diag_embed(nu_.squeeze(-2)) \
            - torch.matmul(Gamma_.transpose(-1, -2) * inv_mu.unsqueeze(-2), Gamma_)  # [bs, k, k]

    inv_Kappa = torch.inverse(Kappa)  # [bs, k, k]

    Gamma_mu = inv_mu.unsqueeze(-1) * Gamma_
    L = Gamma_mu.matmul(inv_Kappa)  # [bs, n, k]
    G1 = grad_output_Gamma * Gamma  # [bs, n, k+1]

    g1 = G1.sum(-1)
    G21 = (g1 * inv_mu).unsqueeze(-1) * Gamma  # [bs, n, k+1]
    g1_L = g1.unsqueeze(-2).matmul(L)  # [bs, 1, k]
    G22 = g1_L.matmul(Gamma_mu.transpose(-1, -2)).transpose(-1, -2) * Gamma  # [bs, n, k+1]
    G23 = - F.pad(g1_L, pad=(0, 1), mode='constant', value=0) * Gamma  # [bs, n, k+1]
    G2 = G21 + G22 + G23  # [bs, n, k+1]

    del g1, G21, G22, G23, Gamma_mu

    g2 = G1.sum(-2).unsqueeze(-1)  # [bs, k+1, 1]
    g2 = g2[:, :-1, :]  # [bs, k, 1]
    G31 = - L.matmul(g2) * Gamma  # [bs, n, k+1]
    G32 = F.pad(inv_Kappa.matmul(g2).transpose(-1, -2), pad=(0, 1), mode='constant', value=0) * Gamma  # [bs, n, k+1]
    G3 = G31 + G32  # [bs, n, k+1]

    grad_C = (-G1 + G2 + G3) / epsilon  # [bs, n, k+1]
    return grad_C


class TopKFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C, mu, nu, epsilon, max_iter):

        with torch.no_grad():
            if epsilon > 1e-2:
                Gamma = sinkhorn_forward(C, mu, nu, epsilon, max_iter)
                if bool(torch.any(Gamma != Gamma)):
                    print('Nan appeared in Gamma, re-computing...')
                    Gamma = sinkhorn_forward_stablized(C, mu, nu, epsilon, max_iter)
            else:
                Gamma = sinkhorn_forward_stablized(C, mu, nu, epsilon, max_iter)
            ctx.save_for_backward(mu, nu, Gamma)
            ctx.epsilon = epsilon
        return Gamma

    @staticmethod
    def backward(ctx, grad_output_Gamma):

        epsilon = ctx.epsilon
        mu, nu, Gamma = ctx.saved_tensors
        # mu [1, n, 1]
        # nu [1, 1, k+1]
        # Gamma [bs, n, k+1]
        with torch.no_grad():
            grad_C = sinkhorn_backward(grad_output_Gamma, Gamma, mu, nu, epsilon)
        return grad_C, None, None, None, None


class TopK_custom(torch.nn.Module):
    def __init__(self, k, epsilon=0.1, max_iter=200):
        super(TopK_custom1, self).__init__()
        self.k = k
        self.epsilon = epsilon
        self.anchors = torch.FloatTensor([k - i for i in range(k + 1)]).view([1, 1, k + 1])
        self.max_iter = max_iter

        if torch.cuda.is_available():
            self.anchors = self.anchors.cuda()

    def forward(self, scores):
        bs, n = scores.size()
        scores = scores.view([bs, n, 1])

        # find the -inf value and replace it with the minimum value except -inf
        scores_ = scores.clone().detach()
        max_scores = torch.max(scores_).detach()
        scores_[scores_ == float('-inf')] = float('inf')
        min_scores = torch.min(scores_).detach()
        filled_value = min_scores - (max_scores - min_scores)
        mask = scores == float('-inf')
        scores = scores.masked_fill(mask, filled_value)

        C = (scores - self.anchors) ** 2
        C = C / (C.max().detach())

        mu = torch.ones([1, n, 1], requires_grad=False) / n
        nu = [1. / n for _ in range(self.k)]
        nu.append((n - self.k) / n)
        nu = torch.FloatTensor(nu).view([1, 1, self.k + 1])

        if torch.cuda.is_available():
            mu = mu.cuda()
            nu = nu.cuda()

        Gamma = TopKFunc.apply(C, mu, nu, self.epsilon, self.max_iter)

        A = Gamma[:, :, :self.k] * n

        return A, None
