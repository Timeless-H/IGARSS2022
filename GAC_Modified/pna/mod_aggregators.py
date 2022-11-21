import math
import torch

EPS = 1e-5


# each aggregator is a function taking as input X (B x N x N x Din), adj (B x N x N), self_loop and device and
# returning the aggregated value of X (B x N x Din) for each dimension

def aggregate_identity(X, adj, self_loop=False, device='cpu'):
    # Y corresponds to the elements of the main diagonal of X
    (_, N, N, _) = X.shape
    Y = torch.sum(torch.mul(X, torch.eye(N).reshape(1, N, N, 1)), dim=2)
    return Y


def aggregate_mean(X, device='cpu'):  # todo: fixed
    # nsample^{-1} * X    i.e. the mean of the neighbours
    B, npoint, nsample, D = X.shape
    # return: [B, npoint, D]

    X_sum = torch.sum(X, dim=2)
    X_mean = torch.div(X_sum, nsample)
    return X_mean


def aggregate_max(X, device='cpu'):
    # return: [B, npoint, D]
    B, npoint, nsample, feat = X.shape

    max = torch.max(X, 2)[0]
    return max


def aggregate_min(X, device='cpu'):
    # return: [B, npoint, D]
    B, npoint, nsample, feat = X.shape

    min = torch.min(X, 2)[0]
    return min


def aggregate_std(X, device='cpu'):
    # sqrt(relu(D^{-1} A X^2 - (D^{-1} A X)^2) + EPS)
    # i.e.  the standard deviation of the features of the neighbours
    # the EPS is added for the stability of the derivative of the square root
    std = torch.sqrt(aggregate_var(X) + EPS)  # sqrt(mean_squares_X - mean_X^2)
    return std


def aggregate_var(X, device='cpu'):
    B, npoint, nsample, feat = X.shape
    # relu(nsamples^{-1} * X^2 - (nsample^{-1} * X)^2)
    # i.e.  the variance of the features of the neighbours

    X_sum_squares = torch.sum(torch.mul(X, X), dim=2)
    X_mean_squares = torch.div(X_sum_squares, nsample)  # nsample^{-1} * X^2
    X_mean = aggregate_mean(X)  # nsample^{-1} * X
    var = torch.relu(X_mean_squares - torch.mul(X_mean, X_mean)) # relu(mean_squares_X - mean_X^2)
    return var


def aggregate_sum(X, device='cpu'):
    # return: [B, npoint, D]

    X_sum = torch.sum(X, dim=2)
    return X_sum


def aggregate_normalised_mean(X, adj, self_loop=False, device='cpu'):
    # D^{-1/2] A D^{-1/2] X
    (B, N, N, _) = X.shape

    if self_loop:  # add self connections
        adj = adj + torch.eye(N, device=device).unsqueeze(0)

    rD = torch.mul(torch.pow(torch.sum(adj, -1, keepdim=True), -0.5), torch.eye(N, device=device)
                   .unsqueeze(0).repeat(B, 1, 1))  # D^{-1/2]
    adj = torch.matmul(torch.matmul(rD, adj), rD)  # D^{-1/2] A' D^{-1/2]

    X_sum = torch.sum(torch.mul(X, adj.unsqueeze(-1)), dim=2)
    return X_sum


def aggregate_softmax(X, adj, self_loop=False, device='cpu'):
    # for each node sum_i(x_i*exp(x_i)/sum_j(exp(x_j)) where x_i and x_j vary over the neighbourhood of the node
    (B, N, N, Din) = X.shape

    if self_loop:  # add self connections
        adj = adj + torch.eye(N, device=device).unsqueeze(0)

    X_exp = torch.exp(X)
    adj = adj.unsqueeze(-1)  # adding extra dimension
    X_exp = torch.mul(X_exp, adj)
    X_sum = torch.sum(X_exp, dim=2, keepdim=True)
    softmax = torch.sum(torch.mul(torch.div(X_exp, X_sum), X), dim=2)
    return softmax


def aggregate_softmin(X, adj, self_loop=False, device='cpu'):
    # for each node sum_i(x_i*exp(-x_i)/sum_j(exp(-x_j)) where x_i and x_j vary over the neighbourhood of the node
    return -aggregate_softmax(-X, adj, self_loop=self_loop, device=device)


def aggregate_moment(X, adj, self_loop=False, device='cpu', n=3):
    # for each node (E[(X-E[X])^n])^{1/n}
    # EPS is added to the absolute value of expectation before taking the nth root for stability

    if self_loop:  # add self connections
        (B, N, _) = adj.shape
        adj = adj + torch.eye(N, device=device).unsqueeze(0)

    D = torch.sum(adj, -1, keepdim=True)
    X_mean = aggregate_mean(X, adj, self_loop=self_loop, device=device)
    X_n = torch.div(torch.sum(torch.mul(torch.pow(X - X_mean.unsqueeze(2), n), adj.unsqueeze(-1)), dim=2), D)
    rooted_X_n = torch.sign(X_n) * torch.pow(torch.abs(X_n) + EPS, 1. / n)
    return rooted_X_n


def aggregate_moment_3(X, adj, self_loop=False, device='cpu'):
    return aggregate_moment(X, adj, self_loop=self_loop, device=device, n=3)


def aggregate_moment_4(X, adj, self_loop=False, device='cpu'):
    return aggregate_moment(X, adj, self_loop=self_loop, device=device, n=4)


def aggregate_moment_5(X, adj, self_loop=False, device='cpu'):
    return aggregate_moment(X, adj, self_loop=self_loop, device=device, n=5)


AGGREGATORS = {'mean': aggregate_mean, 'sum': aggregate_sum, 'max': aggregate_max, 'min': aggregate_min,
               'identity': aggregate_identity, 'std': aggregate_std, 'var': aggregate_var,
               'normalised_mean': aggregate_normalised_mean, 'softmax': aggregate_softmax, 'softmin': aggregate_softmin,
               'moment3': aggregate_moment_3, 'moment4': aggregate_moment_4, 'moment5': aggregate_moment_5}
