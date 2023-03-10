import torch


def euclidean_distance(proto, query):
    """

    :param proto: (n_ways, n_dim)
    :param query: (n_ways * n_queries, n_dim)
    :return: logits: (n_queries, n_ways)
    """

    n_queries = query.shape[0]
    n_proto = proto.shape[0]
    query = query.unsqueeze(1)
    proto = proto.unsqueeze(0).expand(n_queries, n_proto, -1)

    logits = - torch.sum((proto-query) ** 2, dim=2)
    return logits


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

