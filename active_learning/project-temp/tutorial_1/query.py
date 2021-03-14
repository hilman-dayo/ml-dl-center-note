import torch
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.data import DataLoader, SubsetRandomSampler


def random_query(data_loader, query_size=10):
    sample_idx = []
    for batch in data_loader:
        _, _, idx = batch
        sample_idx.extend(idx.tolist())

        if len(sample_idx) >= query_size:
            break

    return sample_idx[:query_size]


def least_confidence_query(model, device, data_loader, query_size=10):
    """LCQ.

    ソフトマックスをネットワークのアウトプットにかけて, `query_size` の長さで
    一番低い確率を持っているデータを抽出.
    """
    confidences = []
    indices = []

    model.eval()

    with torch.no_grad():
        for batch in data_loader:
            data, _, idx = batch
            logits = model(data.to(device))
            probabilities = F.softmax(logits, dim=1)

            most_probable = torch.max(probabilities, dim=1)[0]
            confidences.extend(most_probable.cpu().tolist())
            indices.extend(idx.tolist())

    conf = np.asarray(confidences)
    ind = np.asarray(indices)
    sorted_pool = np.argsort(conf)

    return ind[sorted_pool][:query_size]


def margin_query(model, device, data_loader, query_size=10):
    """MQ.

    画像ずつに対するトップツーの結果の差をもとめ, `query_size` の数で,
    一番差の低い画像を選択.
    """
    margins = []
    indices = []

    model.eval()

    with torch.no_grad():
        for batch in data_loader:
            data, _, idx = batch
            logits = model(data.to(device))
            probabilites = F.softmax(logits, dim=1)

            toptwo = torch.topk(probabilites, 2, dim=1)[0]

            differences = toptwo[:, 0] - toptwo[:, 1]
            margins.extend(torch.abs(differences).cpu().tolist())
            indices.extend(idx.tolist())

    margin = np.asarray(margins)
    index = np.asarray(indices)
    sorted_pool = np.argsort(margin)

    return index[sorted_pool][:query_size]


def query_the_oracle(model, device, dataset, query_size=10, query_strategy="random",
                     interactive=False, pool_size=0, batch_size=128, num_workers=4):
    unlabled_idx = np.nonzero(dataset.unlabeled_mask)[0]
    if pool_size > 0:
        pool_idx = random.sample(range(1, len(unlabled_idx)), pool_size)
        pool_loader = DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers,
            sampler=SubsetRandomSampler(unlabled_idx[pool_idx])
        )
    else:
        pool_loader = DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers,
            sampler=SubsetRandomSampler(unlabled_idx)
        )

    if query_strategy == "margin":
        sample_idx = margin_query(model, device, pool_loader, query_size)
    elif query_strategy == "least_confidence":
        sample_idx = least_confidence_query(model, device, pool_loader, query_size)
    else:
        sample_idx = random_query(pool_loader, query_size)

    for sample in sample_idx:
        if interactive:
            dataset.display(sample)
            new_label = int(input("What is the class? → "))
            dataset.update_label(sample, new_label)
        else:
            dataset.label_from_filename(sample)
