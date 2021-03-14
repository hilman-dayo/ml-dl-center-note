# https://blog.scaleway.com/2020/active-learning-pytorch/
from torch import nn, optim
from torchvision import models
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from data import IndexedDataset
from query import query_the_oracle
from pathlib import Path
import time


# Train and test functions ####################################################
def train(model, device, train_loader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    m_train = 0

    for batch in train_loader:
        data, target, _ = batch
        m_train += data.size(0)
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / m_train


def test(model, device, test_loader, criterion, display=False):
    model.eval()

    test_loss = 0
    n_correct = 0

    one = torch.ones(1, 1).to(device)
    zero = torch.zeros(1, 1).to(device)

    with torch.no_grad():
        for batch in test_loader:
            data, target, _ = batch
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output.squeeze(), target.squeeze()).item()

            prediction = output.argmax(dim=1, keepdim=True)
            # torch.where(output.squeeze() < 0.5, zero, one)
            n_correct += prediction.eq(target.view_as(prediction)).sum().item()

    test_loss /= len(test_loader.dataset)
    if display:
        print("Accuracy on the test set:", (100. * n_correct / len(test_loader.dataset)))
    return test_loss, (100. * n_correct / len(test_loader.dataset))




# Model #######################################################################
class MyModel:
    # Setting data
    train_dir = "/tmp/data/train"
    test_dir = "/tmp/data/test"
    num_queries = 10
    # batch_size = 1024
    batch_size = 8

    def __init__(self, active=True):
        # Model
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.pin_memory = True if torch.cuda.is_available() else False
        self.n_classes = len(list(Path(self.train_dir).iterdir()))
        self.classifier = models.resnet18(pretrained=True)
        self.num_ftrs = self.classifier.fc.in_features
        self.classifier.fc = nn.Linear(self.num_ftrs, self.n_classes)
        self.classifier = self.classifier.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.classifier.parameters(), lr=0.001, momentum=0.9, dampening=0,
            weight_decay=0.001
        )


        test = True if active is False else False
        self.train_set = IndexedDataset(
            self.train_dir, transform=transforms.Compose(
                [transforms.Resize((224, 224)), transforms.ToTensor()]),
                test=test
            )
        self.test_set = IndexedDataset(
            self.test_dir, transform=transforms.Compose(
                [transforms.Resize((224, 224)), transforms.ToTensor()]),
            test=True
        )

        self.test_loader = DataLoader(
            self.test_set, batch_size=self.batch_size, shuffle=False,
            num_workers=10,
            pin_memory=self.pin_memory
        )

epochs_per = 20
if __name__ == "__main__":
    writer = SummaryWriter(log_dir=Path.home() / "al_result")
    for qs in ["normal", "margin", "least_confidence", "random"]:
        if qs == "normal":
            model = MyModel(active=False)
            labeled_loader = None
        else:
            model = MyModel()

        for i, query in enumerate(range(model.num_queries)):
            if qs != "normal":
                model.train_set.filter = False
                query_the_oracle(
                    model.classifier, model.device, model.train_set,
                    query_size=5, query_strategy=qs, interactive=False, pool_size=0
                )
                model.train_set.filter = True

            if qs == "normal":
                if labeled_loader is None:
                    labeled_loader = DataLoader(
                        model.train_set, batch_size=model.batch_size, num_workers=10,
                        pin_memory=model.pin_memory, shuffle=True
                    )
            else:
                labeled_idx = np.where(model.train_set.unlabeled_mask == 0)[0]
                labeled_loader = DataLoader(
                    model.train_set, batch_size=model.batch_size, num_workers=10,
                    sampler=SubsetRandomSampler(labeled_idx),
                    pin_memory=model.pin_memory
                )

            # previous_test_acc = 0
            # current_test_acc = 1
            print(f"=== Query {query + 1} ===")
            train_count = 0
            # while current_test_acc >= previous_test_acc:
            for ii in range(epochs_per):
                start = time.perf_counter()
                train_count += 1
                # previous_test_acc = current_test_acc
                print(f"Train {train_count}")
                n_iter = i * epochs_per + ii

                train_loss = train(
                    model.classifier, model.device, labeled_loader,
                    model.optimizer, model.criterion
                )
                test_loss,  current_test_acc = test(
                    model.classifier, model.device, model.test_loader,
                    model.criterion, display=True
                )

                writer.add_scalar(f"{qs}/loss/train", train_loss, n_iter)
                writer.add_scalar(f"{qs}/loss/test", test_loss, n_iter)
                writer.add_scalar(f"{qs}/accuracy/test", current_test_acc, n_iter)
                writer.add_scalar(
                    f"{qs}/total_time", time.perf_counter() - start, n_iter
                )

            # print(f"Current query final test")
            # test(classifier, device, test_loader, criterion, display=True)


