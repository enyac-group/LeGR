import os
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

import torchvision
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

def get_dataloader(img_size, dataset, datapath, batch_size, no_val):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

    if img_size == 32:
        train_set = eval(dataset)(datapath, True, transforms.Compose([
                transforms.RandomCrop(img_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]), download=True)
        val_set = eval(dataset)(datapath, True, transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]), download=True)

        num_train = len(train_set)
        indices = list(range(num_train))
        split = int(np.floor(0.1 * num_train))

        np.random.seed(int(time.time()))
        np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        test_set = eval(dataset)(datapath, False, transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]), download=True)

        if no_val:
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=batch_size, shuffle=True,
                num_workers=0, pin_memory=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_set, batch_size=batch_size, shuffle=valid_sampler,
                num_workers=0, pin_memory=True
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=batch_size, sampler=train_sampler,
                num_workers=0, pin_memory=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_set, batch_size=batch_size, sampler=valid_sampler,
                num_workers=0, pin_memory=True
            )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=False
        )
    else:
        if dataset == 'ImageNetData':
            # ToTensor and Normalization is done inside SeqImageNetLoader
            train_loader = SeqImageNetLoader('train', batch_size=batch_size, num_workers=8, cuda=True, remainder=False, 
                    transform=transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip()]))

            test_loader = SeqImageNetLoader('val', batch_size=32, num_workers=8, cuda=True, remainder=True, 
                    transform=transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224)]))

            val_loader = test_loader
        else:
            train_set = eval(dataset)(datapath, True, transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))
            val_set = eval(dataset)(datapath, True, transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]))

            num_train = len(train_set)
            indices = list(range(num_train))
            split = int(np.floor(0.1 * num_train))

            np.random.seed(98)
            np.random.shuffle(indices)

            np.random.seed(int(time.time()))

            train_idx, valid_idx = indices[split:], indices[:split]
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)

            test_set = eval(dataset)(datapath, False, transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]))

            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=batch_size, sampler=train_sampler,
                num_workers=0, pin_memory=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_set, batch_size=batch_size, sampler=valid_sampler,
                num_workers=0, pin_memory=True
            )
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                num_workers=0, pin_memory=False)
    return train_loader, val_loader, test_loader

def test(model, data_loader, device='cuda', get_loss=False, n_img=-1):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    if get_loss:
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        loss = np.zeros(0)

    total_len = len(data_loader)
    if n_img > 0 and total_len > int(np.ceil(float(n_img) / data_loader.batch_size)):
        total_len = int(np.ceil(float(n_img) / data_loader.batch_size))
    for i, (batch, label) in enumerate(data_loader):
        if i >= total_len:
            break
        batch, label = batch.to(device), label.to(device)
        output = model(batch)
        if get_loss:
            loss = np.concatenate((loss, criterion(output, label).data.cpu().numpy()))
        pred = output.data.max(1)[1]
        correct += pred.eq(label).sum()
        total += label.size(0)
    
    if get_loss:
        return float(correct)/total*100, loss
    else:
        return float(correct)/total*100


def train(model, train_loader, val_loader, optimizer=None, epochs=10, steps=None, scheduler=None, run_test=True, name='', device='cuda'):
    model.to(device)
    if optimizer is None:
        optimizer = optim.SGD(model.classifier.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)
    if scheduler is None:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Use number of steps as unit instead of epochs
    if steps:
        epochs = int(steps / len(train_loader)) + 1
        if epochs > 1:
            steps = steps % len(train_loader)

    best_acc = 0
    for i in range(epochs):
        print('Epoch: {}'.format(i))
        if i == epochs - 1:
            loss = train_epoch(model, train_loader, optimizer, steps=steps, device=device)
        else:
            loss = train_epoch(model, train_loader, optimizer, device=device)
        scheduler.step()

        if run_test:
            acc = test(model, val_loader)
            print('Testing Accuracy {:.2f}'.format(acc))
            if i and best_acc < acc:
                best_acc = acc
                torch.save(model, os.path.join('ckpt', '{}_best.t7'.format(name)))

def train_epoch(model, train_loader, optimizer=None, steps=None, device='cuda', distillation=None):
    model.to(device)
    model.train()
    losses = np.zeros(0)
    total_loss = 0
    data_t = 0
    train_t = 0 
    criterion = torch.nn.CrossEntropyLoss()
    s = time.time()
    for i, (batch, label) in enumerate(train_loader):
        batch, label = batch.to(device), label.to(device)
        data_t += time.time() - s
        s = time.time()

        model.zero_grad()
        output = model(batch)
        if distillation:
            t_out = distillation.teacher(batch)
            soft_target = F.softmax(t_out/distillation.T, dim=1)
            logp = F.log_softmax(output/distillation.T, dim=1)
            soft_loss = -torch.mean(torch.sum(soft_target)*logp, dim=1)
            soft_loss = soft_loss + distillation.T * distillation.T

            loss = criterion(output, label) + distillation.alpha * soft_loss
            loss.backward()
        else:
            loss = criterion(output, label)
            loss.backward()
        optimizer.step()

        total_loss += loss
        losses = np.concatenate([losses, np.array([loss.item()])])

        train_t += time.time() - s
        length = steps if steps and steps < len(train_loader) else len(train_loader)

        if (i % 100 == 0) or (i == length-1):
            print('Training | Batch ({}/{}) | Loss {:.4f} ({:.4f}) | (PerBatchProfile) Data: {:.3f}s, Net: {:.3f}s'.format(i+1, length, total_loss/(i+1), loss, data_t/(i+1), train_t/(i+1)))
        if i == length-1:
            break
        s = time.time()
    return np.mean(losses)

