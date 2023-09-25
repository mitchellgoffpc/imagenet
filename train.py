import os
import csv
import time
import shutil
import datetime
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from pathlib import Path
from typing import Optional
from omegaconf import OmegaConf
from dataclasses import dataclass
from resnet import ResNet
from dataset import ImageNetDataset

@dataclass
class Config:
    batch_size: int = 32
    learning_rate: float = 3e-4
    resnet_size: int = 18
    checkpoint_path: Optional[str] = None


def get_dataloader(split, bs, rank):
    data_path = Path(__file__).parent / 'data'
    dataset = ImageNetDataset(data_path, split=split, verbose=rank==0)
    sampler = DistributedSampler(dataset)
    return DataLoader(dataset, batch_size=bs, num_workers=16, pin_memory=True, sampler=sampler)

def all_reduce(data, device):
    data = torch.tensor(data, device=device)
    dist.all_reduce(data)
    return data.item()

def train(rank, world_size, config, result_path):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Instantiate the model
    device = torch.device(f'cuda:{rank}')
    model = ResNet(config.resnet_size).to(device)
    model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    if config.checkpoint_path:
        model.load_state_dict(torch.load(config.checkpoint_path))

    # Load the dataset
    train_loader = get_dataloader('train', config.batch_size, rank)
    test_loader = get_dataloader('val', config.batch_size, rank)

    # Create results directory and csv file
    if rank == 0:
        code_path = result_path / 'code'
        code_path.mkdir(parents=True, exist_ok=True)

        for py_file in Path(__file__).parent.glob('*.py'):
            shutil.copy(py_file, code_path)
        with open(result_path / 'config.yaml', 'w') as f:
            f.write(OmegaConf.to_yaml(config))
        with open(result_path / 'results.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'test_loss', 'top1_error', 'top5_error', 'epoch_duration'])

    # Training loop
    for epoch in range(100):
        model.train()
        train_loss = 0
        epoch_start_time = time.time()

        for inputs, labels in (pbar := tqdm(train_loader, leave=False, disable=rank>0)):
            start_time = time.time()
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            step_time = (time.time() - start_time) * 1000
            pbar.set_description(f"Epoch {epoch} | Train Loss: {loss.item():.4f} | Step Time: {step_time:.2f}ms")

        model.eval()
        test_loss = 0
        correct1, correct5, total = 0, 0, 0
        with torch.no_grad():
            for images, labels in (pbar := tqdm(test_loader, leave=False, disable=rank>0)):
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)
                test_loss += loss.item()

                predicted = torch.argmax(outputs, dim=1)
                _, top5_pred = torch.topk(outputs, 5, dim=1)
                total += labels.size(0)
                correct1 += (predicted == labels).sum().item()
                correct5 += (top5_pred.permute(1, 0) == labels).any(dim=0).sum().item()
                pbar.set_description(f"Epoch {epoch} | Test Loss: {loss.item():.4f} | "
                                     f"Top-1 Error: {100 - 100 * correct1 / total:.2f}% | "
                                     f"Top-5 Error: {100 - 100 * correct5 / total:.2f}%")

        # Print report and write results to CSV file
        train_loss, test_loss, correct1, correct5, total = \
            (all_reduce(x, device) for x in (train_loss, test_loss, correct1, correct5, total))

        train_loss = train_loss / len(train_loader)
        test_loss = test_loss / len(test_loader)
        top1_error = 100 - 100 * correct1 / total
        top5_error = 100 - 100 * correct5 / total
        epoch_duration = int(time.time() - epoch_start_time)

        if rank == 0:
            print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | "
                  f"Top-1 Error: {top1_error:.2f}% | Top-5 Error: {top5_error:.2f}% | "
                  f"Duration: {datetime.timedelta(seconds=epoch_duration)}")

            with open(result_path / 'results.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, train_loss, test_loss, top1_error, top5_error, epoch_duration])

        # Save the model checkpoint
        if rank == 0:
            state_dict = {k:v.cpu() for k,v in model.state_dict().items()}
            torch.save(state_dict, result_path / f'checkpoint_{epoch}.ckpt')


if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    os.environ["NCCL_SHM_LOCALITY"] = "1"
    os.environ["NCCL_NSOCKS_PERTHREAD"] = "4"
    os.environ["NCCL_SOCKET_NTHREADS"] = "4"

    schema = OmegaConf.structured(Config)
    config = OmegaConf.merge(schema, OmegaConf.from_cli())
    config = OmegaConf.to_object(config)
    ngpus = torch.cuda.device_count()

    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    result_path = Path(__file__).parent / 'experiments' / current_time

    if ngpus > 1:
        mp.spawn(train, args=(ngpus, config, result_path), nprocs=ngpus, join=True)
    else:
        train(0, 1, config, result_path)
