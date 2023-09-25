import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from classes import IMAGENET2012_CLASSES

SYNSET_LABELS = {k:i for i,k in enumerate(IMAGENET2012_CLASSES)}
SYNSET_NAMES = list(IMAGENET2012_CLASSES.values())

TRANSFORMS_COMMON = [
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
TRANSFORMS_TRAIN = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    *TRANSFORMS_COMMON])
TRANSFORMS_TEST = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    *TRANSFORMS_COMMON])


class ImageNetDataset(Dataset):
    def __init__(self, root_dir, split='train', verbose=True):
        assert split in ('train', 'val', 'test'), f"Invalid split `{split}`"
        assert (Path(root_dir) / split).is_dir(), f"Data for {split} set does not exist yet, run `download.py` to fetch it."
        self.file_paths = []
        self.labels = []
        self.transform = TRANSFORMS_TRAIN if split == 'train' else TRANSFORMS_TEST

        # Load all file paths and labels
        for f in (Path(root_dir) / split).iterdir():
            if f.name.lower().endswith(".jpeg"):
                self.file_paths.append(str(f))
                synset_id = f.name.split('_')[-1].split('.')[0]
                self.labels.append(-1 if split == 'test' else SYNSET_LABELS[synset_id])

        if verbose:
            print(f"Initialized {split} set with {len(self.labels)} samples")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_name = self.file_paths[idx]
        image = Image.open(img_name).convert('RGB')
        image = self.transform(image)
        label = self.labels[idx]
        return image, label


if __name__ == '__main__':
    import torchvision
    from torch.utils.data import DataLoader

    RED = '\u001b[31m'
    GREEN = "\u001b[32m"
    RESET = "\033[m"

    print("Initializing datasets...")
    data_dir = Path(__file__).parent / 'data'
    # train_dataset = ImageNetDataset(data_dir, split='train')
    val_dataset = ImageNetDataset(data_dir, split='val')
    test_dataset = ImageNetDataset(data_dir, split='test')

    print("Benchmarking...")
    from tqdm import trange
    for i in trange(200):
        val_dataset[i]

    print("Evaluating with pre-trained model...")
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT).eval()
    loader = DataLoader(val_dataset, batch_size=10, shuffle=True)

    images, labels = next(iter(loader))
    outputs = model(images)
    predicted = torch.argmax(outputs, dim=1)

    for gt, pred in zip(labels, predicted):
        print(f'True / Pred: {GREEN}{SYNSET_NAMES[gt]}{RESET} / {GREEN if pred == gt else RED}{SYNSET_NAMES[pred]}{RESET}')
