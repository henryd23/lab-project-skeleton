import os
import shutil
import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

def make_loaders(batch_size=32, num_workers=0):
    print("[Dati] Configuro le trasformazioni delle immagini...", flush=True)
    transform = T.Compose([
        T.Resize((224, 224)),  # Resize to fit the input dimensions of the network
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset_root = 'tiny-imagenet/tiny-imagenet-200'
    val_images_root = os.path.join(dataset_root, 'val', 'images')
    val_processed_root = os.path.join(dataset_root, 'val_processed')

    print("[Dati] Indicizzo le immagini di training...", flush=True)
    tiny_imagenet_dataset_train = ImageFolder(root=os.path.join(dataset_root, 'train'), transform=transform)

    annotations_path = os.path.join(dataset_root, 'val', 'val_annotations.txt')
    print("[Dati] Organizzo le immagini di validazione per classe...", flush=True)
    with open(annotations_path) as f:
        annotations = f.readlines()

    progress_interval = max(1, len(annotations) // 5)
    for index, line in enumerate(annotations, start=1):
        fn, cls, *_ = line.strip().split('\t')
        class_dir = os.path.join(val_processed_root, cls)
        os.makedirs(class_dir, exist_ok=True)

        shutil.copyfile(os.path.join(val_images_root, fn), os.path.join(class_dir, fn))
        if index % progress_interval == 0 or index == len(annotations):
            print(
                f"[Dati] Validazione organizzata: {index}/{len(annotations)} "
                f"({index / len(annotations):.0%})",
                flush=True,
            )

    print("[Dati] Indicizzo le immagini di validazione...", flush=True)
    tiny_imagenet_dataset_val = ImageFolder(root=val_processed_root, transform=transform)

    if tiny_imagenet_dataset_train.class_to_idx != tiny_imagenet_dataset_val.class_to_idx:
        raise ValueError('Train and validation class mappings do not match.')

    train_loader = torch.utils.data.DataLoader(tiny_imagenet_dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(tiny_imagenet_dataset_val, batch_size=batch_size, shuffle=False)

    print(
        f"[Dati] Pronti: {len(tiny_imagenet_dataset_train)} immagini training, "
        f"{len(tiny_imagenet_dataset_val)} validazione; "
        f"{len(train_loader)} e {len(val_loader)} batch.",
        flush=True,
    )

    return train_loader, val_loader
