import random
import jittor as jt
import jittor.dataset
from jt_extension.subset import Subset
from jittor import transform
from jittor.dataset import CIFAR100
from PIL import Image

def build_continual_dataloader(args):
    dataloader_list = list()
    # class_mask = list() if args.task_inc or args.train_mask else None

    transform_train = build_transform(True, args)
    transform_val = build_transform(False, args)

    # original_num_classes = 0 # Used to track class offsets for non-split datasets

    if args.dataset.startswith('Split-'):
        dataset_name = args.dataset.replace('Split-','')
        dataset_train, dataset_val = get_dataset(dataset_name, transform_train, transform_val, args)

        args.nb_classes = len(dataset_val.classes)

        splited_dataset, class_mask = split_single_dataset(dataset_train, dataset_val, args)
    else:
        raise NotImplementedError

    for i in range(args.num_tasks):
        if args.dataset.startswith('Split-'):
            dataset_train, dataset_val = splited_dataset[i]
        else:
            raise NotImplementedError

        sampler_train = jittor.dataset.RandomSampler(dataset_train)
        sampler_val = jittor.dataset.SequentialSampler(dataset_val)

        data_loader_train = jittor.dataset.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        data_loader_val = jittor.dataset.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        dataloader_list.append({'train': data_loader_train, 'val': data_loader_val})

    return dataloader_list, class_mask


def get_dataset(dataset_name, transform_train, transform_val, args):
    if dataset_name == 'CIFAR100':
        dataset_train = CIFAR100(root=args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = CIFAR100(root=args.data_path, train=False, download=True, transform=transform_val)
    else:
        raise ValueError(f'Dataset {dataset_name} not supported.')

    return dataset_train, dataset_val


def split_single_dataset(dataset_train, dataset_val, args):
    num_classes = len(dataset_val.classes)
    assert num_classes % args.num_tasks == 0
    classes_per_task = num_classes // args.num_tasks

    labels = [i for i in range(num_classes)]

    split_datasets = list()
    mask = list()

    if args.shuffle:
        random.shuffle(labels)

    for _ in range(args.num_tasks):
        train_split_indices = []
        test_split_indices = []

        scope = labels[:classes_per_task]
        labels = labels[classes_per_task:]

        mask.append(scope)

        for k in range(len(dataset_train.targets)):
            if int(dataset_train.targets[k]) in scope:
                train_split_indices.append(k)

        for h in range(len(dataset_val.targets)):
            if int(dataset_val.targets[h]) in scope:
                test_split_indices.append(h)

        subset_train, subset_val = Subset(dataset_train, train_split_indices), Subset(dataset_val, test_split_indices)

        split_datasets.append([subset_train, subset_val])

    return split_datasets, mask



def build_transform(is_train, args):
    resize_im = args.input_size > 32
    input_size = args.input_size

    if is_train:
        transform_list = [
            jt.transform.RandomResizedCrop(size=input_size, scale=(0.05, 1.0), ratio=(3. / 4., 4. / 3.)),
            jt.transform.RandomHorizontalFlip(p=0.5),
            jt.transform.ToTensor(),
        ]
        composed_transform = transform.Compose(transform_list)
        return composed_transform

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(transform.Resize(size,mode=Image.BICUBIC))
        t.append(transform.CenterCrop(input_size))

    t.append(transform.ToTensor())

    return transform.Compose(t)
