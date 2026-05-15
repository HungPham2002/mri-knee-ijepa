import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from logging import getLogger

logger = getLogger()

class DESSDataset3D(Dataset):
    def __init__(self, dataframe, mri_root, mri_transforms=None):
        self.df = dataframe
        self.mri_root = mri_root
        self.mri_transforms = mri_transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        mri_path = self.mri_root + '/' + str(row.get("mri_path", row.name))
        try:
            mri_data = np.load(mri_path)["data"]
        except Exception:
            mri_data = np.zeros((120, 160, 160), dtype=np.float32)

        mri_data = np.expand_dims(mri_data, 0)  # (1, 120, 160, 160)

        if self.mri_transforms:
            mri_data = self.mri_transforms(mri_data)

        mri_tensor = torch.tensor(mri_data, dtype=torch.float32)
        return mri_tensor, 0

def make_dess3d(
    transform,
    batch_size,
    dataframe_path,
    mri_root,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    drop_last=True
):
    df = pd.read_csv(dataframe_path)
    dataset = DESSDataset3D(dataframe=df, mri_root=mri_root, mri_transforms=transform)
    
    logger.info('DESS 3D dataset created')
    dist_sampler = DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank
    )
    data_loader = DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False
    )
    logger.info('DESS 3D data loader created')

    return dataset, data_loader, dist_sampler
