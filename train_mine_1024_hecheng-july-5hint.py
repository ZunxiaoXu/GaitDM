import os
from os.path import join
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import swanlab
from swanlab.integration.pytorch_lightning import SwanLabLogger
from utils_original.dataset_hecheng import GaitDataset, collate_fn
import torch
from model_mine_1024_hecheng_july_5hint import GaitDiffusion
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.strategies import DDPStrategy
# from remote_pdb import set_trace
# set_trace()

def main():
    works_dir = "./works"
    if not os.path.exists(works_dir):
        os.makedirs(works_dir)

    pl.seed_everything(48)
    dataset_dir = ' '
    train_data = GaitDataset(dataset_dir, set="train", split="LT", num_views=2, start_num=0, end_num=9000000)
    train_loader = DataLoader(train_data, batch_size=5, collate_fn=collate_fn, shuffle=True, num_workers=4, prefetch_factor=2)
    valid_data = GaitDataset(dataset_dir, set="train", split="LT", num_views=2, start_num=0, end_num=9000000)
    valid_loader = DataLoader(valid_data, batch_size=6, collate_fn=collate_fn, shuffle=True, num_workers=4, prefetch_factor=2)

    model = GaitDiffusion()

    logger = SwanLabLogger(
        project=" ",
        experiment_name=' ',
        id=None,
        resume="never",
    )
    ddp_strategy = DDPStrategy(
        find_unused_parameters=True,  # 明确设置
        static_graph=False,  # 动态图支持
        gradient_as_bucket_view=True  # 内存优化
    )
    trainer = pl.Trainer(
        strategy=ddp_strategy,
        precision='16-mixed',
    #    max_steps=15000,
        max_epochs=40,
        check_val_every_n_epoch=9999,  # 关键修改：禁用验证
        accumulate_grad_batches=8,
        gradient_clip_val=1.0,
        default_root_dir=works_dir,
        logger=logger,
        accelerator="auto", 
    #    devices=-1,
        enable_progress_bar=True,
        log_every_n_steps=100,
        callbacks=[
            ModelCheckpoint(
                # monitor='valid_loss',  # 移除监控指标依赖
                save_top_k=-1,
                every_n_epochs=1,
                save_on_train_epoch_end=True  # 确保训练阶段保存
            ),
        ],
    )

    trainer.fit(model, train_loader, valid_loader)


if __name__ == "__main__":
    main()
