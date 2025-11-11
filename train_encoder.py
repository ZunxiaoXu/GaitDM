import os
from os.path import join
import swanlab
from swanlab.integration.pytorch_lightning import SwanLabLogger
# from swanlab.logger import SwanLabLogger
os.environ["WANDB_MODE"] = "offline"
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
# from pytorch_lightning.loggers import SwanLabLogger  
from torch.utils.data import DataLoader

from model_mine_1024_hecheng import GaitVideoEncoder    # _casiab
from utils_original.dataset_hecheng import GaitDataset, collate_fn
# from remote_pdb import set_trace
# set_trace()

def main():
    works_dir = "./works"
    if not os.path.exists(works_dir):
        os.makedirs(works_dir)

    pl.seed_everything(42)

    dataset_dir = "/kaggle/input/casia-b/output"
    dataset_dir = '/home/gait_group/resources/dataset-for-opengait/hecheng'
    train_data = GaitDataset(dataset_dir, set="validation", split="LT", num_views=2, start_num=0, end_num=9000000)
    train_loader = DataLoader(train_data, batch_size=22, collate_fn=collate_fn, shuffle=True, num_workers=4)
    valid_data = GaitDataset(dataset_dir, set="validation", split="LT", num_views=2, start_num=0, end_num=9000000)
    valid_loader = DataLoader(valid_data, batch_size=22, collate_fn=collate_fn, shuffle=True, num_workers=4)  # must be shuffled

    model = GaitVideoEncoder()
    
    logger = SwanLabLogger(
        project="june-gait-encoder",
        name='hecheng-all-validation',
        save_dir=works_dir,
        log_model=False,
        version='hecheng-all-validation',
        # version="...",
    )
    trainer = pl.Trainer(
    #    devices ='0',
        max_epochs=30,
        check_val_every_n_epoch=3,
        gradient_clip_val=1.0,
        default_root_dir=works_dir,
        logger=logger,
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
 #   trainer.fit(model, train_loader)
    trainer.fit(model, train_loader, valid_loader)
    # trainer.fit(model, train_loader, valid_loader, ckpt_path="...")
  #  trainer.save_checkpoint(join(works_dir, "encoder.ckpt"))


if __name__ == "__main__":
   # __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    main()
