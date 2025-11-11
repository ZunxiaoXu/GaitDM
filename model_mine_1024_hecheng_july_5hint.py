import random
import pdb
import socket
import sys
######
# 将模块路径添加到 sys.path
sys.path.append('/home/xuzunxiao/OpenGait-master-4dataloader/opengait/')
from modeling import models
from utils import config_loader
from utils import config_loader, get_ddp_module, init_seeds, params_count, get_msg_mgr
######
from os.path import join
#from new_fn import MyModel
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from diffusers_no_3dconv_no_attn_time import AutoencoderKL
#from diffusers import (
from diffusers_from_base_5hint import (
    DDIMScheduler,
    DPMSolverSinglestepScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
    UNetSpatioTemporalConditionModel,
    AutoencoderKLTemporalDecoder, )
from torch import optim, nn
from tqdm.auto import tqdm
from transformers import VivitModel, VivitConfig
from PIL import Image
######
import utils_original.loss_july as L
import time
from utils_original.gaitset.gaitset import SetNet
######
from os.path import dirname, abspath, join
from collections import deque
from torch.optim.lr_scheduler import StepLR

def frames_process(frames, num_frames, dtype, device, number):   # shape: [f c h w] range: [-1,1]
    if not isinstance(frames, torch.Tensor):
        frames = torch.tensor(frames, dtype=dtype, device=device)  # [f c h w]
    else:
        frames.to(dtype=dtype, device=device)
    f = frames.shape[0]
    if f < num_frames:
        num_pad = num_frames - f
        frames_pad = frames[-1:].repeat_interleave(num_pad, dim=0)  # repeat the last frame
        frames = torch.cat([frames, frames_pad])  # [f c h w]
    else:
        if number == 0:
            start = 0
        elif number == 1:
            start = int((f - num_frames)/2)
        elif number == 2:
            start = f - num_frames
        elif number == -1:
            start = random.randint(0, f - num_frames)
        frames = frames[start: (start + num_frames)]  # pick up 32 frames
    return frames


class GaitAutoencoderKL(pl.LightningModule):
    def __init__(
        self,
        num_frames: int = 32,
        frame_size: int = 64,
        frame_channels: int = 1,
        latent_channels: int = 4,
        scale_factor: int = 8,
        kl_loss_weight: float = 1e-6,
        id_loss_weight: float = 0.01,
        learning_rate: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.frame_channels = frame_channels
        self.latent_channels = latent_channels
        self.scale_factor = scale_factor

        self.kl_loss_weight = kl_loss_weight
        self.id_loss_weight = id_loss_weight
        self.id_loss = L.GaitLoss()

        self.learning_rate = learning_rate

        self.model = AutoencoderKLTemporalDecoder(
            down_block_types=(
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
            ),
            block_out_channels=(128, 256, 512, 512),
            layers_per_block=2,
            sample_size=self.frame_size,
            in_channels=self.frame_channels,
            out_channels=self.frame_channels,
            latent_channels=self.latent_channels,
        )

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.learning_rate)

    def encode(self, x):   # [bs f c h w]
        bs, f, c, h, w = x.shape
        x = x.view(bs * f, c, h, w)
        x = self.model.encode(x).latent_dist.sample()
        x = x.view(bs, f, self.latent_channels, h // self.scale_factor, w // self.scale_factor)
        return x

    def decode(self, z):   # [bs f c h w]
        bs, f, c, h, w = z.shape
        z = z.view(bs * f, c, h, w)
        y = self.model.decode(z, num_frames=f).sample
        y = y.clamp(-1, 1)
        y = y.view(bs, f, self.frame_channels, h * self.scale_factor, w * self.scale_factor)
        return y

    def forward(self, x):   # [bs f c h w]
        bs, f, c, h, w = x.shape
        x = x.view(bs * f, c, h, w)
        latent = self.model.encode(x).latent_dist
        kl = latent.kl()
        z = latent.sample()
        z = z.view(bs, f, self.latent_channels, h // self.scale_factor, w // self.scale_factor)
        y = self.decode(z)
        return z, y, kl

    def batch_process(self, batch):
        x = []
        for i in batch:
            frames = frames_process(
                i["frames"][0],
                num_frames=self.num_frames,
                dtype=self.dtype,
                device=self.device,
            )
            frames = frames.unsqueeze(0)
            x.append(frames)
        x = torch.cat(x)  # [bs f c h w]
        return x

    def _training_step(self, batch):
        x = self.batch_process(batch)
        _, y, kl = self(x)

        rec_loss = F.mse_loss(y, x)
        kl_loss = kl.mean()
        id_loss = self.id_loss(y, x, self.loss_model).to(rec_loss.device)
        loss = rec_loss+ self.id_loss_weight * id_loss + self.kl_loss_weight * kl_loss 

        return loss, rec_loss, kl_loss, id_loss

    def training_step(self, batch):
        loss, rec_loss, kl_loss, id_loss = self._training_step(batch)
        self.log("loss", loss, prog_bar=True)
        self.log("rec_loss", rec_loss, prog_bar=True)
        self.log("kl_loss", kl_loss, prog_bar=True)
        self.log("id_loss", id_loss, prog_bar=True)
        return loss

    def validation_step(self, batch):
        loss, rec_loss, _, id_loss = self._training_step(batch)
        self.log("valid_loss", loss, batch_size=len(batch), prog_bar=True)
        self.log("valid_rec_loss", rec_loss, batch_size=len(batch), prog_bar=True)
        self.log("valid_id_loss", id_loss, batch_size=len(batch), prog_bar=True)
        return loss


class GaitVideoEncoder(pl.LightningModule):
    def __init__(
        self,
        num_frames: int = 32,
        frame_size: int = 64,
        frame_channels: int = 1,
        output_dim: int = 1024,
        learning_rate: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_frames = num_frames
        self.learning_rate = learning_rate

        config = VivitConfig(
            image_size=frame_size,
            num_channels=frame_channels,
            hidden_size=output_dim,
            num_attention_heads=(output_dim // 64),
        )
        self.model = VivitModel(
            config=config,
            add_pooling_layer=True,
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def configure_optimizers(self):
######
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = StepLR(optimizer, step_size=25, gamma=0.1)
######
        return [optimizer], [scheduler]

    def forward(self, x):    # shape: [bs f c h w], range: [0,1]
        return self.model(x).pooler_output

    def batch_process(self, batch):
        x = []
        y = []
        for i in batch:
            frames = frames_process(
                i["frames"][0],
                num_frames= self.num_frames, #min(len(i['frames'][0]), self.num_frames),#, #
                dtype=self.dtype,
                device=self.device,
                number = -1,
            )
            frames = (frames + 1) / 2       # [-1,1] -> [0,1]
            frames = frames.unsqueeze(0)
            x.append(frames)
            frames = frames_process(
                i["frames"][1],
                num_frames= self.num_frames, #min(len(i['frames'][0]), self.num_frames),#
                dtype=self.dtype,
                device=self.device,
                number = -1,
            )
            frames = (frames + 1) / 2
            frames = frames.unsqueeze(0)
            y.append(frames)
        x = torch.cat(x)  # [bs f c h w]
        y = torch.cat(y)
        return x, y

    def _training_step(self, batch):
        x, y = self.batch_process(batch)
        model_in = torch.cat([x, y], dim=0)
        model_out = self(model_in)
        x_emb, y_emb = torch.chunk(model_out, chunks=2, dim=0)

        x_emb = x_emb / x_emb.norm(dim=-1, keepdim=True)
        y_emb = y_emb / y_emb.norm(dim=-1, keepdim=True)
        self.logit_scale.data.clamp_(-np.log(100), np.log(100))
        logits = x_emb @ y_emb.t() * self.logit_scale.exp()
        labels = torch.arange(logits.shape[0], dtype=torch.long, device=self.device)
        loss = (
            F.cross_entropy(logits, labels) +
            F.cross_entropy(logits.t(), labels)
        ) / 2
        acc = (
            ((torch.argmax(logits, 1) == labels).sum() / logits.shape[0]) +
            ((torch.argmax(logits, 0) == labels).sum() / logits.shape[0])
        ) / 2
        return loss, acc

    def training_step(self, batch):
        loss, acc = self._training_step(batch)
        self.log("loss", loss, prog_bar=True)
        self.log("acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch):
        loss, acc = self._training_step(batch)
        self.log("valid_loss", loss, batch_size=len(batch), prog_bar=True)
        self.log("valid_acc", acc, batch_size=len(batch), prog_bar=True)
        return loss


class GaitDiffusion(pl.LightningModule):
    def __init__(
        self,
        num_frames: int = 30,
        frame_size: int = 64,
        frame_channels: int = 1,
        latent_channels: int = 4,
        encoder_output_dim: int = 1024,
        addition_time_embed_dim: int = 256,
        addition_time_ids_dim: int = 2,
        num_train_timesteps: int = 1000,
        vae_scaling_factor: float = 0.439755, #0.18215,
        prediction_type: str = "v_prediction",
        conditioning_dropout_prob: float = 0.1,
        learning_rate: float = 2e-5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.frame_channels = frame_channels
        self.latent_channels = latent_channels
        self.encoder_output_dim = encoder_output_dim
        self.addition_time_embed_dim = addition_time_embed_dim
        self.addition_time_ids_dim = addition_time_ids_dim

        self.num_train_timesteps = num_train_timesteps
        self.vae_scaling_factor = vae_scaling_factor
        self.prediction_type = prediction_type
        self.conditioning_dropout_prob = conditioning_dropout_prob
        self.id_loss = L.GaitLoss()


        self.learning_rate = learning_rate
        self.is_load_from_checkpoint = False
        train_flag = False
        # noise scheduler
        self.scheduler = DDPMScheduler(
        #    beta_start=0.00085,
        #    beta_end=0.012,
            beta_schedule="scaled_linear",
            prediction_type=self.prediction_type,
            num_train_timesteps=self.num_train_timesteps,
        )

        # vae
        down_block_types=("DownEncoderBlock2D","DownEncoderBlock2D",)
        up_block_types=("UpDecoderBlock2D","UpDecoderBlock2D",)
        block_out_channels = (64,64,)
        ae_model = AutoencoderKL(down_block_types=down_block_types,up_block_types=up_block_types,block_out_channels = block_out_channels,)
        ae_model.load_state_dict(torch.load('/home/xuzunxiao/butterfly-diffusion-master/clean/import_autoencoder_from_diffusers_to_train/2_layers_900_epoch_ckpt_ae.pth', map_location=torch.device('cpu')))
        self.vae = ae_model.eval()#.cuda()
        # self.vae = GaitAutoencoderKL()
        self.vae.requires_grad_(False)
# ######
#         self.encoder = GaitVideoEncoder()
#         self.encoder.requires_grad_(False)
#         self.encoder.eval()
# ######
        self.flag = 0
        self.flag_train = 0
        self.flag_test = 0
        self.count_0 =0
        self.count_total =0
        self.pred_dict_data = deque(maxlen=20) 
        self.pred_dict_label = deque(maxlen=20)

        self.unet = UNet2DConditionModel(
            num_class_embeds = 24, # 111084, #  # 14,
            sample_size=64,  # the target image resolution
            in_channels=4,  # the number of input channels, 3 for RGB images
            out_channels=4,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(64, 128, 256), #(64, 128, 128, 256),  # More channels -> more parameters
            cross_attention_dim= 1024,
######
            encoder_hid_dim = 1024,
            addition_embed_type='image_hint',
######
            down_block_types=(
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
            ),
            up_block_types=(
                # "UpBlock2D",  # a regular ResNet upsampling block
                # "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            ),
        )

    def on_load_checkpoint(self, checkpoint):
        self.is_load_from_checkpoint = True

    def setup(self, stage):
        import json
        with open('/home/xuzunxiao/butterfly-diffusion-master/liwenjie-hecheng/my_dict.json', 'r') as file:
            self.my_dict = json.load(file)
#####
######
            # self.encoder.requires_grad_(False)
            # self.encoder.eval()

# #####
#         return [optimizer], [scheduler]
    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.learning_rate)

    def _get_encoder_hidden_states(self, frames):
        cond = (frames + 1) / 2  # [-1,1] -> [0,1]
        cond = torch.cat([cond, cond[:, -1:, :, :, :].repeat(1, 2, 1, 1, 1)], dim=1)
        encoder_hidden_states = self.encoder(cond)  # [bs output_dim]
######
        encoder_hidden_states = encoder_hidden_states.reshape(cond.size(0),-1)   #  b,4096
######
        encoder_hidden_states = encoder_hidden_states.unsqueeze(1)  # [bs seq_len cross_attention_dim]
        return encoder_hidden_states

    def _get_added_time_ids(self, x_view, y_view):
        if len(x_view) > 0 and not isinstance(x_view[0], int):
            x_view = [int(i) for i in x_view]
        if len(y_view) > 0 and not isinstance(y_view[0], int):
            y_view = [int(i) for i in y_view]
        x_view = torch.tensor(x_view, dtype=self.dtype, device=self.device)
        y_view = torch.tensor(y_view, dtype=self.dtype, device=self.device)
        added_time_ids = torch.stack([x_view, y_view], dim=1)
        return added_time_ids

    def forward(
        self,
        x: torch.Tensor,    # shape: [bs f c h w], range: [-1,1]
        y: torch.Tensor,
        x_view: list,
        y_view: list,
        guidance_scale: float = 0, #3.0,
        indexs: int = 0,
        num_inference_steps: int = 20,
    ):
        bs = x.shape[0]
        with torch.no_grad():

            latents = self.vae.encoder(y.reshape(-1, 1, 64,64))
            latents = latents * self.vae_scaling_factor     # [bs f c h w]

            latents1 = latents.reshape(bs, -1, 4, 32, 32)
            index_hint_y_short = [3,9,15,21,27]
            latents_hint_short = latents1[:, index_hint_y_short]
            index_hint_y_long = [3, 3, 3, 3, 3, 3, 9, 9, 9, 9, 9, 9, 15, 15, 15, 15, 15, 15, 21, 21, 21, 21, 21, 21, 27, 27, 27, 27, 27, 27]
            latents_hint_long = latents1[:, index_hint_y_long].reshape(-1, 4, 32, 32)

        # noise

        latent_size = self.frame_size // 2 #self.vae.scale_factor
        size = (bs, self.num_frames, self.latent_channels, latent_size, latent_size)
        latents = torch.randn(size, dtype=self.dtype, device=self.device)     # [bs f c h w]
        latents = latents * self.scheduler.init_noise_sigma

        # embeddings
        embeddings = self._get_encoder_hidden_states(x)

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)

        # diffusion loop
        pbar = tqdm(self.scheduler.timesteps)

        latents_hint_long = latents_hint_long.to(self.device)
        latents = latents.to(self.device)
        self.unet = self.unet.to(self.device)
        embeddings = embeddings.to(self.device)
        self.vae = self.vae.to(self.device)
        latents_hint_short = latents_hint_short.to(self.device)

######
        # added_time_ids
        mapping_dict = { 0: 0, 15: 1,  30: 2, 45: 3, 60: 4, 75: 5, 90: 6, 180: 7, 195: 8, 210: 9, 225: 10, 240: 11, 255: 12,
                         270: 13, 18: 14, 36: 15, 54: 16, 72: 17, 108: 18, 126: 19, 144: 20, 162: 21, 1000: 22, 2000: 23}
        added_time_ids = [mapping_dict[key] for key in y_view]
        added_time_ids = torch.tensor(added_time_ids, dtype=torch.float32)
        added_time_ids = added_time_ids.to(torch.int)
        added_time_ids = torch.repeat_interleave(added_time_ids, 30, dim=0).to(self.device)

        embeddings = torch.repeat_interleave(embeddings, 30, dim=0)
        image_embeds = embeddings.squeeze(1)
######
        latent_out = []
        with torch.no_grad():
            for t in pbar:
                # expand the latents if we are doing classifier free guidance
                latents = latents.reshape(bs, -1, 4, 32, 32)
                latents[:, index_hint_y_short] = latents_hint_short.to(dtype=torch.float32)
                latents = latents.reshape(-1, 4, 32, 32)
                        
                latents_model_input = latents
                latents_model_input = self.scheduler.scale_model_input(latents_model_input, t)

                model_pred = self.unet(
                    latents_model_input,
                    timestep=t,
                    encoder_hidden_states=embeddings,
                    class_labels=added_time_ids,
                    added_cond_kwargs={"image_embeds":image_embeds,"hint":latents_hint_long},
                ).sample

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(model_pred, t, latents).prev_sample
                # update progress bar
                pbar.set_postfix(timestep=t.detach().item())

        # latent space -> pixel values
        latents = latents / self.vae_scaling_factor
        y_recon = self.vae.decoder(latents)
        y_recon[index_hint_y_short,:] = y[0,index_hint_y_short,:]

        return y_recon

    def test_batch_process(self, batch):
        x = []
        y = []
        x_view = []
        y_view = []
        y_indexs = []
        label_name = []
        for i in batch:
######
            label_name.append(i['subject'].split('_')[0]+i['subject'].split('_')[1])
            frames = frames_process(
                i["frames"][0],
                num_frames=  min(len(i['frames'][0]), 10000), #self.num_frames, #
                dtype=self.dtype,
                device=self.device,
                number = -1,
            )
            frames = frames.unsqueeze(0)
            x.append(frames)


            frames = frames_process(
                i["frames"][1],
                num_frames=min(len(i['frames'][1]), 10000), #self.num_frames, # 
                dtype=self.dtype,
                device=self.device,
                number = -1,
            )
            frames = frames.unsqueeze(0)
            y.append(frames)

######
            y_indexs.append(i["indexs"])
            if (len(i["view"][0]) > 5) and (len(i["view"][0])<10):
                x_view.append(1000)
                y_view.append(1000)
            elif '_' in i["view"][0]:
                x_view.append(1000)
                y_view.append(1000) 
            elif len(i["view"][0]) > 10:
                x_view.append(2000)
                y_view.append(2000)
            else:
                x_view.append(int(i["view"][0]))
                y_view.append(int(i["view"][1]))
            # x_view.append(int(i["view"][0]))
            # y_view.append(int(i["view"][1]))
######
            # x_view.append(int(i["view"][0]))
            # y_view.append(int(i["view"][1]))
        try:
            x = torch.cat(x)  # [bs f c h w]
        except:
            ca = 1
        try:
            y = torch.cat(y)
        except:
            ac = 1
        return x, y, x_view, y_view, y_indexs, label_name


    def train_batch_process(self, batch):
        x = []
        y = []
        x_view = []
        y_view = []
        y_indexs = []
        label_name = []
        for i in batch:
######
            label_name.append(i['subject'].split('_')[0]+i['subject'].split('_')[1])
            frames = frames_process(
                i["frames"][0],
                num_frames= self.num_frames, # min(len(i['frames'][0]), 10000), #
                dtype=self.dtype,
                device=self.device,
                number = -1,
            )
            frames = frames.unsqueeze(0)
            x.append(frames)


            frames = frames_process(
                i["frames"][1],
                num_frames=self.num_frames, # min(len(i['frames'][1]), 10000), #
                dtype=self.dtype,
                device=self.device,
                number = -1,
            )
            frames = frames.unsqueeze(0)
            y.append(frames)

######
            y_indexs.append(i["indexs"])
            if (len(i["view"][0]) > 5) and (len(i["view"][0])<10):
                x_view.append(1000)
                y_view.append(1000)
            elif '_' in i["view"][0]:
                x_view.append(1000)
                y_view.append(1000) 
            elif len(i["view"][0]) > 10:
                x_view.append(2000)
                y_view.append(2000)
            else:
                x_view.append(int(i["view"][0]))
                y_view.append(int(i["view"][1]))
            # x_view.append(int(i["view"][0]))
            # y_view.append(int(i["view"][1]))
######
            # x_view.append(int(i["view"][0]))
            # y_view.append(int(i["view"][1]))
        try:
            x = torch.cat(x)  # [bs f c h w]
        except:
            ca = 1
        try:
            y = torch.cat(y)
        except:
            ac = 1
        return x, y, x_view, y_view, y_indexs, label_name


    def gait_loss_forward(self, logits, labels):
        """
            logits: [n, c, p]
            labels: [n]
        """
        eps = 0.1
        scale = 16
        n, c, p = logits.size()
        logits = logits.float()
        labels = labels.unsqueeze(1)
        loss = F.cross_entropy(
                logits*scale, labels.repeat(1, p), label_smoothing=eps)
        return loss

    def _training_step(self, batch, flag_train, train_flag):
        current_epoch = self.current_epoch

        x_all, y_all, x_view, y_view,indexs,label_name = self.train_batch_process(batch)
        bs = len(x_view)

        x = x_all#.reshape(bs,4,30,1,64,64)[:,0,:]
        y = y_all#.reshape(bs,4,30,1,64,64)[:,0,:]

        # pixel values -> latent space
        with torch.no_grad():
            latents = self.vae.encoder(y.reshape(-1, 1, 64,64))
            latents = latents * self.vae_scaling_factor     # [bs f c h w]

            latents1 = latents.reshape(bs, -1, 4, 32, 32)
            index_hint_y_short = [3,9,15,21,27]
            latents_hint_short = latents1[:, index_hint_y_short]
            index_hint_y_long = [3, 3, 3, 3, 3, 3, 9, 9, 9, 9, 9, 9, 15, 15, 15, 15, 15, 15, 21, 21, 21, 21, 21, 21, 27, 27, 27, 27, 27, 27]
            latents_hint_long = latents1[:, index_hint_y_long].reshape(-1, 4, 32, 32)

        # noise
        noise = torch.randn_like(latents, dtype=self.dtype, device=self.device)     # [bs f c h w]

        # timesteps
        timesteps = torch.randint(0, self.num_train_timesteps, (bs,), device=self.device)    # [[bs]]

        # add noise
        timesteps = torch.repeat_interleave(timesteps, 30, dim=0)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        # embeddings
        embeddings = self._get_encoder_hidden_states(x)
 
        # added_time_ids
        mapping_dict = { 0: 0, 15: 1,  30: 2, 45: 3, 60: 4, 75: 5, 90: 6, 180: 7, 195: 8, 210: 9, 225: 10, 240: 11, 255: 12,
                         270: 13, 18: 14, 36: 15, 54: 16, 72: 17, 108: 18, 126: 19, 144: 20, 162: 21, 1000: 22, 2000: 23}
        added_time_ids = [mapping_dict[key] for key in y_view]
        added_time_ids = torch.tensor(added_time_ids, dtype=torch.float32)
        added_time_ids = added_time_ids.to(torch.int)
        added_time_ids = torch.repeat_interleave(added_time_ids, 30, dim=0).to(self.device)

        # predict
        # noisy_latents = noisy_latents.reshape(bs, -1, 4, 32, 32)
        # noisy_latents[:, index_hint_y_short] = latents_hint_short.to(dtype=torch.float32)
        # noisy_latents = noisy_latents.reshape(-1, 4, 32, 32)
        embeddings = torch.repeat_interleave(embeddings, 30, dim=0)
        image_embeds = embeddings.squeeze(1)

        model_pred = self.unet(
            noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=embeddings,
            class_labels=added_time_ids,
            added_cond_kwargs={"image_embeds":image_embeds,"hint":latents_hint_long},
        ).sample


        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.prediction_type}")
        
        target = target.reshape(-1, 4, 32,32)
        noise_loss = F.mse_loss(model_pred, target, reduction="mean")

        latents = self.scheduler.step(model_pred, timesteps, noisy_latents).pred_original_sample
        latents = latents / self.vae_scaling_factor
        y_pred = self.vae.decoder(latents)
        y_pred = y_pred.reshape(-1, 30, 1, 64,64)

        id_loss =  0 #.1*self.id_loss(y, y_pred, self.encoder)

        loss = noise_loss + id_loss

        return loss, noise_loss, id_loss


    def training_step(self, batch):
######
        self.trainer.optimizers[0].param_groups[0]['lr']=1e-4
######
    #    start = time.time()
        train_flag = True
        self.flag_train = self.flag_train +1
        loss, noise_loss, id_loss = self._training_step(batch, flag_train=self.flag_train, train_flag=train_flag)
######
        optimizer1 = self.trainer.optimizers[0]
        # 获取学习率
        lr = optimizer1.param_groups[0]['lr']
######
        self.log("loss", loss, prog_bar=True,sync_dist=True)
        self.log("noise_loss", noise_loss, prog_bar=True,sync_dist=True)
        self.log("id_loss", id_loss, prog_bar=True,sync_dist=True)
        self.log("lr", lr, prog_bar=True,sync_dist=True)

        return loss

    def validation_step(self, batch):
        train_flag = False
    #    start = time.time()
        self.flag_test = self.flag_test + 1

        loss, noise_loss, id_loss = self._training_step(batch, flag_train=self.flag_train, train_flag=train_flag)
        self.log("valid_loss", loss, batch_size=len(batch), prog_bar=True,sync_dist=True)
        self.log("valid_noise_loss", noise_loss, batch_size=len(batch), prog_bar=True,sync_dist=True)
        self.log("valid_id_loss", id_loss, batch_size=len(batch), prog_bar=True,sync_dist=True)


