import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from main import instantiate_from_config

from taming.modules.diffusionmodules.model import Encoder, Decoder
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from taming.modules.vqvae.quantize import GumbelQuantize
from taming.modules.vqvae.quantize import EMAVectorQuantizer

class VQDiffSimpleModel(pl.LightningModule):
    def __init__(self,
                 ddconfig1,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig1)
        self.decoder = Decoder(**ddconfig1)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig1["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig1["z_channels"], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    
    def encode(self, x1, x2):
        # feat equals to cat((feat1, feat2), dim=-1)
        h = self.encoder(x2-x1)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec


    def decode_code(self, code_b):
        raise NotImplementedError

    def forward(self, x1, x2):
        quant, diff, _  = self.encode(x1, x2)
        dec = self.decode(quant)
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        assert x.shape[1] == 2 # 2 frames, while x.shape[0] is batch's dim
        
        assert len(x.shape) == 5    # [batchsize, 2(frame), W, H, 3(RGB)]
        x = x.permute(0, 1, 4, 2, 3).to(memory_format=torch.contiguous_format)
        return x[:, 0].float(), x[:, 1].float()

    def training_step(self, batch, batch_idx):
        x_f1, x_f2 = self.get_input(batch, self.image_key)
        xrec_delta, qloss = self(x_f1, x_f2)

        aeloss, log_dict_ae = self.loss(qloss, x_f2-x_f1, xrec_delta, split="train")

        self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return aeloss

    def validation_step(self, batch, batch_idx):
        x1, x2 = self.get_input(batch, self.image_key)
        xrec_delta, qloss = self(x1, x2)
        aeloss, log_dict_ae = self.loss(qloss, x2, xrec_delta, split="val")
        self.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))

        return [opt_ae, ], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x1, x2 = self.get_input(batch, self.image_key)
        x1 = x1.to(self.device)
        x2 = x2.to(self.device)
        xrec_delta, _ = self(x1, x2)
        # if x.shape[1] > 3:
        #     # colorize with random projection
        #     assert xrec.shape[1] > 3
        #     x = self.to_rgb(x)
        #     xrec = self.to_rgb(xrec)
        log["frame1"] = x1
        log["frame2"] = x2
        log["reconstructions"] = xrec_delta+x1
        log['delta_gt'] = self.to_rgb(x2-x1)
        log['delta_rec'] = self.to_rgb(xrec_delta)
        return log

    def to_rgb(self, x):
        # normalize to [-1, 1]
        return x/2



