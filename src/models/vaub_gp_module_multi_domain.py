from typing import Any, Dict, Tuple, List

import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
import torch.nn.functional as F
import matplotlib.pyplot as plt

from ..utils.utils_gp import calculate_gp_loss
from ..utils.utils_visual import display_reconstructed_and_flip_images_multi, display_umap_for_latent_multi

class VAUBGPMultiDomainModule(LightningModule):

    def __init__(
        self,
        vae_params_list: List[torch.nn.Module],
        optimizer_vae_list: List[torch.optim.Optimizer],
        score_prior: torch.nn.Module,
        optimizer_score: torch.optim.Optimizer,
        unet: torch.nn.Module,
        classifier: torch.nn.Module,
        optimizer_cls: torch.optim.Optimizer,
        gp: torch.nn.Module,
        vaub_lambda: float,
        gp_lambda: float,
        recon_lambda: float,
        block_size: int,
        cls_lambda: float,
        is_vanilla: bool,
        loops: int,
        target_domain_idx: int,
        compile: bool,
        latent_dim: int,
        latent_row_dim: int,
        latent_col_dim: int,
        min_noise_scale: float,
        max_noise_scale: float,
        num_latent_noise_scale: int,
        warm_score_epochs: int,
        init_scale: float,
    ) -> None:

        super().__init__()

        # check if the number of VAEs is equal to the number of optimizers
        assert len(vae_params_list) == len(optimizer_vae_list), "Number of VAEs and optimizers should be the same."
        self.num_domain = len(vae_params_list)

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=['vae_params_list', 'unet', 'classifier'])
        self.automatic_optimization = False

        # initialize all modules
        self.vae_list = []
        for vae in vae_params_list:
            self.vae_list.append(vae(latent_height=latent_row_dim, latent_width=latent_col_dim))

        # reinitialize the params of the VAEs
        [vae.init_weights_fixed(init_scale=init_scale) for vae in self.vae_list]

        self.classifier = classifier(input_dim=latent_dim)
        self.score_model = score_prior(model=unet(in_dim=latent_dim,
                                                  out_dim=latent_dim,
                                                  )
                                       )
        self.gp_model = gp(device=self.device)

        # metric objects for calculating and averaging accuracy across batches
        # self.train_acc = Accuracy(task="multiclass", num_classes=10)
        self.val_src_acc = Accuracy(task="multiclass", num_classes=10)
        self.val_tgt_acc = Accuracy(task="multiclass", num_classes=10)
        self.test_src_acc = Accuracy(task="multiclass", num_classes=10)
        self.test_tgt_acc = Accuracy(task="multiclass", num_classes=10)

        # for averaging loss across batches
        self.train_loss = MeanMetric()

        # # precompute the weighting list for latent noise scale
        self.weighting_list = torch.linspace(1, 1, num_latent_noise_scale)
        self.latent_noise_scale_list = torch.linspace(min_noise_scale, max_noise_scale, num_latent_noise_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        pass

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_src_acc.reset()
        self.val_tgt_acc.reset()

    def vae_loss_lambda(self, recon_x, x, mean, logvar, score=None, DSM=None, weighting=None):
        if isinstance(x, list) and isinstance(recon_x, list):
            for i in range(len(recon_x)):
                if weighting is not None:
                    if i == 0:
                        # recon_loss = nn.functional.mse_loss(recon_x[i], x[i], reduction='sum')/ratio
                        recon_loss = (weighting[:weighting.shape[0] // 2] * nn.functional.mse_loss(recon_x[i], x[i],
                                                                                                   reduction='none')).mean()
                        # print(nn.functional.mse_loss(recon_x[i], x[i], reduction='none').shape)
                    else:
                        recon_loss += (weighting[weighting.shape[0] // 2:] * nn.functional.mse_loss(recon_x[i], x[i],
                                                                                                    reduction='none')).mean()
                        # recon_loss += nn.functional.mse_loss(recon_x[i], x[i], reduction='sum')
                else:
                    if i == 0:
                        # recon_loss = nn.functional.mse_loss(recon_x[i], x[i], reduction='sum')/ratio
                        recon_loss = nn.functional.mse_loss(recon_x[i], x[i], reduction='none').mean()
                        # print(nn.functional.mse_loss(recon_x[i], x[i], reduction='none').shape)
                    else:
                        recon_loss += nn.functional.mse_loss(recon_x[i], x[i], reduction='none').mean()
                        # recon_loss += nn.functional.mse_loss(recon_x[i], x[i], reduction='sum')
        else:
            recon_loss = (weighting * nn.functional.mse_loss(recon_x, x, reduction='none')).mean()
        if weighting is not None:
            kld_encoder_posterior = 0.5 * torch.mean(weighting * (- 1 - logvar))
            # kld_encoder_posterior = 0.5 * torch.sum(- 1 - logvar)
            kld_prior = 0.5 * torch.mean(weighting * (mean.pow(2) + logvar.exp()))
            # kld_prior = 0.5 * torch.sum(mean.pow(2) + logvar.exp())
            kld_loss = kld_encoder_posterior + kld_prior
        else:
            kld_encoder_posterior = 0.5 * torch.mean(- 1 - logvar)
            # kld_encoder_posterior = 0.5 * torch.sum(- 1 - logvar)
            kld_prior = 0.5 * torch.mean(mean.pow(2) + logvar.exp())
            # kld_prior = 0.5 * torch.sum(mean.pow(2) + logvar.exp())
            kld_loss = kld_encoder_posterior + kld_prior
        if score is not None and DSM is None:
            kld_loss = kld_encoder_posterior - score
            kld_prior = - score
        elif DSM is not None:
            kld_loss = kld_encoder_posterior + DSM

        return recon_loss + kld_loss, recon_loss, kld_encoder_posterior, kld_prior, kld_loss

    def training_step(
        self, batch, batch_idx
    ) -> torch.Tensor:

        # with torch.autograd.detect_anomaly(True):

        _, domain_batch_list = batch
        x_batch_list = [domain_batch[0] for domain_batch in domain_batch_list]
        encoded_batch_list = [domain_batch[1] for domain_batch in domain_batch_list]
        label_batch_list = [domain_batch[2] for domain_batch in domain_batch_list]

        optimizer_vae_list, optimizer_score, optimizer_cls = self.optimizers()

        [vae.train() for vae in self.vae_list]

        output_list = [vae(x_batch) for vae, x_batch in zip(self.vae_list, x_batch_list)]

        x_flat_batch_list = [x_batch.view((x_batch.shape[0], -1)) for x_batch in x_batch_list]
        recon_x_flat_list = [output[0].view((output[0].shape[0], -1)) for output in output_list]

        z_list = [output[1].view((output[1].shape[0], -1)) for output in output_list]
        z = torch.vstack(z_list)

        mean_list = [output[2] for output in output_list]
        logvar_list = [output[3] for output in output_list]
        mean, logvar = torch.vstack(mean_list), torch.vstack(logvar_list)

        batch_size = x_batch_list[0].shape[0]
        latent_noise_idx = torch.randint(0, self.hparams.num_latent_noise_scale, (2 * batch_size,))
        weighting = self.weighting_list[latent_noise_idx].view(2 * batch_size, 1)
        latent_noise_scale = self.latent_noise_scale_list[latent_noise_idx]

        # update per loops
        if (((self.global_step % self.hparams.loops) == 0) and
                (self.current_epoch >= self.hparams.warm_score_epochs)):

            [optimizer.zero_grad() for optimizer in optimizer_vae_list]
            optimizer_cls.zero_grad()

            # Score loss
            score = self.score_model.get_mixing_score_fn(
                z, 30*torch.ones(z.shape[0], device=z.device).type(torch.long),
                latent_noise_idx.type(torch.long), detach=True, is_residual=True,
                is_vanilla=self.hparams.is_vanilla,
                alpha=None) - 0.05 * z

            # score = torch.matmul(score.unsqueeze(1), z.unsqueeze(-1)).sum()/z.shape[0]
            score = torch.matmul(score.unsqueeze(1), z.unsqueeze(-1)).sum() / (z.shape[0] * z.shape[1])
            vaub_loss, recon_loss, kld_encoder_posterior, _, kld_loss = (
                self.vae_loss_lambda(recon_x_flat_list,
                                     x_flat_batch_list,
                                     mean, logvar,
                                     score=score,
                                     DSM=None,
                                     weighting=None)
            )
            vaub_loss = self.hparams.vaub_lambda * vaub_loss
            recon_loss = self.hparams.recon_lambda * recon_loss

            # GP loss
            gp_loss = self.gp_model.compute_gp_loss(
                encoded_batch_list,
                z_list,
                block_size=self.hparams.block_size,
            )*self.hparams.gp_lambda
            # gp_loss = calculate_gp_loss([x1.view((x1.shape[0], -1)), x2.view((x1.shape[0], -1))],
            #                             [z1.view((z1.shape[0], -1)), z2.view((z2.shape[0], -1))])

            tot_loss = vaub_loss + recon_loss + gp_loss

            # CLS loss
            classifier_loss = 0
            for domain_idx, (mean_src, label_src) in enumerate(zip(mean_list, label_batch_list)):
                if domain_idx == self.hparams.target_domain_idx:
                    continue
                output_cls = self.classifier(mean_src.view((mean_src.shape[0], -1)).detach())
                classifier_loss += F.cross_entropy(output_cls, label_src, reduction='none').mean()

            # backwarding all losses
            tot_loss.backward(retain_graph=True)
            classifier_loss.backward()

            [optimizer.step() for optimizer in optimizer_vae_list]
            optimizer_cls.step()

            # update and log metrics per batch
            self.log("loss/tot_loss", tot_loss, on_step=True, on_epoch=False, sync_dist=True)
            self.log("loss/vae_loss", vaub_loss, on_step=True, on_epoch=False, sync_dist=True)
            self.log("loss/gp_loss", gp_loss, on_step=True, on_epoch=False, sync_dist=True)
            self.log("loss/classifier_loss", classifier_loss, on_step=True, on_epoch=False, sync_dist=True)

            self.log("loss_detail/recon_loss", recon_loss, on_step=True, on_epoch=False, sync_dist=True)
            self.log("loss_detail/kld_encoder_posterior", kld_encoder_posterior, on_step=True, on_epoch=False, sync_dist=True)
            self.log("loss_detail/kld_loss", kld_loss, on_step=True, on_epoch=False, sync_dist=True)
            self.log("loss_detail/score", score, on_step=True, on_epoch=False, sync_dist=True)

            for domain_idx, (mean_, logvar_)  in enumerate(zip(mean_list, logvar_list)):
                self.log(f"latent_space/mean_src_{domain_idx}", mean_src.mean(), on_step=True, on_epoch=False, sync_dist=True)
                self.log(f"latent_space/var_tgt_{domain_idx}", logvar_.exp().mean(), on_step=True, on_epoch=False, sync_dist=True)

            # self.train_acc(output_cls, label1)
            # self.log("train/acc", self.train_acc, on_step=True, on_epoch=False, sync_dist=True)
            # self.train_loss(tot_loss)

        latent_noise_idx = torch.randint(0, self.hparams.num_latent_noise_scale, (2 * batch_size,))
        latent_noise_scale = self.latent_noise_scale_list[latent_noise_idx]

        [vae.eval() for vae in self.vae_list]
        output_list = [vae(x_batch) for vae, x_batch in zip(self.vae_list, x_batch_list)]

        z_detach = torch.vstack([output[1].view((output[1].shape[0], -1)).detach() for output in output_list])

        # update the score model
        dsm_loss = self.score_model.update_score_fn(z_detach,
                                                    latent_noise_idx=latent_noise_idx,
                                                    optimizer=optimizer_score,
                                                    max_timestep=None,
                                                    is_mixing=True,
                                                    is_residual=True,
                                                    is_vanilla=self.hparams.is_vanilla,
                                                    alpha=None)

        # print(f"dsm_loss: {dsm_loss:.2f}")

        self.log("loss_detail/dsm_loss", dsm_loss, on_step=True, on_epoch=False, sync_dist=True)

        # update and log metrics per epoch
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def model_step(self, batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        _, domain_batch_list = batch

        x_batch_list = [domain_batch[0] for domain_batch in domain_batch_list]
        label_batch_list = [domain_batch[2] for domain_batch in domain_batch_list]

        [vae.eval() for vae in self.vae_list]
        output_list = [vae(x_batch) for vae, x_batch in zip(self.vae_list, x_batch_list)]
        mean_list = [output[2] for output in output_list]
        z_detach_list = [output[1].view((output[1].shape[0], -1)).detach() for output in output_list]

        output_src_list = []
        label_src_list = []

        for domain_idx, (mean_, label_) in enumerate(zip(mean_list, label_batch_list)):
            if domain_idx == self.hparams.target_domain_idx:
                output_tgt = self.classifier(mean_)
                label_tgt = label_
            else:
                output_src_list.append(self.classifier(mean_))
                label_src_list.append(label_)

        # self.val_src_acc(torch.vstack(output_src_list), torch.vstack(label_src_list))
        # self.val_tgt_acc(output_tgt, label_tgt)

        return output_src_list, label_src_list, output_tgt, label_tgt, z_detach_list, label_batch_list

    def validation_step(self, batch, batch_idx: int) -> None:

        _, domain_batch_list = batch
        x_batch_list = [domain_batch[0] for domain_batch in domain_batch_list]

        output_src_list, label_src_list, output_tgt, label_tgt, z_detach_list, label_batch_list = self.model_step(batch)

        # update and log metrics
        self.val_src_acc(torch.vstack(output_src_list), torch.vstack(label_src_list))
        self.val_tgt_acc(output_tgt, label_tgt)

        self.log("val/src_acc", self.val_src_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/tgt_acc", self.val_tgt_acc, on_step=False, on_epoch=True, prog_bar=True)

        logger_experiment = self.logger.experiment

        num_visual_samples = 10

        if self.global_rank == 0:

            for domain_idx in range(self.num_domain):
                fig, axes = plt.subplots(self.num_domain+1, num_visual_samples,
                                         figsize=(num_visual_samples*3/2, 1.5*(self.num_domain+1)))

                display_reconstructed_and_flip_images_multi(axes, self.current_epoch, self.vae_list,
                                                            domain_idx, x_batch_list[domain_idx],
                                                            n_samples=num_visual_samples, )
                logger_experiment.log({
                    f"Original to Flipped for Domain {domain_idx}": fig,
                })
                plt.close()

            fig_total, axes_total = plt.subplots(1, 2, figsize=(20, 10))
            fig_individual, axes_individual = plt.subplots(2, 5, figsize=(40, 8))
            plt_umap = display_umap_for_latent_multi(
                axes_total, axes_individual,
                self.current_epoch,
                z_detach_list,
                label_batch_list,
            )
            logger_experiment.log({
                "UMAP for Latent": fig_total,
                "UMAP for Each Class": fig_individual,
            })
            fig_individual.tight_layout()
            plt.close(fig_total)
            plt.close(fig_individual)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        self.val_src_acc.compute()  # get current val acc
        self.val_tgt_acc.compute()  # get current val acc

    def test_step(self, batch, batch_idx: int) -> None:

        output_src_list, label_src_list, output_tgt, label_tgt, _, _ = self.model_step(batch)

        # update and log metrics
        self.test_src_acc(torch.vstack(output_src_list), torch.vstack(label_src_list))
        self.test_tgt_acc(output_tgt, label_tgt)

        self.log("test/src_acc", self.test_src_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/tgt_acc", self.test_tgt_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """

        self.gp_model.to_device(self.device)

        if self.hparams.compile and stage == "fit":
            self.src_vae = torch.compile(self.src_vae)
            self.tgt_vae = torch.compile(self.tgt_vae)
            self.classifier = torch.compile(self.classifier)
            self.score_model = torch.compile(self.score_model)

    def configure_optimizers(self) -> tuple[list[Any], list[Any]]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer_vae_list = [optimizer_vae_(params=vae_.parameters())
                              for optimizer_vae_, vae_ in zip(self.hparams.optimizer_vae_list, self.vae_list)]
        optimizer_score = self.hparams.optimizer_score(params=self.score_model.parameters())
        optimizer_cls = self.hparams.optimizer_cls(params=self.classifier.parameters())

        # if self.hparams.scheduler is not None:
        #     scheduler = self.hparams.scheduler(optimizer=optimizer)
        #     return {
        #         "optimizer": optimizer,
        #         "lr_scheduler": {
        #             "scheduler": scheduler,
        #             "monitor": "val/loss",
        #             "interval": "epoch",
        #             "frequency": 1,
        #         },
        #     }

        return [optimizer_vae_list, optimizer_score, optimizer_cls], []

