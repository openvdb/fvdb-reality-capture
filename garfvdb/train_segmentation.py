#! /usr/bin/env python
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import itertools
import logging
import math
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, Union

import torch
import torch.nn.functional as F
import tqdm
import tyro
from garfvdb.config import Config
from garfvdb.dataset import (
    GARfVDBInputCollateFn,
    RandomSamplePixels,
    RandomSelectMaskIDAndScale,
    Resize,
    SegmentationDataset,
    TransformDataset,
)
from garfvdb.model import GARfVDBModel
from garfvdb.optim import ExponentaLRWithRampUpScheduler
from garfvdb.util import pca_projection_fast
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose

logging.basicConfig(level=logging.INFO)
logging.getLogger().name = "garfvdb"


class Runner:
    def __init__(
        self,
        cfg: Config,
        checkpoint_path: str,
        segmentation_dataset_path: str,
        device: Union[str, torch.device] = "cuda",
    ):
        self.config = cfg
        self.device = device

        self.val_every_n_steps = 500
        self._test_mode = False  # Private flag for test mode

        # Create tensorboard writer with timestamp for unique runs
        os.makedirs("logs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = f"logs/segmentation_training/{timestamp}"
        self.writer = SummaryWriter(log_dir=log_dir)
        logging.info(f"TensorBoard logs will be saved to: {log_dir}")

        ### Dataset ###
        full_dataset = SegmentationDataset(segmentation_dataset_path)

        # Split into train and validation sets
        val_split = 0.1
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size - 1, 1], generator=torch.Generator().manual_seed(42)
        )

        # For training, randomly sample pixels from each image
        self.train_dataset = TransformDataset(
            self.train_dataset,
            Compose([RandomSamplePixels(self.config.sample_pixels_per_image), RandomSelectMaskIDAndScale()]),
        )
        self.val_dataset = TransformDataset(
            self.val_dataset,
            Compose([RandomSamplePixels(self.config.sample_pixels_per_image), RandomSelectMaskIDAndScale()]),
        )
        # For testing, use the full image
        self.test_dataset = TransformDataset(self.test_dataset, Compose([Resize(1 / 5), RandomSelectMaskIDAndScale()]))

        ### Model ###
        # Scale grouping stats
        grouping_scale_stats = torch.cat(full_dataset.scales)
        self.model = GARfVDBModel(
            checkpoint_path,
            grouping_scale_stats,
            model_config=self.config.model,
            device=device,
        )

        # Optimizer
        # Different parameter groups with separate learning rates
        param_groups = [
            {"params": self.model.mlp.parameters(), "lr": 1e-4},  # Base learning rate for MLP
        ]

        # Add grid parameters with different learning rate if using grid
        if self.config.model.use_grid:
            # For encoder_grids, we need to access the data.jdata parameter as shown in the model's parameters() method
            param_groups.append({"params": [self.model.encoder_grids.data.jdata], "lr": 1e-3})
            if self.config.model.use_grid_conv:
                param_groups.append({"params": self.model.encoder_convnet.parameters(), "lr": 1e-3})

        else:
            # For sh0 model, add the sh0 parameter
            param_groups.append({"params": [self.model.gs_model.sh0], "lr": 1e-3})  # Lower learning rate for sh0

        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=1e-4,
            # param_groups,
            # lr=1e-5,
            # weight_decay=1e-6,
            # eps=1e-15 / lr_batch_rescale,
        )

        # Add ExponentaLRWithRampUpScheduler for each parameter group
        base_lr = self.optimizer.param_groups[0]["lr"]  # MLP learning rate (1e-4)
        # grid_lr = self.optimizer.param_groups[1]["lr"]  # Grid/sh0 learning rate (5e-5)

        # Create scheduler for all parameters using base_lr as reference
        self.scheduler = ExponentaLRWithRampUpScheduler(
            optimizer=self.optimizer,
            lr_init=base_lr,
            lr_final=1e-6,
            max_steps=self.config.num_train_iters,
        )

    @contextmanager
    def test_mode(self):
        """Context manager for test mode.

        Usage:
            with self.test_mode():
                # Code that should run in validation mode
                pass
        """
        previous_mode = self._test_mode
        self._test_mode = True
        self.model.eval()
        try:
            yield
        finally:
            self._test_mode = previous_mode
            if not previous_mode:
                self.model.train()

    def is_test_mode(self) -> bool:
        """Check if we're currently in test mode."""
        return self._test_mode

    def calc_loss(self, enc_feats: torch.Tensor, input: Dict[str, torch.Tensor], step: int = -1) -> torch.Tensor:
        dtype = enc_feats.dtype
        if self.is_test_mode():
            # reduce memory usage by using float16 to accomodate computing loss across the whole image
            dtype = torch.float16

        margin = 1.0

        # Using a product of this form to accomodate 'image' inputs of the form [B, num_samples, C] and [B, H, W, C]
        samples_per_img = math.prod(input["image"].shape[1:-1])

        num_chunks = enc_feats.shape[0]

        if input["mask_id"].is_nested:
            for t in input["mask_id"]:
                print(f"mask id dim: {t.shape}")
        input_id1 = input_id2 = input["mask_id"].flatten().to(dtype)

        # Debug prints
        logging.debug(
            f"calc_loss shapes: enc_feats={enc_feats.shape}, input_id1={input_id1.shape}, mask_id={input['mask_id'].shape}"
        )

        # Expand labels
        labels1_expanded = input_id1.unsqueeze(1).expand(-1, input_id1.shape[0])
        labels2_expanded = input_id2.unsqueeze(0).expand(input_id2.shape[0], -1)

        # Mask for positive/negative pairs across the entire matrix
        mask_full_positive = labels1_expanded == labels2_expanded
        mask_full_negative = ~mask_full_positive

        # # Debug print
        logging.debug(
            f"num_chunks = {num_chunks}, input_id1.shape[0] = {input_id1.shape[0]}, samples_per_img = {samples_per_img}"
        )

        # Create a block mask to only consider pairs within the same image -- no cross-image pairs
        block_mask = torch.kron(  # [samples_per_img*num_chunks, samples_per_img*num_chunks] dtype: torch.bool
            torch.eye(num_chunks, device=labels1_expanded.device, dtype=torch.bool),
            torch.ones(
                (samples_per_img, samples_per_img),
                device=labels1_expanded.device,
                dtype=torch.bool,
            ),
        )  # block-diagonal matrix, to consider only pairs within the same image

        logging.debug(f"block_mask.shape = {block_mask.shape}")

        # Only consider upper triangle to avoid double-counting
        block_mask = torch.triu(
            block_mask, diagonal=0
        )  # [samples_per_img*num_chunks, samples_per_img*num_chunks] dtype: torch.bool
        # Only consider pairs where both points are valid (-1 means not in mask / invalid)
        block_mask = block_mask * (labels1_expanded != -1) * (labels2_expanded != -1)

        # Mask for diagonal elements (i.e., pairs of the same point).
        # Don't consider these pairs for grouping supervision (pulling), since they are trivially similar.
        diag_mask = torch.eye(block_mask.shape[0], device=block_mask.device, dtype=torch.bool)

        scales = input["scales"]

        ####################################################################################
        # Grouping supervision
        ####################################################################################
        total_loss = 0

        # Get instance features - will return a 3D tensor [batch_size, samples_per_img, feat_dim]
        instance_features = self.model.get_mlp_output(enc_feats, scales)

        # Flatten the instance features to match the masking operations
        # [batch_size, samples_per_img, feat_dim] -> [batch_size*samples_per_img, feat_dim]
        instance_features_flat = instance_features.reshape(-1, instance_features.shape[-1])

        # 1. If (A, s_A) and (A', s_A) in same group, then supervise the features to be similar
        mask = torch.where(mask_full_positive * block_mask * (~diag_mask))

        instance_loss_1 = torch.norm(instance_features_flat[mask[0]] - instance_features_flat[mask[1]], p=2, dim=-1)
        if not (mask[0] // samples_per_img == mask[1] // samples_per_img).all():
            logging.error("Loss Function: There's a camera cross-talk issue")

        instance_loss_1_sum = instance_loss_1.nansum()
        total_loss += instance_loss_1_sum
        logging.debug(f"Loss 1: {instance_loss_1_sum.item()}, using {mask[0].shape[0]} pairs")

        if self.is_test_mode():
            # log instance_loss_1
            self.writer.add_scalar("test/instance_loss_1", instance_loss_1.nansum().item(), step)
            # log image of instance_loss_1
            loss_1_img = torch.zeros(input["image"].shape[:-1], device=instance_loss_1.device)
            # Use scatter_reduce_ to accumulate values
            loss_1_img.view(-1).scatter_reduce_(
                0,  # dim to reduce along
                mask[0],  # indices
                instance_loss_1,  # values to scatter
                reduce="sum",  # reduction operation
                include_self=True,  # include values in the output
            )
            # rescale loss_1_img to [0, 1]
            logging.debug("loss_1_img.max() ", loss_1_img.max())
            loss_1_img = loss_1_img / loss_1_img.max()
            self.writer.add_image("test/instance_loss_1_img", loss_1_img, step)
            del loss_1_img

        # 2. If ", then also supervise them to be similar at s > s_A
        # if self.config.use_hierarchy_losses and (not self.config.use_single_scale):

        scale_diff = torch.max(torch.zeros_like(scales), (self.model.model_config.max_grouping_scale - scales))
        larger_scale = scales + scale_diff * torch.rand(size=(1,), device=scales.device)

        # Get larger scale features and flatten
        larger_scale_instance_features = self.model.get_mlp_output(enc_feats, larger_scale)
        larger_scale_instance_features_flat = larger_scale_instance_features.reshape(
            -1, larger_scale_instance_features.shape[-1]
        )

        instance_loss_2 = torch.norm(
            larger_scale_instance_features_flat[mask[0]] - larger_scale_instance_features_flat[mask[1]], p=2, dim=-1
        )
        instance_loss_2_nansum = instance_loss_2.nansum()
        total_loss += instance_loss_2_nansum
        logging.debug(f"Loss 2: {instance_loss_2_nansum.item()}, using {mask[0].shape[0]} pairs")

        if self.is_test_mode():
            # log instance_loss_2
            self.writer.add_scalar("test/instance_loss_2", instance_loss_2_nansum.item(), step)
            # log image of instance_loss_2
            loss_2_img = torch.zeros(input["image"].shape[:-1], device=instance_loss_2.device)
            # Use scatter_reduce_ to accumulate values, using 'sum' as the reduction operation
            loss_2_img.view(-1).scatter_reduce_(
                0,  # dim to reduce along
                mask[0],  # indices
                instance_loss_2,  # values to scatter
                reduce="sum",  # reduction operation
                include_self=True,  # include values in the output
            )
            print("loss_2_img.max() ", loss_2_img.max())
            loss_2_img = loss_2_img / loss_2_img.max()
            self.writer.add_image("test/instance_loss_2_img", loss_2_img, step)
            del loss_2_img

        # 4. Also supervising A, B to be dissimilar at scales s_A, s_B respectively seems to help.
        mask = torch.where(mask_full_negative * block_mask)

        if self.is_test_mode():
            instance_features_flat = instance_features_flat.to(torch.float16)

        instance_loss_4 = F.relu(
            margin - torch.norm(instance_features_flat[mask[0]] - instance_features_flat[mask[1]], p=2, dim=-1)
        )
        instance_loss_4_nansum = instance_loss_4.to(torch.float32).nansum()
        total_loss += instance_loss_4_nansum
        logging.debug(f"Loss 4: {instance_loss_4_nansum.item()}, using {mask[0].shape[0]} pairs")
        if self.is_test_mode():
            # log instance_loss_4
            self.writer.add_scalar("test/instance_loss_4", instance_loss_4_nansum.item(), step)
            # log image of instance_loss_4
            loss_4_img = torch.zeros(
                input["image"].shape[:-1], device=instance_loss_4.device, dtype=instance_loss_4.dtype
            )
            # Use scatter_reduce_ to accumulate values, using 'sum' as the reduction operation
            loss_4_img.view(-1).scatter_reduce_(
                0,  # dim to reduce along
                mask[0],  # indices
                instance_loss_4,  # values to scatter
                reduce="sum",  # reduction operation
                include_self=True,  # include values in the output
            )
            logging.debug("loss_4_img.max() ", loss_4_img.max())
            loss_4_img = loss_4_img / loss_4_img.max()
            self.writer.add_image("test/instance_loss_4_img", loss_4_img, step)

        return total_loss / torch.sum(block_mask * (~diag_mask)).float()

    def train(self):
        trainloader = itertools.cycle(
            DataLoader(
                self.train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=4,
                persistent_workers=True,
                pin_memory=True,
                collate_fn=GARfVDBInputCollateFn,
            )
        )

        # Create validation dataloader
        valloader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=GARfVDBInputCollateFn,
        )

        testloader = DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=GARfVDBInputCollateFn,
        )

        self.model.train()
        self.optimizer.zero_grad()

        pbar = tqdm.tqdm(range(self.config.num_train_iters))

        # Gradient accumulation settings
        gradient_accumulation_steps = 8
        accumulated_loss = 0.0

        for step in pbar:
            minibatch = next(trainloader)

            # Move to device
            for k, v in minibatch.items():
                minibatch[k] = v.to(self.device)

            # Debug prints for first iteration
            if step == 0:
                logging.info(f"Training with sample_pixels_per_image={self.config.sample_pixels_per_image}")
                logging.info(f"Image size: {minibatch['image_full'].shape}")
                logging.info(f"Batch size: {minibatch['image'].shape[0]}")
                logging.info(f"scales shape: {minibatch['scales'].shape}")
                logging.info(f"mask_id shape: {minibatch['mask_id'].shape}")
                logging.info(f"pixel_coords shape: {minibatch['pixel_coords'].shape}")
                logging.info(f"mean2d shape: {self.model.gs_model.means.shape}")
                logging.info(f"Using gradient accumulation over {gradient_accumulation_steps} steps")

            ### Forward pass ###
            cam_enc_feats = self.model.get_enc_feats(minibatch)

            loss = self.calc_loss(cam_enc_feats, minibatch, step)

            # Scale loss by accumulation steps to maintain same effective learning rate
            loss = loss / gradient_accumulation_steps
            accumulated_loss += loss.item()

            # Zero gradients only at the beginning of accumulation cycle
            if step % gradient_accumulation_steps == 0:
                self.optimizer.zero_grad()

            loss.backward()

            # Perform optimizer step and gradient clipping only after accumulating gradients
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                # self.scheduler.step()

                # Log accumulated training loss to tensorboard
                self.writer.add_scalar("train/loss", accumulated_loss, step)
                self.writer.add_scalar("train/learning_rate", self.scheduler.get_last_lr()[0], step)

                # Reset accumulated loss
                accumulated_loss = 0.0

            # For display purposes, show the scaled loss
            display_loss = loss.item() * gradient_accumulation_steps

            pbar.set_postfix(loss=f"{display_loss:.4g}")
            del loss
            torch.cuda.empty_cache()

            # Evaluate on validation set periodically
            if step % self.val_every_n_steps == 0 or step == self.config.num_train_iters - 1:
                val_loss = 0
                with torch.no_grad():
                    for val_batch in valloader:
                        for k, v in val_batch.items():
                            val_batch[k] = v.to(self.device)
                        val_enc_feats = self.model.get_enc_feats(val_batch)
                        val_loss += self.calc_loss(val_enc_feats, val_batch, step).item()
                val_loss /= len(valloader)

                # Log validation loss to tensorboard
                self.writer.add_scalar("val/loss", val_loss, step)

                # Log a sample validation image
                with torch.no_grad():
                    val_batch_zero = next(iter(valloader)).to(self.device)
                    desired_scale = torch.max(val_batch_zero["scales"]) * 0.1
                    val_mask_output, _ = self.model.get_mask_output(val_batch_zero, desired_scale.item())
                    beauty_output = val_batch_zero["image_full"]
                    pca_output = pca_projection_fast(val_mask_output, 3)
                    ### Save images
                    beauty_output = beauty_output.cpu()  # [B, H, W, 3]
                    pca_output = (pca_output.cpu() * 255).type(torch.uint8)  # [B, H, W, 3]
                    self.writer.add_image("val/sample_mask", pca_output.permute(0, 3, 1, 2)[0], step)
                    alpha = 0.7
                    blended = (beauty_output * (1 - alpha) + pca_output * alpha).type(torch.uint8)
                    # Permute dimensions from [B, H, W, C] to [B, C, H, W] for TensorBoard
                    blended = blended.permute(0, 3, 1, 2)
                    self.writer.add_image("val/sample_image", blended[0], step)

                # Log test loss images
                if self.config.model.log_test_images:
                    with self.test_mode(), torch.no_grad():
                        test_loss = 0
                        for test_batch_idx, test_batch in enumerate(testloader):
                            if test_batch_idx != 0:
                                break

                            # Log the image
                            # Permute from [H, W, C] to [C, H, W] for TensorBoard
                            test_image = test_batch["image"][0].cpu().permute(2, 0, 1)
                            self.writer.add_image("test/sample_image", test_image, step)
                            for k, v in test_batch.items():
                                test_batch[k] = v.to(self.device)
                            # Set scales to 0.1
                            test_batch["scales"] = torch.full_like(test_batch["scales"], 0.05)
                            test_enc_feats = self.model.get_enc_feats(test_batch)

                            # Calculate loss on the CPU for memory issues on large images
                            for k, v in test_batch.items():
                                test_batch[k] = v.to("cpu")
                            self.model.to("cpu")
                            test_loss += self.calc_loss(test_enc_feats.to("cpu"), test_batch, step).item()

                            self.model.to(self.device)

                # Save model
                if step % 1000 == 0:
                    dict_to_save = {"mlp": self.model.mlp.state_dict()}
                    if self.config.model.use_grid:
                        dict_to_save["encoder_grids"] = self.model.encoder_grids.data.jdata.detach().cpu()
                    else:
                        dict_to_save["sh0"] = self.model.gs_model.sh0.detach().cpu()
                    torch.save(dict_to_save, f"checkpoints/checkpoint_{step}.pt")

        # Close tensorboard writer
        self.writer.close()


def main(checkpoint_path: str, segmentation_dataset_path: str, config: Config = Config()):
    torch.manual_seed(0)
    runner = Runner(config, checkpoint_path, segmentation_dataset_path)
    runner.train()


if __name__ == "__main__":
    tyro.cli(main)
