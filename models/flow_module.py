from typing import Any
import torch
import time
import os
import random
import wandb
import numpy as np
import pandas as pd
import logging
import torch.distributed as dist
from pytorch_lightning import LightningModule
from analysis import metrics 
from analysis import utils as au
from models.flow_model import FlowModel
from models import utils as mu
from data.interpolant import Interpolant 
from data import utils as du
from data import all_atom
from data import so3_utils
from data import residue_constants
from experiments import utils as eu
from pytorch_lightning.loggers.wandb import WandbLogger
import esm
from data import utils as du
from data import residue_constants
from biotite.sequence.io import fasta
import subprocess
from typing import Optional
from analysis import metrics
import shutil
from datetime import datetime
from openfold.utils.rigid_utils import rot_to_quat, quat_to_rot

class FlowModule(LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self._print_logger = logging.getLogger(__name__)
        self._exp_cfg = cfg.experiment
        self._model_cfg = cfg.model
        self._data_cfg = cfg.data
        self._interpolant_cfg = cfg.interpolant

        # Set-up vector field prediction model
        self.model = FlowModel(cfg.model)

        # Set-up interpolant
        self.interpolant = Interpolant(cfg.interpolant)

        self.validation_epoch_metrics = []
        self.validation_epoch_samples = []
        self.save_hyperparameters()
        self._checkpoint_dir = None
        self._inference_dir = None
        self._time_records = pd.DataFrame(
            columns=['length', 'sample_path', 'start_time', 'sample_time', 'eval_time', 'total_time', 'memory_allocated', 'memory_reserved'])
        print("self._exp_cfg.is_training", self._exp_cfg.is_training)
        if not self._exp_cfg.is_training:
            self._folding_model = esm.pretrained.esmfold_v1().eval()
            self.save_hyperparameters(ignore=["_folding_model"]) 
            for param in self._folding_model.parameters():
                param.requires_grad = False

        

    @property
    def checkpoint_dir(self):
        if self._checkpoint_dir is None:
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    checkpoint_dir = [self._exp_cfg.checkpointer.dirpath]
                else:
                    checkpoint_dir = [None]
                dist.broadcast_object_list(checkpoint_dir, src=0)
                checkpoint_dir = checkpoint_dir[0]
            else:
                checkpoint_dir = self._exp_cfg.checkpointer.dirpath
            self._checkpoint_dir = checkpoint_dir
            os.makedirs(self._checkpoint_dir, exist_ok=True)
        return self._checkpoint_dir

    @property
    def inference_dir(self):
        if self._inference_dir is None:
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    inference_dir = [self._exp_cfg.inference_dir]
                else:
                    inference_dir = [None]
                dist.broadcast_object_list(inference_dir, src=0)
                inference_dir = inference_dir[0]
            else:
                inference_dir = self._exp_cfg.inference_dir
            self._inference_dir = inference_dir
            os.makedirs(self._inference_dir, exist_ok=True)
        return self._inference_dir

    def on_train_start(self):
        self._epoch_start_time = time.time()
        
    def on_train_epoch_end(self):
        epoch_time = (time.time() - self._epoch_start_time) / 60.0
        self.log(
            'train/epoch_time_minutes',
            epoch_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False
        )
        self._epoch_start_time = time.time()

    def model_step(self, noisy_batch: Any):
        training_cfg = self._exp_cfg.training
        loss_mask = noisy_batch['res_mask'] * noisy_batch['diffuse_mask']
        if torch.any(torch.sum(loss_mask, dim=-1) < 1):
            raise ValueError('Empty batch encountered')
        num_batch, num_res = loss_mask.shape

        # Ground truth labels
        gt_trans_1 = noisy_batch['trans_1']
        gt_rotmats_1 = noisy_batch['rotmats_1']
        gt_rotquats_1 = noisy_batch['rotquats_1']
        rotquats_t = noisy_batch['rotquats_t']
        gt_rot_quat_vf = so3_utils.calc_quat_wt_qt_q1(
            rotquats_t, gt_rotquats_1.type(torch.float32))
        if torch.any(torch.isnan(gt_rot_quat_vf)):
            raise ValueError('NaN encountered in gt_rot_quat_vf')

        gt_bb_atoms = all_atom.to_atom37(gt_trans_1, gt_rotmats_1)[:, :, :3]

        # Timestep used for normalization.
        r3_t = noisy_batch['r3_t']
        so3_t = noisy_batch['so3_t']
        r3_norm_scale = 1 - torch.min(
            r3_t[..., None], torch.tensor(training_cfg.t_normalize_clip))
        so3_norm_scale = 1 - torch.min(
            so3_t[..., None], torch.tensor(training_cfg.t_normalize_clip))
        
        # Model output predictions.
        model_output = self.model(noisy_batch)
        pred_trans_1 = model_output['pred_trans']
        pred_rotmats_1 = model_output['pred_rotmats']
        pred_rotquats_1 = model_output['pred_rotquats']
        # pred_rots_quat_vf = so3_utils.calc_quat_wt_qt_q1( #* Exp schedule
        #     rotquats_t, pred_rotquats_1) * 10 * torch.exp( -so3_t[..., None] * 10) 
        pred_rots_quat_vf = so3_utils.calc_quat_wt_qt_q1(
            rotquats_t, pred_rotquats_1)
        if torch.any(torch.isnan(pred_rots_quat_vf)):
            raise ValueError('NaN encountered in pred_rots_quat_vf')

        # Backbone atom loss
        pred_bb_atoms = all_atom.to_atom37(pred_trans_1, pred_rotmats_1)[:, :, :3]
        gt_bb_atoms *= training_cfg.bb_atom_scale / r3_norm_scale[..., None]
        pred_bb_atoms *= training_cfg.bb_atom_scale / r3_norm_scale[..., None]
        loss_denom = torch.sum(loss_mask, dim=-1) * 3
        bb_atom_loss = torch.sum(
            (gt_bb_atoms - pred_bb_atoms) ** 2 * loss_mask[..., None, None],
            dim=(-1, -2, -3)
        ) / loss_denom

        # Translation VF loss
        trans_error = (gt_trans_1 - pred_trans_1) / r3_norm_scale * training_cfg.trans_scale
        trans_loss = training_cfg.translation_loss_weight * torch.sum(
            trans_error ** 2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom
        trans_loss = torch.clamp(trans_loss, max=5)

        # Quat VF loss
        rots_quats_vf_error = (gt_rot_quat_vf - pred_rots_quat_vf) / so3_norm_scale
        rots_quats_vf_loss = training_cfg.rotation_loss_weights * torch.sum(
            rots_quats_vf_error ** 2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom

        # Pairwise distance loss
        gt_flat_atoms = gt_bb_atoms.reshape([num_batch, num_res*3, 3])
        gt_pair_dists = torch.linalg.norm(
            gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1)
        pred_flat_atoms = pred_bb_atoms.reshape([num_batch, num_res*3, 3])
        pred_pair_dists = torch.linalg.norm(
            pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1)

        flat_loss_mask = torch.tile(loss_mask[:, :, None], (1, 1, 3))
        flat_loss_mask = flat_loss_mask.reshape([num_batch, num_res*3])
        flat_res_mask = torch.tile(loss_mask[:, :, None], (1, 1, 3))
        flat_res_mask = flat_res_mask.reshape([num_batch, num_res*3])

        gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
        pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
        pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]

        dist_mat_loss = torch.sum(
            (gt_pair_dists - pred_pair_dists)**2 * pair_dist_mask,
            dim=(1, 2))
        dist_mat_loss /= (torch.sum(pair_dist_mask, dim=(1, 2)) + 1)

        se3_vf_loss = trans_loss + rots_quats_vf_loss
        auxiliary_loss = (
            bb_atom_loss * training_cfg.aux_loss_use_bb_loss
            + dist_mat_loss * training_cfg.aux_loss_use_pair_loss
        )
        auxiliary_loss *= (
            (r3_t[:, 0] > training_cfg.aux_loss_t_pass)
            & (so3_t[:, 0] > training_cfg.aux_loss_t_pass)
        )
        auxiliary_loss *= self._exp_cfg.training.aux_loss_weight
        auxiliary_loss = torch.clamp(auxiliary_loss, max=5)

        se3_vf_loss += auxiliary_loss
        if torch.any(torch.isnan(se3_vf_loss)):
            raise ValueError('NaN loss encountered')
        # print(f"trans_loss: {trans_loss}\n aux_loss: {auxiliary_loss}\n rots_quats_vf_loss: {rots_quats_vf_loss}\n se3_vf_loss: {se3_vf_loss}")
        return {
            "trans_loss": trans_loss,
            "auxiliary_loss": auxiliary_loss,
            "rots_quats_vf_loss": rots_quats_vf_loss,
            "se3_vf_loss": se3_vf_loss
        }
    

    def validation_step(self, batch: Any, batch_idx: int):
        if self._data_cfg.rectify:
            res_mask = batch['sample']['res_mask']
            self.interpolant.set_device(res_mask.device)
            num_batch, num_res = res_mask.shape
            diffuse_mask = batch['sample']['diffuse_mask']
            csv_idx = batch['sample']['csv_idx']
            prot_traj, atom37_traj, _, _ = self.interpolant.sample(
                num_batch,
                num_res,
                self.model,
                trans_1=batch['sample']['trans_1'],
                rotmats_1=batch['sample']['rotmats_1'],
                diffuse_mask=diffuse_mask,
                chain_idx=batch['sample']['chain_idx'],
                res_idx=batch['sample']['res_idx'],
            )
        else:
            res_mask = batch['res_mask']
            self.interpolant.set_device(res_mask.device)
            num_batch, num_res = res_mask.shape
            diffuse_mask = batch['diffuse_mask']
            csv_idx = batch['csv_idx']
            prot_traj, atom37_traj, _, _ = self.interpolant.sample(
                num_batch,
                num_res,
                self.model,
                trans_1=batch['trans_1'],
                rotmats_1=batch['rotmats_1'],
                diffuse_mask=diffuse_mask,
                chain_idx=batch['chain_idx'],
                res_idx=batch['res_idx'],
            )
        samples = atom37_traj[-1].numpy()
        batch_metrics = []
        for i in range(num_batch):
            sample_dir = os.path.join(
                self.checkpoint_dir,
                f'sample_{csv_idx[i].item()}_idx_{batch_idx}_len_{num_res}'
            )
            os.makedirs(sample_dir, exist_ok=True)

            # Write out sample to PDB file
            final_pos = samples[i]
            saved_path = au.write_prot_to_pdb(
                final_pos,
                os.path.join(sample_dir, 'sample.pdb'),
                no_indexing=True
            )
            if isinstance(self.logger, WandbLogger):
                self.validation_epoch_samples.append(
                    [saved_path, self.global_step, wandb.Molecule(saved_path)]
                )

            mdtraj_metrics = metrics.calc_mdtraj_metrics(saved_path)
            ca_idx = residue_constants.atom_order['CA']
            ca_ca_metrics = metrics.calc_ca_ca_metrics(final_pos[:, ca_idx])
            batch_metrics.append((mdtraj_metrics | ca_ca_metrics))

        batch_metrics = pd.DataFrame(batch_metrics)
        self.validation_epoch_metrics.append(batch_metrics)
        
    def on_validation_epoch_end(self):
        if len(self.validation_epoch_samples) > 0:
            self.logger.log_table(
                key='valid/samples',
                columns=["sample_path", "global_step", "Protein"],
                data=self.validation_epoch_samples)
            self.validation_epoch_samples.clear()
        val_epoch_metrics = pd.concat(self.validation_epoch_metrics)
        for metric_name,metric_val in val_epoch_metrics.mean().to_dict().items():
            self._log_scalar(
                f'valid/{metric_name}',
                metric_val,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=len(val_epoch_metrics),
            )
        self.validation_epoch_metrics.clear()

    def _log_scalar(
            self,
            key,
            value,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=None,
            sync_dist=False,
            rank_zero_only=True
        ):
        if sync_dist and rank_zero_only:
            raise ValueError('Unable to sync dist when rank_zero_only=True')
        self.log(
            key,
            value,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
            batch_size=batch_size,
            sync_dist=sync_dist,
            rank_zero_only=rank_zero_only
        )

    def training_step(self, batch: Any, stage: int):
        #* If rectify, A batch is a dictionary with 'sample' and 'noise'.
        step_start_time = time.time()
        if self._data_cfg.rectify:
            self.interpolant.set_device(batch['sample']['res_mask'].device)
            noisy_batch = self.interpolant.rectify_corrupt_batch(batch)
        else:
            self.interpolant.set_device(batch['res_mask'].device)
            noisy_batch = self.interpolant.corrupt_batch(batch)
        if self._interpolant_cfg.self_condition and random.random() > 0.5:
            with torch.no_grad():
                model_sc = self.model(noisy_batch)
                noisy_batch['trans_sc'] = (
                    model_sc['pred_trans'] * noisy_batch['diffuse_mask'][..., None]
                    + noisy_batch['trans_1'] * (1 - noisy_batch['diffuse_mask'][..., None])
                )
        batch_losses = self.model_step(noisy_batch)
        num_batch = batch_losses['trans_loss'].shape[0]
        total_losses = {
            k: torch.mean(v) for k,v in batch_losses.items()
        }
        for k,v in total_losses.items():
            self._log_scalar(
                f"train/{k}", v, prog_bar=False, batch_size=num_batch)
        
        # Losses to track. Stratified across t.
        so3_t = torch.squeeze(noisy_batch['so3_t'])
        self._log_scalar(
            "train/so3_t",
            np.mean(du.to_numpy(so3_t)),
            prog_bar=False, batch_size=num_batch)
        r3_t = torch.squeeze(noisy_batch['r3_t'])
        self._log_scalar(
            "train/r3_t",
            np.mean(du.to_numpy(r3_t)),
            prog_bar=False, batch_size=num_batch)
        for loss_name, loss_dict in batch_losses.items():
            if loss_name == 'rots_vf_loss':
                batch_t = so3_t
            else:
                batch_t = r3_t
            stratified_losses = mu.t_stratified_loss(
                batch_t, loss_dict, loss_name=loss_name)
            for k,v in stratified_losses.items():
                self._log_scalar(
                    f"train/{k}", v, prog_bar=False, batch_size=num_batch)

        # Training throughput
        if self._data_cfg.rectify:
            scaffold_percent = torch.mean(batch['sample']['diffuse_mask'].float()).item()
            self._log_scalar(
                "train/scaffolding_percent",
                scaffold_percent, prog_bar=False, batch_size=num_batch)
            motif_mask = 1 - batch['sample']['diffuse_mask'].float()
            num_motif_res = torch.sum(motif_mask, dim=-1)
            self._log_scalar(
                "train/motif_size", 
                torch.mean(num_motif_res).item(), prog_bar=False, batch_size=num_batch)
            self._log_scalar(
                "train/length", batch['sample']['res_mask'].shape[1], prog_bar=False, batch_size=num_batch)
            self._log_scalar(
                "train/batch_size", num_batch, prog_bar=False)
            step_time = time.time() - step_start_time
            self._log_scalar(
                "train/examples_per_second", num_batch / step_time)
            train_loss = total_losses['se3_vf_loss']
            self._log_scalar(
                "train/loss", train_loss, batch_size=num_batch)
        else:
            scaffold_percent = torch.mean(batch['diffuse_mask'].float()).item()
            self._log_scalar(
                "train/scaffolding_percent",
                scaffold_percent, prog_bar=False, batch_size=num_batch)
            motif_mask = 1 - batch['diffuse_mask'].float()
            num_motif_res = torch.sum(motif_mask, dim=-1)
            self._log_scalar(
                "train/motif_size", 
                torch.mean(num_motif_res).item(), prog_bar=False, batch_size=num_batch)
            self._log_scalar(
                "train/length", batch['res_mask'].shape[1], prog_bar=False, batch_size=num_batch)
            self._log_scalar(
                "train/batch_size", num_batch, prog_bar=False)
            step_time = time.time() - step_start_time
            self._log_scalar(
                "train/examples_per_second", num_batch / step_time)
            train_loss = total_losses['se3_vf_loss']
            self._log_scalar(
                "train/loss", train_loss, batch_size=num_batch)
        return train_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.model.parameters(),
            **self._exp_cfg.optimizer
        )



    def predict_step(self, batch, batch_idx):
        '''
        The main inference function for the model. 
        This function is called by the PyTorch Lightning Trainer.
        Each time this function is called, the model generates a batch of protein structure.
        Dataloader will be used automatically to generate the batch.
        '''
        del batch_idx # Unused
        device = f'cuda:{torch.cuda.current_device()}'
        interpolant = Interpolant(self._infer_cfg.interpolant) 
        interpolant.set_device(device)

        sample_ids = batch['sample_id'].squeeze().tolist()
        sample_ids = [sample_ids] if isinstance(sample_ids, int) else sample_ids
        num_batch = len(sample_ids)


        if 'diffuse_mask' in batch: # motif-scaffolding
            target = batch['target'][0]
            trans_1 = batch['trans_1']
            rotmats_1 = batch['rotmats_1']
            diffuse_mask = batch['diffuse_mask']
            true_bb_pos = all_atom.atom37_from_trans_rot(trans_1, rotmats_1, 1 - diffuse_mask)
            true_bb_pos = true_bb_pos[..., :3, :].reshape(-1, 3).cpu().numpy()
            _, sample_length, _ = trans_1.shape
            sample_dirs = [os.path.join(
                self.inference_dir, target, f'sample_{str(sample_id)}')
                for sample_id in sample_ids]
        else: # unconditional
            sample_length = batch['num_res'].item()
            true_bb_pos = None
            sample_dirs = [os.path.join(
                self.inference_dir, f'length_{sample_length}', f'sample_{str(sample_id)}')
                for sample_id in sample_ids]
            trans_1 = rotmats_1 = diffuse_mask = None
            diffuse_mask = torch.ones(1, sample_length, device=device)


        for i in range(num_batch):
            sample_dir = sample_dirs[i]
            
            os.makedirs(sample_dir, exist_ok=True)
            if 'aatype' in batch:
                aatype = du.to_numpy(batch['aatype'].long())[0]
            else:
                aatype = np.zeros(sample_length, dtype=int)

            start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            start_sample_time = time.time()

            prot_traj, atom37_traj, model_traj, _ = interpolant.sample(
                1, sample_length, self.model,
                trans_1=trans_1, rotmats_1=rotmats_1, diffuse_mask=diffuse_mask,
                record_traj=False,
            )



            finish_sample_time = time.time()
            sample_elapsed_time = finish_sample_time - start_sample_time
            allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)  
            reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)   

            bb_trajs = du.to_numpy(torch.stack(atom37_traj, dim=0).transpose(0, 1))
            bb_traj = bb_trajs[0]
            bb5_traj = np.ascontiguousarray(bb_traj[:, :, :5, :])
            traj_matrix = torch.from_numpy(
                bb5_traj.reshape(bb5_traj.shape[0], -1)
            )
            torch.save(
                traj_matrix,
                os.path.join(sample_dir, 'traj.pt'),
            )

            traj_paths = eu.save_traj(
                bb_traj[-1],
                bb_traj[0],
                np.flip(du.to_numpy(torch.concat(model_traj, dim=0)), axis=0),
                du.to_numpy(diffuse_mask)[0],
                output_dir=sample_dir,
                aatype=aatype,
            )

            if self._infer_cfg.samples.seq_per_sample > 0:
                pdb_path = traj_paths['sample_path']
                sc_output_dir = os.path.join(sample_dir, 'self_consistency')
                os.makedirs(sc_output_dir, exist_ok=True)
                shutil.copy(pdb_path, os.path.join(
                    sc_output_dir, os.path.basename(pdb_path)))
                _ = self.run_self_consistency(
                    sc_output_dir,
                    pdb_path,
                    motif_mask=None
                )
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                self._print_logger.info(f'Done sampling sample {i}: {pdb_path}, allocated {allocated:.2f}GB, reserved {reserved:.2f}GB')
                finish_eval_time = time.time()
                eval_time = finish_eval_time - finish_sample_time
                total_time = finish_eval_time - start_sample_time
                self._time_records = pd.concat(
                    [self._time_records, pd.DataFrame([[sample_length, pdb_path, start_time, sample_elapsed_time, eval_time, total_time, allocated, reserved]], 
                                                    columns=self._time_records.columns)],
                    ignore_index=True
                )
                torch.cuda.empty_cache() 
            else:
                pdb_path = traj_paths['sample_path']
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                finish_eval_time = time.time()
                eval_time = None
                total_time = finish_eval_time - start_sample_time
                self._time_records = pd.concat(
                    [self._time_records, pd.DataFrame([[sample_length, pdb_path, start_time, sample_elapsed_time, eval_time, total_time, allocated, reserved]], 
                                                    columns=self._time_records.columns)],
                    ignore_index=True
                )
                torch.cuda.empty_cache() 

    def on_predict_epoch_end(self):
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = self.global_rank
            gathered_time_records = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(gathered_time_records, self._time_records)

            if rank == 0:
                # merge
                all_time_records = pd.concat(gathered_time_records, ignore_index=True)
                all_time_records['start_time'] = pd.to_datetime(all_time_records['start_time'])
                all_time_records.sort_values(by='start_time', inplace=True)
                csv_file_path = os.path.join(self.inference_dir, 'time_records.csv')
                if os.path.exists(csv_file_path):
                    all_time_records.to_csv(csv_file_path, mode='a', header=False, index=False)
                else:
                    all_time_records.to_csv(csv_file_path, index=False)
        else:
            csv_file_path = os.path.join(self.inference_dir, 'time_records.csv')
            if os.path.exists(csv_file_path):
                self._time_records.to_csv(csv_file_path, mode='a', header=False, index=False)
            else:
                self._time_records.to_csv(csv_file_path, index=False)

    def run_self_consistency(
            self,
            decoy_pdb_dir: str,
            reference_pdb_path: str,
            motif_mask: Optional[np.ndarray]=None):
        """Run self-consistency on design proteins against reference protein.
        
        Args:
            decoy_pdb_dir: directory where designed protein files are stored.
            reference_pdb_path: path to reference protein file
            motif_mask: Optional mask of which residues are the motif.

        Returns:
            Writes ProteinMPNN outputs to decoy_pdb_dir/seqs
            Writes ESMFold outputs to decoy_pdb_dir/esmf
            Writes results in decoy_pdb_dir/sc_results.csv
        """

        # Run PorteinMPNN
        self._print_logger.info(f'Running ProteinMPNN on {decoy_pdb_dir}')
        output_path = os.path.join(decoy_pdb_dir, "parsed_pdbs.jsonl")
        #* parse different chains in the pdb file, and save the results to a jsonl file
        process = subprocess.Popen([
            'python',
            f'{self._infer_cfg.pmpnn_dir}/helper_scripts/parse_multiple_chains.py',
            f'--input_path={decoy_pdb_dir}',
            f'--output_path={output_path}',
        ])
        _ = process.wait()
        num_tries = 0
        ret = -1
        pmpnn_args = [
            'python',
            f'{self._infer_cfg.pmpnn_dir}/protein_mpnn_run.py',
            '--out_folder',
            decoy_pdb_dir,
            '--jsonl_path',
            output_path,
            '--num_seq_per_target',
            str(self._infer_cfg.samples.seq_per_sample),
            '--sampling_temp',
            '0.1',
            '--seed',
            '38',
            '--batch_size',
            '1',
        ]
        gpu_id = torch.cuda.current_device()
        pmpnn_args.append('--device')
        pmpnn_args.append(str(gpu_id))

        while ret < 0:
            try:
                process = subprocess.Popen(
                    pmpnn_args,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT
                )
                ret = process.wait()
            except Exception as e:
                num_tries += 1
                self._print_logger.info(f'Failed ProteinMPNN. Attempt {num_tries}/5')
                torch.cuda.empty_cache()
                if num_tries > 4:
                    raise e
        mpnn_fasta_path = os.path.join(
            decoy_pdb_dir,
            'seqs',
            os.path.basename(reference_pdb_path).replace('.pdb', '.fa')
        )

        # Run ESMFold on each ProteinMPNN sequence and calculate metrics.
        self._print_logger.info(f'Running ESMFold on {mpnn_fasta_path}')
        mpnn_results = {
            'tm_score': [],
            'sample_path': [],
            'header': [],
            'sequence': [],
            'rmsd': [],
        }
        if motif_mask is not None:
            # Only calculate motif RMSD if mask is specified.
            mpnn_results['motif_rmsd'] = []
        esmf_dir = os.path.join(decoy_pdb_dir, 'esmf')
        os.makedirs(esmf_dir, exist_ok=True)
        fasta_seqs = fasta.FastaFile.read(mpnn_fasta_path)
        sample_feats = du.parse_pdb_feats('sample', reference_pdb_path)
        for i, (header, string) in enumerate(fasta_seqs.items()):

            # Run ESMFold
            esmf_sample_path = os.path.join(esmf_dir, f'sample_{i}.pdb')
            _ = self.run_folding(string, esmf_sample_path)
            esmf_feats = du.parse_pdb_feats('folded_sample', esmf_sample_path)
            sample_seq = du.aatype_to_seq(sample_feats['aatype'])

            # Calculate scTM of ESMFold outputs with reference protein
            _, tm_score = metrics.calc_tm_score(
                sample_feats['bb_positions'], esmf_feats['bb_positions'],
                sample_seq, sample_seq)
            rmsd = metrics.calc_aligned_rmsd(
                sample_feats['bb_positions'], esmf_feats['bb_positions'])
            if motif_mask is not None:
                sample_motif = sample_feats['bb_positions'][motif_mask]
                of_motif = esmf_feats['bb_positions'][motif_mask]
                motif_rmsd = metrics.calc_aligned_rmsd(
                    sample_motif, of_motif)
                mpnn_results['motif_rmsd'].append(motif_rmsd)
            mpnn_results['rmsd'].append(rmsd)
            mpnn_results['tm_score'].append(tm_score)
            mpnn_results['sample_path'].append(esmf_sample_path)
            mpnn_results['header'].append(header)
            mpnn_results['sequence'].append(string)

        # Save results to CSV
        csv_path = os.path.join(decoy_pdb_dir, 'sc_results.csv')
        mpnn_results = pd.DataFrame(mpnn_results)
        mpnn_results.to_csv(csv_path)

    def run_folding(self, sequence, save_path):
        """Run ESMFold on sequence."""
        self._folding_model = self._folding_model.to(self.device)
        with torch.no_grad():
            output = self._folding_model.infer_pdb(sequence)

        with open(save_path, "w") as f:
            f.write(output)
        return output
