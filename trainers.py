from collections import defaultdict
import importlib
from os import environ
from functools import partial

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from torch.optim.sgd import SGD

from transformers import get_cosine_schedule_with_warmup
import numpy as np
from transformers import AutoModelForCausalLM
from trl.trainer.utils import DPODataCollatorWithPadding
from accelerate import Accelerator
from tqdm import tqdm

from ZOPrO.ZO.custom_datasets import PrefDataset
from utils import pad_to_length

from typing import Dict, List, Union


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def __call__(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def simple_data_collator(tokenizer, batch):
    return_dict = {}
    labels = [x.pop("labels") if "labels" in x else None for x in batch]
    return_dict["labels"] = labels
    for key in batch[0]:
        values = {"input_ids": [x[key] for x in batch]}
        return_dict[key] = tokenizer.pad(values, return_tensors="pt")["input_ids"]

    return return_dict


class TrainerZO:
    def __init__(
        self,
        model2train,
        tokenizer,
        train_data: Dataset = None,
        eval_data: Dataset = None,
        output_dir=None,
        step_accum=None,
        batch_size=16,
        lr=1e-4,
        weight_decay=0.0,
        epochs=10,
        zo_eps=1e-3,
        use_wandb=False,
        use_cpu=False,
        optimizer = 'SGD'
    ) -> None:
        assert (
            train_data is not None or eval_data is not None
        ), "At least one of train_data or eval_data must be provided."

        self.config = {
            "step_accum": step_accum,
            "weight_decay": weight_decay,
            "zo_eps": zo_eps,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "optimizer" : optimizer,
        }
        self.output_dir = output_dir
        self.accelerator = Accelerator(mixed_precision="bf16", cpu=use_cpu)
        #self.accelerator = Accelerator(mixed_precision="fp16", cpu=use_cpu)
        self.device = self.accelerator.device
        # distributes model if needed
        self.model2train = self.accelerator.prepare(model2train)
        self.train_data = self.accelerator.prepare(
            DataLoader(
                train_data,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=partial(simple_data_collator, tokenizer),
            )
        )
        self.eval_data = self.accelerator.prepare(
            DataLoader(eval_data, batch_size=1, shuffle=True)
        )
        self.step_accum = step_accum if step_accum is not None else 1
        self.weight_decay = weight_decay
        self.zo_eps = zo_eps
        self.epochs = epochs
        self.wandb = (
            importlib.import_module("wandb")
            if use_wandb and self.accelerator.is_main_process
            else None
        )

        # no dropout, see https://github.com/princeton-nlp/MeZO
        self.model2train.eval()

        if train_data is not None:
            # we don't need an optimizer but lr scheduler does
            # so we create a fake optimizer
            mock_optim = SGD(model2train.parameters(), lr=lr)
            self.scheduler = get_cosine_schedule_with_warmup(
                mock_optim,
                100,  # warmup steps
                epochs * len(train_data) / self.step_accum,
            )

        if self.wandb is not None:
            # init wandb project if available
            self.wandb.init(
                project="ZOPrO",
                name=environ["RUN_NAME"] if "RUN_NAME" in environ else None,
                # config=config,
            )
            # config = wandb.config
            self.wandb.define_metric("Train/Step")
            self.wandb.define_metric("Train/*", step_metric="Train/Step")
            self.wandb.define_metric("Val/Step")
            self.wandb.define_metric("Val/*", step_metric="Val/Step")



    def save_checkpoint(self, epoch, step=0, random_seed=None):
        # saves model and training state
        if self.output_dir is None:
            return
        model2save = self.accelerator.unwrap_model(self.model2train)
        model2save.save_pretrained(self.output_dir)
        save_dict = {
            "config": self.config,
            "lr_scheduler": self.scheduler.state_dict(),
        }
        save_dict["epoch"] = epoch
        assert not ((step == 0) ^ (random_seed is None)), "Either pass both or none."
        if random_seed is not None:
            save_dict["random_seed"] = random_seed
            save_dict["step"] = step
        else:
            save_dict["epoch"] += 1  # end of epoch
        torch.save(
            save_dict,
            f"{self.output_dir}/training_state.pt",
        )

    def load_checkpoint(self):
        # loads model and training state
        if self.output_dir is None:
            return 0, 0, None
        self.model2train = AutoModelForCausalLM.from_pretrained(self.output_dir)
        checkpoint = torch.load(f"{self.output_dir}/training_state.pt")
        self.scheduler.load_state_dict(checkpoint["lr_scheduler"])
        self.config = checkpoint["config"]  # FIXME: currently config is not used
        return (
            checkpoint["epoch"],
            checkpoint.get("step", 0),
            checkpoint.get("random_seed", None),
        )

    def perturb_params(self, random_seed, scaling_factor=1):
        # perturb the parameters of model2train
        torch.manual_seed(random_seed)

        for param in self.model2train.parameters():
            if not param.requires_grad:
                continue

            z = torch.normal(
                mean=0,
                std=1,
                size=param.data.size(),
                device=param.data.device,
                dtype=param.data.dtype,
            )
            z *= self.zo_eps
            delta_param = torch.zeros_like(z)
            param.data = param.data + scaling_factor * (z + delta_param)

    def train(self, load_checkpoint=False):
        if load_checkpoint:
            epoch_0, step_0, random_seed_0 = self.load_checkpoint()
        else:
            epoch_0, step_0, random_seed_0 = 0, 0, None
        for epoch in tqdm(range(epoch_0, self.epochs), desc="Epoch Progress"):
            # if self.device != "cpu":
            #     self.eval_epoch(epoch)  # evaluate before training (after partial merging)
            self.train_epoch(epoch, step_0, random_seed_0)
            self.save_checkpoint(epoch)
        self.eval_epoch(epoch)

    # no need to accumulate gradients as we don't have a backward pass
    @torch.no_grad()
    def train_epoch(self, epoch, step_0, random_seed_0):
        accumulated_loss = 0.0
        # Sample the random seed for sampling z
        zo_random_seed = (
            random_seed_0
            if random_seed_0 is not None
            else np.random.randint(1000000000)
        )

        for step, data in enumerate(self.train_data):
            if step < step_0:
                continue
            inputs = data["input_ids"].to(self.device)
            labels = torch.clone(inputs)
            labels[labels == 0] = -100

            accumulated_loss += self.zo_forward(
                inputs, labels, zo_random_seed
            )

            if (step + 1) % self.step_accum == 0:
                self.zo_step(zo_random_seed, accumulated_loss,step+1)
                accumulated_loss = 0.0

            if (step + 1) % 100 == 0:
                self.save_checkpoint(epoch, step, zo_random_seed)

    @torch.inference_mode()
    def eval_epoch(self, epoch):
        accumulated_loss = 0.0
        samples_count = 0
        for step, data in enumerate(self.eval_data):
            inputs = data["input_ids"].to(self.device)
            labels = torch.clone(inputs)
            labels[labels == 0] = -100
            loss = self.model2train(inputs, labels=labels)["loss"]
            accumulated_loss += loss.item()
            samples_count += 1

            if (step + 1) % 10 == 0:
                if self.wandb is not None:
                    wandb_log = {"Val/Loss": loss.item()}
                else:
                    print(f"Step {step+1}: {accumulated_loss}")

        if self.wandb is None:
            print(f"Loss: {accumulated_loss / samples_count}")
        else:
            wandb_log["Val/Step"] = epoch + (step + 1) / len(self.eval_data)
            wandb_log["Val/Final_loss"] = accumulated_loss / samples_count
            self.wandb.log(wandb_log)

    def zo_forward(self, inputs, labels, zo_random_seed):
        self.perturb_params(zo_random_seed, 1)
        loss1 = self.model2train(inputs, labels=labels)["loss"]

        self.perturb_params(zo_random_seed, -2)
        loss2 = self.model2train(inputs, labels=labels)["loss"]

        self.perturb_params(zo_random_seed, 1)
        return ((loss1 - loss2) / (2 * (self.zo_eps))).item(), (
            loss1.item() + loss2.item() / 2
        )


    # this is zo_update from MeZO
    def zo_step(self, zo_random_seed, projected_grad, step,eps=1e-8, beta1=0.9, beta2=0.999):
        """
        Update the parameters with the estimated gradients.
        """
        if torch.isnan(torch.tensor(projected_grad)) or torch.isinf(
            torch.tensor(projected_grad)
        ):
            print("Projected grad is NaN or Inf")
            return

        # Reset the random seed for sampling zs
        torch.manual_seed(zo_random_seed)

        for name, param in self.model2train.named_parameters():
            if not param.requires_grad:
                continue

            # Resample z
            z = torch.normal(
                mean=0,
                std=1,
                size=param.data.size(),
                device=param.data.device,
                dtype=param.data.dtype,
            )

            grad_est = projected_grad * self.zo_eps * z

            if self.config['optimizer'] == 'SGD':
                if (
                    "bias" not in name
                    and "layer_norm" not in name
                    and "layernorm" not in name
                ):
                    param.data = param.data - self.scheduler.get_last_lr()[0] * (
                            grad_est + self.weight_decay * param.data
                    )
                else:
                    param.data = param.data - self.scheduler.get_last_lr()[0] * grad_est

            elif self.config['optimizer'] == 'ADAM':

                if not hasattr(param, 'm'):
                    param.m = torch.zeros_like(param.data)
                if not hasattr(param, 'v'):
                    param.v = torch.zeros_like(param.data)

                param.m = beta1 * param.m + (1 - beta1) * grad_est
                param.v = beta2 * param.v + (1 - beta2) * grad_est ** 2

                m_hat = param.m / (1 - beta1 ** step)
                v_hat = param.v / (1 - beta2 ** step)

                if (
                        "bias" not in name
                        and "layer_norm" not in name
                        and "layernorm" not in name
                ):
                    param.data = param.data - self.scheduler.get_last_lr()[0] * (
                            m_hat / (v_hat.sqrt() + eps) + self.weight_decay * param.data
                    )
                else:
                    param.data = param.data - self.scheduler.get_last_lr()[0] * (
                            m_hat / (v_hat.sqrt() + eps)
                    )

            elif self.config['optimizer'] == 'RMSprop':

                if not hasattr(param, 'v'):
                    param.v = torch.zeros_like(param.data)

                param.v = 0.9 * param.v + 0.1 * grad_est ** 2
                if (
                        "bias" not in name
                        and "layer_norm" not in name
                        and "layernorm" not in name
                ):
                    param.data = param.data - self.scheduler.get_last_lr()[0] * (
                            grad_est / (param.v.sqrt() + eps) + self.weight_decay * param.data
                    )
                else:
                    param.data = param.data - self.scheduler.get_last_lr()[0] * (
                            grad_est / (param.v.sqrt() + eps)
                    )


    
        self.scheduler.step()
#TrainerZO and TrainerZOPrime are for autoregression, I need to create another one class for classifcation.
#TrainerZOPrime for classification
class TrainerZOPrime(TrainerZO):
    def __init__(self,
                 model2train,
                 tokenizer,
                 num_labels,
                 task_name,
                 **kwargs):
        super().__init__(model2train, tokenizer, **kwargs)
        #self.model2train.classifier = nn.Linear(self.model2train.config.hidden_size, num_labels).to(self.device)
        self.num_labels = num_labels
        self.task_name = task_name
        self.model2train = model2train.to(self.device)
        if self.task_name == "hh":

            self.train_data = self.accelerator.prepare(
                DataLoader(
                    kwargs["train_data"],
                    batch_size=self.config["batch_size"],
                    shuffle=True,
                    collate_fn=partial(simple_data_collator, tokenizer),
                )
            )
            self.eval_data = self.accelerator.prepare(
                DataLoader(kwargs["eval_data"], batch_size=1, shuffle=True)
            )

        else:

            self.train_data = kwargs["train_data"]
            self.eval_data = kwargs["eval_data"]

    @torch.no_grad()
    def zo_forward(self, model, inputs):
        model.eval()

        device = next(model.parameters()).device
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)

        outputs = model(**inputs)
        loss = outputs[0].float()
        return loss.detach()

    @torch.no_grad()
    def train_epoch(self, epoch, step_0, random_seed_0):

        accumulated_loss = 0.0
        zo_random_seed = (
            random_seed_0
            if random_seed_0 is not None
            else np.random.randint(1000000000)
        )

        for step, data in enumerate(tqdm(self.train_data, desc=f"Training Epoch {epoch+1}/{self.epochs}")):

            input = {k: v.to(self.device) for k, v in data.items()}

            self.perturb_params(zo_random_seed, scaling_factor=1)
            model = self.model2train.to(self.device)
            loss1 = self.zo_forward(model, input)

            self.perturb_params(zo_random_seed, scaling_factor=-2)
            loss2 = self.zo_forward(model, input)

            projected_grad = ((loss1 - loss2) / (2 * self.zo_eps)).item()
            accumulated_loss += (loss1.item() - loss2.item()) / 2

            #model_dtype = next(self.model2train.parameters()).dtype

            self.perturb_params(zo_random_seed, scaling_factor=1)

            if (step + 1) % self.step_accum == 0:
                self.zo_step(zo_random_seed, projected_grad, step + 1)
                accumulated_loss = 0.0

            if (step + 1) % 100 == 0:
                self.save_checkpoint(epoch, step, zo_random_seed)



    @torch.inference_mode()
    def eval_epoch(self, epoch):
        accumulated_loss = 0.0
        samples_count = 0.0
        correct_count = 0.0

        model = self.model2train.to(self.device)
        model.eval()

        for step, data in enumerate(self.eval_data):

            input = {k: v.to(self.device) for k, v in data.items()}
            loss, logits = model(**input)

            accumulated_loss += loss.float().item()
            if self.num_labels > 1:
                preds = torch.argmax(logits, axis=1)
            elif self.num_labels == 1:
                preds = logits.squeeze()

            labels = input["labels"]
            correct_predictions = (preds == labels).sum().item()
            total_samples = len(labels)

            correct_count += correct_predictions
            samples_count += total_samples

            if (step + 1) % 10 == 0:
                if self.wandb is not None:
                    wandb_log = {"Val/Loss": loss.item()}
                else:
                    print(f"Step {step + 1}: {accumulated_loss}")

        final_loss = accumulated_loss / samples_count
        acc = correct_count / samples_count

        if self.wandb is None:
            print(f"correct_predictions: {correct_count}")
            print(f"total_samples: {samples_count}")
            print(f"Final Loss: {final_loss}")
            print(f"Final Accuracy: {acc}")
        else:
            wandb_log["Val/Step"] = epoch + (step + 1) / len(self.eval_data)
            wandb_log["Val/Final_loss"] = final_loss
            wandb_log["Val/Accuracy"] = acc
            self.wandb.log(wandb_log)


# MeZO with gradient clipping
class TrainerZOClipGrad(TrainerZO):
    def __init__(self, grad_clip=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config["grad_clip"] = grad_clip

    def zo_step(self, zo_random_seed, projected_grad):
        """
            compute the gradient norm clip and call super method
        """
        if torch.isnan(torch.tensor(projected_grad)) or torch.isinf(
            torch.tensor(projected_grad)
        ):
            print("Projected grad is NaN or Inf")
            return

        # Reset the random seed for sampling zs
        torch.manual_seed(zo_random_seed)

        clip_coef = 1  # no clipping by default

        # if clipping get grad norm
        if self.config["grad_clip"]:
            norms = []
            for param in self.model2train.parameters():
                if not param.requires_grad:
                    continue

                # Resample z
                z = torch.normal(
                    mean=0,
                    std=1,
                    size=param.data.size(),
                    device=param.data.device,
                    dtype=param.data.dtype,
                )

                norms.append(
                    torch.linalg.vector_norm(
                        projected_grad * self.zo_eps * z, 2
                    )
                )

            total_norm = torch.linalg.vector_norm(
                torch.stack([norm.to(self.device) for norm in norms]), 2
            )
            max_norm = self.config["grad_clip"]
            clip_coef = max_norm / (total_norm + 1e-6)
            del norms  # free memory

        return super().zo_step(zo_random_seed, projected_grad * clip_coef)


def concatenated_inputs(batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
    """Concatenate the chosen and rejected inputs into a single tensor.

    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

    Returns:
        A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
    """
    max_length = max(batch['chosen_input_ids'].shape[1], batch['rejected_input_ids'].shape[1])
    concatenated_batch = {}
    for k in batch:
        if k.startswith('chosen') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('chosen', 'concatenated')
            concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
    for k in batch:
        if k.startswith('rejected') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('rejected', 'concatenated')
            concatenated_batch[concatenated_key] = torch.cat((
                concatenated_batch[concatenated_key],
                pad_to_length(batch[k], max_length, pad_value=pad_value),
            ), dim=0)
    return concatenated_batch







# this implements MeZO SGD with DPO in the naive way
class TrainerZODPO(TrainerZO):
    def __init__(
        self,
        model_ref,
        train_data: PrefDataset,
        eval_data: PrefDataset,
        dpo_beta: float,
        batch_size: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            train_data=train_data,
            eval_data=eval_data,
            *args,
            **kwargs,
        )
        self.train_data = self.accelerator.prepare(
            DataLoader(
                dataset=train_data,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=DPODataCollatorWithPadding(),
            )
        )
        self.eval_data = self.accelerator.prepare(
            DataLoader(
                dataset=eval_data,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=DPODataCollatorWithPadding(),
            )
        )
        self.beta = dpo_beta

        # we also need the ref model for DPO
        self.model_ref = self.accelerator.prepare(model_ref)
        if self.accelerator.is_main_process:
            # high delta as val loss should not improve much but we don't want it to diverge
            self.early_stopper = EarlyStopper(min_delta=0.5)

    # NOTE: compared to train_epoch in TrainerZO, this train_epoch does evaluation throughout the training (rather than only at the end)
    @torch.no_grad()
    def train_epoch(self, epoch, step_0, random_seed_0):
        accumulated_loss = 0.0
        accumulated_grad = 0.0

        # Sample the random seed for sampling z
        zo_random_seed = (
            random_seed_0
            if random_seed_0 is not None
            else np.random.randint(1000000000)
        )
        eval_interval = max(
            1, len(self.train_data) // 100
        )  # Evaluate every 1% of the training data

        for step, data in enumerate(tqdm(self.train_data)):
            if step < step_0:
                continue

            inputs_pos = data["pos_input_ids"].to(self.device)
            labels_pos = data["pos_labels"].to(self.device)
            labels_pos[labels_pos == 0] = -100

            inputs_neg = data["neg_input_ids"].to(self.device)
            labels_neg = data["neg_labels"].to(self.device)
            labels_neg[labels_neg == 0] = -100

            grad, loss = self.zo_dpo_forward(
                inputs_pos, inputs_neg, labels_pos, labels_neg, zo_random_seed
            )
            accumulated_grad += grad / self.step_accum
            accumulated_loss += loss

            if (step + 1) % self.step_accum == 0:
                if self.wandb is not None:
                    wandb_log = {
                        "Train/Loss": accumulated_loss / self.step_accum,
                        "Train/Step": epoch + step / len(self.train_data),
                    }
                    self.wandb.log(wandb_log)
                self.zo_step(None, zo_random_seed, accumulated_grad)
                accumulated_loss = 0.0
                accumulated_grad = 0.0

            if (step + 1) % 100 == 0:
                self.save_checkpoint(epoch, step, zo_random_seed)

            if (step + 1) % eval_interval == 0:
                self.eval_step(epoch, step - eval_interval, step)

    @torch.inference_mode()
    def eval_step(self, epoch, step_0, step):
        accumulated_loss = defaultdict(float)
        samples_count = defaultdict(int)
        for eval_step, data in enumerate(self.eval_data):
            if eval_step < step_0 // max(
                1, len(self.train_data) // len(self.eval_data)
            ):
                continue
            if eval_step > step // max(
                1, len(self.train_data) // len(self.eval_data)
            ):  # Evaluate only a proportion
                break

            batch_loss = 0.0
            inputs_pos = data["pos_input_ids"].to(self.device)
            labels_pos = data["pos_labels"].to(self.device)
            labels_pos[labels_pos == 0] = -100
            loss = self.model2train(inputs_pos, labels=labels_pos)["loss"]
            batch_loss += loss.item() / 2
            accumulated_loss["loss_pos"] += loss.item()
            samples_count["loss_pos"] += 1

            inputs_neg = data["neg_input_ids"].to(self.device)
            labels_neg = data["neg_labels"].to(self.device)
            labels_neg[labels_neg == 0] = -100
            loss = self.model2train(inputs_neg, labels=labels_neg)["loss"]
            batch_loss += loss.item() / 2
            accumulated_loss["loss_neg"] += loss.item()
            samples_count["loss_neg"] += 1

            if self.wandb is not None and (eval_step + 1) % 10 == 0:
                wandb_log = {
                    f"Val/{key}": accumulated_loss[key] / samples_count[key]
                    for key in accumulated_loss.keys()
                }
                wandb_log["Val/all_loss"] = batch_loss
                wandb_log["Val/Step"] = epoch + (step + 1) / len(self.eval_data)
                self.wandb.log(wandb_log)
            else:
                print(f"Eval Step {eval_step + 1}: {accumulated_loss}")

        if (
            self.accelerator.is_main_process
            and sum(samples_count.values()) > 0
            and self.early_stopper(
                sum(accumulated_loss.values()) / sum(samples_count.values())
            )
        ):
            # stop
            print("Early stopping", flush=True)
            exit(1)

    # eval epoch is not needed as we do eval in train_epoch
    @torch.inference_mode()
    def eval_epoch(self, epoch):
        return

    def zo_dpo_forward(
        self, inputs_pos, inputs_neg, labels_pos, labels_neg, zo_random_seed
    ):
        pos_forward_ref = self.get_logps(
            self.model_ref(inputs_pos)["logits"], labels_pos
        )
        neg_forward_ref = self.get_logps(
            self.model_ref(inputs_neg)["logits"], labels_neg
        )
        ref_ratios = pos_forward_ref - neg_forward_ref
        self.perturb_params(None, zo_random_seed, 1)
        loss1 = self.dpo_loss(
            inputs_pos, inputs_neg, labels_pos, labels_neg, ref_ratios
        )

        self.perturb_params(None, zo_random_seed, -2)
        loss2 = self.dpo_loss(
            inputs_pos, inputs_neg, labels_pos, labels_neg, ref_ratios
        )

        self.perturb_params(None, zo_random_seed, 1)
        return ((loss1 - loss2) / (2 * (self.zo_eps))).item(), (
            loss1.item() + loss2.item() / 2
        )

    def dpo_loss(self, inputs_pos, inputs_neg, labels_pos, labels_neg, ref_ratios):
        # Eq. 22 in https://arxiv.org/pdf/2305.18290
        pos_forward_theta = self.get_logps(self.model(inputs_pos)["logits"], labels_pos)
        neg_forward_theta = self.get_logps(self.model(inputs_neg)["logits"], labels_neg)

        theta_ratios = pos_forward_theta - neg_forward_theta
        logits = theta_ratios - ref_ratios

        loss = -F.logsigmoid(self.beta * logits).mean()
        return loss

    # Credits: https://github.com/eric-mitchell/direct-preference-optimization/blob/main/trainers.py#L90
    def get_logps(self, logits, labels, average_log_prob=False):
        assert logits.shape[:-1] == labels.shape

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = (labels != -100)

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == -100] = 0

        per_token_logps = torch.gather(
            logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
        ).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)


# This implements MeZO SGD with DPO and the gradient trick
class TrainerZODPOPrime(TrainerZODPO):
    def zo_dpo_forward(
        self, inputs_pos, inputs_neg, labels_pos, labels_neg, zo_random_seed
    ):
        pos_grad, _ = self.zo_forward(
            inputs_pos, labels_pos, zo_random_seed=zo_random_seed
        )
        neg_grad, _ = self.zo_forward(
            inputs_neg, labels_neg, zo_random_seed=zo_random_seed
        )

        # Eq. 22 in https://arxiv.org/pdf/2305.18290
        grad = pos_grad - neg_grad
        res_theta_pos = self.model2train(inputs_pos, labels=labels_pos)
        pos_forward_theta = self.get_logps(res_theta_pos["logits"], labels_pos)
        neg_forward_theta = self.get_logps(self.model2train(inputs_neg)["logits"], labels_neg)
        pos_forward_ref = self.get_logps(
            self.model_ref(inputs_pos)["logits"], labels_pos
        )
        neg_forward_ref = self.get_logps(
            self.model_ref(inputs_neg)["logits"], labels_neg
        )

        theta_ratios = pos_forward_theta - neg_forward_theta
        ref_ratios = pos_forward_ref - neg_forward_ref
        logits = theta_ratios - ref_ratios

        loss = -F.logsigmoid(self.beta * logits).mean()
        grad *= self.beta * loss
        return grad.item(), res_theta_pos["loss"].item()
