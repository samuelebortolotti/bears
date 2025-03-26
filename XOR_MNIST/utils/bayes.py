# Bayes module
import math
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import wandb
from datasets import get_dataset
from datasets.utils.base_dataset import BaseDataset
from laplace import Laplace
from models import get_model
from models.utils.utils_problog import build_worlds_queries_matrix
from numpy import ndarray
from scipy.special import softmax
from torch.utils.data import DataLoader, Dataset
from utils import fprint
from utils.checkpoint import load_checkpoint
from utils.conf import set_random_seed
from utils.metrics import evaluate_metrics, vector_to_parameters
from utils.status import progress_bar
from warmup_scheduler import GradualWarmupScheduler


class DatasetPcX(Dataset):
    """Dataset of P(C|X), it is a dictionary dataset"""

    def __init__(self, images, pcx, wq=None):
        """Initialization method

        Args:
            self: instance
            images: images
            pcx: p(c|x)
            wq: query matrix

        Returns:
            None: This function does not return a value.
        """
        self.images = images
        self.pcx = pcx
        print("Len", len(self.images), len(self.pcx))
        self.img_to_key = self._initialize_dict(self.images)
        self.wq = wq

    def _initialize_dict(self, images):
        """Initialize dictionary dataset

        Args:
            self: instance
            images: images

        Returns:
            img_to_key: dictionary having bytes as key and index as value
        """
        img_to_key = dict()
        for i, img in enumerate(images):
            self._add_key_value(img_to_key, img, i)
        return img_to_key

    def _add_key_value(self, dictionary, tensor_key, value):
        """Add key-value to dictionary method

        Args:
            self: instance
            dictionary: dictionary method
            tensor_key: tensor as key
            value: value to put

        Returns:
            None: This function does not return a value.
        """
        tuple_key = tensor_key.detach().cpu().numpy().tobytes()
        dictionary[tuple_key] = value

    def return_value_from_key(self, tensor_key):
        """Retrieve value fom key

        Args:
            self: instance
            tensor_key: tensor as key

        Returns:
            p(c|x) of the images
        """
        tuple_key = tensor_key.detach().cpu().numpy().tobytes()
        index = self.img_to_key[tuple_key]
        return self.pcx[index]

    def _hash_mnist_image(self, image_array):
        """Hash mnist image

        Args:
            self: instance
            image_array: image array

        Returns:
            hash: sha 256 digest
        """
        import hashlib

        flattened_array = image_array.flatten()
        image_bytes = flattened_array.tobytes()
        sha256_hash = hashlib.sha256(image_bytes)

        return sha256_hash.hexdigest()

    def __len__(self):
        """Lenght of the dataset

        Args:
            self: instance

        Returns:
            dim: lenght of the dataset
        """
        return len(self.images)

    def __getitem__(self, index):
        """Retrieve p(c|x) and images from the dataset

        Args:
            self: instance
            index (int): index

        Returns:
            img: image
            pcx: probabilities
        """
        return self.images[index], self.pcx[index]


def activate_dropout(model: nn.Module):
    """Activate dropout in model

    Args:
        model (nn.Module): module

    Returns:
        None: This function does not return a value.
    """
    # enables dropout during test, useful for MC-dropout
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


def save_laplace_state(lll, filename):
    """Save Laplace state function

    Args:
        lll: laplace module
        filename: name of the file where to save things

    Returns:
        None: This function does not return a value.
    """
    # Create a dictionary containing the state of your class
    state = {
        "loss": lll.loss,
        "n_data": lll.n_data,
        "H": lll.H,
        "mean": lll.mean,
        "n_outputs": lll.n_outputs,
        "output_size": lll.model.output_size,
        "n_params": lll.n_params,
        "n_layers": lll.n_layers,
        "prior_precision": lll.prior_precision,
        "prior_mean": lll.prior_mean,
    }

    torch.save(state, filename)
    print(f"Saved state to {filename}")


def load_laplace_state(lll, train_loader, filename):
    """Load Laplace state function

    Args:
        lll: laplace module
        train_loader: train loader
        filename: name of the file where to save things

    Returns:
        None: This function does not return a value.
    """
    state = torch.load(filename)

    # Update the attributes of your class using the loaded state
    lll.loss = state["loss"]
    lll.n_data = state["n_data"]
    lll.H = state["H"]
    lll.mean = state["mean"]
    lll.n_outputs = state["n_outputs"]
    lll.n_params = state["n_params"]
    lll.n_layers = state["n_layers"]
    lll.prior_precision = state["prior_precision"]
    lll.prior_mean = state["prior_mean"]

    if lll.model.last_layer is None:
        X, _ = next(iter(train_loader))
        with torch.no_grad():
            try:
                lll.model.find_last_layer(X[:1].to(lll._device))
            except (TypeError, AttributeError):
                lll.model.find_last_layer(X.to(lll._device))


def montecarlo_dropout_single_batch(
    model: nn.Module,
    batch_samples: torch.tensor,
    num_mc_samples: int = 30,
    apply_softmax: bool = False,
) -> Tuple[ndarray, ndarray, ndarray]:
    """Montecarlo dropout run on a single batch

    Args:
        model (nn.Module): network
        batch_samples (torch.tensor): batch samples
        num_mc_samples (int, default=30): number of montecarlo dropout samples
        apply_softmax (bool, default=False): apply softmax

    Returns:
        label_probability: ensemble stacked probability for the label
        concept_logit: ensemble stacked logits for the concepts
        concept_prob: ensemble stacked probability for the concepts
    """
    model.eval()

    # activate dropout during evaluation
    activate_dropout(model)

    output_dicts = [
        model(batch_samples) for _ in range(num_mc_samples)
    ]  # 30

    label_prob = [
        out_dict["YS"].detach().cpu().numpy()
        for out_dict in output_dicts
    ]  # 30
    concept_logit = [
        out_dict["CS"].detach().cpu().numpy()
        for out_dict in output_dicts
    ]  # 30
    concept_prob = [
        out_dict["pCS"].detach().cpu().numpy()
        for out_dict in output_dicts
    ]  # 30
    ll = [
        out_dict["pCS"].detach().cpu().numpy()
        for out_dict in output_dicts
    ]  # 30

    label_prob = np.stack(label_prob, axis=0)
    concept_logit = np.stack(concept_logit, axis=0)
    concept_prob = np.stack(concept_prob, axis=0)

    if apply_softmax:
        label_prob = softmax(label_prob, axis=2)

    return label_prob, concept_logit, concept_prob


def montecarlo_dropout(
    model,
    loader,
    n_values,
    num_mc_samples: int = 30,
    apply_softmax=False,
) -> List[ndarray]:
    """Montecarlo dropout on a single loader

    Args:
        model (nn.Module): network
        loader: dataset
        n_values (int): values
        batch_samples (torch.tensor): batch samples
        num_mc_samples (int, default=30): number of montecarlo dropout samples
        apply_softmax (bool, default=False): apply softmax

    Returns:
        y_true: groundtruth labels
        gs: groundtruth concepts
        cs: predicted concepts
        gs_separated: groundtruth concepts separated
        cs_separated: predicted concepts separated
        p_cs: probability of concepts
        p_ys: probability of labels
        p_cs_full: probability of concepts full
        p_ys_full: probability of labels full
    """

    # Loop over the dataloader
    for i, data in enumerate(loader):
        images, labels, concepts = data
        images, labels, concepts = (
            images.to(model.device),
            labels.to(model.device),
            concepts.to(model.device),
        )

        # Call MC Dropout
        (
            label_prob_ens,  # (nmod, 256, 19)
            concept_logit_ens,  # (nmod, 256, 2, 10)
            concept_prob_ens,  # (nmod, 256, 2, 10)
        ) = montecarlo_dropout_single_batch(
            model, images, num_mc_samples, apply_softmax
        )

        # Concatenate the output
        if i == 0:
            y_true = labels.detach().cpu().numpy()
            c_true = concepts.detach().cpu().numpy()
            y_pred = label_prob_ens
            c_pred = concept_logit_ens
            pc_pred = concept_prob_ens
        else:
            y_true = np.concatenate(
                [y_true, labels.detach().cpu().numpy()], axis=0
            )
            c_true = np.concatenate(
                [c_true, concepts.detach().cpu().numpy()], axis=0
            )
            y_pred = np.concatenate([y_pred, label_prob_ens], axis=1)
            c_pred = np.concatenate(
                [c_pred, concept_logit_ens], axis=1
            )
            pc_pred = np.concatenate(
                [pc_pred, concept_prob_ens], axis=1
            )

    # Compute the final arrangements
    gs = np.split(
        c_true, c_true.shape[1], axis=1
    )  # splitted groundtruth concepts
    cs = np.split(
        c_pred, c_pred.shape[2], axis=2
    )  # splitted concepts # (nmod, data, 10, 10)
    p_cs = np.split(
        pc_pred, pc_pred.shape[2], axis=2
    )  # splitted concept prob # (nmod, data, 10, 10)

    assert len(gs) == len(cs), f"gs: {gs.shape}, cs: {cs.shape}"

    # [#data, 1]
    # gs = np.concatenate(gs, axis=0).squeeze(1)
    # Print some information for debugging
    gs = np.char.add(gs[0].astype(str), gs[1].astype(str))
    gs = np.where(gs == "-1-1", -1, gs)
    gs = gs.squeeze(-1).astype(int)

    p_cs_1 = np.expand_dims(
        p_cs[0].squeeze(2), axis=-1
    )  # 30, 256, 10, 1
    p_cs_2 = np.expand_dims(
        p_cs[1].squeeze(2), axis=-2
    )  # 30, 256, 1, 10
    p_cs = np.matmul(
        p_cs_1, p_cs_2
    )  # 30, 256, 10, 10 -> # [#modelli, #data, #facts, #facts]
    p_cs = np.reshape(
        p_cs, (*p_cs.shape[:-2], p_cs.shape[-1] * p_cs.shape[-2])
    )  # -> [#modelli, #data, #facts^2]
    p_cs = np.mean(
        p_cs, axis=0
    )  # avg[#modelli, #data, #facts^2] = [#data, #facts^2]

    # mean probabilities of the output
    p_ys = np.mean(y_pred, axis=0)  # -> (256, 19)
    ys = np.argmax(p_ys, axis=1)  # predictions -> (data)

    p_ys_full = p_ys  # all the items of probabilities are considered (#data, #facts^2)
    p_cs_full = p_cs  # all the items of probabilities are considered (#data, #facts^2)
    cs = p_cs.argmax(
        axis=1
    )  # the predicted concept is the argument maximum
    p_cs = p_cs.max(
        axis=1
    )  # only the maximum one is considered (#data,)
    p_ys = p_ys.max(axis=1)  # only the maximum one is considered

    assert gs.shape == cs.shape, f"gs: {gs.shape}, cs: {cs.shape}"

    gs_separated = c_true

    # cs_separated = np.char.mod('%02d', cs)
    decimal_values = np.array([x // n_values for x in cs])
    unit_values = np.array([x % n_values for x in cs])
    cs_separated = np.column_stack((decimal_values, unit_values))

    return (
        y_true,
        gs,
        ys,
        cs,
        gs_separated,
        cs_separated,
        p_cs,
        p_ys,
        p_cs_full,
        p_ys_full,
    )


def _freeze_model_params(model):
    """Freeze model parameters

    Args:
        model (nn.Module): network

    Returns:
        None: This function does not return a value.
    """
    for param in model.parameters():
        param.requires_grad = False


class EarlyStopper:
    """Early Stopper class"""

    def __init__(self, patience=1, min_delta=0):
        """Initialize

        Args:
            self: instance
            patience (int, default=1): patience
            min_delta (int, default=0): minimum delta for early stopper

        Returns:
            None: This function does not return a value.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")
        self.best_weights = None
        self.stuck = False

    def early_stop(self, model, validation_loss):
        """Early stopper, stops if the loss is stuck or does not improve

        Args:
            self: instance
            model (nn.Module): model
            validation_loss: validation loss

        Returns:
            stop (bool): whether to stop or not the model's execution
        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.best_weights = model.state_dict()
            self.stuck = False
        elif validation_loss > (
            self.min_validation_loss + self.min_delta
        ):
            self.counter += 1
            if self.counter >= self.patience:
                model.load_state_dict(self.best_weights)
                return True
            self.stuck = False
        elif validation_loss == self.min_validation_loss:
            if self.stuck:
                self.counter += 1
                if self.counter >= self.patience:
                    model.load_state_dict(self.best_weights)
                    return True
            self.stuck = True
        return False


def deep_ensemble_active(
    seeds: List[int],
    base_model: nn.Module,
    dataset: BaseDataset,
    num_epochs: int,
    args,
    val_loader,
    epsilon=0.01,
    separate_from_others: bool = False,
    lambda_h=10,
    use_wandb=False,
    n_facts=10,
    knowledge_aware_kl: bool = False,
    real_kl: bool = False,
    supervision: List = [],
    weights_base_model="",
):
    """Deep Ensemble version for Kandinsky active learning

    Args:
        seeds (List[int]): seed lists,
        base_model (nn.Module): base model (first element of the ensemble)
        dataset (BaseDataset): dataset
        num_epochs (int): number of epochs as budget
        args: command line arguments
        val_loader: validation loader
        epsilon (float, default=0.01): epsilon value (for the DE term)
        separate_from_others (bool, default=False): whether to use biretta or not
        lambda_h (float, default=10): entropy weight of the biretta loss term
        use_wandb (bool, default=False): whether to use wandb
        n_facts (int, default=10): number of concepts
        knowledge_aware_kl (bool, default=False): use knowledge aware term
        real_kl (bool, default=False): use paper kl
        supervision (List, default=[]): elements to give supervision to
        weights_base_model (str, default=""): weights of the base model to load

    Returns:
        ensemble: models ensemble
    """
    from models.utils.ops import outer_product

    # lambda for kl
    lambda_h = 0.01

    def kl_paper(
        p_model: torch.tensor, p_rest: torch.tensor, k, last_hope=True
    ):
        """KL of the paper

        Args:
            p_model (torch.tensor): probability p(c|x)
            p_rest (torch.tensor): probability p(c|x) of the ensemble
            k (int): number of elements in the ensemble
            last_hope (bool, default=True): whether to use the compressed version

        Returns:
            loss: kl loss according to the paper
        """
        p_model = p_model + 1e-5
        p_model = p_model / (1 + (p_model.shape[1] * 1e-5))

        with torch.no_grad():
            p_rest = p_rest + 1e-5
            p_rest = p_rest / (1 + (p_model.shape[1] * 1e-5))

        if last_hope:
            kl_ew = torch.sum(
                p_model * torch.log(p_model + (k - 1) * p_rest), dim=1
            )
        else:
            ratio = torch.div(p_rest, p_model)
            kl_ew = torch.sum(
                p_model * torch.log(1 + (k - 1) * ratio), dim=1
            )

        return torch.mean(kl_ew, dim=0)

    dataset = get_dataset(args)

    if len(supervision) > 0:
        dataset.give_supervision_to(
            supervision[0], supervision[1], supervision[2]
        )

    # Load dataset, model, loss, and optimizer
    encoder, decoder = dataset.get_backbone()
    n_images, c_split = dataset.get_split()
    base_model = get_model(args, encoder, decoder, n_images, c_split)
    state_dict = torch.load(weights_base_model)
    base_model.load_state_dict(state_dict)
    ensemble = [base_model]

    # model index
    model_idx = 0
    c = np.load(
        f"data/kand-analysis/seed-{args.seed}-{args.dataset}-ftn-{args.finetuning}/c_pred.npy"
    )
    g = np.load(
        f"data/kand-analysis/seed-{args.seed}-{args.dataset}-ftn-{args.finetuning}/c_true.npy"
    )
    y_pred = np.load(
        f"data/kand-analysis/seed-{args.seed}-{args.dataset}-ftn-{args.finetuning}/y_pred.npy"
    )
    y_true = np.load(
        f"data/kand-analysis/seed-{args.seed}-{args.dataset}-ftn-{args.finetuning}/y_true.npy"
    )

    cs = np.split(c, 2, axis=1)
    gs = np.split(g, 2, axis=1)

    tag = ["shapes", "colors"]
    for i in range(2):
        all_c = np.concatenate(
            (cs[i][:, 0], cs[i][:, 1], cs[i][:, 2])
        )
        all_g = np.concatenate(
            (gs[i][:, 0], gs[i][:, 1], gs[i][:, 2])
        )

        if use_wandb:
            wandb.log(
                {
                    f"{args.seed}_{args.finetuning}_conf_mat_{tag[i]}/{model_idx}": wandb.plot.confusion_matrix(
                        probs=None, y_true=all_g, preds=all_c
                    )
                }
            )

    cs = np.split(c, 2, axis=1)
    gs = np.split(g, 2, axis=1)

    shapes_pred = np.concatenate(
        (cs[0][:, 0], cs[0][:, 1], cs[0][:, 2])
    )
    shapes_true = np.concatenate(
        (gs[0][:, 0], gs[0][:, 1], gs[0][:, 2])
    )

    colors_pred = np.concatenate(
        (cs[1][:, 0], cs[1][:, 1], cs[1][:, 2])
    )
    colors_true = np.concatenate(
        (gs[1][:, 0], gs[1][:, 1], gs[1][:, 2])
    )

    all_c = shapes_pred * 3 + colors_pred
    all_g = shapes_true * 3 + colors_true

    if use_wandb:
        wandb.log(
            {
                f"{args.seed}_{args.finetuning}_conf_mat_mondi/{model_idx}": wandb.plot.confusion_matrix(
                    probs=None, y_true=all_g, preds=all_c
                )
            }
        )

    for model_idx, seed in enumerate(seeds):
        model_idx = model_idx + 1

        # setting the seeds
        fprint("Training model with seed", seed, "...")

        set_random_seed(seed)

        # model load and randomize
        model = get_model(args, encoder, decoder, n_images, c_split)
        state_dict = torch.load(weights_base_model)
        model.load_state_dict(state_dict)
        criterion = model.get_loss(args)
        model.start_optim(args)

        model.to(model.device)
        train_loader, val_loader, _ = dataset.get_data_loaders()
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            model.opt, args.exp_decay
        )
        w_scheduler = GradualWarmupScheduler(
            model.opt, 1.0, args.warmup_steps
        )

        # default for warm-up
        model.opt.zero_grad()
        model.opt.step()

        for epoch in range(num_epochs):
            model.train()

            for ti, data in enumerate(train_loader):
                images, labels, concepts = data
                images, labels, concepts = (
                    images.to(model.device),
                    labels.to(model.device),
                    concepts.to(model.device),
                )

                # Generate adversarial examples using x_batch
                out_dict = model(images)
                out_dict.update(
                    {
                        "INPUTS": images,
                        "LABELS": labels,
                        "CONCEPTS": concepts,
                    }
                )

                model.opt.zero_grad()
                loss_original, _ = criterion(out_dict, args)

                c_prb = out_dict["pCS"]

                c_worlds = []
                for i in range(3):
                    list_concepts = torch.split(
                        c_prb[:, i, :], 3, dim=-1
                    )
                    p_w_image = (
                        outer_product(*list_concepts)
                        .unsqueeze(1)
                        .view(-1, 1, 3**6)
                    )

                    c_worlds.append(p_w_image)

                c_worlds = torch.cat(c_worlds, dim=1)

                # Append the new tensor [batch_size,3,729]
                model_itself_pc_x = c_worlds

                total_dist = 0

                if len(ensemble) > 0:

                    # Create the ensembles world probabilities
                    pcx_list_ensemble = list()

                    for m in ensemble:
                        out_dict = m(images)
                        # Get concept probability
                        c_prb_other = out_dict["pCS"]

                        other_c_worlds = []
                        for i in range(3):
                            list_concepts = torch.split(
                                c_prb[:, i, :], 3, dim=-1
                            )
                            p_w_image = (
                                outer_product(*list_concepts)
                                .unsqueeze(1)
                                .view(-1, 1, 3**6)
                            )

                            other_c_worlds.append(p_w_image)

                        c_prb_other = torch.cat(other_c_worlds, dim=1)

                        # other p(c|x)
                        pcx_list_ensemble.append(c_prb_other)

                    pcx_list_ensemble = torch.stack(pcx_list_ensemble)
                    other_m_pc_x_mean = torch.mean(
                        pcx_list_ensemble, dim=0
                    )

                    for i in range(3):
                        p_t = model_itself_pc_x[:, i, :]
                        p_ens = other_m_pc_x_mean[:, i, :]
                        distance = lambda_h * (
                            1
                            + kl_paper(
                                p_t, p_ens, len(ensemble) + 1, True
                            )
                            / (6 * math.log(3))
                        )  # remove last hope

                        total_dist += distance

                    if use_wandb:
                        wandb.log(
                            {
                                f"{args.seed}_{args.finetuning}_it": ti,
                                f"{args.seed}_{args.finetuning}_epoch": epoch,
                                f"{args.seed}_{args.finetuning}_loss_original": loss_original,
                                f"{args.seed}_{args.finetuning}_total_dist": total_dist,
                                f"{args.seed}_{args.finetuning}_full_loss": loss_original
                                + total_dist,
                            }
                        )

                    loss_original += total_dist

                loss_original.backward()
                model.opt.step()

                if ti % 10 == 0:
                    progress_bar(
                        ti,
                        len(train_loader) - 9,
                        epoch,
                        loss_original.item(),
                    )

            # update at end of the epoch
            if epoch < args.warmup_steps:
                w_scheduler.step()
            else:
                scheduler.step()
                if hasattr(criterion, "grade"):
                    criterion.update_grade(epoch)

        model.eval()

        # Evaluate performances on VAL
        (
            y_true,
            c_true,
            y_pred,
            c_pred,
            p_cs,
            p_ys,
            p_cs_all,
            p_ys_all,
        ) = evaluate_metrics(model, val_loader, args, last=True)

        if True:
            import os

            os.makedirs(
                f"data/kand-analysis/seed_{seed}_biretta-{args.dataset}-{model_idx}-{args.finetuning}",
                exist_ok=True,
            )

            np.save(
                f"data/kand-analysis/seed_{seed}_biretta-{args.dataset}-{model_idx}-{args.finetuning}/y_true.npy",
                y_true,
            )
            np.save(
                f"data/kand-analysis/seed_{seed}_biretta-{args.dataset}-{model_idx}-{args.finetuning}/c_true.npy",
                c_true,
            )
            np.save(
                f"data/kand-analysis/seed_{seed}_biretta-{args.dataset}-{model_idx}-{args.finetuning}/y_pred.npy",
                y_pred,
            )
            np.save(
                f"data/kand-analysis/seed_{seed}_biretta-{args.dataset}-{model_idx}-{args.finetuning}/c_pred.npy",
                c_pred,
            )
            np.save(
                f"data/kand-analysis/seed_{seed}_biretta-{args.dataset}-{model_idx}-{args.finetuning}/p_cs.npy",
                p_cs,
            )
            np.save(
                f"data/kand-analysis/seed_{seed}_biretta-{args.dataset}-{model_idx}-{args.finetuning}/p_ys.npy",
                p_ys,
            )
            np.save(
                f"data/kand-analysis/seed_{seed}_biretta-{args.dataset}-{model_idx}-{args.finetuning}/p_cs_all.npy",
                p_cs_all,
            )

            c = np.load(
                f"data/kand-analysis/seed_{seed}_biretta-{args.dataset}-{model_idx}-{args.finetuning}/c_pred.npy"
            )
            g = np.load(
                f"data/kand-analysis/seed_{seed}_biretta-{args.dataset}-{model_idx}-{args.finetuning}/c_true.npy"
            )
            y_pred = np.load(
                f"data/kand-analysis/seed_{seed}_biretta-{args.dataset}-{model_idx}-{args.finetuning}/y_pred.npy"
            )
            y_true = np.load(
                f"data/kand-analysis/seed_{seed}_biretta-{args.dataset}-{model_idx}-{args.finetuning}/y_true.npy"
            )

            cs = np.split(c, 2, axis=1)
            gs = np.split(g, 2, axis=1)

            tag = ["shapes", "colors"]
            for i in range(2):
                all_c = np.concatenate(
                    (cs[i][:, 0], cs[i][:, 1], cs[i][:, 2])
                )
                all_g = np.concatenate(
                    (gs[i][:, 0], gs[i][:, 1], gs[i][:, 2])
                )

                wandb.log(
                    {
                        f"{args.seed}_{args.finetuning}_conf_mat_{tag[i]}/{model_idx}": wandb.plot.confusion_matrix(
                            probs=None, y_true=all_g, preds=all_c
                        )
                    }
                )

            cs = np.split(c, 2, axis=1)
            gs = np.split(g, 2, axis=1)

            shapes_pred = np.concatenate(
                (cs[0][:, 0], cs[0][:, 1], cs[0][:, 2])
            )
            shapes_true = np.concatenate(
                (gs[0][:, 0], gs[0][:, 1], gs[0][:, 2])
            )

            colors_pred = np.concatenate(
                (cs[1][:, 0], cs[1][:, 1], cs[1][:, 2])
            )
            colors_true = np.concatenate(
                (gs[1][:, 0], gs[1][:, 1], gs[1][:, 2])
            )

            all_c = shapes_pred * 3 + colors_pred
            all_g = shapes_true * 3 + colors_true

            if use_wandb:
                wandb.log(
                    {
                        f"{args.seed}_{args.finetuning}_conf_mat_mondi/{model_idx}": wandb.plot.confusion_matrix(
                            probs=None, y_true=all_g, preds=all_c
                        )
                    }
                )

        ensemble.append(model)

    fprint("Done!\nTotal length of the ensemble: ", len(ensemble))

    return ensemble


def deep_ensemble(
    seeds: List[int],
    dataset: BaseDataset,
    num_epochs: int,
    args,
    val_loader,
    epsilon=0.01,
    separate_from_others: bool = False,
    lambda_h=1,  # 10,
    use_wandb=False,
    n_facts=10,
    knowledge_aware_kl: bool = False,
    real_kl: bool = False,
):
    """Deep Ensemble version for Test

    Args:
        seeds (List[int]): seed lists,
        dataset (BaseDataset): dataset
        num_epochs (int): number of epochs
        dataset (BaseDataset): dataset
        num_epochs (int): number of epochs as budget
        args: command line arguments
        val_loader: validation loader
        epsilon (float, default=0.01): epsilon value (for the DE term)
        separate_from_others (bool, default=False): whether to use biretta or not
        lambda_h (float, default=10): entropy weight of the biretta loss term
        use_wandb (bool, default=False): whether to use wandb
        n_facts (int, default=10): number of concepts
        knowledge_aware_kl (bool, default=False): use knowledge aware term
        real_kl (bool, default=False): use paper kl

    Returns:
        ensemble: models ensemble
    """
    import wandb
    from datasets.utils.base_dataset import get_loader

    def wandb_log_step_bears(i, epoch, loss_ce, loss_kl, prefix):
        wandb.log(
            {
                f"{prefix}loss-ce": loss_ce,
                f"{prefix}loss-kl": loss_kl,
                f"{prefix}epoch": epoch,
                f"{prefix}step": i,
            }
        )

    def wandb_log_step_deep_ens(i, epoch, loss_ce, loss_adv, prefix):
        wandb.log(
            {
                f"{prefix}loss-ce": loss_ce,
                f"{prefix}loss-adv": loss_adv,
                f"{prefix}epoch": epoch,
                f"{prefix}step": i,
            }
        )

    def wandb_log_val(i, epoch, loss, prefix):
        wandb.log(
            {
                f"{prefix}loss-val": loss,
                f"{prefix}epoch": epoch,
                f"{prefix}step": i,
            }
        )

    def wandb_log_epoch(**kwargs):
        # log accuracies
        epoch = kwargs["epoch"]
        acc = kwargs["acc"]
        c_acc = kwargs["cacc"]
        val_loss = kwargs["valloss"]
        prefix = kwargs["prefix"]

        wandb.log(
            {
                f"{prefix}acc-val": acc,
                f"{prefix}c-acc-val": c_acc,
                f"{prefix}epoch": epoch,
                f"{prefix}mean-loss-val": val_loss,
            }
        )

        lr = kwargs["lr"]
        wandb.log({f"{prefix}lr": lr})

    # Method explained in "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles"
    ensemble = []

    if separate_from_others:
        print("Doing a separation with KL...")

    def kl_paper(
        p_model: torch.tensor, p_rest: torch.tensor, k, last_hope=True
    ):
        p_model = p_model + 1e-5
        p_rest = p_rest + 1e-5
        p_model = p_model / (1 + (p_model.shape[1] * 1e-5))
        p_rest = p_rest / (1 + (p_rest.shape[1] * 1e-5))

        if not last_hope:
            ratio = torch.div(p_rest, p_model)
            kl_ew = torch.sum(
                p_model * torch.log(1 + (k - 1) * ratio), dim=1
            )
        else:
            kl_ew = torch.sum(
                p_model * torch.log(p_model + (k - 1) * p_rest), dim=1
            )

        return torch.mean(kl_ew, dim=0)

    for _, seed in enumerate(seeds):
        # setting the seeds
        fprint("Training model with seed", seed, "...")

        set_random_seed(seed)

        dataset = get_dataset(args)

        # Load dataset, model, loss, and optimizer
        encoder, decoder = dataset.get_backbone()
        n_images, c_split = dataset.get_split()
        train_loader, val_loader, _ = dataset.get_data_loaders()
        pcx_loader = get_loader(
            dataset.dataset_train,
            args.batch_size,
            num_workers=4,
            val_test=True,
        )

        # model
        model = get_model(args, encoder, decoder, n_images, c_split)
        criterion = model.get_loss(args)
        model.start_optim(args)
        model.to(model.device)

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            model.opt, args.exp_decay
        )
        w_scheduler = GradualWarmupScheduler(
            model.opt, 1.0, args.warmup_steps
        )

        # Training loop for one model in the ensemble
        model.train()

        # default for warm-up
        model.opt.zero_grad()
        model.opt.step()

        wq = build_worlds_queries_matrix(2, n_facts, "addmnist")
        wq = wq.to(model.device)

        # early stopper
        early_stopper = EarlyStopper(
            patience=5, min_delta=0.001
        )  # prev 0.01

        for epoch in range(num_epochs):
            model.train()

            for ti, data in enumerate(train_loader):
                images, labels, concepts = data
                images, labels, concepts = (
                    images.to(model.device),
                    labels.to(model.device),
                    concepts.to(model.device),
                )

                # make the image carry the gradient
                images.requires_grad = True

                model.opt.zero_grad()

                # Generate adversarial examples using x_batch
                out_dict = model(images)
                out_dict.update(
                    {
                        "INPUTS": images,
                        "LABELS": labels,
                        "CONCEPTS": concepts,
                    }
                )

                loss_original, _ = criterion(out_dict, args)

                if not separate_from_others:
                    # Call the backwards here
                    loss_original.backward()

                    # Generate adversarial examples
                    adversarial_batch = (
                        images + epsilon * images.grad.sign()
                    )

                    # Compute adversarial loss
                    out_dict_adversarial = model(adversarial_batch)
                    out_dict_adversarial.update(
                        {
                            "INPUTS": images,
                            "LABELS": labels,
                            "CONCEPTS": concepts,
                        }
                    )

                    loss_adversarial, _ = criterion(
                        out_dict_adversarial, args
                    )

                    # Minimize the combined loss l(θm, xbatch, ybatch) + l(θm, advbatch, advbatch) w.r.t. θm
                    loss_adversarial.backward()

                    if use_wandb:
                        wandb_log_step_deep_ens(
                            ti,
                            epoch,
                            loss_original,
                            loss_adversarial,
                            f"seed_{seed}_deep-ens-",
                        )

                else:

                    c_prb = out_dict["pCS"]
                    c_prb_1 = c_prb[:, 0, :]
                    c_prb_2 = c_prb[:, 1, :]

                    from models.utils.ops import outer_product

                    model_itself_pc_x = outer_product(
                        c_prb_1, c_prb_2
                    ).view(
                        c_prb_1.shape[0],
                        c_prb_1.shape[1] * c_prb_1.shape[1],
                    )

                    total_dist = 0

                    if len(ensemble) > 0:
                        if knowledge_aware_kl:
                            model_itself_pc_x = (
                                compute_pw_knowledge_filter(
                                    c_prb_1=c_prb_1,
                                    c_prb_2=c_prb_2,
                                    labels=labels,
                                    wq=wq,
                                )
                            )

                        # Create the ensembles world probabilities
                        pcx_list_ensemble = list()

                        for m in ensemble:
                            out_dict = m(images)
                            # Get concept probability
                            c_prb_other = out_dict["pCS"]
                            # other p(c|x)
                            other_pcx = compute_pw_knowledge_filter(
                                c_prb_1=c_prb_other[:, 0, :],
                                c_prb_2=c_prb_other[:, 1, :],
                                labels=labels,
                                wq=wq,
                            )
                            pcx_list_ensemble.append(other_pcx)

                        if real_kl:
                            pcx_list_ensemble = torch.stack(
                                pcx_list_ensemble
                            )
                            other_m_pc_x_mean = torch.mean(
                                pcx_list_ensemble, dim=0
                            )
                            distance = lambda_h * kl_paper(
                                model_itself_pc_x,
                                other_m_pc_x_mean,
                                len(ensemble) + 1,
                                False,
                            )
                        else:
                            crit_kl = nn.KLDivLoss(
                                reduction="batchmean"
                            )()
                            distance = -lambda_h * torch.mean(
                                torch.stack(
                                    list(
                                        crit_kl(
                                            model_itself_pc_x.log(),
                                            other_m_pc_x,
                                        )
                                        for other_m_pc_x in pcx_list_ensemble
                                    )
                                ),
                                dim=0,
                            )

                        total_dist += distance

                    if use_wandb:
                        wandb_log_step_bears(
                            ti,
                            epoch,
                            loss_original,
                            total_dist,
                            f"seed_{seed}_biretta-",
                        )

                    loss_original += total_dist
                    loss_original.backward()

                model.opt.step()

            # update at end of the epoch
            if epoch < args.warmup_steps:
                w_scheduler.step()
            else:
                scheduler.step()
                if hasattr(criterion, "grade"):
                    criterion.update_grade(epoch)

            model.eval()

            validation_loss = 0

            for i, data in enumerate(val_loader):
                images, labels, concepts = data
                images, labels, concepts = (
                    images.to(model.device),
                    labels.to(model.device),
                    concepts.to(model.device),
                )

                out_dict = model(images)
                out_dict.update(
                    {
                        "INPUTS": images,
                        "LABELS": labels,
                        "CONCEPTS": concepts,
                    }
                )

                curr_val_loss, _ = criterion(out_dict, args)

                validation_loss += curr_val_loss

                if use_wandb:
                    prefix = (
                        f"seed_{seed}_biretta-"
                        if separate_from_others
                        else f"seed_{seed}_deep-ens-"
                    )
                    wandb_log_val(ti, epoch, curr_val_loss, prefix)

            validation_loss = validation_loss / len(val_loader)

            tloss, cacc, yacc, f1 = evaluate_metrics(
                model, val_loader, args
            )
            if use_wandb is not None:
                prefix = (
                    f"seed_{seed}_biretta-"
                    if separate_from_others
                    else f"seed_{seed}_deep-ens-"
                )
                wandb_log_epoch(
                    prefix=prefix,
                    epoch=epoch,
                    acc=yacc,
                    cacc=cacc,
                    valloss=validation_loss,
                    lr=float(scheduler.get_last_lr()[0]),
                )

            fprint(
                f"># Epoch {epoch}: val loss equal to {validation_loss}"
            )

            if early_stopper.early_stop(model, validation_loss):
                break

        model.eval()

        if args.checkout is not None:
            if args.real_kl:
                PATH = f"data/ckpts/deepens_dset-{args.dataset}-bears-{separate_from_others}-model-{args.model}-seed-ensmember-{seed}-joint-{args.joint}-real-kl-{args.real_kl}.pt"
            else:
                PATH = f"data/ckpts/deepens_dset-{args.dataset}-bears-{separate_from_others}-model-{args.model}-seed-ensmember-{seed}-joint-{args.joint}.pt"
            torch.save(model.state_dict(), PATH)

        # freeze the parameters of the model in the ensemble
        _freeze_model_params(model)

        ensemble.append(model)

    fprint("Done!\nTotal length of the ensemble: ", len(ensemble))

    return ensemble


def update_pcx_dataset(model, dataset, pcx_loader, batch_size):
    """Update dataset with model p(c|x)

    Args:
        model (nn.Module): model
        dataset: dataset
        pcx_loader: dataset loader
        batch_size: batch_size

    Returns:
        dataset: modified dataset
    """
    indexes = 0

    for _, data in enumerate(pcx_loader):
        images, labels, concepts = data
        images, labels, concepts = (
            images.to(model.device),
            labels.to(model.device),
            concepts.to(model.device),
        )

        out_dict = model(images)

        # Get concept probability
        c_prb = out_dict["pCS"]
        c_prb_1 = c_prb[:, 0, :]
        c_prb_2 = c_prb[:, 1, :]

        to_append = torch.chunk(
            torch.cat((c_prb_1, c_prb_2), axis=1),
            images.size(0),
            dim=0,
        )

        j = 0
        for i in range(indexes, indexes + images.size(0)):
            dataset.pcx[i].append(to_append[j])
            j += 1
        indexes += images.size(0)

    return dataset


def populate_pcx_dataset(
    model, pcx_loader, batch_size
) -> Tuple[DatasetPcX, DataLoader]:
    """Initialize dataset with model p(c|x)

    Args:
        model (nn.Module): model
        pcx_loader: dataset loader
        batch_size: batch_size

    Returns:
        dataset: modified dataset
    """
    c_prb_list = list()
    images_list = list()

    for _, data in enumerate(pcx_loader):
        images, labels, concepts = data
        images, labels, concepts = (
            images.to(model.device),
            labels.to(model.device),
            concepts.to(model.device),
        )

        out_dict = model(images)

        # Get concept probability
        c_prb = out_dict["pCS"]
        c_prb_1 = c_prb[:, 0, :]
        c_prb_2 = c_prb[:, 1, :]

        # Append the new tensor
        c_prb_tmp = torch.chunk(
            torch.cat((c_prb_1, c_prb_2), axis=1),
            images.size(0),
            dim=0,
        )
        tmp_img_list = torch.chunk(images, images.size(0), dim=0)

        images_list.extend(tmp_img_list)

        for sublist in c_prb_tmp:
            c_prb_list.append([sublist])

    dataset = DatasetPcX(images=images_list, pcx=c_prb_list)

    return dataset


def compute_pw_knowledge_filter(c_prb_1, c_prb_2, labels, wq):
    """Compute p(w|x) with knowledge filter

    Args:
        c_prb_1 (ndarray): concept probability 1
        c_prb_2 (ndarray): concept probability 2
        labels: labels
        wq: world query

    Returns:
        p(w|x): world probability with knowledge filter
    """
    w_prob_list = list()

    for i in range(c_prb_1.size(0)):
        c_prb_1_matrix = torch.unsqueeze(c_prb_1[i], -1)
        c_prb_2_matrix = torch.unsqueeze(c_prb_2[i], 0)

        w_prob = c_prb_1_matrix.matmul(c_prb_2_matrix)
        w_prob = w_prob.view(-1)

        w_prob = w_prob * wq[:, labels[i]]

        # normalization:
        w_prob += 1e-5
        w_prob = w_prob / w_prob.sum()

        w_prob_list.append(w_prob)

    return torch.stack(w_prob_list, dim=0)


def populate_pcx_dataset_knowledge_aware(
    model, pcx_loader, n_facts
) -> Tuple[DatasetPcX, DataLoader]:
    """Initialize the dataset with knowledge aware p(w|x)

    Args:
        model (nn.Module): network
        pcx_loader: dataloader
        n_facts (int): number of concepts

    Returns:
        dataset: dataset
    """
    wq = build_worlds_queries_matrix(2, n_facts, "addmnist")
    wq = wq.to(model.device)

    w_prb_filtered_list = list()
    images_list = list()
    labels_list = list()

    for _, data in enumerate(pcx_loader):
        images, labels, concepts = data
        images, labels, concepts = (
            images.to(model.device),
            labels.to(model.device),
            concepts.to(model.device),
        )

        out_dict = model(images)

        # Get concept probability
        c_prb = out_dict["pCS"]
        c_prb_1 = c_prb[:, 0, :]
        c_prb_2 = c_prb[:, 1, :]

        # wprob list
        w_prob_tmp = []

        for i in range(c_prb_1.size(0)):
            c_prb_1_matrix = torch.unsqueeze(c_prb_1[i], -1)
            c_prb_2_matrix = torch.unsqueeze(c_prb_2[i], 0)

            w_prob = c_prb_1_matrix.matmul(c_prb_2_matrix)
            w_prob = w_prob.view(-1)

            w_prob = w_prob * wq[:, labels[i]]

            # normalization:
            w_prob += 1e-5
            with torch.no_grad():
                Z = w_prob.sum()
            w_prob = w_prob / Z

            w_prob = torch.unsqueeze(w_prob, 0)
            w_prob_tmp.append(w_prob)

        # Append the new tensor
        tmp_img_list = torch.chunk(images, images.size(0), dim=0)
        tmp_label_list = torch.chunk(labels, images.size(0), dim=0)

        images_list.extend(tmp_img_list)
        labels_list.extend(tmp_label_list)

        for sublist in w_prob_tmp:
            w_prb_filtered_list.append([sublist])

    dataset = DatasetPcX(
        images=images_list, pcx=w_prb_filtered_list, wq=wq
    )

    return dataset


def update_pcx_dataset_knowledge_aware(
    model, dataset, pcx_loader, n_facts
):
    """Update the dataset with knowledge aware p(w|x)

    Args:
        model (nn.Module): network
        dataset: dataset
        pcx_loader: dataloader
        n_facts (int): number of concepts

    Returns:
        dataset: modified dataset
    """
    indexes = 0
    for _, data in enumerate(pcx_loader):
        images, labels, concepts = data
        images, labels, concepts = (
            images.to(model.device),
            labels.to(model.device),
            concepts.to(model.device),
        )

        out_dict = model(images)

        # Get concept probability
        c_prb = out_dict["pCS"]
        c_prb_1 = c_prb[:, 0, :]
        c_prb_2 = c_prb[:, 1, :]

        # wprob list
        w_prob_tmp = []

        for i in range(c_prb_1.size(0)):
            c_prb_1_matrix = torch.unsqueeze(c_prb_1[i], -1)
            c_prb_2_matrix = torch.unsqueeze(c_prb_2[i], 0)

            w_prob = c_prb_1_matrix.matmul(c_prb_2_matrix)
            w_prob = w_prob.view(-1)

            w_prob = w_prob * dataset.wq[:, labels[i]]

            # normalization:
            w_prob += 1e-5
            with torch.no_grad():
                Z = w_prob.sum()
            w_prob = w_prob / Z

            w_prob = torch.unsqueeze(w_prob, dim=0)
            w_prob_tmp.append(w_prob)

        j = 0
        for i in range(indexes, indexes + images.size(0)):
            dataset.pcx[i].append(w_prob_tmp[j])
            j += 1
        indexes += images.size(0)

    return dataset


def get_predictions(model, loader):
    """Get predictions from a model

    Args:
        model (nn.Module): network
        loader: dataloader

    Returns:
        out_dict: output dictionary
    """
    for _, data in enumerate(loader):
        images, labels, concepts = data
        images, labels, concepts = (
            images.to(model.device),
            labels.to(model.device),
            concepts.to(model.device),
        )

        out_dict = model(images)
        out_dict.update(
            {"INPUTS": images, "LABELS": labels, "CONCEPTS": concepts}
        )
    return out_dict


def ensemble_single_predict(
    models: List[nn.Module],
    batch_samples: torch.tensor,
    apply_softmax=False,
):
    """Single prediction from an ensemble of models

    Args:
        models (List[nn.Module]): ensemble
        batch_samples: batch samples
        apply_softmax (bool, default=False): whether to apply stofmax

    Returns:
        label_prob_ens: ensemble label probability
        concept_logit_ens: ensemble concepts logits
        concept_prob_ens: ensemble concepts probability
    """
    for model in models:
        model.eval()

    output_dicts = [model(batch_samples) for model in models]

    # get out the different output
    label_prob = [
        out_dict["YS"].detach().cpu().numpy()
        for out_dict in output_dicts
    ]
    concept_logit = [
        out_dict["CS"].detach().cpu().numpy()
        for out_dict in output_dicts
    ]
    concept_prob = [
        out_dict["pCS"].detach().cpu().numpy()
        for out_dict in output_dicts
    ]

    label_prob_ens = np.stack(label_prob, axis=0)
    concept_logit_ens = np.stack(concept_logit, axis=0)
    concept_prob_ens = np.stack(concept_prob, axis=0)

    if apply_softmax:
        label_prob_ens = softmax(label_prob_ens, axis=2)

    return label_prob_ens, concept_logit_ens, concept_prob_ens


def ensemble_single_la_predict(
    models: List[nn.Module],
    batch_samples: torch.tensor,
    apply_softmax=False,
):
    """Single prediction from an ensemble of Laplace models

    Args:
        models (List[nn.Module]): ensemble
        batch_samples: batch samples
        apply_softmax (bool, default=False): whether to apply stofmax

    Returns:
        label_prob_ens: ensemble label probability
        concept_logit_ens: ensemble concepts logits
        concept_prob_ens: ensemble concepts probability
    """
    for model in models:
        model.eval()

    output_dicts = [model(batch_samples) for model in models]

    # get out the different output
    label_prob = [
        out_dict["YS"].detach().cpu().numpy()
        for out_dict in output_dicts
    ]
    concept_logit = [
        out_dict["CS"].detach().cpu().numpy()
        for out_dict in output_dicts
    ]
    concept_prob = [
        out_dict["pCS"].detach().cpu().numpy()
        for out_dict in output_dicts
    ]

    label_prob_ens = np.stack(label_prob, axis=0)
    concept_logit_ens = np.stack(concept_logit, axis=0)
    concept_prob_ens = np.stack(concept_prob, axis=0)

    if apply_softmax:
        label_prob_ens = softmax(label_prob_ens, axis=2)

    return label_prob_ens, concept_logit_ens, concept_prob_ens


def ensemble_predict(
    ensemble, loader, n_values: int, apply_softmax=False
) -> List[ndarray]:
    """Ensemble predict

    Args:
        ensemble (List[nn.Module]): ensemble
        loader: dataset
        n_values (int): values
        apply_softmax (bool, default=False): apply softmax

    Returns:
        y_true: groundtruth labels
        gs: groundtruth concepts
        cs: predicted concepts
        gs_separated: groundtruth concepts separated
        cs_separated: predicted concepts separated
        p_cs: probability of concepts
        p_ys: probability of labels
        p_cs_full: probability of concepts full
        p_ys_full: probability of labels full
    """

    device = ensemble[0].device

    # Loop over the dataloader
    for i, data in enumerate(loader):
        images, labels, concepts = data
        images, labels, concepts = (
            images.to(device),
            labels.to(device),
            concepts.to(device),
        )

        # Call Ensemble predict
        (label_prob_ens, concept_logit_ens, concept_prob_ens) = (
            ensemble_single_predict(ensemble, images, apply_softmax)
        )

        # Concatenate the output
        if i == 0:
            y_true = labels.detach().cpu().numpy()
            c_true = concepts.detach().cpu().numpy()
            y_pred = label_prob_ens
            c_pred = concept_logit_ens
            pc_pred = concept_prob_ens
        else:
            y_true = np.concatenate(
                [y_true, labels.detach().cpu().numpy()], axis=0
            )
            c_true = np.concatenate(
                [c_true, concepts.detach().cpu().numpy()], axis=0
            )
            y_pred = np.concatenate([y_pred, label_prob_ens], axis=1)
            c_pred = np.concatenate(
                [c_pred, concept_logit_ens], axis=1
            )
            pc_pred = np.concatenate(
                [pc_pred, concept_prob_ens], axis=1
            )

    # Compute the final arrangements
    gs = np.split(
        c_true, c_true.shape[1], axis=1
    )  # splitted groundtruth concepts
    cs = np.split(
        c_pred, c_pred.shape[2], axis=2
    )  # splitted concepts # (nmod, data, 10, 10)
    p_cs = np.split(
        pc_pred, pc_pred.shape[2], axis=2
    )  # splitted concept prob # (nmod, data, 10, 10)

    assert len(gs) == len(cs), f"gs: {gs.shape}, cs: {cs.shape}"

    # [#data, 1]
    # gs = np.concatenate(gs, axis=0).squeeze(1)
    # Print some information for debugging
    gs = np.char.add(gs[0].astype(str), gs[1].astype(str))
    gs = np.where(gs == "-1-1", -1, gs)
    gs = gs.squeeze(-1).astype(int)

    p_cs_1 = np.expand_dims(
        p_cs[0].squeeze(2), axis=-1
    )  # 30, 256, 10, 1
    p_cs_2 = np.expand_dims(
        p_cs[1].squeeze(2), axis=-2
    )  # 30, 256, 1, 10
    p_cs = np.matmul(
        p_cs_1, p_cs_2
    )  # 30, 256, 10, 10 -> # [#modelli, #data, #facts, #facts]
    p_cs = np.reshape(
        p_cs, (*p_cs.shape[:-2], p_cs.shape[-1] * p_cs.shape[-2])
    )  # -> [#modelli, #data, #facts^2]
    p_cs = np.mean(
        p_cs, axis=0
    )  # avg[#modelli, #data, #facts^2] = [#data, #facts^2]

    # mean probabilities of the output
    p_ys = np.mean(y_pred, axis=0)  # -> (256, 19)
    ys = np.argmax(p_ys, axis=1)  # predictions -> (data)

    p_ys_full = p_ys  # all the items of probabilities are considered (#data, #facts^2)
    p_cs_full = p_cs  # all the items of probabilities are considered (#data, #facts^2)
    cs = p_cs.argmax(
        axis=1
    )  # the predicted concept is the argument maximum
    p_cs = p_cs.max(
        axis=1
    )  # only the maximum one is considered (#data,)
    p_ys = p_ys.max(axis=1)  # only the maximum one is considered

    assert gs.shape == cs.shape, f"gs: {gs.shape}, cs: {cs.shape}"

    gs_separated = c_true

    # cs_separated = np.char.mod('%02d', cs)
    decimal_values = np.array([x // n_values for x in cs])
    unit_values = np.array([x % n_values for x in cs])
    cs_separated = np.column_stack((decimal_values, unit_values))

    return (
        y_true,
        gs,
        ys,
        cs,
        gs_separated,
        cs_separated,
        p_cs,
        p_ys,
        p_cs_full,
        p_ys_full,
    )


def laplace_approximation(
    model: nn.Module, device, train_loader, val_loader
):
    """Performs the Laplace Approximation

    Args:
        model (nn.Module): network
        device: device
        train_loader: train dataloader
        val_loader: validation dataloader

    Returns:
        la: laplace model
    """
    from laplace.curvature import AsdlGGN
    from torch.utils.data import DataLoader

    # Wrapper DataLoader
    class WrapperDataLoader(DataLoader):
        """WrapperDataLoader. It returns samples according to what the Laplace library expect"""

        def __init__(self, original_dataloader, **kwargs):
            """Initialize method of the WrapperDataLoader

            Args:
                self: instance
                original_dataloader: dataloader to convert
                kwargs: key-value arguments

            Returns:
                None: This function does not return a value.
            """
            super(WrapperDataLoader, self).__init__(
                dataset=original_dataloader.dataset, **kwargs
            )

        def __iter__(self):
            """Iter method of the dataloader

            Args:
                self: instance

            Returns:
                iter: dataset iterator
            """
            # Get the iterator from the original DataLoader
            original_iterator = super(
                WrapperDataLoader, self
            ).__iter__()

            for original_batch in original_iterator:
                modified_batch = [
                    original_batch[0],
                    original_batch[1].to(torch.long),
                ]
                yield modified_batch

    # Wrapper Model
    class WrapperModel(nn.Module):
        """WrapperModel. It performs predictions according to what the Laplace library expect"""

        def __init__(self, original_model, device, output_all=False):
            """Initialize method of the WrapperModel

            Args:
                self: instance
                original_model (nn.Module): model to convert
                device: device
                output_all (bool, default=False): whether to output all or only the y

            Returns:
                None: This function does not return a value.
            """
            super(WrapperModel, self).__init__()
            self.original_model = original_model
            self.original_model.to(device)
            self.output_all = output_all
            self.model_possibilities = list()
            self.device = device

        def forward(self, input_batch):
            """Forward method of the WrapperModel

            Args:
                self: instance
                input_batch: input batch

            Returns:
                ys: y probabilities if output_all = False
                out: concatenation of labels probability, concepts logits and concepts probabilities
            """
            batch_size = input_batch.shape[0]

            # Call the forward method of the model
            original_output = self.original_model(input_batch)
            # Modify the output to return only the output

            ys = original_output["YS"]  # torch.Size([batch, 19])
            ys = ys.to(dtype=torch.float)
            py = original_output["CS"]  # torch.Size([batch, 2, 10])
            pCS = original_output["pCS"]  # torch.Size([batch, 2, 10])

            # torch.Size([batch, 19]) torch.Size([batch, 2, 10]) torch.Size([batch, 2, 10])
            if not self.output_all:
                return ys

            # I want to flat all the tensors in this way:
            return torch.cat(
                (
                    ys,
                    py.reshape(batch_size, -1),
                    pCS.reshape(batch_size, -1),
                ),
                dim=1,
            )

        def get_ensembles(self, la_model, n_models):
            """Retrieve an esenmble out of the laplace model

            Args:
                self: instance
                la_model: laplace model
                n_models (int): how many models to extract

            Returns:
                ensemble (list[nn.Module]): ensemble of models
            """
            import copy

            ensembles = []
            for i, mp in enumerate(self.model_possibilities):
                # substituting to the current model one of the possible parameters
                vector_to_parameters(
                    mp, la_model.model.last_layer.parameters()
                )
                # Retrieve the current model and append it
                ensembles.append(
                    copy.deepcopy(la_model.model.model.original_model)
                )

                if i == n_models - 1:
                    break

            # restore original model
            vector_to_parameters(
                la_model.mean, la_model.model.last_layer.parameters()
            )

            # return an ensembles of models
            return ensembles

    # wrap the dataloaders
    la_training_loader = WrapperDataLoader(train_loader)
    la_val_loader = WrapperDataLoader(val_loader)

    # wrap the model
    la_model = WrapperModel(model, device)
    la_model.to(device)

    la = Laplace(
        la_model,
        "classification",
        subset_of_weights="last_layer",  # subset_of_weights='subnetwork',
        hessian_structure="kron",  # hessian_structure='full', # hessian_structure='diag', # hessian_structure='kron',
        backend=AsdlGGN,
    )

    la.fit(la_training_loader)
    la.optimize_prior_precision(
        method="marglik", val_loader=la_val_loader
    )

    # Enabling last layer output all
    la.model.model.output_all = True

    return la


def laplace_single_prediction(
    la,
    sample_batch: ndarray,
    output_classes: int,
    num_concepts: int,
    apply_softmax=False,
):
    """Performs the laplace prediction. It is a placeholder to let laplace instantiate the models from sampling

    Args:
        la: laplace model
        sample_batch: batch of samples
        output_classes (int): number of ouput classes
        num_concepts (int): number of concepts
        apply_softmax (bool, default=False): whether to apply softmax

    Returns:
        prediction: recovered laplace predictions
    """

    pred = la(sample_batch, pred_type="nn", link_approx="mc")

    """
    What does la.forward do?

    def _nn_predictive_samples(self, X, n_samples=100):
        fs = list()
        for sample in self.sample(n_samples):
            vector_to_parameters(sample, self.model.last_layer.parameters())
            f = self.model(X.to(self._device))
            fs.append(f.detach() if not self.enable_backprop else f)
        vector_to_parameters(self.mean, self.model.last_layer.parameters())
        fs = torch.stack(fs)
        if self.likelihood == 'classification':
            fs = torch.softmax(fs, dim=-1)
        return fs

    Basically, it samples a series of parameters (the sample is a multivariate gaussian whose variance and mean have been
    learned during within the fit command execution), then it instantiate the network based on such parameters and then performs 
    the predictions. This is done until the end. Afterwards it employs the softmax, which is not needed in our case.
    Hence, a quickfix could be to comment that line in our installed library, otherwise to try to "invert" the softmax
    computation. However, this induces to a non-reliable result.

    Here is the sampling operation. It samples 100 items of size self.n_params

    def sample(self, n_samples=100):
        samples = torch.randn(n_samples, self.n_params, device=self._device)
        samples = self.posterior_precision.bmm(samples, exponent=-0.5)
        return self.mean.reshape(1, self.n_params) + samples.reshape(n_samples, self.n_params)
    
    It seems like it batch multply the samples (from a gaussian centered in zero with 1 as variance), and then rescale it
    """
    recovered_pred = recover_predictions_from_laplace(
        pred,
        sample_batch.shape[0],
        output_classes,
        num_concepts,
        apply_softmax,
    )

    return recovered_pred


def recover_predictions_from_laplace(
    la_prediction,
    batch_size,
    output_classes: int = 19,
    num_concepts: int = 10,
    apply_softmax=False,
):
    """From the WrappedModel that laplace uses, retrieve the predictions in the format we expect

    Args:
        la_prediction: laplace prediction
        batch_size (int): batch size
        output_classes (int, default=19): number of classes as output
        num_concepts (int, default=10): number of concepts
        apply_softmax (bool, default=False): whether to apply softmax

    Returns:
        out_dict: dictionary of predictions
    """
    # Recovering shape
    ys = la_prediction[
        :, :output_classes
    ]  # take all until output_classes
    py = la_prediction[
        :, output_classes : output_classes + 2 * num_concepts
    ]  # take all from output_classes until output_classes+2*num_concepts
    py = py.reshape(
        batch_size, 2, num_concepts
    )  # reshape it correctly
    pCS = la_prediction[
        :, output_classes + 2 * num_concepts :
    ]  # take all from the previous to the end
    pCS = pCS.reshape(
        batch_size, 2, num_concepts
    )  # reshape it correctly

    if apply_softmax:
        import torch.nn.functional as F

        print("Qui sono questo", py.shape)
        py = F.softmax(py, dim=1)

    return {"YS": ys, "CS": py, "pCS": pCS}


def laplace_prediction(
    laplace_model,
    device,
    loader,
    n_ensembles: int,
    output_classes: int,
    num_concepts: int,
    apply_softmax=False,
) -> List[ndarray]:
    """Laplace prediction

    Args:
        laplace_model: laplace model
        device: device
        loader: dataloader
        n_ensembles (int): number of ensemble
        output_classes (int): output classes
        num_concepts (int): number of concepts
        apply_softmax (bool, default=False): whether to apply softmax

    Returns:
        y_true: groundtruth labels
        gs: groundtruth concepts
        cs: predicted concepts
        gs_separated: groundtruth concepts separated
        cs_separated: predicted concepts separated
        p_cs: probability of concepts
        p_ys: probability of labels
        p_cs_full: probability of concepts full
        p_ys_full: probability of labels full
    """

    # Loop over the dataloader
    for i, data in enumerate(loader):
        images, labels, concepts = data
        images, labels, concepts = (
            images.to(device),
            labels.to(device),
            concepts.to(device),
        )

        # prediction
        _ = laplace_single_prediction(
            laplace_model,
            images,
            output_classes,
            num_concepts,
            apply_softmax,
        )

        # Call Laplace ensembles
        ensemble = laplace_model.model.model.get_ensembles(
            laplace_model, n_ensembles
        )

        # Call Ensemble predict
        (label_prob_ens, concept_logit_ens, concept_prob_ens) = (
            ensemble_single_la_predict(
                ensemble, images, apply_softmax
            )
        )

        # Concatenate the output
        if i == 0:
            y_true = labels.detach().cpu().numpy()
            c_true = concepts.detach().cpu().numpy()
            y_pred = label_prob_ens
            c_pred = concept_logit_ens
            pc_pred = concept_prob_ens
        else:
            y_true = np.concatenate(
                [y_true, labels.detach().cpu().numpy()], axis=0
            )
            c_true = np.concatenate(
                [c_true, concepts.detach().cpu().numpy()], axis=0
            )
            y_pred = np.concatenate([y_pred, label_prob_ens], axis=1)
            c_pred = np.concatenate(
                [c_pred, concept_logit_ens], axis=1
            )
            pc_pred = np.concatenate(
                [pc_pred, concept_prob_ens], axis=1
            )

    # Compute the final arrangements
    gs = np.split(
        c_true, c_true.shape[1], axis=1
    )  # splitted groundtruth concepts
    cs = np.split(
        c_pred, c_pred.shape[2], axis=2
    )  # splitted concepts # (nmod, data, 10, 10)
    p_cs = np.split(
        pc_pred, pc_pred.shape[2], axis=2
    )  # splitted concept prob # (nmod, data, 10, 10)

    assert len(gs) == len(cs), f"gs: {gs.shape}, cs: {cs.shape}"

    # [#data, 1]
    # gs = np.concatenate(gs, axis=0).squeeze(1)
    # Print some information for debugging
    gs = np.char.add(gs[0].astype(str), gs[1].astype(str))
    gs = np.where(gs == "-1-1", -1, gs)
    gs = gs.squeeze(-1).astype(int)

    p_cs_1 = np.expand_dims(
        p_cs[0].squeeze(2), axis=-1
    )  # 30, 256, 10, 1
    p_cs_2 = np.expand_dims(
        p_cs[1].squeeze(2), axis=-2
    )  # 30, 256, 1, 10
    p_cs = np.matmul(
        p_cs_1, p_cs_2
    )  # 30, 256, 10, 10 -> # [#modelli, #data, #facts, #facts]
    p_cs = np.reshape(
        p_cs, (*p_cs.shape[:-2], p_cs.shape[-1] * p_cs.shape[-2])
    )  # -> [#modelli, #data, #facts^2]
    p_cs = np.mean(
        p_cs, axis=0
    )  # avg[#modelli, #data, #facts^2] = [#data, #facts^2]

    # mean probabilities of the output
    p_ys = np.mean(y_pred, axis=0)  # -> (256, 19)
    ys = np.argmax(p_ys, axis=1)  # predictions -> (data)

    p_ys_full = p_ys  # all the items of probabilities are considered (#data, #facts^2)
    p_cs_full = p_cs  # all the items of probabilities are considered (#data, #facts^2)
    cs = p_cs.argmax(
        axis=1
    )  # the predicted concept is the argument maximum
    p_cs = p_cs.max(
        axis=1
    )  # only the maximum one is considered (#data,)
    p_ys = p_ys.max(axis=1)  # only the maximum one is considered

    assert gs.shape == cs.shape, f"gs: {gs.shape}, cs: {cs.shape}"

    gs_separated = c_true

    # cs_separated = np.char.mod('%02d', cs)
    decimal_values = np.array([x // num_concepts for x in cs])
    unit_values = np.array([x % num_concepts for x in cs])
    cs_separated = np.column_stack((decimal_values, unit_values))

    return (
        y_true,
        gs,
        ys,
        cs,
        gs_separated,
        cs_separated,
        p_cs,
        p_ys,
        p_cs_full,
        p_ys_full,
    )
