# -*- coding: utf-8 -*-
"""
This files's functions are almost written by SENN's authors to train SENN.
We modified so as to fit the semi-supervised fashion.
"""

import builtins
import copy
import math
import os
import pdb
import random
import shutil

# standard imports
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import wandb
from aggregators_BDD import CBM_aggregator, additive_scalar_aggregator
from conceptizers_BDD import image_fcc_conceptizer
from DPL.dpl import DPL
from DPL.dpl_auc import DPL_AUC
from laplace import Laplace
from numpy import ndarray
from parametrizers import dfc_parametrizer, image_parametrizer
from scipy.special import softmax

# Local imports
from SENN.utils import AverageMeter
from torch.autograd import Variable
from torch.utils.data import Dataset
from worlds_BDD import (
    compute_forward_prob,
    compute_forward_stop_groundtruth,
    compute_forward_stop_prob,
    compute_left,
    compute_left_groundtruth,
    compute_output_probability,
    compute_right,
    compute_right_groundtruth,
    compute_stop_prob,
    convert_np_array_to_binary,
)


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")
        self.best_weights = None
        self.stuck = False

    def early_stop(self, model, validation_loss):
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


def add_previous_dot(string, prefix):
    parts = string.split(".")
    return parts[0] + prefix + "." + parts[1]


def join_row(row):
    return "".join(row)


def fprint(*args, **kwargs):
    """
    Flushing print
    """
    builtins.print(*args, **kwargs)
    sys.stdout.flush()


class DatasetPcX(Dataset):
    def __init__(
        self,
        images,
        pcx,
        mode=False,
        w_fstop_prob_list=[],
        w_left_prob_list=[],
        w_right_prob_list=[],
        FS_w_q=None,
        L_w_q=None,
        R_w_q=None,
    ):
        self.images = images
        self.pcx = pcx
        fprint("Len", len(self.images), len(self.pcx))
        self.img_to_key = self._initialize_dict(self.images)
        self.w_fstop_prob_list = w_fstop_prob_list
        self.w_left_prob_list = w_left_prob_list
        self.w_right_prob_list = w_right_prob_list
        self.FS_w_q = FS_w_q
        self.L_w_q = L_w_q
        self.R_w_q = R_w_q
        self.mode = mode

    def _initialize_dict(self, images):
        img_to_key = dict()
        for i, img in enumerate(images):
            self._add_key_value(img_to_key, img, i)
        return img_to_key

    def _add_key_value(self, dictionary, tensor_key, value):
        tuple_key = tensor_key.detach().cpu().numpy().tobytes()
        dictionary[tuple_key] = value

    def return_value_from_key(self, tensor_key):
        tuple_key = tensor_key.detach().cpu().numpy().tobytes()
        index = self.img_to_key[tuple_key]
        if not self.mode:
            return self.pcx[index]
        return (
            self.w_fstop_prob_list[index],
            self.w_left_prob_list[index],
            self.w_right_prob_list[index],
        )

    def _hash_mnist_image(self, image_array):
        import hashlib

        flattened_array = image_array.flatten()
        image_bytes = flattened_array.tobytes()
        sha256_hash = hashlib.sha256(image_bytes)

        return sha256_hash.hexdigest()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if not self.mode:
            return self.images[index], self.pcx[index]
        return (
            self.images[index],
            self.w_fstop_prob_list[index],
            self.w_left_prob_list[index],
            self.w_right_prob_list[index],
        )


def set_random_seed(seed: int) -> None:
    """
    Sets the seeds at a certain value.
    :param seed: the value to be set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ===============================================================================
# ====================      REGULARIZER UTILITIES    ============================
# ===============================================================================
"""
def compute jacobian:
Inputs: 
    x: encoder's output
    fx: prediction
    device: GPU or CPU
Return:
    J: Jacobian
NOTE: This function is not modified from original SENN
"""


# function borrowed (what were you thinking!? :)) from Laplace which applies a set of weights to a model
def _vector_to_parameters(vec: torch.Tensor, parameters) -> None:
    r"""Convert one vector to the parameters

    Args:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    """
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError(
            "expected torch.Tensor, but got: {}".format(
                torch.typename(vec)
            )
        )

    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.data = (
            vec[pointer : pointer + num_param].view_as(param).data
        )

        # Increment the pointer
        pointer += num_param


# ===============================================================================
# ==================================   TESTER    ================================
# ===============================================================================

"""
Class ClassificationTester:
"""


class ClassificationTester:
    def __init__(self, model, args, device):
        # hyparparameters used in the loss function
        self.lambd = (
            args.theta_reg_lambda
            if ("theta_reg_lambda" in args)
            else 1e-6
        )  # for regularization strenght
        self.eta = (
            args.h_labeled_param
            if ("h_labeled_param" in args)
            else 0.0
        )  # for wealky supervised
        self.gamma = (
            args.info_hypara if ("info_hypara" in args) else 0.0
        )  # for wealky supervised
        self.w_entropy = (
            args.w_entropy if ("w_entropy" in args) else 0.0
        )

        # set the seed
        self.seed = args.seed

        # use the gradient norm conputation
        self.norm = 2

        # set models
        self.model = model

        # weight of losses
        self.c_freq = np.load("BDD/c_freq.npy")
        self.y_freq = np.load("BDD/y_freq.npy")

        # others
        self.args = args
        self.cuda = args.cuda
        self.device = device

        self.nclasses = args.nclasses

        if False and args.model_name == "dpl":
            fprint("Selected CE_for_loop")
            self.prediction_criterion = self.BCE_forloop
        elif args.model_name in ["dpl_auc", "dpl_auc_pcbm", "dpl"]:
            fprint("Selected CE_for_loop")
            self.prediction_criterion = self.CE_forloop

        # save stuff to train ensembles with the same parameters
        self.model_name = args.model_name
        self.h_type = args.h_type
        self.nconcepts = args.nconcepts
        self.nconcepts_labeled = args.nconcepts_labeled
        self.concept_dim = args.concept_dim
        self.h_sparsity = args.h_sparsity
        self.senn = args.senn
        self.cbm = args.cbm
        self.batch_size = args.batch_size
        self.n_models = args.n_models
        self.theta_dim = args.theta_dim

        self.learning_h = True

        # acumulate loss to make loss figure
        self.loss_history = []
        self.val_loss_history = []

        # for the ensembles, an array of loss_history
        self.loss_histories = []
        self.val_loss_histories = []

        # use to fprint error, loss
        self.print_freq = args.print_freq

        """
        select optimizer
        self.optimizer: [conceptizer, parametrizer, aggregator]
        self.aux_optimizer: [conceptizer, parametrizer, aggregator] for aux. output
        self.pre_optimizer: for pretrained model
        """
        if args.opt == "adam":
            optim_betas = (0.9, 0.999)
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=args.lr, betas=optim_betas
            )
        elif args.opt == "rmsprop":
            self.optimizer = optim.RMSprop(
                self.model.parameters(), lr=args.lr
            )
        elif args.opt == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
                momentum=0.9,
            )

        # set other stuff
        self.opt = args.opt
        self.lr = args.lr
        self.weight_decay = args.weight_decay

        # set scheduler for learning rate
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=args.exp_decay_lr
        )

    def validate(self, val_loader, epoch, fold=None, name=""):

        # initialization of fprint's values
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        topc1 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        end = time.time()

        loss_y, loss_c, loss_h = 0, 0, 0

        for i, (inputs, targets, concepts) in enumerate(val_loader):

            # get the inputs
            if self.cuda:
                inputs, targets, concepts = (
                    inputs.cuda(self.device),
                    targets.cuda(self.device),
                    concepts.cuda(self.device),
                )

            # compute output
            output = self.model(inputs)

            # prediction_criterion is defined in __init__ of "class GradPenaltyTrainer"
            pred_loss = self.prediction_criterion(output, targets)

            ################### EM ############################

            # save loss (only value) to the all_losses list
            loss_y += pred_loss.cpu().data.numpy() / len(val_loader)
            all_losses = {"prediction": pred_loss.cpu().data.numpy()}

            # compute loss of known concets and discriminator
            h_loss, hh_labeled = (
                self.concept_learning_loss_for_weak_supervision(
                    inputs,
                    all_losses,
                    concepts,
                    self.args.cbm,
                    self.args.senn,
                    epoch,
                )
            )

            loss_h += self.entropy_loss(
                hh_labeled, all_losses, epoch
            ).cpu().data.numpy() / len(val_loader)

            loss_c += h_loss.data.cpu().numpy() / len(val_loader)

            # add to loss_history
            all_losses["iter"] = epoch

            #######################################################
            # measure accuracy and record loss
            if self.nclasses > 4:
                # mainly use this line (current)
                prec1, _ = self.accuracy(
                    output.data, targets, topk=(1, 5)
                )
            elif self.nclasses == 3:
                prec1, _ = self.accuracy(
                    output.data, targets, topk=(1, 3)
                )
            else:
                prec1, _ = self.binary_accuracy(
                    output.data, targets
                ), [100]

            # update each value of fprint's values
            losses.update(
                pred_loss.data.cpu().numpy(), inputs.size(0)
            )
            top1.update(prec1[0], inputs.size(0))

            # measure accuracy of concepts
            hh_labeled, _, _ = self.model.conceptizer(inputs)
            if not self.args.senn:
                err = self.concept_error(hh_labeled.data, concepts)

                # update fprint's value
                topc1.update(err, inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if not self.args.senn:
                # fprint values of i-th iteration in the current epoch
                if i % self.print_freq == 0:
                    fprint(
                        "Val: [{0}/{1}]\t"
                        "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                        "Loss {loss.val:.4f} ({loss.avg:.4f})\t".format(
                            i,
                            len(val_loader),
                            batch_time=batch_time,
                            loss=losses,
                        )
                    )
            else:
                # fprint values of i-th iteration in the current epoch
                if i % self.print_freq == 0:
                    fprint(
                        "Val: [{0}/{1}]\t"
                        "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                        "Loss {loss.val:.4f} ({loss.avg:.4f})\t".format(
                            i,
                            len(val_loader),
                            batch_time=batch_time,
                            loss=losses,
                        )
                    )
        val_loss_dict = {"iter": epoch, f"{name} prediction": loss_y}

        if self.args.wandb is not None:
            wandb_dict = {}
            for key in val_loss_dict.keys():
                wandb_dict.update({"val-" + key: val_loss_dict[key]})

            wandb.log(wandb_dict)

        self.val_loss_history.append(val_loss_dict)

        # top1.avg: use whether models save or not
        return top1.avg, loss_y

    def test(self, test_loader, save_file_name, fold=None):

        # initialization of fprint's values
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        topc1 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        end = time.time()

        # open the save file
        fp = open(save_file_name, "a")
        for i, (inputs, targets, concepts) in enumerate(test_loader):

            # get the inputs
            if self.cuda:
                inputs, targets, concepts = (
                    inputs.cuda(self.device),
                    targets.cuda(self.device),
                    concepts.cuda(self.device),
                )

            # compute output
            output = self.model(inputs)

            # prediction_criterion is defined in __init__ of "class GradPenaltyTrainer"
            loss = self.prediction_criterion(output, targets)

            # measure accuracy and record loss
            if self.nclasses > 4:
                # mainly use this line (current)
                prec1, _ = self.accuracy(
                    output.data, targets, topk=(1, 5)
                )
            elif self.nclasses == 3:
                prec1, _ = self.accuracy(
                    output.data, targets, topk=(1, 3)
                )
            else:
                prec1, _ = self.binary_accuracy(
                    output.data, targets
                ), [100]

            # update each value of fprint's values
            losses.update(loss.data.cpu().numpy(), inputs.size(0))
            top1.update(prec1[0], inputs.size(0))

            # measure accuracy of concepts
            hh_labeled, hh, _ = self.model.conceptizer(inputs)
            if not self.args.senn:
                err = self.concept_error(hh_labeled.data, concepts)

                # update fprint's value
                topc1.update(err, inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if not self.args.senn:
                # if you use gpu, send results to cpu
                if self.cuda:
                    pred_labels = output.data.to("cpu")
                    # pred_labels = np.argmax(output.data.to("cpu"), axis=1)
                    concept_labels = hh_labeled.data.to("cpu")
                    concept_nolabels = hh.data.to("cpu")
                    attr = concepts.data.to("cpu")
                else:
                    pred_labels = output.data
                    concept_labels = hh_labeled
                    concept_nolabels = hh
                    attr = concepts

                # save to the file
                for j in range(len(targets)):
                    for k in range(len(targets[0])):
                        fp.write("%f," % (targets[j][k]))
                    for k in range(len(pred_labels[0])):
                        fp.write("%f," % (pred_labels[j][k]))
                    for k in range(concept_labels.shape[1]):
                        fp.write("%f," % (concept_labels[j][k]))
                    for k in range(concept_nolabels.shape[1]):
                        fp.write("%f," % (concept_nolabels[j][k]))
                    fp.write(",")
                    for k in range(attr.shape[1]):
                        fp.write("%f," % (attr[j][k]))
                    fp.write("\n")

                # fprint values of i-th iteration
                """
                if i % self.print_freq == 0:
                    fprint('Test on '+fold+': [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                           i, len(test_loader), batch_time=batch_time, loss=losses))
                """
            else:
                # if you use gpu, send results to cpu
                if self.cuda:
                    pred_labels = output.data.to("cpu")
                    # pred_labels = np.argmax(output.data.to("cpu"), axis=1)
                    concept_nolabels = hh.data.to("cpu")
                    attr = concepts.data.to("cpu")
                else:
                    pred_labels = output.data
                    concept_nolabels = hh
                    attr = concepts

                # save to the file
                for j in range(len(targets)):
                    for k in range(len(targets[0])):
                        fp.write("%f," % (targets[j][k]))
                        # fp.write("%d,%d,"%(targets[j][k],pred_labels[j][k]))
                    for k in range(len(pred_labels[0])):
                        fp.write("%f," % (pred_labels[j][k]))
                    for k in range(concept_nolabels.shape[1]):
                        fp.write("%f," % (concept_nolabels[j][k]))
                    fp.write(",")
                    for k in range(attr.shape[1]):
                        fp.write("%f," % (attr[j][k]))
                    fp.write("\n")

                # fprint values of i-th iteration
                if i % self.print_freq == 0:
                    fprint(
                        "Test: [{0}/{1}]\t"
                        "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                        "Loss {loss.val:.4f} ({loss.avg:.4f})\t".format(
                            i,
                            len(test_loader),
                            batch_time=batch_time,
                            loss=losses,
                        )
                    )

        # close the save_file_name
        fp.close()

    def save_model_params(self, model, save_path, file_name):
        filename = os.path.join(save_path, file_name)
        torch.save(model.state_dict(), filename)

    # Compute the worlds probability for each kind of bayesian model
    def worlds_probability(
        self,
        loader,
        output_classes: int,
        num_concepts: int,
        n_ensembles: int = 30,
        apply_softmax=False,
    ):
        raise NotImplementedError("Not implemented")

    # compute the pc_x distance for each kind of bayesian model
    def p_c_x_distance(
        self,
        loader,
    ):
        raise NotImplementedError("Not implemented")

    def compute_accuracy_f1(self, y_pred, y_true):
        from sklearn.metrics import accuracy_score, f1_score

        y_trues = torch.split(y_true, 1, dim=-1)
        y_preds = torch.split(y_pred, 2, dim=-1)

        y_preds_list = list(y_preds)

        # Modify each tensor in the list
        for i in range(len(y_preds_list)):
            y_preds_list[i] = torch.argmax(
                y_preds_list[i], dim=1
            ).unsqueeze(-1)

        all_true = torch.cat(y_trues[:4], dim=-1)
        all_pred = torch.cat(y_preds_list[:4], dim=-1)

        # Convert to numpy arrays
        all_true_np = all_true.detach().cpu().numpy()
        all_pred_np = all_pred.detach().cpu().numpy()

        # Compute accuracy and F1 score
        accuracy = accuracy_score(all_true_np, all_pred_np)
        f1 = f1_score(all_true_np, all_pred_np, average="weighted")

        return accuracy, f1

    # just a mean l2 distance
    def mean_l2_distance(self, vectors):
        num_vectors = len(vectors)
        distances = []

        for i in range(num_vectors):
            for j in range(i + 1, num_vectors):  # exclude itself
                distance = np.linalg.norm(vectors[i] - vectors[j])
                distances.append(distance)

        mean_distance = np.mean(distances)
        return mean_distance

    def concept_error(self, output, target):
        err = torch.Tensor(1).fill_(
            (output.round().eq(target)).float().mean() * 100
        )
        err = (100.0 - err.data[0]) / 100
        return err

    def binary_accuracy(self, output, target):
        """Computes the accuracy"""
        return torch.Tensor(1).fill_(
            (output.round().eq(target)).float().mean() * 100
        )

    def accuracy(self, output, target, topk=(1,), numpy=False):
        if numpy:
            maxk = max(topk)
            batch_size = target.shape[0]

            # Get the indices of the topk predictions
            pred = np.argpartition(output, -maxk)[:, -maxk:]

            # Check if each predicted class matches the target class
            correct = (
                pred == target
            )  # (pred == target.reshape((-1, 1)))

            # If topk = (1,5), then, k=1 and k=5
            res = []
            for k in topk:
                correct_k = np.sum(
                    correct[:, :k].any(axis=1).astype(float)
                )
                res.append(correct_k * 100.0 / batch_size)

            return res

        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        # pred = pred.t()
        # correct = pred.eq(target.view(1, -1).expand_as(pred))

        correct = pred.eq(target.long())

        # if topk = (1,5), then, k=1 and k=5
        res = []
        for k in topk:
            correct_k = (
                correct[:k].view(-1).float().sum(0, keepdim=True)
            )
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    # plot losses -> it just takes the values from loss_history and val_loss_history
    def plot_losses(self, name, save_path=None):
        loss_types = [
            k for k in self.loss_history[0].keys() if k != "iter"
        ]
        losses = {k: [] for k in loss_types}
        iters = []
        for e in self.loss_history:
            iters.append(e["iter"])
            for k in loss_types:
                losses[k].append(e[k])
        fig, ax = plt.subplots(
            1, len(loss_types), figsize=(4 * len(loss_types), 5)
        )
        if len(loss_types) == 1:
            ax = [ax]  # Hacky, fix
        for i, k in enumerate(loss_types):
            ax[i].plot(iters, losses[k])
            ax[i].set_title("Loss: {}".format(k))
            ax[i].set_xlabel("Iters")
            ax[i].set_ylabel("Loss")
        if save_path is not None:
            plt.savefig(
                save_path + f"/training_losses_{name}.pdf",
                bbox_inches="tight",
                format="pdf",
                dpi=300,
            )

        print(f"Saving {save_path}/training_losses_{name}.pdf")

        #### VALIDATION
        plt.close()

        loss_types = [
            k for k in self.val_loss_history[0].keys() if k != "iter"
        ]
        losses = {k: [] for k in loss_types}
        iters = []
        for e in self.val_loss_history:
            iters.append(e["iter"])
            for k in loss_types:
                losses[k].append(e[k])
        fig, ax = plt.subplots(
            1, len(loss_types), figsize=(4 * len(loss_types), 5)
        )
        if len(loss_types) == 1:
            ax = [ax]  # Hacky, fix
        for i, k in enumerate(loss_types):
            ax[i].plot(iters, losses[k])
            ax[i].set_title("Loss: {}".format(k))
            ax[i].set_xlabel("Epoch")
            ax[i].set_ylabel("Loss")
        if save_path is not None:
            plt.savefig(
                save_path + f"/validation_losses_{name}.pdf",
                bbox_inches="tight",
                format="pdf",
                dpi=300,
            )

    def BCE_forloop(self, tar, pred):
        loss = F.binary_cross_entropy(tar[0, :4], pred[0, :4])

        for i in range(1, len(tar)):
            loss = loss + F.binary_cross_entropy(
                tar[i, :4], pred[i, :4]
            )
        return loss

    def CE_forloop(self, y_pred, y_true):
        y_trues = torch.split(y_true, 1, dim=-1)
        y_preds = torch.split(y_pred, 2, dim=-1)

        loss = 0
        for i in range(4):

            true = y_trues[i].view(-1)
            pred = y_preds[i]

            loss_i = F.nll_loss(pred.log(), true.to(torch.long))
            loss += loss_i / 4

            assert loss_i > 0, pred.log()

        return loss

    def concept_learning_loss_for_weak_supervision(
        self, inputs, all_losses, concepts, cbm, senn, epoch
    ):

        # compute predicted known concepts by inputs
        # real uses the discriminator's loss
        hh_labeled_list, h_x, real = self.model.conceptizer(inputs)
        concepts = concepts.to(self.device)

        if not senn:

            # compute losses of known concepts
            # for i in range(21):

            #     lamb = self.c_freq[i] * (1 - self.c_freq[i])
            #     w_i  = 1 / self.c_freq[i] if concepts[0, i] == 1 else 1 / (1 - self.c_freq[i])
            #     labeled_loss = lamb * w_i * F.binary_cross_entropy(hh_labeled_list[0, i], concepts[0, i].to(self.device))
            #     for j in range(1,len(hh_labeled_list)):
            #         w_i  = 1 / self.c_freq[i] if concepts[j, i] == 1 else 1 / (1 - self.c_freq[i])
            #         labeled_loss = labeled_loss + lamb * w_i* F.binary_cross_entropy(hh_labeled_list[j,i], concepts[j, i].to(self.device))

            labeled_loss = torch.zeros([])
            if (-1) in self.args.which_c:
                labeled_loss = labeled_loss + F.binary_cross_entropy(
                    hh_labeled_list[0], concepts[0].to(self.device)
                )
                for j in range(1, len(hh_labeled_list)):
                    labeled_loss = (
                        labeled_loss
                        + F.binary_cross_entropy(
                            hh_labeled_list[j],
                            concepts[j].to(self.device),
                        )
                    )
                    # labeled_loss = labeled_loss + torch.nn.BCELoss()
                    F.binary_cross_entropy(
                        hh_labeled_list[j],
                        concepts[j].to(self.device),
                    )
            else:
                L = len(self.args.which_c)
                for i in range(21):
                    if i in self.args.which_c:
                        labeled_loss = (
                            F.binary_cross_entropy(
                                hh_labeled_list[0, i],
                                concepts[0, i].to(self.device),
                            )
                            / L
                        )
                        for j in range(1, len(hh_labeled_list)):
                            labeled_loss = (
                                labeled_loss
                                + F.binary_cross_entropy(
                                    hh_labeled_list[j, i],
                                    concepts[j, i].to(self.device),
                                )
                                / L
                            )

            # MSE loss version for known concepts
            # labeled_loss = F.mse_loss(hh_labeled_list,concepts)
            # labeled_loss = labeled_loss*len(concepts[0])

        if cbm:  # Standard CBM does not use decoder
            info_loss = self.eta * labeled_loss
        elif senn:
            info_loss = info_loss
        else:
            info_loss += self.eta * labeled_loss

        if not senn:
            # save loss (only value) to the all_losses list
            all_losses["labeled_h"] = (
                self.eta * labeled_loss.data.cpu().numpy()
            )

        # use in def train_batch (class GradPenaltyTrainer)
        return info_loss, hh_labeled_list

    def entropy_loss(self, pred_c, all_losses, epoch):

        # compute predicted known concepts by inputs
        # real uses the discriminator's loss
        avg_c = torch.mean(pred_c, dim=0)

        total_ent = -avg_c[0] * torch.log(avg_c[0]) - (
            1 - avg_c[0]
        ) * torch.log(1 - avg_c[0])
        total_ent /= np.log(2)
        for i in range(1, 21):
            ent_i = -avg_c[i] * torch.log(avg_c[i]) - (
                1 - avg_c[i]
            ) * torch.log(1 - avg_c[i])
            ent_i /= np.log(2)

            assert ent_i <= 1 and ent_i >= 0, (ent_i, avg_c[i])

            total_ent += ent_i

        total_ent = total_ent / 21
        assert total_ent >= 0 and total_ent <= 1, total_ent

        # save loss (only value) to the all_losses list
        all_losses["entropy"] = total_ent.data.cpu().numpy()

        # use in def train_batch (class GradPenaltyTrainer)
        return (1 - total_ent) * self.w_entropy

    # test the model (on the current model) and then save the results
    def test_and_save_csv(
        self,
        test_loader,
        save_file_name,
        fold=None,
        dropout=False,
        pcbm=False,
    ):

        print("Saving ", save_file_name, "...")

        def _activate_dropout():
            # enables dropout during test, useful for MC-dropout
            for m in self.model.modules():
                if m.__class__.__name__.startswith("Dropout"):
                    print("Trovato")
                    m.train()

        def _deactivate_dropout():
            # deactivates dropout during test, useful for MC-dropout
            for m in self.model.modules():
                if m.__class__.__name__.startswith("Dropout"):
                    m.eval()

        # initialization of print's values
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        topc1 = AverageMeter()

        if dropout:
            _activate_dropout()

        # switch to evaluate mode
        self.model.eval()

        end = time.time()

        # open the save file
        fp = open(save_file_name, "a")
        for i, (inputs, targets, concepts) in enumerate(test_loader):

            # get the inputs
            if self.cuda:
                inputs, targets, concepts = (
                    inputs.cuda(self.device),
                    targets.cuda(self.device),
                    concepts.cuda(self.device),
                )

            # compute output
            output = self.model(inputs)

            # prediction_criterion is defined in __init__ of "class GradPenaltyTrainer"
            loss = self.prediction_criterion(output, targets)

            # measure accuracy and record loss
            if self.nclasses > 4:
                # mainly use this line (current)
                prec1, _ = self.accuracy(
                    output.data, targets, topk=(1, 5)
                )
            elif self.nclasses == 3:
                prec1, _ = self.accuracy(
                    output.data, targets, topk=(1, 3)
                )
            else:
                prec1, _ = self.binary_accuracy(
                    output.data, targets
                ), [100]

            # update each value of print's values
            losses.update(loss.data.cpu().numpy(), inputs.size(0))
            top1.update(prec1[0], inputs.size(0))

            if pcbm:
                self.model.ignore_prob_log = True
                hh_labeled = self.model(inputs)
                self.model.ignore_prob_log = False
                _, hh, _ = self.model.conceptizer(inputs)
            else:
                # measure accuracy of concepts
                hh_labeled, hh, _ = self.model.conceptizer(inputs)

            if not self.args.senn:
                err = self.concept_error(hh_labeled.data, concepts)
                # update print's value
                topc1.update(err, inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if not self.args.senn:
                # if you use gpu, send results to cpu
                if self.cuda:
                    pred_labels = output.data.to("cpu")
                    # pred_labels = np.argmax(output.data.to("cpu"), axis=1)
                    concept_labels = hh_labeled.data.to("cpu")
                    concept_nolabels = hh.data.to("cpu")
                    attr = concepts.data.to("cpu")
                else:
                    pred_labels = output.data
                    concept_labels = hh_labeled
                    concept_nolabels = hh
                    attr = concepts

                # save to the file

                for j in range(len(targets)):
                    for k in range(len(targets[0])):
                        fp.write("%f," % (targets[j][k]))
                        # fp.write("%d,%d,"%(targets[j][k],pred_labels[j][k]))
                    for k in range(len(pred_labels[0])):
                        fp.write("%f," % (pred_labels[j][k]))
                    for k in range(concept_labels.shape[1]):
                        fp.write("%f," % (concept_labels[j][k]))
                    for k in range(concept_nolabels.shape[1]):
                        fp.write("%f," % (concept_nolabels[j][k]))
                    fp.write(",")
                    for k in range(attr.shape[1]):
                        fp.write("%f," % (attr[j][k]))
                    fp.write("\n")

                # print values of i-th iteration
                """
                if i % self.print_freq == 0:
                    fprint('Test on '+fold+': [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                           i, len(test_loader), batch_time=batch_time, loss=losses))
                """
            else:
                # if you use gpu, send results to cpu
                if self.cuda:
                    pred_labels = output.data.to("cpu")
                    # pred_labels = np.argmax(output.data.to("cpu"), axis=1)
                    concept_nolabels = hh.data.to("cpu")
                    attr = concepts.data.to("cpu")
                else:
                    pred_labels = output.data
                    concept_nolabels = hh
                    attr = concepts

                # save to the file
                for j in range(len(targets)):
                    for k in range(len(targets[0])):
                        fp.write("%f," % (targets[j][k]))
                        # fp.write("%d,%d,"%(targets[j][k],pred_labels[j][k]))
                    for k in range(len(pred_labels[0])):
                        fp.write("%f," % (pred_labels[j][k]))
                    for k in range(concept_nolabels.shape[1]):
                        fp.write("%f," % (concept_nolabels[j][k]))
                    fp.write(",")
                    for k in range(attr.shape[1]):
                        fp.write("%f," % (attr[j][k]))
                    fp.write("\n")

                """
                # print values of i-th iteration
                if i % self.print_freq == 0:
                    fprint('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                           i, len(test_loader), batch_time=batch_time, loss=losses))
                """

        if dropout:
            _deactivate_dropout()

        # close the save_file_name
        fp.close()

    # get the ensemble from the bayes method employed
    def get_ensemble_from_bayes(self, n_ensemble):
        raise NotImplemented("Not implemented for this method")

    # get the concept probability (factorized, not actual) for the ensemble
    def get_concept_probability_factorized_ensemble(
        self, ensemble, loader
    ):
        ensemble_c_prb = []

        for model in ensemble:
            model.eval()
            device = model.device

            c_prb = None
            c_true = None
            for i, data in enumerate(loader):
                images, labels, concepts = data
                images, labels, concepts = (
                    images.to(device),
                    labels.to(device),
                    concepts.to(device),
                )

                model.return_both_concept_out_prob = True
                lab, concept = model(images)
                model.return_both_concept_out_prob = False

                if i == 0:
                    c_prb = concept.detach().cpu().numpy()
                    c_true = concepts.detach().cpu().numpy()
                else:
                    c_prb = np.concatenate(
                        [c_prb, concept.detach().cpu().numpy()],
                        axis=0,
                    )
                    c_true = np.concatenate(
                        [c_true, concepts.detach().cpu().numpy()],
                        axis=0,
                    )

            # the groundtruth world
            gt_factorized_max = np.max(c_true, axis=-1)
            # it is basically a list of indexes where the maximum value (1) occours
            gt_factorized = [
                np.where(row == gt_factorized_max[i])[0]
                for i, row in enumerate(c_true)
            ]
            ensemble_c_prb.append(c_prb)

        # Average for each model in the ensemble
        avg_c_prb = np.mean(ensemble_c_prb, axis=0)

        # basically, take all the ensemble predictions and average them
        return avg_c_prb, gt_factorized


class Frequentist(ClassificationTester):
    def __init__(self, model, args, device):
        # call superclass
        super(Frequentist, self).__init__(model, args, device)

    def setup(
        self,
        train_loader,
        train_loader_no_shuffle,
        seeds,
        val_loader=None,
        epochs=10,
        save_path=None,
        separate_from_others=False,
        epsilon=0.01,
        lambda_h=1.0,
        lambda_kl=1.0,
    ):
        # No setup needed for the frequentist model
        pass

    def test_and_save(self, test_loader, save_file_name, fold=None):
        # just add frequentist at the end of the file name and then save it
        current_model_name = add_previous_dot(
            save_file_name, "_frequentist"
        )
        super().test_and_save(test_loader, current_model_name, fold)

    def frequentist_batch_prediction(
        self, batch_samples, apply_softmax=False
    ):
        # single prediction for the standard frequentist model
        self.model.eval()

        # activate the double return
        self.model.return_both_concept_out_prob = True
        lab, concept = self.model(batch_samples)
        # deactivate the double return
        self.model.return_both_concept_out_prob = False

        label_prob = lab.detach().cpu().numpy()
        concept_prob = concept.detach().cpu().numpy()

        label_prob = np.stack(label_prob, axis=0)
        concept_prob = np.stack(concept_prob, axis=0)

        if apply_softmax:
            label_prob = softmax(label_prob, axis=2)

        return label_prob, concept_prob

    def frequentist_prediction(self, loader, apply_softmax=False):
        # just the standard frequentist prediction
        for i, data in enumerate(loader):
            images, labels, concepts = data
            images, labels, concepts = (
                images.to(self.model.device),
                labels.to(self.model.device),
                concepts.to(self.model.device),
            )

            (label_prob, concept_prob) = (  # (256, 4)  # (256, 21)
                self.frequentist_batch_prediction(
                    images, apply_softmax
                )
            )

            # Concatenate the output
            if i == 0:
                y_true = labels.detach().cpu().numpy()
                c_true = concepts.detach().cpu().numpy()
                y_pred = label_prob
                pc_pred = concept_prob
            else:
                y_true = np.concatenate(
                    [y_true, labels.detach().cpu().numpy()], axis=0
                )
                c_true = np.concatenate(
                    [c_true, concepts.detach().cpu().numpy()], axis=0
                )
                y_pred = np.concatenate([y_pred, label_prob], axis=0)
                pc_pred = np.concatenate(
                    [pc_pred, concept_prob], axis=0
                )

        return y_true, c_true, y_pred, pc_pred

    def worlds_probability(
        self,
        loader,
        output_classes: int,
        num_concepts: int,
        n_ensembles: int = 30,
        apply_softmax=False,
    ):

        # get the prediction
        y_true, c_true, y_pred_org, pc_pred = (
            self.frequentist_prediction(loader, apply_softmax)
        )

        # unsqueeze it as it was an ensemble of a single prediction
        y_pred = np.expand_dims(y_pred_org, axis=0)
        pc_pred_ext = np.expand_dims(pc_pred, axis=0)

        fstop_prob = compute_forward_stop_prob(
            pc_pred_ext
        )  # data, possibleworlds
        left_prob = compute_left(pc_pred_ext)  # data, possibleworlds
        right_prob = compute_right(
            pc_pred_ext
        )  # data, possibleworlds
        y_prob = compute_output_probability(
            y_pred
        )  # data, possibleworlds

        # put all the probabilities together
        w_probs = [fstop_prob, left_prob, right_prob]

        # do the same for the predictions and the associated probability value
        w_predictions = []
        w_predictions_prob_value = []
        for prob in w_probs:
            w_predictions.append(np.argmax(prob, axis=-1))  # data, 1
            w_predictions_prob_value.append(
                np.max(prob, axis=-1)
            )  # data, 1

        # compute the ground_truth (which is simply the binary representation up of that slice)
        fstop_ground = compute_forward_stop_groundtruth(
            c_true
        )  # data, 1
        left_ground = compute_left_groundtruth(c_true)  # data, 1
        right_ground = compute_right_groundtruth(c_true)  # data, 1

        # w grountruths
        w_groundtruths = [fstop_ground, left_ground, right_ground]

        y_preds = torch.split(torch.tensor(y_prob), 2, dim=-1)
        y_trues = torch.split(torch.tensor(y_true), 1, dim=-1)

        y_preds_list = list(y_preds)
        y_preds_prob_list = list(y_preds)

        # Modify each array in the list
        for i in range(len(y_preds_list)):
            y_preds_list_tmp = y_preds_list[i].numpy()
            y_preds_list[i] = np.expand_dims(
                np.argmax(y_preds_list_tmp, axis=1), axis=-1
            )
            y_preds_prob_list[i] = np.expand_dims(
                np.max(y_preds_list_tmp, axis=1), axis=-1
            )

        y_true = np.concatenate(y_trues[:4], axis=-1)
        y_predictions = np.concatenate(y_preds_list[:4], axis=-1)
        y_predictions_prob = np.concatenate(
            y_preds_prob_list[:4], axis=-1
        )

        pc_prob = pc_pred
        pc_pred = (pc_prob > 0.5).astype(float)

        return (
            y_true,
            y_prob,
            y_predictions,
            y_predictions_prob,
            w_probs,
            w_predictions,
            w_groundtruths,
            w_predictions_prob_value,
            c_true,
            pc_prob,
            pc_pred,
            y_pred_org,
        )

    # silly ensemble to make things work
    def get_ensemble_from_bayes(self, n_ensemble):
        return [self.model]


class MCDropout(ClassificationTester):
    def __init__(self, model, args, device):
        super(MCDropout, self).__init__(model, args, device)

    def setup(
        self,
        train_loader,
        train_loader_no_shuffle,
        seeds,
        val_loader=None,
        epochs=10,
        save_path=None,
        separate_from_others=False,
        epsilon=0.01,
        lambda_h=1.0,
        lambda_kl=1.0,
    ):
        # No setup needed for the MCDropout model
        pass

    def _activate_dropout(self):
        # enables dropout during test, useful for MC-dropout
        for m in self.model.modules():
            if m.__class__.__name__.startswith("Dropout"):
                print("Trovato")
                m.train()

    def _deactivate_dropout(self):
        # deactivates dropout during test, useful for MC-dropout
        for m in self.model.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.eval()

    def test_and_save(self, test_loader, save_file_name, fold=None):
        # activates dropout for the model
        self._activate_dropout()
        # then call the save method with the different kind of evaluation procedures
        for i in range(self.num_mc_samples):
            current_model_name = add_previous_dot(
                save_file_name, f"_montecarlo_{i}"
            )
            super().test_and_save(
                test_loader, current_model_name, fold
            )

    def _montecarlo_dropout_single_batch(
        self,
        batch_samples: torch.tensor,
        num_mc_samples: int = 30,
        apply_softmax: bool = False,
    ):
        # basic montecarlo procedure for a single batch
        self.model.eval()

        # activate dropout during evaluation
        self._activate_dropout()

        # save the number of mc samples
        self.num_mc_samples = num_mc_samples

        # activate the double return
        self.model.return_both_concept_out_prob = True
        output_list = [
            self.model(batch_samples) for _ in range(num_mc_samples)
        ]  # 30
        # deactivate the double return
        self.model.return_both_concept_out_prob = False
        self._deactivate_dropout()

        label_prob = [
            lab.detach().cpu().numpy() for lab, _ in output_list
        ]  # 30
        concept_prob = [
            concept.detach().cpu().numpy()
            for _, concept in output_list
        ]  # 30

        label_prob = np.stack(label_prob, axis=0)
        concept_prob = np.stack(concept_prob, axis=0)

        if apply_softmax:
            label_prob = softmax(label_prob, axis=2)

        return label_prob, concept_prob

    def mc_dropout_predictions(
        self,
        loader,
        num_mc_samples: int = 30,
        apply_softmax=False,
    ):
        # dropout predictions, as one would expect
        for i, data in enumerate(loader):
            images, labels, concepts = data
            images, labels, concepts = (
                images.to(self.model.device),
                labels.to(self.model.device),
                concepts.to(self.model.device),
            )

            # Call MC Dropout
            (
                label_prob_ens,
                concept_prob_ens,
            ) = self._montecarlo_dropout_single_batch(  # (nmod, 256, 4)  # (nmod, 256, 21)
                images, num_mc_samples, apply_softmax
            )

            # Concatenate the output
            if i == 0:
                y_true = labels.detach().cpu().numpy()
                c_true = concepts.detach().cpu().numpy()
                y_pred = label_prob_ens
                pc_pred = concept_prob_ens
            else:
                y_true = np.concatenate(
                    [y_true, labels.detach().cpu().numpy()], axis=0
                )
                c_true = np.concatenate(
                    [c_true, concepts.detach().cpu().numpy()], axis=0
                )
                y_pred = np.concatenate(
                    [y_pred, label_prob_ens], axis=1
                )
                pc_pred = np.concatenate(
                    [pc_pred, concept_prob_ens], axis=1
                )

        return y_true, c_true, y_pred, pc_pred

    def worlds_probability(
        self,
        loader,
        output_classes: int,
        num_concepts: int,
        n_ensembles: int = 30,
        apply_softmax=False,
    ):

        y_true, c_true, y_pred, pc_pred = self.mc_dropout_predictions(
            loader, n_ensembles, apply_softmax
        )

        # no need to expand the dimensions as here we are still considering the ensemble, hence
        # we have the first dimension valid

        fstop_prob = compute_forward_stop_prob(
            pc_pred
        )  # data, possibleworlds
        left_prob = compute_left(pc_pred)  # data, possibleworlds
        right_prob = compute_right(pc_pred)  # data, possibleworlds
        y_prob = compute_output_probability(
            y_pred
        )  # data, possibleworlds

        w_probs = [fstop_prob, left_prob, right_prob]

        w_predictions = []
        w_predictions_prob_value = []
        for prob in w_probs:
            w_predictions.append(np.argmax(prob, axis=-1))  # data, 1
            w_predictions_prob_value.append(
                np.max(prob, axis=-1)
            )  # data, 1

        fstop_ground = compute_forward_stop_groundtruth(
            c_true
        )  # data, 1
        left_ground = compute_left_groundtruth(c_true)  # data, 1
        right_ground = compute_right_groundtruth(c_true)  # data, 1

        # w grountruths
        w_groundtruths = [fstop_ground, left_ground, right_ground]

        # get the predictions, probabilities and the groundtruth not one-ho
        y_mean_prob = np.mean(y_pred, axis=0)
        y_preds = torch.split(torch.tensor(y_mean_prob), 2, dim=-1)
        y_trues = torch.split(torch.tensor(y_true), 1, dim=-1)

        y_preds_list = list(y_preds)
        y_preds_prob_list = list(y_preds)

        # Modify each array in the list
        for i in range(len(y_preds_list)):
            y_preds_list_tmp = y_preds_list[i].numpy()
            y_preds_list[i] = np.expand_dims(
                np.argmax(y_preds_list_tmp, axis=1), axis=-1
            )
            y_preds_prob_list[i] = np.expand_dims(
                np.max(y_preds_list_tmp, axis=1), axis=-1
            )

        y_true = np.concatenate(y_trues[:4], axis=-1)
        y_predictions = np.concatenate(y_preds_list[:4], axis=-1)
        y_predictions_prob = np.concatenate(
            y_preds_prob_list[:4], axis=-1
        )

        pc_pred_mean = np.mean(pc_pred, axis=0)
        pc_prob = pc_pred_mean
        pc_pred = (pc_prob > 0.5).astype(float)

        return (
            y_true,
            y_prob,
            y_predictions,
            y_predictions_prob,
            w_probs,
            w_predictions,
            w_groundtruths,
            w_predictions_prob_value,
            c_true,
            pc_prob,
            pc_pred,
            y_mean_prob,
        )

    # MC DROPOUT
    def test_and_save_csv(
        self, test_loader, save_file_name, fold=None, pcbm=False
    ):

        print("Saving ", save_file_name, "...")

        # initialization of print's values
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        topc1 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        end = time.time()

        # open the save file
        fp = open(save_file_name, "a")
        for i, (inputs, targets, concepts) in enumerate(test_loader):

            # get the inputs
            if self.cuda:
                inputs, targets, concepts = (
                    inputs.cuda(self.device),
                    targets.cuda(self.device),
                    concepts.cuda(self.device),
                )

            # compute output
            (
                label_prob_ens,
                concept_prob_ens,
            ) = self._montecarlo_dropout_single_batch(  # (nmod, 256, 4)  # (nmod, 256, 21)
                inputs, 30, False
            )

            output = torch.tensor(np.mean(label_prob_ens, axis=0))
            pc_prob = torch.tensor(np.mean(concept_prob_ens, axis=0))

            if self.cuda:
                output, pc_prob = output.cuda(
                    self.device
                ), pc_prob.cuda(self.device)

            # prediction_criterion is defined in __init__ of "class GradPenaltyTrainer"
            loss = self.prediction_criterion(output, targets)

            # measure accuracy and record loss
            if self.nclasses > 4:
                # mainly use this line (current)
                prec1, _ = self.accuracy(
                    output.data, targets, topk=(1, 5)
                )
            elif self.nclasses == 3:
                prec1, _ = self.accuracy(
                    output.data, targets, topk=(1, 3)
                )
            else:
                prec1, _ = self.binary_accuracy(
                    output.data, targets
                ), [100]

            # update each value of print's values
            losses.update(loss.data.cpu().numpy(), inputs.size(0))
            top1.update(prec1[0], inputs.size(0))

            # measure accuracy of concepts
            _, hh, _ = self.model.conceptizer(inputs)
            if not self.args.senn:
                err = self.concept_error(pc_prob.data, concepts)

                # update print's value
                topc1.update(err, inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if not self.args.senn:
                # if you use gpu, send results to cpu
                if self.cuda:
                    pred_labels = output.data.to("cpu")
                    # pred_labels = np.argmax(output.data.to("cpu"), axis=1)
                    concept_labels = pc_prob.data.to("cpu")
                    concept_nolabels = hh.data.to("cpu")
                    attr = concepts.data.to("cpu")
                else:
                    pred_labels = output.data
                    concept_labels = pc_prob
                    concept_nolabels = hh
                    attr = concepts

                # save to the file
                for j in range(len(targets)):
                    for k in range(len(targets[0])):
                        fp.write("%f," % (targets[j][k]))
                        # fp.write("%d,%d,"%(targets[j][k],pred_labels[j][k]))
                    for k in range(len(pred_labels[0])):
                        fp.write("%f," % (pred_labels[j][k]))
                    for k in range(concept_labels.shape[1]):
                        fp.write("%f," % (concept_labels[j][k]))
                    for k in range(concept_nolabels.shape[1]):
                        fp.write("%f," % (concept_nolabels[j][k]))
                    fp.write(",")
                    for k in range(attr.shape[1]):
                        fp.write("%f," % (attr[j][k]))
                    fp.write("\n")

                # print values of i-th iteration
                """
                if i % self.print_freq == 0:
                    fprint('Test on '+fold+': [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                           i, len(test_loader), batch_time=batch_time, loss=losses))
                """
            else:
                # if you use gpu, send results to cpu
                if self.cuda:
                    pred_labels = output.data.to("cpu")
                    # pred_labels = np.argmax(output.data.to("cpu"), axis=1)
                    concept_nolabels = hh.data.to("cpu")
                    attr = concepts.data.to("cpu")
                else:
                    pred_labels = output.data
                    concept_nolabels = hh
                    attr = concepts

                # save to the file
                for j in range(len(targets)):
                    for k in range(len(targets[0])):
                        fp.write("%f," % (targets[j][k]))
                        # fp.write("%d,%d,"%(targets[j][k],pred_labels[j][k]))
                    for k in range(len(pred_labels[0])):
                        fp.write("%f," % (pred_labels[j][k]))
                    for k in range(concept_nolabels.shape[1]):
                        fp.write("%f," % (concept_nolabels[j][k]))
                    fp.write(",")
                    for k in range(attr.shape[1]):
                        fp.write("%f," % (attr[j][k]))
                    fp.write("\n")

                # print values of i-th iteration
                if i % self.print_freq == 0:
                    fprint(
                        "Test: [{0}/{1}]\t"
                        "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                        "Loss {loss.val:.4f} ({loss.avg:.4f})\t".format(
                            i,
                            len(test_loader),
                            batch_time=batch_time,
                            loss=losses,
                        )
                    )

        # close the save_file_name
        fp.close()

    def p_c_x_distance(
        self,
        loader,
    ):
        # model predictions for all the categorization we have made
        model_pred = [[], [], []]

        for _ in range(self.num_mc_samples):
            self.model.eval()
            self._activate_dropout()

            for i, data in enumerate(loader):
                images, labels, concepts = data
                images, labels, concepts = (
                    images.to(self.model.device),
                    labels.to(self.model.device),
                    concepts.to(self.model.device),
                )

                # activate the double return
                self.model.return_both_concept_out_prob = True
                output = self.model(images)
                # deactivate the double return
                self.model.return_both_concept_out_prob = False
                self._deactivate_dropout()

                label_prob, concept_prob = output

                concept_prob = np.expand_dims(
                    concept_prob.detach().cpu().numpy(), axis=0
                )

                if i == 0:
                    c_prb = concept_prob
                else:
                    c_prb = np.concatenate(
                        [c_prb, concept_prob], axis=1
                    )

            # get the worlds probabilities by computing the matmultiplication
            fstop_prob = compute_forward_stop_prob(
                c_prb
            )  # data, possibleworlds
            left_prob = compute_left(c_prb)  # data, possibleworlds
            right_prob = compute_right(c_prb)  # data, possibleworlds

            # append the alpha matrices to the list of lists
            model_pred[0].append(fstop_prob)
            model_pred[1].append(left_prob)
            model_pred[2].append(right_prob)

        # compute the distance on those lists
        return (
            self.mean_l2_distance(model_pred[0]),
            self.mean_l2_distance(model_pred[1]),
            self.mean_l2_distance(model_pred[2]),
        )

    # just the same model with dropout activated
    def get_ensemble_from_bayes(self, n_ensemble):
        self._activate_dropout()
        return [self.model for _ in range(n_ensemble)]


class LaplaceBayes(ClassificationTester):
    def __init__(self, model, args, device):
        super(LaplaceBayes, self).__init__(model, args, device)

    def setup(
        self,
        train_loader,
        train_loader_no_shuffle,
        seeds,
        val_loader=None,
        epochs=10,
        save_path=None,
        separate_from_others=False,
        epsilon=0.01,
        lambda_h=1.0,
        lambda_kl=1.0,
    ):
        # Laplace model is given in the setup, which performs the laplace approximation
        # same stuff as always, compute the hessian around the MAP point on the loss function, as use it
        # as a multivariate gaussian distribution to sample new model weigths
        self.laplace_model = self.laplace_approximation(
            train_loader, val_loader
        )

    # get the ensemble, and run the test on that specific ensemble!
    def test_and_save(self, test_loader, save_file_name, fold=None):
        ensemble = self.laplace_model.model.model.get_ensembles(
            self.laplace_model, self.n_ensembles
        )
        for i in range(self.num_mc_samples):
            current_model_name = add_previous_dot(
                save_file_name, f"_laplace_{i}"
            )
            self.model = ensemble[i]  # override the model
            super().test_and_save(
                test_loader, current_model_name, fold
            )

    def laplace_approximation(self, train_loader, val_loader):
        # usual laplace approximation hook
        from laplace.curvature import AsdlGGN
        from torch.utils.data import DataLoader

        def new_model_copy(to_be_copied):
            # only "fcc" conceptizer use, otherwise cannot use (not modifile so as to fit this task...)
            if self.h_type == "fcc":
                conceptizer1 = image_fcc_conceptizer(
                    2048,
                    self.nconcepts,
                    self.nconcepts_labeled,
                    self.concept_dim,
                    self.h_sparsity,
                    self.senn,
                )
            elif self.h_type == "cnn":
                fprint("[ERROR] please use fcc network")
                sys.exit(1)
            else:
                fprint("[ERROR] please use fcc network")
                sys.exit(1)

            parametrizer1 = dfc_parametrizer(
                2048,
                1024,
                512,
                256,
                128,
                self.nconcepts,
                self.theta_dim,
                layers=4,
            )

            if self.cbm == True:
                aggregator = CBM_aggregator(
                    self.concept_dim,
                    self.nclasses,
                    self.nconcepts_labeled,
                )
            else:
                aggregator = additive_scalar_aggregator(
                    self.concept_dim, self.nclasses
                )

            if self.model_name == "dpl":
                model = DPL(
                    conceptizer1,
                    parametrizer1,
                    aggregator,
                    self.cbm,
                    self.senn,
                    self.device,
                )
            elif self.model_name == "dpl_auc":
                model = DPL_AUC(
                    conceptizer1,
                    parametrizer1,
                    aggregator,
                    self.cbm,
                    self.senn,
                    self.device,
                )

            # send models to device you want to use
            model = model.to(self.device)
            model.load_state_dict(
                copy.deepcopy(to_be_copied.state_dict())
            )
            return model

        # Wrapper DataLoader
        class WrapperDataLoader(DataLoader):
            def __init__(self, original_dataloader, **kwargs):
                super(WrapperDataLoader, self).__init__(
                    dataset=original_dataloader.dataset, **kwargs
                )

            def __iter__(self):
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
            def __init__(
                self, original_model, device, output_all=False
            ):
                super(WrapperModel, self).__init__()
                self.original_model = original_model
                self.original_model.to(device)
                self.output_all = output_all
                self.model_possibilities = list()
                self.device = device

            def forward(self, input_batch):
                batch_size = input_batch.shape[0]

                # Call the forward method of the model
                original_output = self.original_model(input_batch)
                concept_p_x, h_x, _ = self.original_model.conceptizer(
                    input_batch
                )

                # torch.Size([batch, 19]) torch.Size([batch, 2, 10]) torch.Size([batch, 2, 10])
                if not self.output_all:
                    return original_output

                # I want to flat all the tensors in this way:
                return torch.cat(
                    (original_output, concept_p_x), dim=1
                )

            def get_ensembles(self, la_model, n_models):

                ensembles = []

                torch.set_printoptions(profile="full")
                np.set_printoptions(threshold=sys.maxsize)

                for i, mp in enumerate(self.model_possibilities):
                    _vector_to_parameters(
                        mp, la_model.model.last_layer.parameters()
                    )
                    ensembles.append(
                        new_model_copy(
                            la_model.model.model.original_model
                        )
                    )
                    if i == n_models - 1:
                        break

                # restore original model
                _vector_to_parameters(
                    la_model.mean,
                    la_model.model.last_layer.parameters(),
                )
                # return an ensembles of models
                return ensembles

        # wrap the dataloaders
        la_training_loader = WrapperDataLoader(train_loader)
        la_val_loader = WrapperDataLoader(val_loader)

        # wrap the model
        la_model = WrapperModel(
            new_model_copy(self.model), self.model.device
        )
        la_model.to(self.model.device)

        la = Laplace(
            la_model,
            "classification",
            subset_of_weights="last_layer",  # subset_of_weights='subnetwork',
            hessian_structure="diag",  # hessian_structure='full', # hessian_structure='diag', # hessian_structure='kron',
            backend=AsdlGGN,
            backend_kwargs={"boia": True},
        )

        return self._fit_la_model(
            la, la_training_loader, la_val_loader
        )

    # fit the laplace model
    def _fit_la_model(self, la, la_training_loader, la_val_loader):
        fprint("Doing Laplace fit...")
        la.fit(la_training_loader)
        la.optimize_prior_precision(
            method="marglik", val_loader=la_val_loader
        )
        # Enabling last layer output all
        la.model.model.output_all = True
        return la

    def test_and_save_csv(
        self, test_loader, save_file_name, fold=None, pcbm=False
    ):

        # initialization of print's values
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        topc1 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        # open the save file
        fp = open(save_file_name, "a")
        for i, (inputs, targets, concepts) in enumerate(test_loader):

            # get the inputs
            if self.cuda:
                inputs, targets, concepts = (
                    inputs.cuda(self.device),
                    targets.cuda(self.device),
                    concepts.cuda(self.device),
                )

            _ = self.laplace_single_prediction(
                self.laplace_model, inputs, 5, 21, False
            )

            # Call Laplace ensembles
            ensemble = self.laplace_model.model.model.get_ensembles(
                self.laplace_model, 30
            )

            # Call Ensemble predict (same as ensemble)
            (label_prob_ens, concept_prob_ens) = (
                self.ensemble_single_la_predict(
                    ensemble, inputs, False
                )
            )

            output = torch.tensor(np.mean(label_prob_ens, axis=0))
            pc_prob = torch.tensor(np.mean(concept_prob_ens, axis=0))

            if self.cuda:
                output, pc_prob = output.cuda(
                    self.device
                ), pc_prob.cuda(self.device)

            # prediction_criterion is defined in __init__ of "class GradPenaltyTrainer"
            loss = self.prediction_criterion(output, targets)

            # measure accuracy and record loss
            if self.nclasses > 4:
                # mainly use this line (current)
                prec1, _ = self.accuracy(
                    output.data, targets, topk=(1, 5)
                )
            elif self.nclasses == 3:
                prec1, _ = self.accuracy(
                    output.data, targets, topk=(1, 3)
                )
            else:
                prec1, _ = self.binary_accuracy(
                    output.data, targets
                ), [100]

            # update each value of print's values
            losses.update(loss.data.cpu().numpy(), inputs.size(0))
            top1.update(prec1[0], inputs.size(0))

            # measure accuracy of concepts
            _, hh, _ = self.model.conceptizer(inputs)
            if not self.args.senn:
                err = self.concept_error(pc_prob.data, concepts)

                # update print's value
                topc1.update(err, inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if not self.args.senn:
                # if you use gpu, send results to cpu
                if self.cuda:
                    pred_labels = output.data.to("cpu")
                    # pred_labels = np.argmax(output.data.to("cpu"), axis=1)
                    concept_labels = pc_prob.data.to("cpu")
                    concept_nolabels = hh.data.to("cpu")
                    attr = concepts.data.to("cpu")
                else:
                    pred_labels = output.data
                    concept_labels = pc_prob
                    concept_nolabels = hh
                    attr = concepts

                # save to the file
                for j in range(len(targets)):
                    for k in range(len(targets[0])):
                        fp.write("%f," % (targets[j][k]))
                        # fp.write("%d,%d,"%(targets[j][k],pred_labels[j][k]))
                    for k in range(len(pred_labels[0])):
                        fp.write("%f," % (pred_labels[j][k]))
                    for k in range(concept_labels.shape[1]):
                        fp.write("%f," % (concept_labels[j][k]))
                    for k in range(concept_nolabels.shape[1]):
                        fp.write("%f," % (concept_nolabels[j][k]))
                    fp.write(",")
                    for k in range(attr.shape[1]):
                        fp.write("%f," % (attr[j][k]))
                    fp.write("\n")

                # print values of i-th iteration
                if i % self.print_freq == 0:
                    fprint(
                        "Test on " + fold + ": [{0}/{1}]\t"
                        "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                        "Loss {loss.val:.4f} ({loss.avg:.4f})\t".format(
                            i,
                            len(test_loader),
                            batch_time=batch_time,
                            loss=losses,
                        )
                    )

            else:
                # if you use gpu, send results to cpu
                if self.cuda:
                    pred_labels = output.data.to("cpu")
                    # pred_labels = np.argmax(output.data.to("cpu"), axis=1)
                    concept_nolabels = hh.data.to("cpu")
                    attr = concepts.data.to("cpu")
                else:
                    pred_labels = output.data
                    concept_nolabels = hh
                    attr = concepts

                # save to the file
                for j in range(len(targets)):
                    for k in range(len(targets[0])):
                        fp.write("%f," % (targets[j][k]))
                        # fp.write("%d,%d,"%(targets[j][k],pred_labels[j][k]))
                    for k in range(len(pred_labels[0])):
                        fp.write("%f," % (pred_labels[j][k]))
                    for k in range(concept_nolabels.shape[1]):
                        fp.write("%f," % (concept_nolabels[j][k]))
                    fp.write(",")
                    for k in range(attr.shape[1]):
                        fp.write("%f," % (attr[j][k]))
                    fp.write("\n")

                # print values of i-th iteration
                if i % self.print_freq == 0:
                    fprint(
                        "Test: [{0}/{1}]\t"
                        "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                        "Loss {loss.val:.4f} ({loss.avg:.4f})\t".format(
                            i,
                            len(test_loader),
                            batch_time=batch_time,
                            loss=losses,
                        )
                    )

        # close the save_file_name
        fp.close()

    def laplace_prediction(
        self,
        laplace_model,
        loader,
        n_ensembles: int,
        output_classes: int,
        num_concepts: int,
        apply_softmax=False,
    ):
        # save the number of ensembles
        self.n_ensembles = n_ensembles

        # Loop over the dataloader
        for i, data in enumerate(loader):
            images, labels, concepts = data
            images, labels, concepts = (
                images.to(self.model.device),
                labels.to(self.model.device),
                concepts.to(self.model.device),
            )

            # prediction (this one is only needed for making laplace store the models inside the WrapperModel)
            _ = self.laplace_single_prediction(
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

            # Call Ensemble predict (same as ensemble)
            (label_prob_ens, concept_prob_ens) = (
                self.ensemble_single_la_predict(
                    ensemble, images, apply_softmax
                )
            )

            # Concatenate the output
            if i == 0:
                y_true = labels.detach().cpu().numpy()
                c_true = concepts.detach().cpu().numpy()
                y_pred = label_prob_ens
                pc_pred = concept_prob_ens
            else:
                y_true = np.concatenate(
                    [y_true, labels.detach().cpu().numpy()], axis=0
                )
                c_true = np.concatenate(
                    [c_true, concepts.detach().cpu().numpy()], axis=0
                )
                y_pred = np.concatenate(
                    [y_pred, label_prob_ens], axis=1
                )
                pc_pred = np.concatenate(
                    [pc_pred, concept_prob_ens], axis=1
                )

        return y_true, c_true, y_pred, pc_pred

    # Use a prediction with laplace (this is basically a deep ensemble internally)
    def laplace_single_prediction(
        self,
        la,
        sample_batch: ndarray,
        output_classes: int,
        num_concepts: int,
        apply_softmax=False,
    ):
        pred = la(sample_batch, pred_type="nn", link_approx="mc")
        recovered_pred = self.recover_predictions_from_laplace(
            pred,
            sample_batch.shape[0],
            output_classes,
            num_concepts,
            apply_softmax,
        )
        return recovered_pred

    # from the laplace predicion (due to the wrapper hook) get the original things predicted
    def recover_predictions_from_laplace(
        self,
        la_prediction,
        batch_size,
        output_classes: int = 19,
        num_concepts: int = 10,
        apply_softmax=False,
    ):
        # Recovering shape
        ys = la_prediction[
            :, :output_classes
        ]  # take all until output_classes
        pCS = la_prediction[
            :, output_classes:
        ]  # take all from output_classes until the end

        if apply_softmax:
            import torch.nn.functional as F

            py = F.softmax(py, dim=1)

        return ys, pCS

    # single batch prediction for the laplace model
    def ensemble_single_la_predict(
        self, models, batch_samples: torch.tensor, apply_softmax=False
    ):

        # activate the double return
        for model in models:
            model.eval()
            model.return_both_concept_out_prob = True

        output_list = [model(batch_samples) for model in models]

        # deactivate the double return
        for model in models:
            model.return_both_concept_out_prob = False

        # get out the different output
        label_prob = [
            lab.detach().cpu().numpy() for lab, _ in output_list
        ]  # 30
        concept_prob = [
            concept.detach().cpu().numpy()
            for _, concept in output_list
        ]  # 30

        label_prob = np.stack(label_prob, axis=0)
        concept_prob = np.stack(concept_prob, axis=0)

        if apply_softmax:
            label_prob = softmax(label_prob, axis=2)

        return label_prob, concept_prob

    # worlds probabilty as always
    def worlds_probability(
        self,
        loader,
        output_classes: int,
        num_concepts: int,
        n_ensembles: int = 30,
        apply_softmax=False,
    ):

        y_true, c_true, y_pred, pc_pred = self.laplace_prediction(
            self.laplace_model,
            loader,
            n_ensembles,
            output_classes,
            num_concepts,
            apply_softmax,
        )

        fstop_prob = compute_forward_stop_prob(
            pc_pred
        )  # data, possibleworlds
        left_prob = compute_left(pc_pred)  # data, possibleworlds
        right_prob = compute_right(pc_pred)  # data, possibleworlds
        y_prob = compute_output_probability(
            y_pred
        )  # data, possibleworlds

        w_probs = [fstop_prob, left_prob, right_prob]

        w_predictions = []
        w_predictions_prob_value = []
        for prob in w_probs:
            w_predictions.append(np.argmax(prob, axis=-1))  # data, 1
            w_predictions_prob_value.append(
                np.max(prob, axis=-1)
            )  # data, 1

        fstop_ground = compute_forward_stop_groundtruth(
            c_true
        )  # data, 1
        left_ground = compute_left_groundtruth(c_true)  # data, 1
        right_ground = compute_right_groundtruth(c_true)  # data, 1

        # w grountruths
        w_groundtruths = [fstop_ground, left_ground, right_ground]

        # get the predictions, probabilities and the groundtruth not one-ho
        y_mean_prob = np.mean(y_pred, axis=0)
        y_preds = torch.split(torch.tensor(y_mean_prob), 2, dim=-1)
        y_trues = torch.split(torch.tensor(y_true), 1, dim=-1)

        y_preds_list = list(y_preds)
        y_preds_prob_list = list(y_preds)

        # Modify each array in the list
        for i in range(len(y_preds_list)):
            y_preds_list_tmp = y_preds_list[i].numpy()
            y_preds_list[i] = np.expand_dims(
                np.argmax(y_preds_list_tmp, axis=1), axis=-1
            )
            y_preds_prob_list[i] = np.expand_dims(
                np.max(y_preds_list_tmp, axis=1), axis=-1
            )

        y_true = np.concatenate(y_trues[:4], axis=-1)
        y_predictions = np.concatenate(y_preds_list[:4], axis=-1)
        y_predictions_prob = np.concatenate(
            y_preds_prob_list[:4], axis=-1
        )

        pc_pred_mean = np.mean(pc_pred, axis=0)
        pc_prob = pc_pred_mean
        pc_pred = (pc_prob > 0.5).astype(float)

        return (
            y_true,
            y_prob,
            y_predictions,
            y_predictions_prob,
            w_probs,
            w_predictions,
            w_groundtruths,
            w_predictions_prob_value,
            c_true,
            pc_prob,
            pc_pred,
            y_mean_prob,
        )

    # distance, similar if not identical as the one I did with ensemble and mcdropout
    def p_c_x_distance(
        self,
        loader,
    ):
        model_pred = [[], [], []]

        ensemble = self.laplace_model.model.model.get_ensembles(
            self.laplace_model, self.n_ensembles
        )

        for j in range(len(ensemble)):
            self.model = ensemble[j]
            self.model.eval()

            for i, data in enumerate(loader):
                images, labels, concepts = data
                images, labels, concepts = (
                    images.to(self.model.device),
                    labels.to(self.model.device),
                    concepts.to(self.model.device),
                )

                # activate the double return
                self.model.return_both_concept_out_prob = True
                output = self.model(images)
                # deactivate the double return
                self.model.return_both_concept_out_prob = False

                label_prob, concept_prob = output
                concept_prob = np.expand_dims(
                    concept_prob.detach().cpu().numpy(), axis=0
                )

                if i == 0:
                    c_prb = concept_prob
                else:
                    c_prb = np.concatenate(
                        [c_prb, concept_prob], axis=1
                    )

            fstop_prob = compute_forward_stop_prob(
                c_prb
            )  # data, possibleworlds
            left_prob = compute_left(c_prb)  # data, possibleworlds
            right_prob = compute_right(c_prb)  # data, possibleworlds

            # append the alpha matrices to the list of lists
            model_pred[0].append(fstop_prob)
            model_pred[1].append(left_prob)
            model_pred[2].append(right_prob)

        return (
            self.mean_l2_distance(model_pred[0]),
            self.mean_l2_distance(model_pred[1]),
            self.mean_l2_distance(model_pred[2]),
        )

    def get_ensemble_from_bayes(self, n_ensemble, inputs):
        _ = self.laplace_single_prediction(
            self.laplace_model, inputs, 5, 21, False
        )

        ensemble = self.laplace_model.model.model.get_ensembles(
            self.laplace_model, n_ensemble
        )
        print("Ense", len(ensemble), type(ensemble))
        return ensemble


class DeepEnsembles(ClassificationTester):
    def __init__(self, model, args, device, name):
        self.name = name
        self.n_models = args.n_models
        self.exp_decay_lr = args.exp_decay_lr
        self.knowledge_aware_kl = args.knowledge_aware_kl

        # call superclass
        super(DeepEnsembles, self).__init__(model, args, device)

    # as previously, override the current model and then deepens!
    def test_and_save(self, test_loader, save_file_name, fold=None):
        for i in range(len(self.ensemble)):
            current_model_name = add_previous_dot(
                save_file_name, f"_{self.name}_{i}"
            )
            self.model = self.ensemble[i]  # override the model
            super().test_and_save(
                test_loader, current_model_name, fold
            )

    def test_and_save_csv(
        self, test_loader, save_file_name, fold=None, pcbm=False
    ):

        print("Saving...", save_file_name)

        # initialization of print's values
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        topc1 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        end = time.time()

        # open the save file
        fp = open(save_file_name, "a")
        for i, (inputs, targets, concepts) in enumerate(test_loader):

            # get the inputs
            if self.cuda:
                inputs, targets, concepts = (
                    inputs.cuda(self.device),
                    targets.cuda(self.device),
                    concepts.cuda(self.device),
                )

            # compute output
            (label_prob_ens, concept_prob_ens) = (
                self._ensemble_single_predict(
                    self.ensemble, inputs, False
                )
            )

            output = torch.tensor(np.mean(label_prob_ens, axis=0))
            pc_prob = torch.tensor(np.mean(concept_prob_ens, axis=0))

            if self.cuda:
                output, pc_prob = output.cuda(
                    self.device
                ), pc_prob.cuda(self.device)

            # prediction_criterion is defined in __init__ of "class GradPenaltyTrainer"
            loss = self.prediction_criterion(
                torch.tensor(output), targets
            )

            # measure accuracy and record loss
            if self.nclasses > 4:
                # mainly use this line (current)
                prec1, _ = self.accuracy(
                    output.data, targets, topk=(1, 5)
                )
            elif self.nclasses == 3:
                prec1, _ = self.accuracy(
                    output.data, targets, topk=(1, 3)
                )
            else:
                prec1, _ = self.binary_accuracy(
                    output.data, targets
                ), [100]

            # update each value of print's values
            losses.update(loss.data.cpu().numpy(), inputs.size(0))
            top1.update(prec1[0], inputs.size(0))

            # measure accuracy of concepts
            _, hh, _ = self.model.conceptizer(inputs)
            if not self.args.senn:
                err = self.concept_error(pc_prob.data, concepts)

                # update print's value
                topc1.update(err, inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if not self.args.senn:
                # if you use gpu, send results to cpu
                if self.cuda:
                    pred_labels = output.data.to("cpu")
                    # pred_labels = np.argmax(output.data.to("cpu"), axis=1)
                    concept_labels = pc_prob.data.to("cpu")
                    concept_nolabels = hh.data.to("cpu")
                    attr = concepts.data.to("cpu")
                else:
                    pred_labels = output.data
                    concept_labels = pc_prob
                    concept_nolabels = hh
                    attr = concepts

                # save to the file
                for j in range(len(targets)):
                    for k in range(len(targets[0])):
                        fp.write("%f," % (targets[j][k]))
                        # fp.write("%d,%d,"%(targets[j][k],pred_labels[j][k]))
                    for k in range(len(pred_labels[0])):
                        fp.write("%f," % (pred_labels[j][k]))
                    for k in range(concept_labels.shape[1]):
                        fp.write("%f," % (concept_labels[j][k]))
                    for k in range(concept_nolabels.shape[1]):
                        fp.write("%f," % (concept_nolabels[j][k]))
                    fp.write(",")
                    for k in range(attr.shape[1]):
                        fp.write("%f," % (attr[j][k]))
                    fp.write("\n")

                # print values of i-th iteration
                if i % self.print_freq == 0:
                    fprint(
                        "Test on " + fold + ": [{0}/{1}]\t"
                        "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                        "Loss {loss.val:.4f} ({loss.avg:.4f})\t".format(
                            i,
                            len(test_loader),
                            batch_time=batch_time,
                            loss=losses,
                        )
                    )

            else:
                # if you use gpu, send results to cpu
                if self.cuda:
                    pred_labels = output.data.to("cpu")
                    # pred_labels = np.argmax(output.data.to("cpu"), axis=1)
                    concept_nolabels = hh.data.to("cpu")
                    attr = concepts.data.to("cpu")
                else:
                    pred_labels = output.data
                    concept_nolabels = hh
                    attr = concepts

                # save to the file
                for j in range(len(targets)):
                    for k in range(len(targets[0])):
                        fp.write("%f," % (targets[j][k]))
                        # fp.write("%d,%d,"%(targets[j][k],pred_labels[j][k]))
                    for k in range(len(pred_labels[0])):
                        fp.write("%f," % (pred_labels[j][k]))
                    for k in range(concept_nolabels.shape[1]):
                        fp.write("%f," % (concept_nolabels[j][k]))
                    fp.write(",")
                    for k in range(attr.shape[1]):
                        fp.write("%f," % (attr[j][k]))
                    fp.write("\n")

                # print values of i-th iteration
                if i % self.print_freq == 0:
                    fprint(
                        "Test: [{0}/{1}]\t"
                        "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                        "Loss {loss.val:.4f} ({loss.avg:.4f})\t".format(
                            i,
                            len(test_loader),
                            batch_time=batch_time,
                            loss=losses,
                        )
                    )

        # close the save_file_name
        fp.close()

    # populate the dataset with p(c|x) for the deep diversification
    def _populate_pcx_dataset(self, model, pcx_loader, batch_size):
        c_prb_list = list()
        images_list = list()

        for _, data in enumerate(pcx_loader):
            images, labels, concepts = data
            images, labels, concepts = (
                images.to(model.device),
                labels.to(model.device),
                concepts.to(model.device),
            )

            # Get concept probability
            p_c_x, c_x, _ = model.conceptizer(images)

            # Append the new tensor
            c_prb_tmp = torch.chunk(p_c_x, images.size(0), dim=0)
            tmp_img_list = torch.chunk(images, images.size(0), dim=0)

            images_list.extend(tmp_img_list)

            for sublist in c_prb_tmp:
                c_prb_list.append([sublist])

        dataset = DatasetPcX(images=images_list, pcx=c_prb_list)

        return dataset

    def save_model_params_all(
        self, save_path, separate_from_others, lambda_h
    ):
        for i, model in enumerate(self.ensemble):
            file_name = (
                f"dens-{i}-seed-{self.seed}-lambda-h-{lambda_h}-real_kl.pth"
                if not separate_from_others
                else f"biretta-{i}-seed-{self.seed}-lambda-h-{lambda_h}.pth"
            )
            super().save_model_params(model, save_path, file_name)

    def _populate_pcx_dataset_knowledge_aware(
        self, model, pcx_loader, n_facts
    ):
        print("Initializing PWX database...")

        from DPL.utils_problog import (
            build_world_queries_matrix_complete_FS,
            build_world_queries_matrix_L,
            build_world_queries_matrix_R,
        )

        FS_w_q = build_world_queries_matrix_complete_FS().to(
            self.model.device
        )
        L_w_q = build_world_queries_matrix_L().to(self.model.device)
        R_w_q = build_world_queries_matrix_R().to(self.model.device)

        images_list = list()

        w_fstop_prob_list = list()
        w_left_prob_list = list()
        w_right_prob_list = list()

        for _, data in enumerate(pcx_loader):
            images, labels, concepts = data
            images, labels, concepts = (
                images.to(model.device),
                labels.to(model.device),
                concepts.to(model.device),
            )

            # Get concept probability
            p_c_x, c_x, _ = model.conceptizer(images)

            p_c_x_ext = torch.unsqueeze(p_c_x, dim=0)

            fstop_prob = compute_forward_stop_prob(
                p_c_x_ext, numpy=False
            )  # data, possibleworlds
            left_prob = compute_left(
                p_c_x_ext, numpy=False
            )  # data, possibleworlds
            right_prob = compute_right(
                p_c_x_ext, numpy=False
            )  # data, possibleworlds

            # wfstop list
            w_fstop_tmp = []
            w_left_tmp = []
            w_right_tmp = []

            label_indices = labels.long()

            FS_w_q_transposed = FS_w_q.t()
            L_w_q_transposed = L_w_q.t()
            R_w_q_transposed = R_w_q.t()

            # Extracting probabilities using indexing and performing element-wise multiplication
            fstop_prob = (
                fstop_prob * FS_w_q_transposed[label_indices[:, 0], :]
            )
            left_prob = (
                left_prob * L_w_q_transposed[label_indices[:, 1], :]
            )
            right_prob = (
                right_prob * R_w_q_transposed[label_indices[:, 2], :]
            )

            # Adding a small constant for normalization
            fstop_prob += 1e-5
            left_prob += 1e-5
            right_prob += 1e-5

            # Computing normalization constants
            Z_fs = fstop_prob.sum(dim=1, keepdim=True)
            Z_l = left_prob.sum(dim=1, keepdim=True)
            Z_r = right_prob.sum(dim=1, keepdim=True)

            # Normalizing probabilities
            fstop_prob /= Z_fs
            left_prob /= Z_l
            right_prob /= Z_r

            # Reshaping and appending to temporary lists
            w_fstop_tmp = list(
                map(
                    lambda x: x.squeeze(0),
                    torch.split(fstop_prob, 1, dim=0),
                )
            )
            w_left_tmp = list(
                map(
                    lambda x: x.squeeze(0),
                    torch.split(left_prob, 1, dim=0),
                )
            )
            w_right_tmp = list(
                map(
                    lambda x: x.squeeze(0),
                    torch.split(right_prob, 1, dim=0),
                )
            )

            # Append the new tensor
            tmp_img_list = torch.chunk(images, images.size(0), dim=0)

            images_list.extend(tmp_img_list)

            for sublist in w_fstop_tmp:
                w_fstop_prob_list.append([sublist])

            for sublist in w_left_tmp:
                w_left_prob_list.append([sublist])

            for sublist in w_right_tmp:
                w_right_prob_list.append([sublist])

        dataset = DatasetPcX(
            images=images_list,
            pcx=[],
            mode=True,
            w_fstop_prob_list=w_fstop_prob_list,
            w_left_prob_list=w_left_prob_list,
            w_right_prob_list=w_right_prob_list,
            FS_w_q=FS_w_q,
            L_w_q=L_w_q,
            R_w_q=R_w_q,
        )

        return dataset

    # update the dataset for the deep diversification
    def _update_pcx_dataset(
        self, model, dataset, pcx_loader, batch_size
    ):
        indexes = 0

        for _, data in enumerate(pcx_loader):
            images, labels, concepts = data
            images, labels, concepts = (
                images.to(model.device),
                labels.to(model.device),
                concepts.to(model.device),
            )

            # Get concept probability
            p_c_x, c_x, _ = model.conceptizer(images)

            to_append = torch.chunk(p_c_x, images.size(0), dim=0)

            j = 0
            for i in range(indexes, indexes + images.size(0)):
                dataset.pcx[i].append(to_append[j])
                j += 1
            indexes += images.size(0)

        return dataset

    def _update_pcx_dataset_knowledge_aware(
        self, model, dataset, pcx_loader, n_facts
    ):
        print("Updating PWX database...")
        indexes = 0

        for _, data in enumerate(pcx_loader):
            images, labels, concepts = data
            images, labels, concepts = (
                images.to(model.device),
                labels.to(model.device),
                concepts.to(model.device),
            )

            # Get concept probability
            p_c_x, c_x, _ = model.conceptizer(images)

            p_c_x_ext = torch.unsqueeze(p_c_x, dim=0)

            fstop_prob = compute_forward_stop_prob(
                p_c_x_ext, numpy=False
            )  # data, possibleworlds
            left_prob = compute_left(
                p_c_x_ext, numpy=False
            )  # data, possibleworlds
            right_prob = compute_right(
                p_c_x_ext, numpy=False
            )  # data, possibleworlds

            # wfstop list
            w_fstop_tmp = []
            w_left_tmp = []
            w_right_tmp = []

            label_indices = labels.long()

            FS_w_q_transposed = dataset.FS_w_q.t()
            L_w_q_transposed = dataset.L_w_q.t()
            R_w_q_transposed = dataset.R_w_q.t()

            # Extracting probabilities using indexing and performing element-wise multiplication
            fstop_prob = (
                fstop_prob * FS_w_q_transposed[label_indices[:, 0], :]
            )
            left_prob = (
                left_prob * L_w_q_transposed[label_indices[:, 1], :]
            )
            right_prob = (
                right_prob * R_w_q_transposed[label_indices[:, 2], :]
            )

            # Adding a small constant for normalization
            fstop_prob += 1e-5
            left_prob += 1e-5
            right_prob += 1e-5

            # Computing normalization constants
            Z_fs = fstop_prob.sum(dim=1, keepdim=True)
            Z_l = left_prob.sum(dim=1, keepdim=True)
            Z_r = right_prob.sum(dim=1, keepdim=True)

            # Normalizing probabilities
            fstop_prob /= Z_fs
            left_prob /= Z_l
            right_prob /= Z_r

            # Reshaping and appending to temporary lists
            w_fstop_tmp = list(
                map(
                    lambda x: x.squeeze(0),
                    torch.split(fstop_prob, 1, dim=0),
                )
            )
            w_left_tmp = list(
                map(
                    lambda x: x.squeeze(0),
                    torch.split(left_prob, 1, dim=0),
                )
            )
            w_right_tmp = list(
                map(
                    lambda x: x.squeeze(0),
                    torch.split(right_prob, 1, dim=0),
                )
            )

            j = 0
            for i in range(indexes, indexes + images.size(0)):
                dataset.w_fstop_prob_list[i].append(w_fstop_tmp[j])
                dataset.w_left_prob_list[i].append(w_left_tmp[j])
                dataset.w_right_prob_list[i].append(w_right_tmp[j])
                j += 1
            indexes += images.size(0)

        return dataset

    def give_full_worlds(self, model_itself_pc_x):

        p_c_x_ext = torch.unsqueeze(model_itself_pc_x, dim=0)

        f_prob = compute_forward_prob(
            p_c_x_ext, numpy=False
        )  # data, possibleworlds
        stop_prob = compute_forward_prob(
            p_c_x_ext, numpy=False
        )  # data, possibleworlds
        left_prob = compute_left(
            p_c_x_ext, numpy=False
        )  # data, possibleworlds
        right_prob = compute_right(
            p_c_x_ext, numpy=False
        )  # data, possibleworlds

        return f_prob, stop_prob, left_prob, right_prob

    def compute_pw_knowledge_filter(
        self, model_itself_pc_x, labels, wfs, wl, wR
    ):

        p_c_x_ext = torch.unsqueeze(model_itself_pc_x, dim=0)

        fstop_prob = compute_forward_stop_prob(
            p_c_x_ext, numpy=False
        ).to(
            self.model.device
        )  # data, possibleworlds
        left_prob = compute_left(p_c_x_ext, numpy=False).to(
            self.model.device
        )  # data, possibleworlds
        right_prob = compute_right(p_c_x_ext, numpy=False).to(
            self.model.device
        )  # data, possibleworlds

        label_indices = labels.long()

        FS_w_q_transposed = wfs.t()
        L_w_q_transposed = wl.t()
        R_w_q_transposed = wR.t()

        # Extracting probabilities using indexing and performing element-wise multiplication
        print(FS_w_q_transposed.device, fstop_prob.device)
        fstop_prob = (
            fstop_prob * FS_w_q_transposed[label_indices[:, 0], :]
        )
        left_prob = (
            left_prob * L_w_q_transposed[label_indices[:, 1], :]
        )
        right_prob = (
            right_prob * R_w_q_transposed[label_indices[:, 2], :]
        )

        # Adding a small constant for normalization
        fstop_prob += 1e-5
        left_prob += 1e-5
        right_prob += 1e-5

        # Computing normalization constants
        Z_fs = fstop_prob.sum(dim=1, keepdim=True)
        Z_l = left_prob.sum(dim=1, keepdim=True)
        Z_r = right_prob.sum(dim=1, keepdim=True)

        # Normalizing probabilities
        fstop_prob /= Z_fs
        left_prob /= Z_l
        right_prob /= Z_r

        # Reshaping and appending to temporary lists
        w_fstop_tmp = list(
            map(
                lambda x: x.squeeze(0),
                torch.split(fstop_prob, 1, dim=0),
            )
        )
        w_left_tmp = list(
            map(
                lambda x: x.squeeze(0),
                torch.split(left_prob, 1, dim=0),
            )
        )
        w_right_tmp = list(
            map(
                lambda x: x.squeeze(0),
                torch.split(right_prob, 1, dim=0),
            )
        )

        w_fstop_tmp = torch.stack(w_fstop_tmp, dim=0)
        w_left_tmp = torch.stack(w_left_tmp, dim=0)
        w_right_tmp = torch.stack(w_right_tmp, dim=0)

        return w_fstop_tmp, w_left_tmp, w_right_tmp

        # return F.log_softmax(w_fstop_tmp.unsqueeze(0), -1), F.log_softmax(w_left_tmp.unsqueeze(0), -1), F.log_softmax(w_right_tmp.unsqueeze(0), -1)

    # for the setup just train the ensembles
    def setup(
        self,
        train_loader,
        train_loader_no_shuffle,
        seeds,
        val_loader=None,
        epochs=10,
        save_path=None,
        separate_from_others=False,
        epsilon=0.01,
        lambda_h=1.0,
        lambda_kl=1.0,
    ):
        from DPL.utils_problog import (
            build_world_queries_matrix_complete_FS,
            build_world_queries_matrix_L,
            build_world_queries_matrix_R,
        )

        if separate_from_others:
            self.FS_w_q = build_world_queries_matrix_complete_FS().to(
                self.model.device
            )
            self.L_w_q = build_world_queries_matrix_L().to(
                self.model.device
            )
            self.R_w_q = build_world_queries_matrix_R().to(
                self.model.device
            )
        self.train_ensembles(
            train_loader,
            train_loader_no_shuffle,
            seeds,
            val_loader,
            epochs,
            save_path,
            separate_from_others,
            epsilon,
            lambda_h,
            lambda_kl,
        )

    # train the ensembles with deep separation with KL
    def train_ensembles(
        self,
        train_loader,
        train_loader_no_shuffle,
        seeds,
        val_loader=None,
        epochs=10,
        save_path=None,
        separate_from_others=False,
        epsilon=0.01,
        lambda_h=1.0,
        lambda_kl=1.0,
    ):
        # ensemble is a instance variable
        self.ensemble = []
        separation_dset = None

        pcx_loader = train_loader_no_shuffle

        if separate_from_others:
            fprint("Doing a separation with KL...")

        # only for other approach
        self.criterion_kl = nn.KLDivLoss(reduction="batchmean")

        for i, seed in enumerate(seeds):
            fprint("Training model with seed", seed, "...")

            set_random_seed(seed)

            # prepare model
            self._prepare_model()

            self.model.train()

            # train single model
            self.train_single_model(
                i,
                separation_dset,
                train_loader,
                val_loader,
                epochs,
                save_path,
                separate_from_others,
                epsilon,
                lambda_h,
                lambda_kl,
            )

            # freeze the parameters and set it to evaluation
            self.model.eval()

            # do the update of the dataset

            """
            if separate_from_others and (separation_dset is None):
                if self.knowledge_aware_kl:
                    t1 = time.time()
                    separation_dset = self._populate_pcx_dataset_knowledge_aware(
                        self.model, pcx_loader, 21
                    )
                    t2 = time.time()
                    print("Dataset creation", t2 - t1)
                else:
                    separation_dset = self._populate_pcx_dataset(
                        self.model, 
                        pcx_loader,
                        self.batch_size
                    )
            elif separate_from_others:
                if self.knowledge_aware_kl:
                    t1 = time.time()
                    separation_dset = self._update_pcx_dataset_knowledge_aware(
                        self.model,
                        separation_dset,
                        pcx_loader,
                        21
                    )
                    t2 = time.time()
                    print("Dataset update", t2 - t1)
                else:
                    separation_dset = self._update_pcx_dataset(
                        self.model,
                        separation_dset,
                        pcx_loader,
                        self.batch_size
                    )
            """
            self.ensemble.append(self.model)

        fprint(
            "Done!\nTotal length of the ensemble: ",
            len(self.ensemble),
        )

    # train the single model
    def train_single_model(
        self,
        model_idx,
        separation_dset,
        train_loader,
        val_loader=None,
        epochs=10,
        save_path=None,
        separate_from_others=False,
        epsilon=0.01,
        lambda_h=1.9,
        lambda_kl=1.0,
    ):
        best_prec1 = 0

        # validate evaluation
        if val_loader is not None:
            val_prec1 = 1
            val_prec1, best_loss = self.validate(val_loader, 0)

        if self.args.wandb is not None:
            wandb.log({f"start-lr-model-{model_idx}": self.args.lr})

        early_stopper = EarlyStopper(
            patience=5, min_delta=0.001
        )  # prev 0.01

        for epoch in range(epochs):
            # go to train_epoch function
            self.train_epoch_single_model(
                model_idx,
                separation_dset,
                epoch,
                train_loader,
                val_loader,
                separate_from_others,
                epsilon,
                lambda_h,
                lambda_kl,
            )

            if self.args.wandb is not None:
                wandb.log(
                    {
                        f"lr-model-{model_idx}": float(
                            self.scheduler.get_last_lr()[0]
                        )
                    }
                )

            # # validate evaluation
            if val_loader is not None:
                val_prec1 = 1
                val_prec1, last_loss = self.validate(
                    val_loader, epoch + 1, name=f"{model_idx}"
                )

            # remember best prec@1 and save checkpoint
            is_best = last_loss < best_loss
            best_loss = min(last_loss, best_loss)

            if early_stopper.early_stop(self.model, last_loss):
                break

        # save history in the array for plotting them later
        self.loss_histories.append(self.loss_history)
        self.val_loss_histories.append(self.val_loss_history)

        # clear the current arrays
        self.loss_history = []
        self.val_loss_history = []

        # end message
        fprint("Training done")

    def train_batch_single_model(
        self,
        model_idx,
        separation_dset,
        inputs,
        targets,
        concepts,
        epoch,
        separate_from_others=False,
        epsilon=0.01,
        lambda_h=10,
        lambda_kl=10,
    ):
        def kl_paper(p_model: torch.tensor, p_rest: torch.tensor, k):
            # with torch.no_grad():
            p_model = p_model + 1e-5
            p_model = p_model / (1 + (p_model.shape[1] * 1e-5))

            p_rest = p_rest + 1e-5
            p_rest = p_rest / (1 + (p_rest.shape[1] * 1e-5))

            ratio = torch.div(p_rest, p_model)
            kl_ew = torch.sum(
                p_model * torch.log(1 + (k - 1) * ratio), dim=1
            )

            return torch.mean(kl_ew, dim=0)

        self.optimizer.zero_grad()

        inputs, targets, concepts = (
            Variable(inputs),
            Variable(targets),
            Variable(concepts),
        )

        inputs.requires_grad = True

        pred = self.model(inputs)

        # Calculate loss
        pred_loss = self.prediction_criterion(pred, targets)

        # save loss (only value) to the all_losses list
        all_losses = {
            f"model {model_idx} prediction": pred_loss.cpu().data.numpy()
        }

        # compute loss of known concets and discriminator
        fprint(f"\tCE Loss: {pred_loss}")

        def entropy(probabilities: torch.tensor, concept=False):
            def concept_entropy(p):
                return -torch.sum(p * torch.log2(p)) / (
                    len(p) * math.log(2)
                )

            def single_concept_entropy(p):
                positive = p * torch.log2(p)
                negative = (1 - p) * torch.log2(1 - p)
                entropy_per_element = -(positive + negative)
                return entropy_per_element.mean()

            probabilities = probabilities + 1e-5
            probabilities = probabilities / (
                1 + (probabilities.shape[1] * 1e-5)
            )

            from functorch import vmap

            if concept:
                entropies = vmap(concept_entropy)(probabilities)
            else:
                entropies = vmap(single_concept_entropy)(
                    probabilities
                )
            return torch.mean(entropies, dim=0)

        h_loss, hh_labeled = (
            self.concept_learning_loss_for_weak_supervision(
                inputs,
                all_losses,
                concepts,
                self.args.cbm,
                self.args.senn,
                epoch,
            )
        )

        # add entropy on concepts
        ent_loss = self.entropy_loss(hh_labeled, all_losses, epoch)

        # Get the concept hh_labeled
        hh_labeled, hh, _ = self.model.conceptizer(inputs)

        loss = pred_loss + h_loss + ent_loss

        if not separate_from_others:
            # total loss to train models
            loss.backward()

            # Generate adversarial examples
            adversarial_batch = inputs + 0.01 * inputs.grad.sign()

            # Compute adversarial loss
            out_dict_adversarial = self.model(adversarial_batch)

            loss_adversarial = self.prediction_criterion(
                out_dict_adversarial, targets
            )

            # Minimize the combined loss l(θm, xbatch, ybatch) + l(θm, advbatch, advbatch) w.r.t. θm
            loss_adversarial.backward()
        else:

            # get the
            # concept probability
            model_itself_pc_x, c_x = hh_labeled, hh  # size: [512, 21]
            torch.autograd.set_detect_anomaly(True)

            with torch.no_grad():
                e_pc = 1 - entropy(model_itself_pc_x)

            # Adding entropy
            if model_idx == 0:
                loss = (
                    loss + (1 - entropy(model_itself_pc_x)) * lambda_h
                )
                print(
                    "\tEntropy loss:",
                    (1 - entropy(model_itself_pc_x)).item(),
                )
                all_losses.update(
                    {
                        f"model {model_idx} all loss": loss.cpu().data.numpy()
                    }
                )
                all_losses.update(
                    {
                        f"entropy {model_idx} loss": e_pc.cpu().data.numpy()
                    }
                )

            # usual update for the deep diversification thing
            if model_idx > 0:
                if self.knowledge_aware_kl:
                    (
                        model_itself_pfs_x,
                        model_itself_pleft_x,
                        model_itself_right_x,
                    ) = self.compute_pw_knowledge_filter(
                        model_itself_pc_x=model_itself_pc_x,
                        labels=targets,
                        wfs=self.FS_w_q,
                        wl=self.L_w_q,
                        wR=self.R_w_q,
                    )

                    (
                        model_itself_pf_x,
                        model_itself_pstop_x,
                        model_itself_pleft_x,
                        model_itself_right_x,
                    ) = self.give_full_worlds(model_itself_pc_x)

                    pf_list_ensemble = list()
                    pstop_list_ensemble = list()
                    pleftx_list_ensemble = list()
                    prightx_list_ensemble = list()

                    for m in self.ensemble:
                        m.ignore_prob_log = True
                        other_model_pc_x = m(inputs)
                        m.ignore_prob_log = False
                        (
                            other_model_itself_pfs_x,
                            other_model_itself_pleft_x,
                            other_model_itself_right_x,
                        ) = self.compute_pw_knowledge_filter(
                            model_itself_pc_x=other_model_pc_x,
                            labels=targets,
                            wfs=self.FS_w_q,
                            wl=self.L_w_q,
                            wR=self.R_w_q,
                        )
                        (
                            other_f_prob,
                            other_stop_prob,
                            other_left_prob,
                            other_right_prob,
                        ) = self.give_full_worlds(other_model_pc_x)
                        pf_list_ensemble.append(other_f_prob)
                        pstop_list_ensemble.append(other_stop_prob)
                        pleftx_list_ensemble.append(other_left_prob)
                        prightx_list_ensemble.append(other_right_prob)

                    # mean forward stop
                    pf_list_ensemble = torch.stack(pf_list_ensemble)
                    other_pf_mean = torch.mean(
                        pf_list_ensemble, dim=0
                    )  # .unsqueeze(0)

                    pstop_list_ensemble = torch.stack(
                        pstop_list_ensemble
                    )
                    other_pstop_mean = torch.mean(
                        pstop_list_ensemble, dim=0
                    )  # .unsqueeze(0)

                    # mean left
                    pleftx_list_ensemble = torch.stack(
                        pleftx_list_ensemble
                    )
                    other_pleftx_mean = torch.mean(
                        pleftx_list_ensemble, dim=0
                    )  # .unsqueeze(0)

                    # mean right
                    prightx_list_ensemble = torch.stack(
                        prightx_list_ensemble
                    )
                    other_prightx_mean = torch.mean(
                        prightx_list_ensemble, dim=0
                    )  # .unsqueeze(0)

                    with torch.no_grad():
                        kl_forward = kl_paper(
                            model_itself_pf_x,
                            other_pf_mean,
                            len(self.ensemble) + 1,
                            False,
                        )
                        kl_stop = kl_paper(
                            model_itself_pstop_x,
                            other_pstop_mean,
                            len(self.ensemble) + 1,
                            False,
                        )
                        kl_left = kl_paper(
                            model_itself_pleft_x,
                            other_pleftx_mean,
                            len(self.ensemble) + 1,
                            False,
                        )
                        kl_right = kl_paper(
                            model_itself_right_x,
                            other_prightx_mean,
                            len(self.ensemble) + 1,
                            False,
                        )
                        kl = kl_forward + kl_stop + kl_left + kl_right
                        entropy_pc_x = entropy(
                            model_itself_pc_x, True
                        )
                        entropy_f = entropy(model_itself_pf_x, True)
                        entropy_s = entropy(
                            model_itself_pstop_x, True
                        )
                        entropy_l = entropy(
                            model_itself_pleft_x, True
                        )
                        entropy_r = entropy(
                            model_itself_right_x, True
                        )

                    distance = (
                        kl_paper(
                            model_itself_pf_x,
                            other_pf_mean,
                            len(self.ensemble) + 1,  # , False
                        )
                        + kl_paper(
                            model_itself_pstop_x,
                            other_pstop_mean,
                            len(self.ensemble) + 1,  # , False
                        )
                        + kl_paper(
                            model_itself_pleft_x,
                            other_pleftx_mean,
                            len(self.ensemble) + 1,  # , False
                        )
                        + kl_paper(
                            model_itself_right_x,
                            other_prightx_mean,
                            len(self.ensemble) + 1,  # , False
                        )
                    )

                    assert kl >= 0

                    (
                        model_itself_pfs_x_2,
                        model_itself_pleft_x_2,
                        model_itself_right_x_2,
                    ) = self.compute_pw_knowledge_filter(
                        model_itself_pc_x=model_itself_pc_x,
                        labels=targets,
                        wfs=self.FS_w_q,
                        wl=self.L_w_q,
                        wR=self.R_w_q,
                    )
                    distance = distance / (math.log(2) * 21) + 1
                    distance = lambda_kl * distance
                    distance = (
                        distance
                        + (1 - entropy(model_itself_pc_x)) * lambda_h
                    )

                    print("kl", kl.item())
                    print("entropy p(c|x)", entropy_pc_x.item())
                    print("entropy f", entropy_f.item())
                    print("entropy s", entropy_s.item())
                    print("entropy l", entropy_l.item())
                    print("entropy r", entropy_r.item())

                    all_losses.update(
                        {
                            f"model {model_idx} kl_f (v)": kl_forward.cpu().data.numpy()
                        }
                    )
                    all_losses.update(
                        {
                            f"model {model_idx} kl_s (v)": kl_stop.cpu().data.numpy()
                        }
                    )
                    all_losses.update(
                        {
                            f"model {model_idx} kl_left (v)": kl_left.cpu().data.numpy()
                        }
                    )
                    all_losses.update(
                        {
                            f"model {model_idx} kl_right (v)": kl_right.cpu().data.numpy()
                        }
                    )
                    all_losses.update(
                        {
                            f"model {model_idx} entropy p(c|x) (^)": entropy_pc_x.cpu().data.numpy()
                        }
                    )
                    all_losses.update(
                        {
                            f"model {model_idx} entropy f (^)": entropy_f.cpu().data.numpy()
                        }
                    )
                    all_losses.update(
                        {
                            f"model {model_idx} entropy s (^)": entropy_s.cpu().data.numpy()
                        }
                    )
                    all_losses.update(
                        {
                            f"model {model_idx} entropy l (^)": entropy_l.cpu().data.numpy()
                        }
                    )
                    all_losses.update(
                        {
                            f"model {model_idx} entropy r (^)": entropy_r.cpu().data.numpy()
                        }
                    )
                    all_losses.update(
                        {
                            f"model {model_idx} objective (v)": distance.cpu().data.numpy()
                        }
                    )

                    """
                    distance = - lambda_h * (
                        torch.mean(
                            torch.stack(
                                list(
                                    self.criterion_kl(model_itself_pfs_x, other_m_pc_x) 
                                    for other_m_pc_x in pfsx_list_ensemble
                                )
                            ), dim=0
                        ) + 
                        torch.mean(
                            torch.stack(
                                list(
                                    self.criterion_kl(model_itself_pleft_x, other_m_pc_x) 
                                    for other_m_pc_x in pleftx_list_ensemble
                                )
                            ), dim=0
                        ) +
                        torch.mean(
                            torch.stack(
                                list(
                                    self.criterion_kl(model_itself_right_x, other_m_pc_x) 
                                    for other_m_pc_x in prightx_list_ensemble
                                )
                            ), dim=0
                        )
                    )
                    """
                else:
                    # Create the ensembles world probabilities
                    pcx_list_ensemble = list()

                    for m in self.ensemble:
                        m.ignore_prob_log = True
                        other_model_pc_x = m(inputs)
                        m.ignore_prob_log = False

                        pcx_list_ensemble.append(other_model_pc_x)

                    pcx_list_ensemble = torch.stack(
                        pcx_list_ensemble, dim=0
                    )
                    pcx_list_ensemble = torch.mean(
                        pcx_list_ensemble, dim=0
                    )

                    """
                    distance = - lambda_h * torch.mean(
                        torch.stack(
                            list(
                                self.criterion_kl(model_itself_pc_x, other_m_pc_x) 
                                for other_m_pc_x in pcx_list_ensemble
                            )
                        ), dim=0
                    )
                    distance = - lambda_h * self.criterion_kl(
                        model_itself_pc_x, pcx_list_ensemble
                    )
                    """
                    print(
                        model_itself_pc_x.shape,
                        pcx_list_ensemble.shape,
                    )
                    distance = lambda_h * kl_paper(
                        model_itself_pc_x,
                        pcx_list_ensemble,
                        len(self.ensemble) + 1,
                    )

                loss += distance

                # fprint(f"\tKL Loss: {distance}")
                # fprint(f"Tot Loss: {loss}")

            loss.backward()

        # update each model
        self.optimizer.step()

        return pred, loss, all_losses, hh_labeled, inputs

    def train_epoch_single_model(
        self,
        model_idx,
        separation_dset,
        epoch,
        train_loader,
        val_loader=None,
        separate_from_others=False,
        epsilon=0.01,
        lambda_h=1.0,
        lambda_kl=1.0,
    ):

        # initialization of fprint's values
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        topc1 = AverageMeter()

        # switch to train mode
        self.model.train()
        end = time.time()

        for i, (inputs, targets, concepts) in enumerate(
            train_loader, 0
        ):

            # measure data loading time
            data_time.update(time.time() - end)

            # get the inputs
            if self.cuda:
                inputs = inputs.cuda(self.device)
                concepts = concepts.cuda(self.device)
                targets = targets.cuda(self.device)

            # go to def train_batch (class GradPenaltyTrainer)
            outputs, loss, loss_dict, hh_labeled, pretrained_out = (
                self.train_batch_single_model(
                    model_idx,
                    separation_dset,
                    inputs,
                    targets,
                    concepts,
                    epoch,
                    separate_from_others,
                    epsilon,
                    lambda_h,
                    lambda_kl,
                )
            )

            if self.args.wandb is not None:
                wandb.log(loss_dict)
                wandb.log({"step": i, "epoch": epoch})

            # add to loss_history
            loss_dict["iter"] = i + (len(train_loader) * epoch)
            self.loss_history.append(loss_dict)

            # measure accuracy and record loss
            if self.nclasses > 4:
                # mainly use this line (current)
                prec1, _ = self.accuracy(
                    outputs.data, targets.data, topk=(1, 5)
                )
            elif self.nclasses in [3, 4]:
                prec1, _ = self.accuracy(
                    outputs.data,
                    targets.data,
                    topk=(1, self.nclasses),
                )
            else:
                prec1, _ = self.binary_accuracy(
                    outputs.data, targets.data
                ), [100]

            # update each value of fprint's values
            losses.update(
                loss.data.cpu().numpy(), pretrained_out.size(0)
            )
            top1.update(prec1[0], pretrained_out.size(0))

            if not self.args.senn:
                # measure accuracy of concepts
                err = self.concept_error(hh_labeled.data, concepts)

                # update fprint's value
                topc1.update(err, pretrained_out.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if not self.args.senn:
                # fprint values of i-th iteration in the current epoch
                if i % self.print_freq == 0:
                    fprint(
                        "Epoch: [{0}][{1}/{2}]  "
                        "Time {batch_time.val:.2f} ({batch_time.avg:.2f})  "
                        "Loss {loss.val:.4f} ({loss.avg:.4f})  ".format(
                            epoch,
                            i,
                            len(train_loader),
                            batch_time=batch_time,
                            data_time=data_time,
                            loss=losses,
                        )
                    )
            else:
                # fprint values of i-th iteration in the current epoch
                if i % self.print_freq == 0:
                    fprint(
                        "Epoch: [{0}][{1}/{2}]  "
                        "Time {batch_time.val:.2f} ({batch_time.avg:.2f})  "
                        "Loss {loss.val:.4f} ({loss.avg:.4f})  ".format(
                            epoch,
                            i,
                            len(train_loader),
                            batch_time=batch_time,
                            data_time=data_time,
                            loss=losses,
                        )
                    )
        # optimizer's schedule update based on epoch
        self.scheduler.step(epoch)

    # prepare a new model, I need to build up all the things from scratch
    def _prepare_model(self):
        # only "fcc" conceptizer use, otherwise cannot use (not modifile so as to fit this task...)
        if self.h_type == "fcc":
            conceptizer1 = image_fcc_conceptizer(
                2048,
                self.nconcepts,
                self.nconcepts_labeled,
                self.concept_dim,
                self.h_sparsity,
                self.senn,
            )
        elif self.h_type == "cnn":
            fprint("[ERROR] please use fcc network")
            sys.exit(1)
        else:
            fprint("[ERROR] please use fcc network")
            sys.exit(1)

        parametrizer1 = dfc_parametrizer(
            2048,
            1024,
            512,
            256,
            128,
            self.nconcepts,
            self.theta_dim,
            layers=4,
        )

        if self.cbm == True:
            aggregator = CBM_aggregator(
                self.concept_dim,
                self.nclasses,
                self.nconcepts_labeled,
            )
        else:
            aggregator = additive_scalar_aggregator(
                self.concept_dim, self.nclasses
            )

        if self.model_name == "dpl":
            model = DPL(
                conceptizer1,
                parametrizer1,
                aggregator,
                self.cbm,
                self.senn,
                self.device,
            )
        elif self.model_name == "dpl_auc":
            model = DPL_AUC(
                conceptizer1,
                parametrizer1,
                aggregator,
                self.cbm,
                self.senn,
                self.device,
            )

        # send models to device you want to use
        self.model = model.to(self.device)

        # optimizer
        if self.opt == "adam":
            optim_betas = (0.9, 0.999)
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=self.lr, betas=optim_betas
            )
        elif self.opt == "rmsprop":
            self.optimizer = optim.RMSprop(
                self.model.parameters(), lr=self.lr
            )
        elif self.opt == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                momentum=0.9,
            )

        # set scheduler for learning rate
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=self.exp_decay_lr
        )

    # freeze the models parameters
    def _freeze_model_params(self):
        for param in self.model.parameters():
            param.requires_grad = False

    # perform a single prediction
    def _ensemble_single_predict(
        self, models, batch_samples: torch.tensor, apply_softmax=False
    ):
        for model in models:
            model.eval()
            model.return_both_concept_out_prob = True

        output_list = [model(batch_samples) for model in models]

        for model in models:
            model.return_both_concept_out_prob = False

        # get out the different output
        label_prob = [
            lab.detach().cpu().numpy() for lab, _ in output_list
        ]  # 30
        concept_prob = [
            concept.detach().cpu().numpy()
            for _, concept in output_list
        ]  # 30

        label_prob = np.stack(label_prob, axis=0)
        concept_prob = np.stack(concept_prob, axis=0)

        if apply_softmax:
            label_prob = softmax(label_prob, axis=2)

        return label_prob, concept_prob

    # perform a prediction on the ensemble
    def ensemble_predict(self, loader, apply_softmax=False):
        device = self.model.device

        # Loop over the dataloader
        for i, data in enumerate(loader):
            images, labels, concepts = data
            images, labels, concepts = (
                images.to(device),
                labels.to(device),
                concepts.to(device),
            )

            # Call Ensemble predict
            (label_prob_ens, concept_prob_ens) = (
                self._ensemble_single_predict(
                    self.ensemble, images, apply_softmax
                )
            )

            # Concatenate the output
            if i == 0:
                y_true = labels.detach().cpu().numpy()
                c_true = concepts.detach().cpu().numpy()
                y_pred = label_prob_ens
                pc_pred = concept_prob_ens
            else:
                y_true = np.concatenate(
                    [y_true, labels.detach().cpu().numpy()], axis=0
                )
                c_true = np.concatenate(
                    [c_true, concepts.detach().cpu().numpy()], axis=0
                )
                y_pred = np.concatenate(
                    [y_pred, label_prob_ens], axis=1
                )
                pc_pred = np.concatenate(
                    [pc_pred, concept_prob_ens], axis=1
                )

        return y_true, c_true, y_pred, pc_pred

    # produce, as before the worlds probabilities
    def worlds_probability(
        self,
        loader,
        output_classes: int,
        num_concepts: int,
        n_ensembles: int = 30,
        apply_softmax=False,
    ):

        y_true, c_true, y_pred, pc_pred = self.ensemble_predict(
            loader, apply_softmax
        )

        fstop_prob = compute_forward_stop_prob(
            pc_pred
        )  # data, possibleworlds
        left_prob = compute_left(pc_pred)  # data, possibleworlds
        right_prob = compute_right(pc_pred)  # data, possibleworlds
        y_prob = compute_output_probability(
            y_pred
        )  # data, possibleworlds

        w_probs = [fstop_prob, left_prob, right_prob]

        w_predictions = []
        w_predictions_prob_value = []
        for prob in w_probs:
            w_predictions.append(np.argmax(prob, axis=-1))  # data, 1
            w_predictions_prob_value.append(
                np.max(prob, axis=-1)
            )  # data, 1

        fstop_ground = compute_forward_stop_groundtruth(
            c_true
        )  # data, 1
        left_ground = compute_left_groundtruth(c_true)  # data, 1
        right_ground = compute_right_groundtruth(c_true)  # data, 1

        # w grountruths
        w_groundtruths = [fstop_ground, left_ground, right_ground]

        # get the predictions, probabilities and the groundtruth not one-ho
        y_mean_prob = np.mean(y_pred, axis=0)
        y_preds = torch.split(torch.tensor(y_mean_prob), 2, dim=-1)
        y_trues = torch.split(torch.tensor(y_true), 1, dim=-1)

        y_preds_list = list(y_preds)
        y_preds_prob_list = list(y_preds)

        # Modify each array in the list
        for i in range(len(y_preds_list)):
            y_preds_list_tmp = y_preds_list[i].numpy()
            y_preds_list[i] = np.expand_dims(
                np.argmax(y_preds_list_tmp, axis=1), axis=-1
            )
            y_preds_prob_list[i] = np.expand_dims(
                np.max(y_preds_list_tmp, axis=1), axis=-1
            )

        y_true = np.concatenate(y_trues[:4], axis=-1)
        y_predictions = np.concatenate(y_preds_list[:4], axis=-1)
        y_predictions_prob = np.concatenate(
            y_preds_prob_list[:4], axis=-1
        )

        pc_pred_mean = np.mean(pc_pred, axis=0)
        pc_prob = pc_pred_mean
        pc_pred = (pc_prob > 0.5).astype(float)

        return (
            y_true,
            y_prob,
            y_predictions,
            y_predictions_prob,
            w_probs,
            w_predictions,
            w_groundtruths,
            w_predictions_prob_value,
            c_true,
            pc_prob,
            pc_pred,
            y_mean_prob,
        )

    # plot losses with overwritten histories
    def plot_losses(self, name, save_path=None):
        for i in range(len(self.loss_histories)):
            # override loss history
            self.loss_history = self.loss_histories[i]
            self.val_loss_history = self.val_loss_histories[i]
            curr_name = name + f"_{i}"
            super().plot_losses(curr_name, save_path)

    # same thing as before for the p_c_x distance
    def p_c_x_distance(
        self,
        loader,
    ):

        model_pred = [[], [], []]

        for j in range(len(self.ensemble)):
            self.model = self.ensemble[j]
            self.model.eval()

            for i, data in enumerate(loader):
                images, labels, concepts = data
                images, labels, concepts = (
                    images.to(self.model.device),
                    labels.to(self.model.device),
                    concepts.to(self.model.device),
                )

                # activate the double return
                self.model.return_both_concept_out_prob = True
                output = self.model(images)
                # deactivate the double return
                self.model.return_both_concept_out_prob = False

                label_prob, concept_prob = output
                concept_prob = np.expand_dims(
                    concept_prob.detach().cpu().numpy(), axis=0
                )

                if i == 0:
                    c_prb = concept_prob
                else:
                    c_prb = np.concatenate(
                        [c_prb, concept_prob], axis=1
                    )

            fstop_prob = compute_forward_stop_prob(
                c_prb
            )  # data, possibleworlds
            left_prob = compute_left(c_prb)  # data, possibleworlds
            right_prob = compute_right(c_prb)  # data, possibleworlds

            # append the alpha matrices to the list of lists
            model_pred[0].append(fstop_prob)
            model_pred[1].append(left_prob)
            model_pred[2].append(right_prob)

        return (
            self.mean_l2_distance(model_pred[0]),
            self.mean_l2_distance(model_pred[1]),
            self.mean_l2_distance(model_pred[2]),
        )

    # get the bayes ensemble as always
    def get_ensemble_from_bayes(self, n_ensemble):
        ensemble = self.ensemble
        return ensemble


# Mein Freund ClassificationTesterFactory
class ClassificationTesterFactory:
    @staticmethod
    def get_model(
        name: str, model, args, device
    ) -> ClassificationTester:
        if name == "frequentist":
            return Frequentist(model, args, device)
        elif name == "mcdropout":
            return MCDropout(model, args, device)
        elif name == "laplace":
            return LaplaceBayes(model, args, device)
        elif name == "deepensembles":
            return DeepEnsembles(model, args, device, name)
        elif name == "resense":
            return DeepEnsembles(model, args, device, name)
        else:
            raise ValueError(
                "The chosen model is not valid: chosen", name
            )
