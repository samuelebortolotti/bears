# Module which contains the code for training a model and the active learning setup
import math
import os
import sys

import numpy as np
import torch
import wandb
from datasets.utils.base_dataset import BaseDataset
from models.mnistdpl import MnistDPL
from torchvision.utils import make_grid
from utils import fprint
from utils.bayes import deep_ensemble_active
from utils.dpl_loss import ADDMNIST_DPL
from utils.generative import conditional_gen, recon_visaulization
from utils.metrics import (
    evaluate_metrics,
    evaluate_metrics_ensemble,
    evaluate_mix,
    mean_entropy,
)
from utils.status import progress_bar
from utils.wandb_logger import *
from warmup_scheduler import GradualWarmupScheduler


def active_start(model, seed, ensemble=[]):
    """Saves the starting point of the model if it does not exists, and loads it if it exists.

    Args:
        model (nn.Module): network
        seed (int): random seed
        ensemble (List[nn.Module]): ensemble of network

    Returns:
        None: This function does not return a value.
    """
    model_filename = (
        f"data/ckpts/minikandinsky-kanddpl-dis-{seed}-end.pt"
    )

    if os.path.exists(model_filename):
        state_dict = torch.load(model_filename)
        model.load_state_dict(state_dict)
        print(f"Model loaded: {model_filename}")
    else:
        state_dict = model.state_dict()
        torch.save(state_dict, model_filename)
        print(f"Model saved: {model_filename}")

    # load the same start for the ensemble
    if len(ensemble) > 0:
        for i in range(len(ensemble)):
            state_dict = torch.load(model_filename)
            ensemble[i].load_state_dict(state_dict)


def return_metrics(
    y_true,
    y_pred,
    c_true,
    c_pred,
):
    """Computes metrix on labels and concepts and then returns them

    Args:
        y_true: groundtruth labels
        y_pred: label predictions
        c_true: groundtruth concepts
        c_pred: concepts predictions

    Returns:
        yac (float): label accuracy
        cac (float): concept accuracy
        cf1 (float): concept f1 score
        yf1 (float): label f1 score
    """
    yac, yf1 = evaluate_mix(y_true, y_pred)
    cac, cf1 = evaluate_mix(c_true, c_pred)

    return yac, cac, cf1, yf1


def train_active(
    model: MnistDPL, dataset: BaseDataset, _loss: ADDMNIST_DPL, args
):
    """Active learning

    Args:
        model (MnistDPL): network
        dataset (BaseDataset): dataset Kandinksy
        _loss (ADDMNIST_DPL): loss function
        args: parsed args

    Returns:
        None: This function does not return a value.
    """

    # set wandb if specified
    if args.wandb is not None:
        fprint("\n---wandb on\n")
        wandb.init(
            project=args.project,
            entity=args.wandb,
            name=str(args.dataset) + "_" + str(args.model),
            config=args,
        )

    """ACTIVE LEARNING SETUP
    To set up your active-learning, at the moment, you have to modify this parameters by hand.

        save_me (bool): allows to make the model save it's status, which will be loaded by future models
        random_selection (bool): selects the objets to give the supervision to
        n_ensembles (int): number of ensembles for biretta

    """
    save_me = False
    random_selection = True
    n_ensembles = 5

    """GLOBAL VARIABLES
    
        entropy_array (ndarray): contains the sorted values of the entropy per each objects
        figures (ndarray): contains the index of the figures of the objects sorted by the objects with the highest entropy
        objects (ndarray): contains the index of the object in the figure sorted by the objects with the highest entropy
        indices (ndarray): contains the index of the sample sorted by the objects with the highest entropy
    """
    entropy_array, figures, objects, indices = (
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([]),
    )

    """Metrics:
    
        acc_c (List[Tuple]): accuracy on concepts, the first dimension is frequentist, second is biretta
        acc_y (List[Tuple]): accuracy on concepts, the first dimension is frequentist, second is biretta
        iterations (List[int]): iterations count
    """
    acc_c = [[], []]  # first dimension is freq, second is biretta
    acc_y = [[], []]  # first dimension is freq, second is biretta
    iterations = []

    # DO NOT APPLY BIRETTA FIRST
    for biretta in [False, True]:

        """SAMPLES TO GIVE SUPERVISION TO:
            - first dimension is sample index
            - second dimension is figure index
            - third dimension is the object index

        IT RESETS FOR FREQUENTIST AND FOR BIRETTA
        """
        chosen_samples = [[], [], []]

        # LOOP FOR ACTIVE LEARNING SAMPLES
        for active_setup_samples in range(0, 80, 10):

            # append iterations
            iterations.append(active_setup_samples)

            # set finetuning equal to the number of samples to give supervision to
            args.finetuning = active_setup_samples

            # BASICALLY 10 RED SQUARES -> this is done by hand
            starting_img_idx = [0, 1, 2, 3, 4, 5, 11, 12, 13, 14]
            starting_figure_idx = [1, 1, 1, 1, 0, 0, 2, 2, 1, 1]
            starting_obj_idx = [0, 0, 2, 2, 0, 0, 1, 1, 2, 2]

            # ELEMENTS I WANT TO GIVE SUPERVISION TO
            my_data_idx = []
            my_figure_idx = []
            my_obj_idx = []

            # IF THERE ARE SAMPLES TO GIVE SUPERVISION TO, THEN:
            if active_setup_samples > 0:
                print("Adding samples", 10)

                # temporal lists containing initial superivion + chosen supervision
                tmp_data_idx = starting_img_idx + chosen_samples[0]
                tmp_figure_idx = (
                    starting_figure_idx + chosen_samples[1]
                )
                tmp_obj_idx = starting_obj_idx + chosen_samples[2]

                # counter of selected elements
                samples_counter = 0

                # loop through indices, figures, objects
                # reminders: indices, figures, objects are the sorted vectors according to highest entropy samples
                for indice, figure, oggetto in zip(
                    indices, figures, objects
                ):
                    if random_selection:
                        # random supervision
                        indice, figure, oggetto = (
                            np.random.randint(0, 4000),
                            np.random.randint(0, 3),
                            np.random.randint(0, 3),
                        )

                    # already present, then move
                    if (
                        indice in tmp_data_idx
                        and figure in tmp_figure_idx
                        and oggetto in tmp_obj_idx
                    ):
                        continue

                    # append samples
                    samples_counter += 1
                    chosen_samples[0].append(int(indice))
                    chosen_samples[1].append(int(figure))
                    chosen_samples[2].append(int(oggetto))

                    # whenever 10 is reached, break the loop
                    if samples_counter == 10:
                        break

                # my supervision samples
                my_data_idx = chosen_samples[0]
                my_figure_idx = chosen_samples[1]
                my_obj_idx = chosen_samples[2]

            # merging the lists
            data_idx = starting_img_idx + my_data_idx
            figure_idx = starting_figure_idx + my_figure_idx
            obj_idx = starting_obj_idx + my_obj_idx

            # give supervision to:
            dataset.give_supervision_to(data_idx, figure_idx, obj_idx)

            fprint(f"Giving supervision to #{len(data_idx)} samples")

            # Default Setting for Training
            model.to(model.device)
            train_loader, val_loader, test_loader = (
                dataset.get_data_loaders()
            )
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                model.opt, args.exp_decay
            )
            w_scheduler = GradualWarmupScheduler(
                model.opt, 1.0, args.warmup_steps
            )

            # train loader without shuffling
            train_loader_as_val = dataset.get_train_loader_as_val()

            # ensemble lists
            ensemble = []

            fprint("\n--- Start of Training ---\n")

            # default for warm-up
            model.opt.zero_grad()
            model.opt.step()

            active_start(model, args.seed, [])

            if biretta:
                # training ensembles
                ensemble = deep_ensemble_active(
                    seeds=[
                        i + args.seed + 1
                        for i in range(n_ensembles - 1)
                    ],
                    base_model=model,
                    dataset=dataset,
                    num_epochs=args.n_epochs,
                    args=args,
                    val_loader=val_loader,
                    epsilon=0.01,
                    separate_from_others=True,
                    lambda_h=0.1,
                    use_wandb=False,
                    n_facts=model.n_facts,
                    knowledge_aware_kl=True,
                    real_kl=True,
                    supervision=[data_idx, figure_idx, obj_idx],
                    weights_base_model=f"data/ckpts/minikandinsky-kanddpl-dis-{args.seed}-end.pt",
                )
            else:
                # normal training
                for epoch in range(args.n_epochs):
                    model.train()

                    ys, y_true = None, None

                    for i, data in enumerate(train_loader):
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

                        model.opt.zero_grad()
                        loss, losses = _loss(out_dict, args)

                        loss.backward()
                        model.opt.step()

                        if args.wandb is not None:
                            wandb_log_step_prefix(
                                f"{args.seed}_{args.finetuning}_freq",
                                i,
                                epoch,
                                loss.item(),
                                losses,
                            )

                        if i % 10 == 0:
                            progress_bar(
                                i,
                                len(train_loader) - 9,
                                epoch,
                                loss.item(),
                            )

                    model.eval()
                    tloss, cacc, yacc, f1 = evaluate_metrics(
                        model, val_loader, args
                    )

                    # update at end of the epoch
                    if epoch < args.warmup_steps:
                        w_scheduler.step()
                    else:
                        scheduler.step()
                        if hasattr(_loss, "grade"):
                            _loss.update_grade(epoch)

                    ### LOGGING ###
                    fprint(
                        "\n  ACC C", cacc, "  ACC Y", yacc, "F1 Y", f1
                    )

                    # simple early stopper
                    if yacc > 95:
                        break

                    if args.wandb is not None:
                        wandb_log_epoch_prefix(
                            prefix=f"{args.seed}_{args.finetuning}_freq",
                            epoch=epoch,
                            acc=yacc,
                            cacc=cacc,
                            tloss=tloss,
                            lr=float(scheduler.get_last_lr()[0]),
                        )

                # Evaluate performances on val or test
                if args.validate:
                    (
                        y_true,
                        c_true,
                        y_pred,
                        c_pred,
                        p_cs,
                        p_ys,
                        p_cs_all,
                        p_ys_all,
                    ) = evaluate_metrics(
                        model, val_loader, args, last=True
                    )
                else:
                    (
                        y_true,
                        c_true,
                        y_pred,
                        c_pred,
                        p_cs,
                        p_ys,
                        p_cs_all,
                        p_ys_all,
                    ) = evaluate_metrics(
                        model, test_loader, args, last=True
                    )

            if not biretta:
                tloss, cacc, yacc, f1 = evaluate_metrics(
                    model, val_loader, args
                )
            else:
                tloss, cacc, yacc, f1 = evaluate_metrics_ensemble(
                    ensemble, val_loader, args
                )

            print("Y-ACC", yacc, "C-ACC", cacc)

            # append cacc and yacc
            acc_c[int(biretta)].append(cacc)
            acc_y[int(biretta)].append(yacc)

            entropy_array, figures, objects, indices = (
                concept_supervision_selection(
                    train_loader_as_val, model, ensemble
                )
            )

            # print all, and print as many as finetuning says
            np.set_printoptions(threshold=sys.maxsize)

            fprint(
                "Most confused elements figures",
                figures[: args.finetuning].astype(int),
            )
            fprint(
                "Most confused elements objects",
                objects[: args.finetuning].astype(int),
            )
            fprint(
                "Most confused elements entropy",
                entropy_array[: args.finetuning],
            )
            fprint(
                "Most confused elements indices",
                indices[: args.finetuning].astype(int),
            )
            fprint(
                "Max entropy",
                max(entropy_array),
                "min entropy",
                min(entropy_array),
            )

            ensemble_string = (
                "" if len(ensemble) == 0 else "-ensemble"
            )

            if not biretta:
                # save numpy values for debugging

                fprint(
                    f"data/kand-analysis/seed-{args.seed}-{args.dataset}-ftn-{args.finetuning}{ensemble_string}"
                )
                os.makedirs(
                    f"data/kand-analysis/seed-{args.seed}-{args.dataset}-ftn-{args.finetuning}{ensemble_string}/",
                    exist_ok=True,
                )

                np.save(
                    f"data/kand-analysis/seed-{args.seed}-{args.dataset}-ftn-{args.finetuning}{ensemble_string}/y_true.npy",
                    y_true,
                )
                np.save(
                    f"data/kand-analysis/seed-{args.seed}-{args.dataset}-ftn-{args.finetuning}{ensemble_string}/c_true.npy",
                    c_true,
                )
                np.save(
                    f"data/kand-analysis/seed-{args.seed}-{args.dataset}-ftn-{args.finetuning}{ensemble_string}/y_pred.npy",
                    y_pred,
                )
                np.save(
                    f"data/kand-analysis/seed-{args.seed}-{args.dataset}-ftn-{args.finetuning}{ensemble_string}/c_pred.npy",
                    c_pred,
                )
                np.save(
                    f"data/kand-analysis/seed-{args.seed}-{args.dataset}-ftn-{args.finetuning}{ensemble_string}/p_cs.npy",
                    p_cs,
                )
                np.save(
                    f"data/kand-analysis/seed-{args.seed}-{args.dataset}-ftn-{args.finetuning}{ensemble_string}/p_ys.npy",
                    p_ys,
                )
                np.save(
                    f"data/kand-analysis/seed-{args.seed}-{args.dataset}-ftn-{args.finetuning}{ensemble_string}/p_cs_all.npy",
                    p_cs_all,
                )
                np.save(
                    f"data/kand-analysis/seed-{args.seed}-{args.dataset}-ftn-{args.finetuning}{ensemble_string}/figures.npy",
                    figures,
                )
                np.save(
                    f"data/kand-analysis/seed-{args.seed}-{args.dataset}-ftn-{args.finetuning}{ensemble_string}/objects.npy",
                    objects,
                )
                np.save(
                    f"data/kand-analysis/seed-{args.seed}-{args.dataset}-ftn-{args.finetuning}{ensemble_string}/entropy_array.npy",
                    entropy_array,
                )
                np.save(
                    f"data/kand-analysis/seed-{args.seed}-{args.dataset}-ftn-{args.finetuning}{ensemble_string}/indices.npy",
                    indices,
                )

                if save_me:
                    # save the base model
                    base_model_name = f"data/ckpts/minikandinsky-kanddpl-dis-{args.seed}-end-biretta.pt"
                    state_dict = model.state_dict()
                    torch.save(state_dict, base_model_name)
                    print(f"Model saved: {base_model_name}")
                    save_me = False

                c = np.load(
                    f"data/kand-analysis/seed-{args.seed}-{args.dataset}-ftn-{args.finetuning}{ensemble_string}/c_pred.npy"
                )
                g = np.load(
                    f"data/kand-analysis/seed-{args.seed}-{args.dataset}-ftn-{args.finetuning}{ensemble_string}/c_true.npy"
                )
                y_pred = np.load(
                    f"data/kand-analysis/seed-{args.seed}-{args.dataset}-ftn-{args.finetuning}{ensemble_string}/y_pred.npy"
                )
                y_true = np.load(
                    f"data/kand-analysis/seed-{args.seed}-{args.dataset}-ftn-{args.finetuning}{ensemble_string}/y_true.npy"
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
                            f"{args.seed}_{args.finetuning}_freq_conf_mat_{tag[i]}": wandb.plot.confusion_matrix(
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

                wandb.log(
                    {
                        f"{args.seed}_{args.finetuning}_freq_conf_mat_mondi": wandb.plot.confusion_matrix(
                            probs=None, y_true=all_g, preds=all_c
                        )
                    }
                )

    acc_c = np.asarray(acc_c)
    acc_y = np.asarray(acc_y)

    suffix = ""
    if random_selection:
        suffix = "_random"

    suffix = "_biretta"

    np.save(
        f"data/kand-analysis/acc_c_{args.seed}{suffix}.npy", acc_c
    )
    np.save(
        f"data/kand-analysis/acc_y_{args.seed}{suffix}.npy", acc_y
    )

    wandb.finish()


def concept_supervision_selection(loader, model, ensemble=[]):
    """Method which computes the entropy per object and allows the selection of the object with the highest entropy values

    Args:
        loader: not shuffled train loader
        model (nn.Module): network
        ensemble (List[nn.Module]): ensemble if provided

    Returns:
        entropy_array: sorted entropy values
        figures_array: sorted figures index arrays
        objects_array: sorted objects index arrays
        indices: sorted sample index array
    """

    def ova_entropy_per_object_ensemble(p_list, n_figure, n_object):
        """OVA entropy per object in the ensemble

        Args:
            p_list: list of probability vector
            n_figure (int): index of figures
            n_object (int): index of objects

        Returns:
            entropy_value: entropy value per object
        """
        world_prob = []

        for p in p_list:
            # normalization
            p[n_figure, n_object, :] += 1e-5
            p[n_figure, n_object, :] /= 1 + (p.shape[0] * 1e-5)

            # get color and shape
            color_worlds = np.asarray(list(p[n_figure, n_object, :3]))
            shape_worlds = np.asarray(list(p[n_figure, n_object, 3:]))

            # compute the outer product
            world_prob.append(
                np.outer(color_worlds, shape_worlds).flatten()
            )

        # mean in the ensemble
        world_prob = np.stack(world_prob, axis=0)
        world_prob = np.mean(world_prob, axis=0)

        # entropy
        entropy_value = -np.sum(world_prob * np.log(world_prob)) / (
            math.log(9)
        )

        return entropy_value

    def ova_entropy_per_object(p, n_figure, n_object):
        """OVA entropy per object

        Args:
            p: probability vector
            n_figure (int): index of figures
            n_object (int): index of objects

        Returns:
            entropy_value: entropy value per object
        """
        p[n_figure, n_object, :] += 1e-5
        p[n_figure, n_object, :] /= 1 + (p.shape[0] * 1e-5)

        color_worlds = np.asarray(list(p[n_figure, n_object, :3]))
        shape_worlds = np.asarray(list(p[n_figure, n_object, 3:]))

        world_prob = np.outer(color_worlds, shape_worlds).flatten()
        entropy_value = -np.sum(world_prob * np.log(world_prob)) / (
            math.log(9)
        )

        return entropy_value

    probs = []
    p_list = []

    # FREQUENTIST
    if len(ensemble) == 0:

        # get predictions
        for i, data in enumerate(loader):
            images, labels, concepts = data
            images, labels, concepts = (
                images.to(model.device),
                labels.to(model.device),
                concepts.to(model.device),
            )

            logits = model(images, activate_simple_concepts=True)
            logits = list(torch.split(logits, 3, dim=-1))
            for i in range(len(logits)):
                logits[i] = torch.nn.functional.softmax(
                    logits[i], dim=-1
                )
            logits = torch.cat(logits, dim=-1)
            probs.append(logits.detach().cpu().numpy())

        probs = np.concatenate(
            probs, axis=0
        )  # data, images, objects, probabilities

    else:

        # ENSEMBLES
        for m in ensemble:
            probs = []
            for i, data in enumerate(loader):
                images, labels, concepts = data
                images, labels, concepts = (
                    images.to(model.device),
                    labels.to(model.device),
                    concepts.to(model.device),
                )

                logits = m(images, activate_simple_concepts=True)
                logits = list(torch.split(logits, 3, dim=-1))
                for i in range(len(logits)):
                    logits[i] = torch.nn.functional.softmax(
                        logits[i], dim=-1
                    )
                logits = torch.cat(logits, dim=-1)
                probs.append(logits.detach().cpu().numpy())

            probs = np.concatenate(
                probs, axis=0
            )  # data, images, objects, probabilities
            p_list.append(probs)

    # RETURN VALUES INITIALIZATION
    entropy_array = np.array([])
    figures_array = np.array([])
    objects_array = np.array([])
    indices = np.array([])

    for obj in range(9):
        # get indexes of figures and objects
        n_figure = obj // 3
        n_object = obj % 3

        # for all elements get entropies
        for i in range(probs.shape[0]):
            if len(ensemble) == 0:
                entropies = ova_entropy_per_object(
                    probs[i],
                    n_figure=n_figure,
                    n_object=n_object,
                )
            else:
                entropies = ova_entropy_per_object_ensemble(
                    p_list,
                    n_figure=n_figure,
                    n_object=n_object,
                )
            entropy_array = np.concatenate(
                (entropy_array, entropies), axis=None
            )
            figures_array = np.concatenate(
                (figures_array, n_figure), axis=None
            )
            objects_array = np.concatenate(
                (objects_array, n_object), axis=None
            )
            indices = np.concatenate((indices, i), axis=None)

    # rank indices in descending order according to entropy values
    ranked_indices = np.argsort(entropy_array)
    ranked_indices = ranked_indices[::-1]

    return (
        entropy_array[ranked_indices],
        figures_array[ranked_indices],
        objects_array[ranked_indices],
        indices[ranked_indices],
    )


def train(
    model: MnistDPL, dataset: BaseDataset, _loss: ADDMNIST_DPL, args
):
    """TRAINING

    Args:
        model (MnistDPL): network
        dataset (BaseDataset): dataset Kandinksy
        _loss (ADDMNIST_DPL): loss function
        args: parsed args

    Returns:
        None: This function does not return a value.
    """

    # Default Setting for Training
    model.to(model.device)
    train_loader, val_loader, test_loader = dataset.get_data_loaders()
    dataset.print_stats()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        model.opt, args.exp_decay
    )
    w_scheduler = GradualWarmupScheduler(
        model.opt, 1.0, args.warmup_steps
    )

    if args.wandb is not None:
        fprint("\n---wandb on\n")
        wandb.init(
            project=args.project,
            entity=args.wandb,
            name=str(args.dataset) + "_" + str(args.model),
            config=args,
        )

    fprint("\n--- Start of Training ---\n")

    # default for warm-up
    model.opt.zero_grad()
    model.opt.step()

    for epoch in range(args.n_epochs):
        model.train()

        ys, y_true = None, None

        for i, data in enumerate(train_loader):
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

            model.opt.zero_grad()
            loss, losses = _loss(out_dict, args)

            # print(loss)

            loss.backward()
            model.opt.step()

            if ys is None:
                ys = out_dict["YS"]
                y_true = out_dict["LABELS"]
            else:
                ys = torch.concatenate((ys, out_dict["YS"]), dim=0)
                y_true = torch.concatenate(
                    (y_true, out_dict["LABELS"]), dim=0
                )

            if args.wandb is not None:
                wandb_log_step(i, epoch, loss.item(), losses)

            if i % 10 == 0:
                progress_bar(
                    i, len(train_loader) - 9, epoch, loss.item()
                )

        y_pred = torch.argmax(ys, dim=-1)

        print(
            "\n Train acc: ",
            (y_pred == y_true).sum().item() / len(y_true) * 100,
            "%",
            len(y_true),
        )

        model.eval()
        tloss, cacc, yacc, f1 = evaluate_metrics(
            model, val_loader, args
        )

        # update at end of the epoch
        if epoch < args.warmup_steps:
            w_scheduler.step()
        else:
            scheduler.step()
            if hasattr(_loss, "grade"):
                _loss.update_grade(epoch)

        ### LOGGING ###
        fprint("  ACC C", cacc, "  ACC Y", yacc)

        if args.wandb is not None:
            wandb_log_epoch(
                epoch=epoch,
                acc=yacc,
                cacc=cacc,
                tloss=tloss,
                lr=float(scheduler.get_last_lr()[0]),
            )

    # Evaluate performances on val or test
    if args.validate:
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
    else:
        (
            y_true,
            c_true,
            y_pred,
            c_pred,
            p_cs,
            p_ys,
            p_cs_all,
            p_ys_all,
        ) = evaluate_metrics(model, test_loader, args, last=True)

    yac, yf1 = evaluate_mix(y_true, y_pred)
    cac, cf1 = evaluate_mix(c_true, c_pred)
    h_c = mean_entropy(p_cs_all, model.n_facts)

    fprint(f"Concepts:\n    ACC: {cac}, F1: {cf1}")
    fprint(f"Labels:\n      ACC: {yac}, F1: {yf1}")
    fprint(f"Entropy:\n     H(C): {h_c}")

    if args.wandb is not None:
        K = max(max(y_pred), max(y_true))

        wandb.log({"test-y-acc": yac * 100, "test-y-f1": yf1 * 100})
        wandb.log({"test-c-acc": cac * 100, "test-c-f1": cf1 * 100})

        wandb.log(
            {
                "cf-labels": wandb.plot.confusion_matrix(
                    None,
                    y_true,
                    y_pred,
                    class_names=[str(i) for i in range(K + 1)],
                ),
            }
        )
        K = max(np.max(c_pred), np.max(c_true))
        wandb.log(
            {
                "cf-concepts": wandb.plot.confusion_matrix(
                    None,
                    c_true,
                    c_pred,
                    class_names=[str(i) for i in range(K + 1)],
                ),
            }
        )

        if hasattr(model, "decoder"):
            list_images = make_grid(
                conditional_gen(model),
                nrow=8,
            )
            images = wandb.Image(
                list_images, caption="Generated samples"
            )
            wandb.log({"Conditional Gen": images})

            list_images = make_grid(
                recon_visaulization(out_dict), nrow=8
            )
            images = wandb.Image(
                list_images, caption="Reconstructed samples"
            )
            wandb.log({"Reconstruction": images})

        wandb.finish()
