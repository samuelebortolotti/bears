# Module which identifies an LTN loss
import itertools

import ltn
import torch
from utils.normal_kl_divergence import kl_divergence


class identity(torch.nn.Module):
    """Identity module"""

    def __init__(self):
        """Init module

        Args:
            self: instance

        Returns:
            None: This function does not return a value.
        """
        super(identity, self).__init__()

    def forward(self, x, d):
        return torch.gather(x, 1, d)


class ADDMNIST_SAT_AGG(torch.nn.Module):
    def __init__(
        self, loss, task="addition", nr_classes=19, pcbm=False
    ) -> None:
        super().__init__()
        self.base_loss = loss
        self.task = task
        self.pcbm = pcbm
        self.beta = 0.001

        if task == "addition":
            self.nr_classes = 19
        elif task == "product":
            self.nr_classes = 81
        elif task == "multiop":
            self.nr_classes = 15

        self.iden = identity()
        self.ltn = ltn.Predicate(self.iden)

        self.grade = 2  # 10

    def update_grade(self, epoch):
        """Update grade function

        Args:
            self: instance
            epoch: epoch number

        Returns:
            None: This function does not return a value.
        """
        # if (epoch) % 2 == 0 and epoch != 0:
        #    self.grade += 1 if self.grade < 9 else 0

        # if (epoch) % 3 == 0 and epoch != 0:
        #     self.grade += 2 if self.grade < 9 else 0

        print("Currently in", epoch, "grade:", self.grade)
        if epoch in range(0, 4):
            self.grade = 2
        if epoch in range(4, 8):
            self.grade = 3
        if epoch in range(8, 12):
            self.grade = 6
        if epoch in range(12, 20):
            self.grade = 9
        if epoch in range(20, 100):
            self.grade = 10

    def forward(self, out_dict, args):
        """Forward module

        Args:
            self: instance
            out_dict: output dictionary
            args: command line arguments

        Returns:
            loss: loss value
        """
        loss, losses = self.base_loss(out_dict, args)

        # load from dict
        Ys = out_dict["LABELS"]  # groundtruth labels
        pCs = out_dict["pCS"]

        prob_digit1, prob_digit2 = pCs[:, 0, :], pCs[:, 1, :]

        if self.task == "addition":
            sat_loss = ADDMNISTsat_agg_loss(
                self.ltn, prob_digit1, prob_digit2, Ys, self.grade
            )
        elif self.task == "product":
            sat_loss = PRODMNISTsat_agg_loss(
                self.ltn, prob_digit1, prob_digit2, Ys, self.grade
            )
        elif self.task == "multiop":
            sat_loss = MULTIOPsat_agg_loss(
                self.ltn, prob_digit1, prob_digit2, Ys, self.grade
            )

        losses.update({"sat-loss": sat_loss.item()})

        # print("SAT LOSS:", sat_loss.item())

        if self.pcbm:
            kl_div = 0

            mus = out_dict["MUS"]
            logvars = out_dict["LOGVARS"]
            for i in range(2):
                kl_div += kl_divergence(mus[i], logvars[i])

            losses.update({"kl-div": kl_div})

        return loss + sat_loss + self.beta * kl_div, losses


def ADDMNISTsat_agg_loss(eltn, p1, p2, labels, grade):
    """Addmnist sat agg loss

    Args:
        eltn: eltn
        p1: probability of the first concept
        p2: probability of the second concept
        labels: labels
        grade: grade

    Returns:
        loss: loss value
    """
    max_c = p1.size(-1)

    # convert to variables
    bit1 = ltn.Variable("bit1", p1)  # , trainable=True)
    bit2 = ltn.Variable("bit2", p2)  # , trainable=True)
    y_true = ltn.Variable("labels", labels)

    # variables in LTN
    b_1 = ltn.Variable("b_1", torch.tensor(range(max_c)))
    b_2 = ltn.Variable("b_2", torch.tensor(range(max_c)))

    And = ltn.Connective(ltn.fuzzy_ops.AndProd())
    Exists = ltn.Quantifier(
        ltn.fuzzy_ops.AggregPMean(p=2), quantifier="e"
    )
    Forall = ltn.Quantifier(
        ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f"
    )
    SatAgg = ltn.fuzzy_ops.SatAgg()

    sat_agg = Forall(
        ltn.diag(bit1, bit2, y_true),
        Exists(
            [b_1, b_2],
            And(eltn(bit1, b_1), eltn(bit2, b_2)),
            cond_vars=[b_1, b_2, y_true],
            cond_fn=lambda d1, d2, z: torch.eq(
                (d1.value + d2.value), z.value
            ),
            p=grade,
        ),
    )

    return 1 - sat_agg.value


def PRODMNISTsat_agg_loss(eltn, p1, p2, labels, grade):
    """Prodmnist sat agg loss

    Args:
        eltn: eltn
        p1: probability of the first concept
        p2: probability of the second concept
        labels: labels
        grade: grade

    Returns:
        loss: loss value
    """
    max_c = p1.size(-1)

    # convert to variables
    bit1 = ltn.Variable("bit1", p1)  # , trainable=True)
    bit2 = ltn.Variable("bit2", p2)  # , trainable=True)
    y_true = ltn.Variable("labels", labels)

    # print(bit1)
    # print(bit2)
    # print(y_true)

    # variables in LTN
    b_1 = ltn.Variable("b_1", torch.tensor(range(max_c)))
    b_2 = ltn.Variable("b_2", torch.tensor(range(max_c)))

    And = ltn.Connective(ltn.fuzzy_ops.AndProd())
    Exists = ltn.Quantifier(
        ltn.fuzzy_ops.AggregPMean(p=2), quantifier="e"
    )
    Forall = ltn.Quantifier(
        ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f"
    )
    SatAgg = ltn.fuzzy_ops.SatAgg()

    sat_agg = Forall(
        ltn.diag(bit1, bit2, y_true),
        Exists(
            [b_1, b_2],
            And(eltn(bit1, b_1), eltn(bit2, b_2)),
            cond_vars=[b_1, b_2, y_true],
            cond_fn=lambda b_1, b_2, z: torch.eq(
                b_1.value * b_2.value, z.value
            ),
            p=grade,
        ),
    ).value
    return 1 - sat_agg


def MULTIOPsat_agg_loss(eltn, p1, p2, labels, grade):
    """Multioperation sat agg loss

    Args:
        eltn: eltn
        p1: probability of the first concept
        p2: probability of the second concept
        labels: labels
        grade: grade

    Returns:
        loss: loss value
    """
    max_c = p1.size(-1)

    # convert to variables
    bit1 = ltn.Variable("bit1", p1)  # , trainable=True)
    bit2 = ltn.Variable("bit2", p2)  # , trainable=True)
    y_true = ltn.Variable("labels", labels)

    # variables in LTN
    b_1 = ltn.Variable("b_1", torch.tensor(range(max_c)))
    b_2 = ltn.Variable("b_2", torch.tensor(range(max_c)))

    And = ltn.Connective(ltn.fuzzy_ops.AndProd())
    Exists = ltn.Quantifier(
        ltn.fuzzy_ops.AggregPMean(p=2), quantifier="e"
    )
    Forall = ltn.Quantifier(
        ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f"
    )
    SatAgg = ltn.fuzzy_ops.SatAgg()

    sat_agg = Forall(
        ltn.diag(bit1, bit2, y_true),
        Exists(
            [b_1, b_2],
            And(eltn(bit1, b_1), eltn(bit2, b_2)),
            cond_vars=[b_1, b_2, y_true],
            cond_fn=lambda b_1, b_2, z: torch.eq(
                b_1.value**2 + b_2.value**2 + b_1.value * b_2.value,
                z.value,
            ),
            p=grade,
        ),
    ).value
    return 1 - sat_agg
