import torch
import torch.nn.functional as F
from DPL.dpl_auc import DPL_AUC


class DPL_AUC_PCBM(DPL_AUC):
    def __init__(
        self, conceptizer, parametrizer, aggregator, cbm, senn, device
    ):
        self.gaussian_vars = False
        super(DPL_AUC_PCBM, self).__init__(
            conceptizer, parametrizer, aggregator, cbm, senn, device
        )

    def forward(self, x):
        # Get concepts, h_x_labeled is known, h_x is unknown concepts
        _, h_x, (mu, logvar) = self.conceptizer(x)
        # h_x_labeled = h_x_labeled_raw.view(-1,h_x_labeled_raw.shape[1], 1)

        # z stuff
        z_plus = F.normalize(self.conceptizer.positives, p=2, dim=-1)
        z_nega = F.normalize(self.conceptizer.negatives, p=2, dim=-1)

        z_tot = torch.cat([z_nega, z_plus]).unsqueeze(-2)

        # sampling
        prob_Cs, c_logits = [], []
        latents = F.normalize(mu, p=2, dim=-1)
        logsigma = torch.clip(logvar, max=10)

        if self.gaussian_vars:
            return latents, logsigma

        pred_embeddings = self._sample_gaussian_tensors(
            latents, logsigma, 100
        )
        concept_logit, concept_prob = self._compute_distance(
            pred_embeddings, z_tot
        )

        # prob_Cs = concept_prob[..., 1]
        # print(concept_logit.shape)
        # c_logits = torch.sigmoid(concept_logit[...,1])
        # print(concept_logit[...,1].shape)

        # store known concepts
        # if not self.senn:
        squeezed_c_logits = torch.unsqueeze(
            concept_prob[..., 1], dim=-1
        )
        # store (known+unknown) concepts
        self.concepts = torch.cat((squeezed_c_logits, h_x), dim=1)
        # store known concepts
        self.concepts_labeled = squeezed_c_logits
        # else:
        #     self.concepts = c_logits

        if self.ignore_prob_log:
            return concept_prob[..., 1]

        # if you use cbm, aggregator does not use unknown concepts, even if you define it
        out = self.proglob_pred()

        if self.return_both_concept_out_prob:
            return out, c_logits

        return out

    def _batchwise_cdist(self, samples1, samples2, eps=1e-6):
        if len(samples1.size()) not in [3, 4, 5] or len(
            samples2.size()
        ) not in [
            3,
            4,
            5,
        ]:
            raise RuntimeError(
                "expected: 4-dim tensors, got: {}, {}".format(
                    samples1.size(), samples2.size()
                )
            )

        if samples1.shape[1] == samples2.shape[1]:
            samples1 = samples1.unsqueeze(2)
            samples2 = samples2.unsqueeze(3)
            samples1 = samples1.unsqueeze(1)
            samples2 = samples2.unsqueeze(0)
            result = torch.sqrt(
                ((samples1 - samples2) ** 2).sum(-1) + eps
            )
            return result.view(*result.shape[:-2], -1)
        else:
            raise RuntimeError(
                f"samples1 ({samples1.size()}) and samples2 ({samples2.size()}) dimensionalities "
                "are non-broadcastable."
            )

    def _compute_distance(
        self,
        pred_embeddings,
        z_tot,
        negative_scale=None,
        shift=None,
        reduction="mean",
    ):
        negative_scale = (
            self.conceptizer.negative_scale
            if negative_scale is None
            else negative_scale
        )

        distance = self._batchwise_cdist(pred_embeddings, z_tot)

        distance = distance.permute(0, 2, 3, 1)

        logits = -negative_scale.view(1, -1, 1, 1) * distance
        prob = torch.nn.functional.softmax(logits, dim=-1)
        if reduction == "none":
            return logits, prob
        return logits.mean(axis=-2), prob.mean(axis=-2)

    def _sample_gaussian_tensors(self, mu, logsigma, num_samples):
        eps = torch.randn(
            mu.size(0),
            mu.size(1),
            num_samples,
            mu.size(2),
            dtype=mu.dtype,
            device=mu.device,
        )
        samples_sigma = eps.mul(
            torch.exp(logsigma.unsqueeze(2) * 0.5)
        )
        samples = samples_sigma.add_(mu.unsqueeze(2))
        return samples
