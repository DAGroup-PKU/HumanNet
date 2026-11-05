"""Order-two UniPC inference with immutable schedule coefficients."""

from dataclasses import dataclass, replace

import numpy as np
import torch


@dataclass(frozen=True)
class Update:
    ratio: torch.Tensor
    alpha_phi: torch.Tensor
    alpha_b: torch.Tensor
    rks: tuple
    rhos: torch.Tensor | None


def coefficients(scheduler, index, order, *, corrector, device, dtype):
    """Evaluate native scalar expressions once, on their original devices."""
    sigmas = scheduler.sigmas
    target, source = (index, index - 1) if corrector else (index + 1, index)
    alpha_t, sigma_t = scheduler._sigma_to_alpha_sigma_t(sigmas[target])
    alpha_s0, sigma_s0 = scheduler._sigma_to_alpha_sigma_t(sigmas[source])
    lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
    lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
    h = lambda_t - lambda_s0
    rks = []
    for offset in range(1, order):
        previous = index - offset - 1 if corrector else index - offset
        alpha_si, sigma_si = scheduler._sigma_to_alpha_sigma_t(sigmas[previous])
        rks.append((torch.log(alpha_si) - torch.log(sigma_si) - lambda_s0) / h)
    hh = -h
    h_phi_1 = torch.expm1(hh)
    b_h = torch.expm1(hh)
    if corrector and order == 2:
        all_rks = torch.tensor([*rks, 1.0], device=device)
        h_phi_k = h_phi_1 / hh - 1
        rows, rhs = ([], [])
        factorial = 1
        for row in range(1, order + 1):
            rows.append(torch.pow(all_rks, row - 1))
            rhs.append(h_phi_k * factorial / b_h)
            factorial *= row + 1
            h_phi_k = h_phi_k / hh - 1 / factorial
        rhos = torch.linalg.solve(torch.stack(rows), torch.tensor(rhs, device=device)).to(dtype)
    elif corrector or order == 2:
        rhos = torch.tensor([0.5], dtype=dtype, device=device)
    else:
        rhos = None
    return Update(sigma_t / sigma_s0, alpha_t * h_phi_1, alpha_t * b_h, tuple(rks), rhos)


class UniPCPlan:
    def __init__(self, config, device):
        # Preserve the FP32 endpoint before NumPy constructs the inference grid.
        endpoint = torch.tensor(1.0 - 1.0 / 1000, dtype=torch.float32).item()
        sigmas = np.linspace(endpoint, 0, config.num_steps + 1)[:-1]
        sigmas = config.flow_shift * sigmas / (1 + (config.flow_shift - 1) * sigmas)
        self.timesteps = tuple(int(t) for t in sigmas * 1000)
        self.sigmas = torch.from_numpy(np.concatenate((sigmas, [0])).astype(np.float32))
        self.updates = []
        previous_order = 0
        for index in range(config.num_steps):
            corrector = (
                coefficients(
                    self, index, previous_order, corrector=True, device=device, dtype=torch.float32
                )
                if index
                else None
            )
            if corrector is not None and config.corrector_weights is not None:
                corrector = replace(
                    corrector,
                    rhos=torch.tensor(
                        config.corrector_weights[index], dtype=torch.float32, device=device
                    ),
                )
            order = min(2, config.num_steps - index, index + 1)
            predictor = coefficients(
                self, index, order, corrector=False, device=device, dtype=torch.float32
            )
            self.updates.append((corrector, predictor))
            previous_order = order

    @staticmethod
    def _sigma_to_alpha_sigma_t(sigma):
        return 1 - sigma, sigma

    def run(self, initial, velocity):
        """Match native [B,C,...] einsum dimensions and operation order exactly."""
        current = initial.unsqueeze(0)
        outputs = [None, None]
        last_sample = None
        for index, timestep in enumerate(self.timesteps):
            pred = velocity(current.squeeze(0), timestep).unsqueeze(0)
            converted = current - self.sigmas[index] * pred
            corrector, predictor = self.updates[index]
            m0 = outputs[-1]
            if corrector is not None:
                differences = [
                    (outputs[-offset - 2] - m0) / rk for offset, rk in enumerate(corrector.rks)
                ]
                corr = (
                    torch.einsum(
                        "k,bkc...->bc...", corrector.rhos[:-1], torch.stack(differences, dim=1)
                    )
                    if differences
                    else 0
                )
                x_t = corrector.ratio * last_sample - corrector.alpha_phi * m0
                current = (
                    x_t - corrector.alpha_b * (corr + corrector.rhos[-1] * (converted - m0))
                ).to(current.dtype)
            outputs = [outputs[-1], converted]
            last_sample = current
            differences = [
                (outputs[-offset - 2] - converted) / rk for offset, rk in enumerate(predictor.rks)
            ]
            prediction = (
                torch.einsum("k,bkc...->bc...", predictor.rhos, torch.stack(differences, dim=1))
                if differences
                else 0
            )
            x_t = predictor.ratio * current - predictor.alpha_phi * converted
            current = (x_t - predictor.alpha_b * prediction).to(current.dtype)
        return current.squeeze(0)
