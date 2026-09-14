import torch
import numpy as np

P = 5
N_QUBITS = 12


class QAOA:
    """Дифференцируемый симулятор схемы QAOA глубины p=5 для 12-кубитной модели Изинга.

    J фиксирована, h — вектор линейных членов (вход). По заданным углам gamma, beta
    считает квантовое состояние и метрику P(ground). Углы не подбирает — оценивает.
    Все операции на torch и дифференцируемы по углам: можно обучать модель backprop'ом.
    """

    def __init__(self, J, device="cpu"):
        self.device = device
        self.n = J.shape[0]
        self.p = P
        self.dim = 2 ** self.n

        J = torch.as_tensor(J, dtype=torch.float32, device=device)
        J = (J + J.T) / 2
        J.fill_diagonal_(0)
        self.J = J

        bits = torch.arange(self.dim, device=device)
        x = ((bits.unsqueeze(1) >> torch.arange(self.n - 1, -1, -1, device=device)) & 1).float()
        self.S = 2 * x - 1
        self.quad = 0.5 * torch.einsum("ij,ki,kj->k", self.J, self.S, self.S)

    def energies(self, h):
        h = torch.as_tensor(h, dtype=torch.float32, device=self.device)
        if h.ndim == 1:
            h = h.unsqueeze(0)
        return self.quad.unsqueeze(0) + h @ self.S.T

    def _mixer(self, state, beta):
        cb = torch.cos(beta).to(torch.complex64)
        sb = (-1j * torch.sin(beta)).to(torch.complex64)
        B = state.shape[0]
        cbk = cb.view(B, 1, 1)
        sbk = sb.view(B, 1, 1)
        for k in range(self.n):
            v = state.view(B, 2 ** k, 2, 2 ** (self.n - 1 - k))
            a = v[:, :, 0, :]
            c = v[:, :, 1, :]
            state = torch.stack([cbk * a + sbk * c, sbk * a + cbk * c], dim=2).reshape(B, self.dim)
        return state

    def state(self, h, gamma, beta):
        E = self.energies(h)
        B = E.shape[0]
        gamma = torch.as_tensor(gamma, dtype=torch.float32, device=self.device)
        beta = torch.as_tensor(beta, dtype=torch.float32, device=self.device)
        if gamma.ndim == 1:
            gamma = gamma.unsqueeze(0).expand(B, -1)
        if beta.ndim == 1:
            beta = beta.unsqueeze(0).expand(B, -1)
        psi = torch.full((B, self.dim), 1 / np.sqrt(self.dim), dtype=torch.complex64, device=self.device)
        for l in range(self.p):
            psi = psi * torch.exp(1j * (gamma[:, l].unsqueeze(1) * E))
            psi = self._mixer(psi, beta[:, l])
        return psi

    def probs(self, h, gamma, beta):
        return self.state(h, gamma, beta).abs() ** 2

    def p_ground(self, h, gamma, beta):
        E = self.energies(h)
        prob = self.probs(h, gamma, beta)
        gmin = E.min(dim=1, keepdim=True).values
        mask = (E <= gmin + 1e-9).float()
        return (prob * mask).sum(dim=1)


if __name__ == "__main__":
    J = np.load("J.npy")
    h = np.load("h_train.npy")

    qaoa = QAOA(torch.tensor(J, dtype=torch.float32))

    gamma = torch.rand(len(h), P, requires_grad=True)
    beta = torch.rand(len(h), P, requires_grad=True)

    opt = torch.optim.Adam([gamma, beta], lr=0.05)
    for step in range(200):
        opt.zero_grad()
        loss = -qaoa.p_ground(torch.tensor(h, dtype=torch.float32), gamma, beta).mean()
        loss.backward()
        opt.step()

    print(qaoa.p_ground(torch.tensor(h, dtype=torch.float32), gamma, beta).mean().item())
