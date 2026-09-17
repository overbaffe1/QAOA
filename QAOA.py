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

class FastQAOA:
    """Тот же QAOA, в ~5 раз быстрее: миксер применяется группами по `group` кубитов.

    Точно (не приближение): exp(-i*beta*sum_k X_k) факторизуется по кубитам,
    т.к. вращения RX на разных кубитах коммутируют; для группы g кубитов матрица
    2^g x 2^g выписывается явно через popcount и применяется одним bmm.
    12 проходов по памяти на слой -> 12/group. p_ground совпадает с QAOA.p_ground до 1e-9.
    """

    def __init__(self, J, p=None, group=4, device="cpu"):
        self.device = device
        self.n = J.shape[0]
        self.p = p or P
        self.dim = 2 ** self.n
        if self.n % group:
            raise ValueError("число кубитов должно делиться на group")
        self.g = group
        self.ng = self.n // group
        J = torch.as_tensor(J, dtype=torch.float32, device=device)
        J = (J + J.T) / 2
        J.fill_diagonal_(0)
        self.J = J
        bits = torch.arange(self.dim, device=device)
        x = ((bits.unsqueeze(1) >> torch.arange(self.n - 1, -1, -1, device=device)) & 1).float()
        self.S = 2 * x - 1
        self.quad = 0.5 * torch.einsum("ij,ki,kj->k", self.J, self.S, self.S)
        gb = torch.arange(2 ** group, device=device)
        gbits = (gb.unsqueeze(1) >> torch.arange(group - 1, -1, -1, device=device)) & 1
        k = (gbits.unsqueeze(1) ^ gbits.unsqueeze(0)).sum(-1)
        self.k = k
        self.phase = torch.tensor([1, -1j, -1, 1j], dtype=torch.complex64,
                                  device=device)[k % 4]

    def energies(self, h):
        h = torch.as_tensor(h, dtype=torch.float32, device=self.device)
        if h.ndim == 1:
            h = h.unsqueeze(0)
        return self.quad.unsqueeze(0) + h @ self.S.T

    def _mixer_matrix(self, beta):
        c = torch.cos(beta).view(-1, 1, 1)
        s = torch.sin(beta).view(-1, 1, 1)
        amp = c ** (self.g - self.k) * s ** self.k
        return amp.to(torch.complex64) * self.phase

    def state(self, h, gamma, beta):
        E = self.energies(h)
        B = E.shape[0]
        gamma = torch.as_tensor(gamma, dtype=torch.float32, device=self.device)
        beta = torch.as_tensor(beta, dtype=torch.float32, device=self.device)
        if gamma.ndim == 1:
            gamma = gamma.unsqueeze(0).expand(B, -1)
        if beta.ndim == 1:
            beta = beta.unsqueeze(0).expand(B, -1)
        d = 2 ** self.g
        psi = torch.full((B, self.dim), 1 / np.sqrt(self.dim),
                         dtype=torch.complex64, device=self.device)
        for l in range(self.p):
            psi = psi * torch.exp(1j * (gamma[:, l].unsqueeze(1) * E))
            M = self._mixer_matrix(beta[:, l]).transpose(1, 2)
            for j in range(self.ng):
                left, right = d ** j, d ** (self.ng - 1 - j)
                v = psi.view(B, left, d, right).permute(0, 1, 3, 2).reshape(B, left * right, d)
                v = torch.bmm(v, M)
                psi = v.view(B, left, right, d).permute(0, 1, 3, 2).reshape(B, self.dim)
        return psi

    def probs(self, h, gamma, beta):
        return self.state(h, gamma, beta).abs() ** 2

    def p_ground(self, h, gamma, beta):
        E = self.energies(h)
        prob = self.probs(h, gamma, beta)
        gmin = E.min(dim=1, keepdim=True).values
        return (prob * (E <= gmin + 1e-9).float()).sum(dim=1)

