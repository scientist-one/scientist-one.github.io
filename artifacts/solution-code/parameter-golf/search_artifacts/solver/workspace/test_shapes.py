import torch
rows, cols = 64, 128
r = 4
E = torch.randn(rows, cols)
H = torch.randn(cols, cols)
H = H.T @ H + torch.eye(cols)

H_diag = H.diag()
H_diag_sqrt = torch.sqrt(H_diag)
H_diag_inv_sqrt = 1.0 / H_diag_sqrt

E_prime = E * H_diag_sqrt.unsqueeze(0)
U, S, Vh = torch.linalg.svd(E_prime, full_matrices=False)
sqrt_S = torch.sqrt(S[:r])
A = (U[:, :r] * sqrt_S).contiguous()
B_prime = (Vh[:r, :].T * sqrt_S).T.contiguous()
B = (B_prime * H_diag_inv_sqrt.unsqueeze(0)).contiguous()

print("E_prime shape:", E_prime.shape)
print("A shape:", A.shape)
print("B shape:", B.shape)

L = torch.linalg.cholesky(H)
L_inv = torch.linalg.inv(L)
E_new = E.clone()
E_weighted = E_new @ L
U, S, Vh = torch.linalg.svd(E_weighted, full_matrices=False)
sqrt_S = torch.sqrt(S[:r])
A2 = (U[:, :r] * sqrt_S).contiguous()
B_w = (Vh[:r, :].T * sqrt_S).T.contiguous()
B2 = (B_w @ L_inv).contiguous()

print("A2 shape:", A2.shape)
print("B2 shape:", B2.shape)
