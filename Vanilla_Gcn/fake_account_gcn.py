"""
Fake Account Detection using a Graph Convolutional Network (GCN)
Pure Python: no imports, no NumPy, no ML library. Everything built by hand.
"""

E = 2.718281828459045


# ---------------------------------------------------------------
# 1. Small helper functions (our own matrix library)
# ---------------------------------------------------------------
def zeros(r, c):
    return [[0.0] * c for _ in range(r)]


def transpose(M):
    return [[M[i][j] for i in range(len(M))] for j in range(len(M[0]))]


def matmul(A, B):
    n, m, p = len(A), len(B), len(B[0])
    C = zeros(n, p)
    for i in range(n):
        for k in range(m):
            a = A[i][k]
            if a != 0.0:
                for j in range(p):
                    C[i][j] += a * B[k][j]
    return C


def relu(M):
    return [[x if x > 0 else 0.0 for x in row] for row in M]


def softmax_rows(M):
    out = []
    for row in M:
        mx = max(row)
        ex = [E ** (x - mx) for x in row]
        s = sum(ex)
        out.append([x / s for x in ex])
    return out


def argmax(row):
    best = 0
    for i in range(1, len(row)):
        if row[i] > row[best]:
            best = i
    return best


class RNG:
    """Tiny random generator (LCG) so we need no 'random' module."""
    def __init__(self, seed=7):
        self.s = seed

    def uniform(self, lo, hi):
        self.s = (1103515245 * self.s + 12345) % 2147483648
        return lo + (hi - lo) * (self.s / 2147483648)


# ---------------------------------------------------------------
# 2. Build the normalized graph:  A_hat = D^-1/2 (A + I) D^-1/2
# ---------------------------------------------------------------
def build_adjacency(n, edges):
    A = zeros(n, n)
    for u, v in edges:
        A[u][v] = 1.0
        A[v][u] = 1.0
    for i in range(n):          # self-loops
        A[i][i] = 1.0
    return A


def normalize(A):
    n = len(A)
    deg = [sum(row) for row in A]
    A_hat = zeros(n, n)
    for i in range(n):
        for j in range(n):
            if A[i][j] != 0.0:
                A_hat[i][j] = A[i][j] / ((deg[i] ** 0.5) * (deg[j] ** 0.5))
    return A_hat


# ---------------------------------------------------------------
# 3. The GCN model
# ---------------------------------------------------------------
class GCN:
    def __init__(self, in_dim, hidden, out_dim):
        rng = RNG()
        self.W1 = [[rng.uniform(-0.8, 0.8) for _ in range(hidden)] for _ in range(in_dim)]
        self.W2 = [[rng.uniform(-0.8, 0.8) for _ in range(out_dim)] for _ in range(hidden)]

    def forward(self, A_hat, X):
        self.AX = matmul(A_hat, X)
        self.Z1 = matmul(self.AX, self.W1)
        self.H1 = relu(self.Z1)
        self.AH = matmul(A_hat, self.H1)
        self.P = softmax_rows(matmul(self.AH, self.W2))
        return self.P

    def backward(self, A_hat, Y, train_idx, lr):
        n, c = len(self.P), len(self.P[0])
        dZ2 = zeros(n, c)
        for i in train_idx:                       # loss only on labeled nodes
            for j in range(c):
                dZ2[i][j] = (self.P[i][j] - Y[i][j]) / len(train_idx)
        dW2 = matmul(transpose(self.AH), dZ2)
        dH1 = matmul(A_hat, matmul(dZ2, transpose(self.W2)))
        dZ1 = [[dH1[i][j] if self.Z1[i][j] > 0 else 0.0
                for j in range(len(dH1[0]))] for i in range(n)]
        dW1 = matmul(transpose(self.AX), dZ1)
        for i in range(len(self.W1)):
            for j in range(len(self.W1[0])):
                self.W1[i][j] -= lr * dW1[i][j]
        for i in range(len(self.W2)):
            for j in range(len(self.W2[0])):
                self.W2[i][j] -= lr * dW2[i][j]


# ---------------------------------------------------------------
# 4. INPUT DATA
# ---------------------------------------------------------------
# Features (all scaled 0..1): [account_age, posts_per_day, following/followers ratio]
names = ["Asha", "Bala", "Chitra", "Dev", "Esha", "Farid", "Gita", "Hari",      # 0-7
         "bot_01", "bot_02", "bot_03", "bot_04", "bot_05", "bot_06"]            # 8-13

X = [
    [0.9, 0.3, 0.2], [0.8, 0.2, 0.3], [0.7, 0.4, 0.1], [0.9, 0.2, 0.2],
    [0.6, 0.3, 0.4], [0.8, 0.5, 0.2], [0.7, 0.3, 0.3], [0.5, 0.4, 0.3],
    [0.1, 0.9, 0.9], [0.2, 0.8, 0.8], [0.1, 0.9, 0.7], [0.2, 0.7, 0.9],
    [0.1, 0.8, 0.8], [0.3, 0.6, 0.6],
]

# Connections (who follows / interacts with whom)
edges = [
    (0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (1, 6), (0, 7),  # real group
    (8, 9), (8, 10), (9, 11), (10, 11), (11, 12), (12, 13), (9, 13), (8, 12),        # bot group
    (7, 13), (4, 10),                                                               # a few cross links
]

# Known labels (0 = real, 1 = fake). Only 4 accounts are verified by a human.
known = {0: 0, 1: 0, 8: 1, 9: 1}
true_labels = [0] * 8 + [1] * 6          # ground truth, used ONLY to measure accuracy


# ---------------------------------------------------------------
# 5. TRAIN
# ---------------------------------------------------------------
def main():
    n = len(X)
    A_hat = normalize(build_adjacency(n, edges))
    train_idx = sorted(known.keys())
    Y = [[0.0, 0.0] for _ in range(n)]
    for i, lab in known.items():
        Y[i][lab] = 1.0

    model = GCN(in_dim=3, hidden=4, out_dim=2)
    print("Accounts:", n, "| Connections:", len(edges), "| Labeled accounts:", len(known))
    print("-" * 60)

    for epoch in range(1, 301):
        P = model.forward(A_hat, X)
        mean_conf = sum(P[i][known[i]] for i in train_idx) / len(train_idx)
        model.backward(A_hat, Y, train_idx, lr=0.5)
        if epoch == 1 or epoch % 50 == 0:
            test = [i for i in range(n) if i not in known]
            acc = sum(1 for i in test if argmax(P[i]) == true_labels[i]) / len(test)
            print(f"Epoch {epoch:3d} | confidence on labeled: {mean_conf:.3f} | accuracy on unlabeled: {acc:.2f}")

    # ---------------- OUTPUT ----------------
    P = model.forward(A_hat, X)
    print("-" * 60)
    print(f"{'Account':<10}{'P(fake)':>9}   {'Predicted':<10}{'Actual':<8}")
    for i in range(n):
        pred = "FAKE" if argmax(P[i]) == 1 else "REAL"
        act = "FAKE" if true_labels[i] == 1 else "REAL"
        tag = "(labeled)" if i in known else ""
        print(f"{names[i]:<10}{P[i][1]:>9.3f}   {pred:<10}{act:<8}{tag}")


if __name__ == "__main__":
    main()