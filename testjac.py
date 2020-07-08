import numpy as np

def g(r):
    return r

def gprime(r):
    return r

y = np.array([[0, 0, 0], [1, 2, 3], [8, -1, 5], [10, 11, 12]])
n = y.shape[0]
tmp = np.repeat(y[:, :, np.newaxis], y.shape[0], axis=2)
norm = np.sqrt(((tmp - tmp.transpose())**2).sum(axis=1))
r_hat = np.rollaxis(tmp-tmp.transpose(), 2, 0)

# Step 1
def rrT(r):
    r = r[:, np.newaxis]
    return r@r.transpose()

B = np.apply_along_axis(rrT, 2, r_hat)


with np.errstate(divide='ignore', invalid='ignore'):
    # Ignore divide by 0 warnings
    # All NaNs are removed below
    B = (
            B*(
            (gprime(norm)-g(norm)/norm)[:, :, np.newaxis, np.newaxis]
            .repeat(B.shape[2], axis=2).repeat(B.shape[3], axis=3))
        + (np.identity(3)[np.newaxis, np.newaxis, :, :]
            .repeat(B.shape[0], axis=0).repeat(B.shape[1], axis=1))*
            (g(norm)/norm)[:, :, np.newaxis, np.newaxis]
            .repeat(B.shape[2], axis=2).repeat(B.shape[3], axis=3))

    B[np.isnan(B)] = 0

# Step 2: compute the diagonal

B[range(n), range(n), :, :] = -B.sum(axis=0)

# Step 3: Build block matrix
print(B.reshape(n,n,3,3).swapaxes(1,2).reshape(3*n, -1))
