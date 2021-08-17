# Author: Remi Flamary <remi.flamary@unice.fr>
#         Kilian Fatras <kilian.fatras@irisa.fr>
#
# License: MIT License

# sphinx_gallery_thumbnail_number = 4

import numpy as np
import matplotlib.pylab as pl
import ot
import ot.plot

n = 5  # nb samples

mu_s = np.array([0, 0])
cov_s = np.array([[1, 0], [0, 1]])

print('mu_s: {}'.format(mu_s))
print('cov_s: {}'.format(cov_s))
input('')

mu_t = np.array([4, 4])
cov_t = np.array([[1, -.8], [-.8, 1]])

print('mu_t: {}'.format(mu_t))
print('cov_t: {}'.format(cov_t))
input('')

xs = ot.datasets.make_2D_samples_gauss(n, mu_s, cov_s)
print('xs: ')
print(xs)
input('')

xt = ot.datasets.make_2D_samples_gauss(n, mu_t, cov_t)
print('xt: ')
print(xt)
input('')

a, b = np.ones((n,)) / n, np.ones((n,)) / n  # uniform distribution on samples
a[0] = 0.25
a[1] = 0.15
print('a: ')
print(a)
input('')

print('b: ')
print(b)
input('')

# loss matrix
M = ot.dist(xs, xt)
print('M prev: ')
print(M)
input('')

M /= M.max()
print('M after: ')
print(M)
input('')


pl.figure(1)
pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
pl.legend(loc=0)
pl.title('Source and target distributions')

pl.figure(2)
pl.imshow(M, interpolation='nearest')
pl.title('Cost matrix M')

G0 = ot.emd(a, b, M)
print('G0: ')
print(G0)
input('')


pl.figure(3)
pl.imshow(G0, interpolation='nearest')
pl.title('OT matrix G0')

pl.figure(4)
ot.plot.plot2D_samples_mat(xs, xt, G0, c=[.5, .5, 1])
pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
pl.legend(loc=0)
pl.title('OT matrix with samples')