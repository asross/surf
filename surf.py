import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import autograd.numpy as np
from autograd import elementwise_grad

def _flt(X):
  if isinstance(X, np.ndarray):
    return X.astype(float)
  elif isinstance(X, list):
    return np.array(X).astype(float)
  else:
    return X

class Surface():
  def __init__(self, f, normal_direction=-1.0):
    self.f = lambda x: f(_flt(x))
    self.normdir = normal_direction

  def jacobian(self, X):
    return np.array([
      elementwise_grad(lambda x: self.f(x)[i])(_flt(X))
      for i in range(3)
    ])

  def hessian(self, X):
    return np.array([
      [elementwise_grad(lambda x: self.jacobian(x)[i][j])(_flt(X))
      for j in range(2)]
      for i in range(3)
    ])

  def normal(self, X):
    J = self.jacobian(X)
    normals = self.normdir * np.cross(J[:,0,:], J[:,1,:], axis=0)
    return normals / np.linalg.norm(normals, axis=0)

  def metric_tensor(self, X):
    J = self.jacobian(X)
    return np.array([[
      np.sum(J[:,0,:] * J[:,0,:], axis=0),
      np.sum(J[:,0,:] * J[:,1,:], axis=0)
    ], [
      np.sum(J[:,1,:] * J[:,0,:], axis=0),
      np.sum(J[:,1,:] * J[:,1,:], axis=0)
    ]])

  def area_element(self, X):
    g = np.rollaxis(self.metric_tensor(X), 2)
    return np.linalg.det(g)

  def shape_tensor(self, X):
    norm = self.normal(X)
    hess = self.hessian(X)
    return np.array([[
      np.sum(hess[:,0,0,:] * norm, axis=0),
      np.sum(hess[:,0,1,:] * norm, axis=0),
    ], [
      np.sum(hess[:,1,0,:] * norm, axis=0),
      np.sum(hess[:,1,1,:] * norm, axis=0),
    ]])

  def weingarten_map(self, X, roll=True):
    g = np.rollaxis(self.metric_tensor(X), 2)
    h = np.rollaxis(self.shape_tensor(X), 2)
    res = np.einsum("aio, abi -> abo", np.linalg.inv(g), h)
    if roll:
      return np.rollaxis(res, 0, 3)
    else:
      return res

  def gauss_curvature(self, X):
    h = self.weingarten_map(X, roll=False)
    return np.linalg.det(h)

  def mean_curvature(self, X):
    h = self.weingarten_map(X, roll=False)
    return np.trace(h, axis1=1, axis2=2)

  dA = area_element
  K = gauss_curvature
  H = mean_curvature
  first_fun_form = metric_tensor
  second_fun_form = shape_tensor

  def plot(self, show='area_element', grid=None, over=[], ax=None, cb=True, title=True):
    if grid is None:
      if len(over) == 3:
        ulims, vlims, res = over
      elif len(over) == 2:
        ulims, vlims = over
        res = 50
      else:
        assert(False)
      grid = np.meshgrid(np.linspace(*ulims, res), np.linspace(*vlims, res))

    U, V = grid
    X = np.vstack((U.ravel(), V.ravel()))
    F = self.f(X).reshape([3]+ list(U.shape))

    if ax is None:
      ax = plt.subplot(111, projection='3d')

    def colorplot(colors, cmap=mpl.cm.bwr):
      vlim = max(abs(colors.min()), colors.max())
      vlim = max(vlim, 1e-8)
      norm = mpl.colors.Normalize(vmin=-vlim, vmax=vlim)
      ax.plot_surface(*F,facecolors=cmap(norm(colors)))
      if cb:
        m = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        m.set_array([])
        plt.colorbar(m, label=show)

    if show in ['area_element', 'dA']:
      detg = self.area_element(X).reshape(list(U.shape))
      colorplot(detg)
    elif show == 'normal':
      norm = self.normal(X).reshape([3] + list(U.shape))
      ax.quiver(*F, *norm, color='black', length=0.33)
      ax.plot_wireframe(*F)
    elif show in ['gauss_curvature', 'K']:
      K = self.gauss_curvature(X).reshape(list(U.shape))
      colorplot(K)
    elif show in ['mean_curvature', 'H']:
      H = self.mean_curvature(X).reshape(list(U.shape))
      colorplot(H)
    else:
      ax.plot_wireframe(*F)

    plt.axis('equal')
    if title:
      plt.title(show or 'surface')
