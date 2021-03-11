import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import transforms as transforms
from sklearn.preprocessing import StandardScaler

def fetch_scaled_data(path):
    '''
    Fetch the data and perform standard scaling
    
    Parameters
    ----------
    path: string
        The path of the data
    
    Returns
    -------
    omics_T: np.ndarray
        Scaled omics data for further analysis
    omics.index:np.ndarray
        Variables information
    '''
    omics = pd.read_csv(path)
    omics.index = omics.iloc[:,0]
    omics = omics.iloc[:,1:-2]
    omics_T = omics.T
    scaler = StandardScaler()
    scaler.fit(omics_T)
    omics_T = scaler.transform(omics_T)
    return omics_T, list(omics.index)

def confidence_ellipse(x, y, ax, n_std = 3.0, **kwargs):
    '''
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    '''
    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width = ell_radius_x * 2, height=ell_radius_y*2, facecolor='none', **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def vip(x, y, model):
    '''
    Calculation of the VIP score of the PLS model

    Parameters
    ----------
    x: numpy.ndarray
        omics matrics of the project
    y: numpy.ndarray
        sample group data
    model: sklearn.cross_decomposition.PLSRegression
        PLS model

    Returns
    -------
    VIP value: float
    '''
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_

    m, p = x.shape
    _, h = t.shape
    
    vips = np.zeros((p, ))
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)

    for i in range(p):
        weight = np.array([(w[i,j]/np.linalg.norm(w[:,j]))**2 for j in range(h)])
        vips[i] = np.sqrt(p*(s.T @ weight)/total_s)
    return vips