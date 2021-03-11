import pandas as pd
import numpy as np
import networkx as nx 
from utils import confidence_ellipse
from utils import vip
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from scipy.stats import spearmanr
from matplotlib import pyplot as plt

def pca_analysis(omics_matrix, sample_group, variables, suffix):
    '''
    Perform PCA analysis on the omics_matrix and generate image

    Parameters
    ----------
    omics_matrix: numpy.ndarray
        scaled omics matrix
    sample_group: numpy.ndarray
        sample group data
    variables: numpy.ndarray
        variable information
    suffix: string
        suffix of the output
    
    Returns
    -------
    None
    '''
    # PCA analysis
    print("Preparing for PCA analysis for {}......".format(suffix))
    pca = PCA()
    omics_pca = pca.fit_transform(omics_matrix)
    explained_ratio = pca.explained_variance_ratio_
    scatter = omics_pca[:,:2]

    # Process the group data
    groups = sample_group.iloc[:,1]
    group = groups.unique()
    group_scatter = []
    for g in group:
        m = groups[groups==g].index.tolist()
        group_scatter.append((g, scatter[m[0]:m[-1]+1, :]))
    
    # Export PCA loadings
    print("Exporting PCA loading for {}......".format(suffix))
    loadings = pca.components_.T[:,:2]
    loadings = pd.DataFrame(loadings, index=variables, columns=["PC1", "PC2"])
    loadings.to_csv("output_params/{}_pca_loadings.csv".format(suffix))
    print("PCA loading exported for {}.".format(suffix))
    
    # Visualize scatter
    print("Generating PCA plot for {}......".format(suffix))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    color = ["dodgerblue","orange","red","yellow","black"]
    re_color = color[:len(group)]
    for i, (g, m) in enumerate(group_scatter):
        ax.scatter(m[:,0], m[:,1], label=g, color=re_color[i])
        confidence_ellipse(m[:, 0], m[:, 1], ax, edgecolor=re_color[i])
    ax.legend()
    ax.set_xlabel("PC1({:.2f}%)".format(explained_ratio[0]*100))
    ax.set_ylabel("PC2({:.2f}%)".format(explained_ratio[1]*100))
    ax.set_title("PCA plot of {}".format(suffix))
    fig.savefig("output_figures/{}_pca.png".format(suffix), dpi=300)
    print("PCA plot generated for {}.".format(suffix))

def volcano_plot(path, suffix):
    '''
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    path:
        path of the omics data
    
    Returns
    -------
    None
    '''
    print("Preparing for generating the volcano plot for {}......".format(suffix))
    omics = pd.read_csv(path)
    omics.iloc[:,-2] = np.log2(omics.iloc[:, -2])
    omics.iloc[:,-1] = -np.log10(omics.iloc[:,-1])
    up = omics[(omics.iloc[:,-2]>1) & (omics.iloc[:,-1]>2)]
    down = omics[(omics.iloc[:, -2]<-1) & (omics.iloc[:, -1]>2)]
    ns = omics[(omics.iloc[:, -2]<1) & (omics.iloc[:, -2]>-1) | (omics.iloc[:, -1] < 2)]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(ns.iloc[:,-2], ns.iloc[:,-1], color="grey")
    ax.scatter(up.iloc[:,-2], up.iloc[:,-1], color="red")
    ax.scatter(down.iloc[:,-2], down.iloc[:,-1], color="blue")
    ax.axhline(2, linestyle="--", color="gray")
    ax.axvline(1, linestyle="--", color="gray")
    ax.axvline(-1, linestyle="--", color="gray")
    ax.set_title("Volcano plot of {}".format(suffix))
    ax.set_xlabel("log2FC")
    ax.set_ylabel("-log10P")
    fig.savefig("output_figures/{}_volcano.png".format(suffix),dpi=300)
    print("Volcano plot generated for {}.".format(suffix))

def pls_analysis(omics_matrix, sample_group, variables, suffix):
    '''
    Perform PLS regression on the omics_matrix and generate image

    Parameters
    ----------
    omics_matrix: numpy.ndarray
        scaled omics matrix
    sample_group: numpy.ndarray
        sample group data
    variables: numpy.ndarray
        variable information
    suffix: string
        suffix of the output
    
    Returns
    -------
    None
    '''
    # PLS analysis
    if len(sample_group.iloc[:,1].unique()) != 2:
        raise ValueError("To use PLS, the group number must be 2.")
    groups = sample_group.iloc[:,1]
    group = groups.unique()
    groups_binary = []
    for i in groups:
        if i == group[0]:
            groups_binary.append(0)
        else:
            groups_binary.append(1)
    print("Preparing for PLS analysis for {}......".format(suffix))
    pls = PLSRegression(n_components=2)
    omics_pls = pls.fit(omics_matrix,groups_binary).transform(omics_matrix)
    scatter = omics_pls

    # Process group data
    group_scatter = []
    for g in group:
        m = groups[groups==g].index.tolist()
        group_scatter.append((g, scatter[m[0]:m[-1]+1, :]))
    
    # Export PCA loadings
    print("Exporting PLS VIP for {}......".format(suffix))
    vips = vip(omics_matrix, groups_binary, pls)
    loadings = pd.DataFrame(vips, index=variables, columns=["VIP score"])
    loadings.to_csv("output_params/{}_pls_vips.csv".format(suffix))
    print("PLS VIP exported for {}.".format(suffix))
    
    # Visualize scatter
    print("Generating PLS-DA plot for {}......".format(suffix))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    color = ["dodgerblue","orange"]
    re_color = color[:len(group)]
    for i, (g, m) in enumerate(group_scatter):
        ax.scatter(m[:,0], m[:,1], label=g, color=re_color[i])
        confidence_ellipse(m[:, 0], m[:, 1], ax, edgecolor=re_color[i])
    ax.legend()
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PLS plot of {}".format(suffix))
    fig.savefig("output_figures/{}_pls.png".format(suffix), dpi=300)
    print("PLS plot generated for {}.".format(suffix))

def spearman_analysis(omics_1, variable_1, omics_2, variable_2, suffix, suffix_1, suffix_2, top_n, network_threshold):
    '''
    Perform spearman analysis for the two omics matrices.

    Parameters
    ----------
    omics_1: numpy.ndarray
        the matrix of omics 1
    variable_1: numpy.ndarray
        the compounds in omics 1
    omics_2: numpy.ndarray
        the matrix of omics 2
    variable_2: numpy.ndarray
        the compounds in omics 2
    suffix: string
        the suffix of the output data
    suffix_1: string
        the suffix of the omics_1 data
    suffix_2: string
        the suffix of the omics_2 data
    top_n: interger
        the top_n compounds included in the analysis
    network_threshold: float
        threshold of the network analysis
    
    Returns
    -------
    None
    '''
    print("Preparing for spearman analysis......")
    shape_1 = min(omics_1.shape[1], top_n)
    shape_2 = min(omics_2.shape[1], top_n)
    omics_1 = omics_1[:,:shape_1]
    omics_2 = omics_2[:,:shape_2]
    om_hstack = np.hstack((omics_1, omics_2))
    # Generate spearman matrix
    spearman_raw = spearmanr(om_hstack).correlation
    spearman_matrix = spearman_raw[-shape_2:, :shape_1]
    print("Generating spearman matrix......")
    spearman_df = pd.DataFrame(spearman_matrix, columns=variable_1[:shape_1], index=variable_2[:shape_2])
    spearman_df.to_csv("output_params/spearman_{}.csv".format(suffix))
    print("Spearman matrix exported.")

    # Generate heatmap
    plt.clf()
    plt.xticks(ticks=np.arange(shape_1), labels=variable_1[:shape_1],rotation=45)
    plt.yticks(ticks=np.arange(shape_2), labels=variable_2[:shape_2])
    plt.imshow(spearman_matrix, cmap="cool",interpolation="nearest", vmin=-1, vmax=1)
    plt.colorbar(cax=None, ax=None, shrink=0.5)
    plt.xlabel(suffix_1)
    plt.ylabel(suffix_2)
    plt.savefig("output_figures/spearman_{}.png".format(suffix), dpi=300)

    # Generate network
    print("Preparing network analysis......")
    spearman_raw = pd.DataFrame(om_hstack).corr()
    spearman_raw.columns = list(variable_1)[:shape_1] + list(variable_2)[:shape_2]
    spearman_raw.index = variable_1[:shape_1]+variable_2[:shape_2]
    lst_symbols = spearman_raw.columns.to_list()
    lst_from = []
    lst_to = []
    lst_corr_coeff = []
    for sym_from in lst_symbols:
        for sym_to in lst_symbols:
            if sym_from != sym_to:
                corr_coef = spearman_raw.loc[sym_from, sym_to]
                if corr_coef > network_threshold:
                    lst_from.append(sym_from)
                    lst_to.append(sym_to)
                    lst_corr_coeff.append(corr_coef)
    df_net = pd.DataFrame({"from":lst_from, "to":lst_to, "corr_coeff": lst_corr_coeff})
    plt.clf()
    G = nx.from_pandas_edgelist(df_net, "from", "to")
    nx.draw_spring(G, with_labels=True, node_color="cornflowerblue", node_size=600, edge_color="black",linewidths=1, font_size=10, width=1, alpha=0.5)
    plt.savefig("output_figures/network_{}.png".format(suffix), dpi=300)
    plt.clf()
    nx.draw_circular(G, with_labels=True, node_color="cornflowerblue", node_size=600, edge_color="black", linewidths=1,font_size=10, width=1, alpha=0.5)
    plt.savefig("output_figures/network_{}_circular.png".format(suffix),dpi=300)
    print("Network exported.")

    
