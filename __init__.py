#!/usr/bin/python
import utils
import data
import pandas as pd

def main():
    suffix_1 = input("Please input the name of the first omics matrix\n")
    suffix_2 = input("Please input the name of the second omics matrix\n")
    suffix = input("Please input the name of the correlation analysis\n")
    top_n = int(input("Please input the number of the top n compounds included in the correlation analysis\n"))
    threshold = float(input("Please input the threshold for network analysis(e.g. 0.4)\n"))
    print("\n#0 Preprocessing")
    print("Preprocessing the data......")
    omics_1, comp_1 = utils.fetch_scaled_data("input_data/omics_1.csv")
    omics_2, comp_2 = utils.fetch_scaled_data("input_data/omics_2.csv")
    sample_group = pd.read_csv("input_data/sample_group.csv")
    print("Preprocessing success.")
    print("\n#1 Volcano plot")
    data.volcano_plot("input_data/omics_1.csv", suffix_1)
    data.volcano_plot("input_data/omics_2.csv", suffix_2)
    print("\n#2 PCA analysis")
    data.pca_analysis(omics_1, sample_group, comp_1, suffix_1)
    data.pca_analysis(omics_2, sample_group, comp_2, suffix_2)
    print("\n#3 PLS analysis")
    data.pls_analysis(omics_1, sample_group, comp_1, suffix_1)
    data.pls_analysis(omics_2, sample_group, comp_2, suffix_2)
    print("\n#4 Spearman correlation matrix")
    data.spearman_analysis(omics_1, comp_1, omics_2, comp_2, suffix, suffix_1, suffix_2, top_n, threshold)


if __name__ == "__main__":
    main()
