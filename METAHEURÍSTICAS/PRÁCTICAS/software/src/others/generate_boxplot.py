import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def main():
    results_folder = "./results/"
    fnames:list[str] = [f for f in os.listdir(results_folder) if ".csv" in f]
    for fname in fnames:
        df = pd.read_csv(results_folder + fname)
        p = sns.catplot(data=df, x="alg", y="fitness", aspect=2.2, height=5, kind="box")
        p.set(xlabel="Algoritmo", ylabel="Fitness")
        p.set_xticklabels(rotation=45, ha="right", fontsize=7)
        dataset_name_clean = fname.replace("_results.csv", "").replace(".csv", "")
        p.figure.suptitle(f"Dataset: {dataset_name_clean}", y=1.02, fontsize=16)
        p.figure.subplots_adjust(bottom=0.22)
        foutput = results_folder + fname.replace(".csv", ".png")
        p.savefig(foutput)
        plt.close(p.figure)

        print(f"\nResumen para {fname}:")
        resumen = df.groupby('alg')['fitness'].agg(['max', 'mean', 'std'])
        print(resumen)
    
if __name__ == '__main__':
    main()
