import analyzer
import classifier
import regressor
import clustering
import pandas as pd

# classification = True
# regression = True
# cluster = True


from sklearn.datasets import load_iris
X, y = load_iris(as_frame=True,return_X_y=True)
df = pd.concat([X, y],axis=1)

def main():

    A = analyzer.Analyzer(df=df, target_labels=['target'])

    A.plot_correlationMatrix(save_plot=True, include_target=True)
    #A.plot_boxPlot(save_plot=True)

    #A.plot_pairPlot(save_plot=True,include_target=True)

    # output = {}
    # if classification == True:
    #     output['classification'] = classifier()
    # if regression == True:
    #     output['regression'] = regressor()
    # if cluster == True:
    #     output['clustering'] = clustering()

    
if __name__ == '__main__':
    main()