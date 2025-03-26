import analyzer
import classifier
import regressor
import clustering
import pandas as pd


from utils.utils import load_JSON


# classification = True
# regression = True
# cluster = True


path = ""#"/Users/koa/Documents/ML Course/test dir"

#hyperparam_grid = load_JSON('/Users/koa/Documents/ML Course/capstone/bootcamp-ml-framework/hyperparams.json')['classification']
#print(hyperparam_grid)

def main():
    from sklearn.datasets import load_iris
    X, y = load_iris(as_frame=True, return_X_y=True)
    df = pd.concat([X, y], axis=1)

    A = analyzer.Analyzer(df=df, target_labels=["target"])

    for col in A.df.drop("target", axis=1).columns:
        A.set_col_dtype(col, float)

    A.scale()

    C = classifier.ClassificationModelOrganizer(
        df=A.get_frame(), 
        target_labels=A.target_labels, 
        file_directory=path
    )

    C.create_model("log_reg")   # models with no int hyper params work fine
    #C.create_model("KNN") 
    C.create_model('SVC') # models with int hyper params raise the confusing error.


    #C.create_model("ANN", input_shape=X.iloc[0].shape, n_classes=pd.unique(y).shape[0])

    C.fit_set()

    C.score_set()
    
    #print(C.x_test, C.y_test)
    print("Log Reg: ",C.models["log_reg"].metrics)
    #print("ANN: ", C.models["ANN"].metrics)

    #print(C.models["log_reg"].hyperparams)





    # A.plot_correlationMatrix(save_plot=True, include_target=True)
    # A.plot_boxPlot(save_plot=True)

    # A.plot_pairPlot(save_plot=True,include_target=True)


if __name__ == "__main__":
    main()
