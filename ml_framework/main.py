import analyzer
import classifier
import regressor
import clustering
from utils.utils import load_JSON, read_to_frame

import pandas as pd


path = "/Users/koa/Documents/ML Course/test dir/"


# Global Parameter
analysis = True
classification = False
regression = False
cluster = False

n_trials = 100
seed = 0

save_files = True


# To demo or test on a smaller subset of the full dataset
demo = False
demo_rows = 5000
demo_trials = 20




classification_hyperparam_grid = load_JSON(
    "/Users/koa/Documents/ML Course/capstone/bootcamp-ml-framework/hyperparams.json"
)["classification"]

regression_hyperparam_grid = load_JSON(
    "/Users/koa/Documents/ML Course/capstone/bootcamp-ml-framework/hyperparams.json"
)["regression"]

clustering_hyperparam_grid = load_JSON(
    "/Users/koa/Documents/ML Course/capstone/bootcamp-ml-framework/hyperparams.json"
)["clustering"]



def main():
    global n_trials
    global save_files
    df = read_to_frame(path + "diamonds.csv")


    if demo is True:
        df = df.sample(frac=1, random_state=seed).iloc[:demo_rows, :]
        n_trials = demo_trials


    if analysis is True:

        A = analyzer.Analyzer(df=df, seed=seed, dir_path=path+"analysis/")

        A.drop_columns(cols_to_drop=["Unnamed: 0"])
        A.set_col_dtype("color", "category"), A.set_col_dtype("clarity", "category"), A.set_col_dtype("cut", "category"), 
        A.plot_histograms_categorical(save_plot=save_files)
        A.plot_histograms_numeric(save_plot=save_files)
        A.encode_features(encoder="ord", cols_to_encode=["color"], categories=[["J", "I", "H", "G", "F", "E", "D"]])
        A.encode_features(encoder="ord", cols_to_encode=["clarity"], categories=[["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]])
        A.encode_target(encoder="ord", categories=[["Fair", "Good", "Very Good", "Premium", "Ideal"]])
        A.scale(scale_target=False)

        A.plot_correlationMatrix(save_plot=save_files, include_target=save_files)
        A.plot_boxPlot(save_plot=save_files)



    if classification is True:

        A = analyzer.Analyzer(df=df, target_labels=["cut"], seed=seed, dir_path=path)

        A.drop_columns(cols_to_drop=["Unnamed: 0"])
        A.encode_features(encoder="ord", cols_to_encode=["color"], categories=[["J", "I", "H", "G", "F", "E", "D"]])
        A.encode_features(encoder="ord", cols_to_encode=["clarity"], categories=[["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]])
        A.encode_target(encoder="ord", categories=[["Fair", "Good", "Very Good", "Premium", "Ideal"]])
        A.scale(scale_target=False)

        #print("Corr to target: \n", A.calc_corr_to_target())
        #print("Corr to features: \n", A.calc_corr_matrix(include_target=False).mean())
        A.cut_features_by_corr(t_corr_threshold=0.05)
        #print(A.get_frame())

        C = classifier.ClassificationModelOrganizer(
            df=A.get_frame(),
            target_labels=A.target_labels,
            file_directory=path + "classification/",
            hyperparam_grid=classification_hyperparam_grid,
            seed=seed,
        )

        C.create_model("log_reg")
        C.create_model("KNN")
        C.create_model("SVC")
        C.create_model("decision_tree")
        C.create_model("XG_boost")
        C.create_model("random_forest")
        C.create_model("ANN", input_shape=C.x.iloc[0].shape, n_classes=pd.unique(C.y).shape[0])

        C.fit_set(n_trials=n_trials, keras_tuner_search_method="hyperband")

        C.score_set()
        print(C.compare_models(plot=True, save_plot=save_files, save_dir=C.file_directory))
        if save_files is True:
            C.save_models(dir_path=C.file_directory)

    if regression is True:
        
        A = analyzer.Analyzer(df=df, target_labels=["price"], seed=seed, dir_path=path)

        A.drop_columns(cols_to_drop=["Unnamed: 0"])
        A.encode_features(
            encoder="ord",
            cols_to_encode=["color"],
            categories=[["J", "I", "H", "G", "F", "E", "D"]],
        )
        A.encode_features(
            encoder="ord",
            cols_to_encode=["clarity"],
            categories=[["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]],
        )
        A.encode_features(
            encoder="ord",
            cols_to_encode=['cut'],
            categories=[["Fair", "Good", "Very Good", "Premium", "Ideal"]],
        )
        A.scale(scale_target=False)


        A.cut_features_by_corr(t_corr_threshold=0.05)

        print("print test")
        print(A.get_frame().head())

        R = regressor.RegressionModelOrganizer(
            df=A.get_frame(),
            target_labels=A.target_labels,
            file_directory=path + "regression/",
            hyperparam_grid=regression_hyperparam_grid,
            seed=seed,
        )

        R.create_model("lin_reg")
        R.create_model("KNN")
        R.create_model("SVR")
        R.create_model("decision_tree")
        R.create_model("XG_boost")
        R.create_model("random_forest")
        R.create_model("ANN", input_shape=R.x.iloc[0].shape)
        
        R.fit_set(n_trials=n_trials, keras_tuner_search_method="hyperband")

        # R.load_model(R.file_directory+"lin_reg.pkl",'lin_reg')
        # R.load_model(R.file_directory+"KNN.pkl",'KNN')
        # R.load_model(R.file_directory+"SVR.pkl",'SVR')
        # R.load_model(R.file_directory+"decision_tree.pkl",'decision_tree')      
        # R.load_model(R.file_directory+"random_forest.pkl",'random_forest')            
        # R.load_model(R.file_directory+"XG_boost.pkl",'XG_boost')        

        R.score_set()
        print(R.compare_models(plot=True, save_plot=save_files, save_dir=R.file_directory))
        if save_files is True:
            R.save_models(dir_path=R.file_directory)




    if cluster is True:
        A = analyzer.Analyzer(df=df, target_labels=[], seed=seed, dir_path=path)

        A.drop_columns(cols_to_drop=["Unnamed: 0"])
        A.encode_features(
            encoder="ord",
            cols_to_encode=["color"],
            categories=[["J", "I", "H", "G", "F", "E", "D"]],
        )
        A.encode_features(
            encoder="ord",
            cols_to_encode=["clarity"],
            categories=[["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]],
        )
        A.encode_features(
            encoder="ord",
            cols_to_encode=['cut'],
            categories=[["Fair", "Good", "Very Good", "Premium", "Ideal"]],
        )
        A.scale()

        Clu = clustering.ClusteringModelOrganizer(
            df=A.get_frame(),
            file_directory=path+"clustering/",
            hyperparam_grid = clustering_hyperparam_grid,
            seed = seed,
        )

        Clu.create_model('k_means')
        Clu.create_model('agglomerative')
        #Clu.create_model('mean_shift')
        #Clu.create_model('dbscan')

        Clu.fit_set(param_tuning=False,
                    hyper_params={
                        "k_means": {'n_clusters': 2, 'init': 'random', 'tol': 0.010924873680680089},
                        "agglomerative": {'n_clusters': 2, 'metric': 'euclidean', 'linkage': 'ward'},
                    })

        Clu.score_set()
        print(Clu.compare_models(plot=True, save_plot=save_files, save_dir=Clu.file_directory))
        if save_files is True:
            Clu.save_models(dir_path=Clu.file_directory)
        Clu.visualize_clustering(dimension_reduction_method='PCA', save_plot=save_files, save_dir=Clu.file_directory, save_name="clustering_visualization_pca")
        Clu.visualize_clustering(dimension_reduction_method='TSNE', save_plot=save_files, save_dir=Clu.file_directory, save_name="clustering_visualization_tsne")


if __name__ == "__main__":
    main()



