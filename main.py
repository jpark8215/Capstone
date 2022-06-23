import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import project

while True:

    print("1> View graphs")
    print("2> Training set performance")
    print("3> PREDICTION on test set")
    print("0> Exit Application\n")

    selected = input("Please enter an option above: ")

    if selected == '0':
        sys.exit()

    # option 1
    if selected == '1':
        print("Hello. Welcome to vaccine predictor! \n"
              "* The following three charts show vaccine status of population we gathered data from. \n"
              "* The first chart will show you the proportion of vaccine status on the training population. \n"
              "  0 means no vaccine received and 1 means vaccine received. \n"
              "* The second chart will show you the H1N1 vaccine status based on the concerns. \n"
              "  0 means no vaccine received or has no concerns. And 1 means vaccine received or has concerns. \n"
              "* The third chart will show the H1N1 and seasonal flu vaccine status based on opinion on vaccine. \n"
              "  0 means no vaccine received or negative opinion. And 1 means vaccine received or positive opinion. \n")

        # first chart
        project.vac_rate()
        # second chart
        project.vac_rate_concern()
        # third chart
        cols_to_plot = [
            'opinion_h1n1_vacc_effective',
            'opinion_h1n1_risk',
            'opinion_seas_vacc_effective',
            'opinion_seas_risk',
        ]

        fig, ax = plt.subplots(len(cols_to_plot), 2, figsize=(10, len(cols_to_plot) * 3))
        for idx, col in enumerate(cols_to_plot):
            project.vac_rate_op(col, 'h1n1_vaccine', project.df_all, ax=ax[idx, 0])
            project.vac_rate_op(col, 'seasonal_vaccine', project.df_all, ax=ax[idx, 1])

        ax[0, 0].legend(loc='best', title='vaccine status')
        # ax[0, 0].legend(loc='best', title='H1N1 vaccine')
        # ax[0, 1].legend(loc='best', title='Seasonal vaccine')
        fig.tight_layout()
        plt.show()

    # option 2
    if selected == '2':
        print("* The chart will show you AUC-ROC curve for H1N1 and seasonal flu vaccine prediction on training set. \n"
              "AUC - ROC curve is a classification problem performance measurement. \n"
              "AOC is the area under the ROC curve, which is probability \n"
              "that a randomly chosen positive case is ranked higher than a randomly chosen negative case. \n"
              "ROC is calculated by the true positive rate (TPR) against the false positive rate (FPR).\n"
              "* The accuracy score on training set is as shown below. \n")
        # accuracy score of training set
        project.full_pipeline.fit(project.df1, project.df2)

        org_data = project.df2
        eval_data = project.full_pipeline.predict_proba(project.df1)

        predict_data = pd.DataFrame({"h1n1_vaccine": eval_data[0][:, 1], "seasonal_vaccine": eval_data[1][:, 1]},
                                    index=project.df2.index)
        predict_data["h1n1_vaccine"] = predict_data["h1n1_vaccine"].round(0)
        predict_data["seasonal_vaccine"] = predict_data["seasonal_vaccine"].round(0)

        print("Accuracy Score:", (accuracy_score(org_data, predict_data) * 100).__round__(2), "%\n")
        print("H1N1 Vaccine Accuracy Score:",
              (accuracy_score(org_data["h1n1_vaccine"], predict_data["h1n1_vaccine"]) * 100).__round__(2), "%")
        print("Seasonal Vaccine Accuracy Score:",
              (accuracy_score(org_data["seasonal_vaccine"], predict_data["seasonal_vaccine"]) * 100).__round__(2),
              "%\n")

        # auc-roc curve chart for training set
        fig, ax = plt.subplots(1, 2, figsize=(6, 3))

        project.plot_roc(project.y_eval['h1n1_vaccine'], project.eval_predict_set['h1n1_vaccine'],
                         'h1n1_vaccine', ax=ax[0])
        project.plot_roc(project.y_eval['seasonal_vaccine'], project.eval_predict_set['seasonal_vaccine'],
                         'seasonal_vaccine', ax=ax[1])
        fig.tight_layout()
        plt.show()

    # option 3
    if selected == '3':
        # test set prediction
        print(project.test_predict_set)

        print("* The chart shows you the vaccination status proportion of the test set population. \n "
              "  0 means no vaccine received and 1 means vaccine received. \n")
        project.test_set()

        project.search_participant(
            int(input("Please enter the participant's id you would like to look up between 26707 and 53414: ")))
