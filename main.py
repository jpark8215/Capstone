
import sys
import project
import matplotlib.pyplot as plt

while True:

    print("1> View graphics")
    print("2> Prediction")
    print("0> Exit Application\n")

    selected = input("Please enter an option above: ")

    if selected == '0':
        sys.exit()

    #
    if selected == '1':

        print("Hello. Welcome to vaccine predictor! \n "
              "The following three charts show vaccine status of population we gathered data from. \n"
              "The first chart will show you the proportion of vaccine status based on the total population. \n"
              "0 means no vaccine received and 1 means vaccine received. \n"
              "The second chart will show you the H1N1 and seasonal flu vaccine status based on their concerns. \n"
              "0 means no vaccine received or has no concerns. And 1 means vaccine received or has concerns. \n"
              "The third chart will show the H1N1 and seasonal flu vaccine status based on opinion on vaccine. \n"
              "0 means no vaccine received or negative opinion. And 1 means vaccine received or positive opinion")
        project.vac_rate()
        project.vac_rate_concern()
        # project.vac_rate_op(col, target, df_all, ax=None)
        plt.show()

    #
    if selected == '2':
        # project.plot_roc()
        print(project.test_predict_set)
