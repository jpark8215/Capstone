# import
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# columns in features
cols = ["respondent_id",
        "h1n1_concern",
        "h1n1_knowledge",
        "behavioral_antiviral_meds",
        "behavioral_avoidance",
        "behavioral_face_mask",
        "behavioral_wash_hands",
        "behavioral_large_gatherings",
        "behavioral_outside_home",
        "behavioral_touch_face",
        "doctor_recc_h1n1",
        "doctor_recc_seasonal",
        "chronic_med_condition",
        "child_under_6_months",
        "health_worker",
        "health_insurance",
        "opinion_h1n1_vacc_effective",
        "opinion_h1n1_risk",
        "opinion_h1n1_sick_from_vacc",
        "opinion_seas_vacc_effective",
        "opinion_seas_risk",
        "opinion_seas_sick_from_vacc",
        "age_group",
        "education",
        "race",
        "sex",
        "income_poverty",
        "marital_status",
        "rent_or_own",
        "employment_status",
        "hhs_geo_region",
        "census_msa",
        "household_adults",
        "household_children",
        "employment_industry",
        "employment_occupation",
        "h1n1_vaccine",
        "seasonal_vaccine"]

# load dataset from training set files
df1 = pd.read_csv('training_set_features.csv', index_col="respondent_id")
df2 = pd.read_csv('training_set_labels.csv', index_col="respondent_id")

# merge training set files on respondent id
df_all = df1.merge(df2, how="left", on="respondent_id")


# distribution of h1n1 vaccine and seasonal vaccine status
# display in horizontal graphs
def vac_rate():
    fig, ax = plt.subplots(2, 1)
    n_obs = df_all.shape[0]

    (df_all['h1n1_vaccine'].value_counts().div(n_obs).plot.barh(title="Proportion of H1N1 Vaccine", ax=ax[0]))
    ax[0].set_ylabel("H1N1 Vaccine")
    (df_all['seasonal_vaccine'].value_counts().div(n_obs).plot.barh(title="Proportion of Seasonal Vaccine", ax=ax[1]))
    ax[1].set_ylabel("Seasonal Vaccine")
    fig.tight_layout()
    plt.show()


# split merged dataset in features and target variable
feature_cols = ["h1n1_concern",
                "h1n1_knowledge",
                "behavioral_antiviral_meds",
                "behavioral_avoidance",
                "behavioral_face_mask",
                "behavioral_wash_hands",
                "behavioral_large_gatherings",
                "behavioral_outside_home",
                "behavioral_touch_face",
                "doctor_recc_h1n1",
                "doctor_recc_seasonal",
                "chronic_med_condition",
                "child_under_6_months",
                "health_worker",
                "health_insurance",
                "opinion_h1n1_vacc_effective",
                "opinion_h1n1_risk",
                "opinion_h1n1_sick_from_vacc",
                "opinion_seas_vacc_effective",
                "opinion_seas_risk",
                "opinion_seas_sick_from_vacc",
                "age_group",
                "education",
                "race",
                "sex",
                "income_poverty",
                "marital_status",
                "rent_or_own",
                "employment_status",
                # "hhs_geo_region",
                "census_msa",
                "household_adults",
                "household_children",
                # "employment_industry",
                # "employment_occupation"
                ]


# vaccine rate base on disease concerns
# display in horizontal graphs
def vac_rate_concern():
    # count of observations for each combination of two variables
    h1n1_counts = (df_all[['h1n1_concern', 'h1n1_vaccine']].groupby(['h1n1_concern', 'h1n1_vaccine']).size().unstack(
        'h1n1_vaccine'))

    # rate of vaccination for each level of h1n1_concern
    h1n1_concern_counts = h1n1_counts.sum(axis='columns')
    props = h1n1_counts.div(h1n1_concern_counts, axis='index')

    # rate of vaccination for each level of h1n1_concern
    ax = props.plot.barh(title="Rate of H1N1 Vaccine")
    ax.invert_yaxis()
    ax.set_xlabel('H1N1 vaccine rate')
    ax.legend(loc='best', title='H1N1 vaccine')
    plt.show()


# vaccine rate base on opinions
# display in horizontal graphs
def vac_rate_op(col, target, df_all, ax=None):
    # col: column name of feature variable
    # target: column name of target variable
    # df_all: dataframe that contains columns `col` and `target`

    feature_counts = (df_all[[target, col]].groupby([target, col]).size().unstack(target))
    group_counts = feature_counts.sum(axis='columns')
    prop_bar = feature_counts.div(group_counts, axis='index')

    prop_bar.plot(kind="barh", stacked=True, ax=ax)
    ax.invert_yaxis()
    ax.legend().remove()


# train model using logistic regression
# set a random seed for reproducibility
RANDOM_SEED = 15

# replace non-numeric object values into numeric values
df1['age_group'].replace(['18 - 34 Years', '35 - 44 Years', '45 - 54 Years', '55 - 64 Years', '65+ Years'],
                         [0, 1, 2, 3, 4], inplace=True)
df1['education'].replace(['< 12 Years', '12 Years', 'Some College', 'College Graduate'], [0, 1, 2, 3], inplace=True)
df1['race'].replace(['White', 'Black', 'Hispanic', 'Other or Multiple'], [0, 1, 2, 3], inplace=True)
df1['sex'].replace(['Male', 'Female'], [0, 1], inplace=True)
df1['income_poverty'].replace(['Below Poverty', '<= $75,000, Above Poverty', '> $75,000'], [0, 1, 2], inplace=True)
df1['marital_status'].replace(['Not Married', 'Married'], [0, 1], inplace=True)
df1['rent_or_own'].replace(['Rent', 'Own'], [0, 1], inplace=True)
df1['employment_status'].replace(['Not in Labor Force', 'Unemployed', 'Employed'], [0, 1, 2], inplace=True)
df1['census_msa'].replace(['Non-MSA', 'MSA, Not Principle  City', 'MSA, Principle City'], [0, 1, 2], inplace=True)


# process numeric values
numeric_cols = df1.columns[df1.dtypes != "object"].values

# chain preprocessing into a pipeline object into a tuple of (numeric value, sklearn transformer)
# Z-score scaling scales and shifts features so that they have zero mean and unit variance
# NA mean imputation fills missing values with the mean value of the training set
data_process = Pipeline([('Z-score scaling', StandardScaler()), ('simple_imputer', SimpleImputer(strategy='mean'))])
transformer = ColumnTransformer(transformers=[("numeric", data_process, numeric_cols)], remainder="drop")
estimators = MultiOutputClassifier(estimator=LogisticRegression(penalty="l2", C=1))

full_pipeline = Pipeline([("preprocessor", transformer), ("estimators", estimators)])


# split data
X_train, X_eval, y_train, y_eval = train_test_split(df1, df2, test_size=0.25, shuffle=True, stratify=df2,
                                                    random_state=RANDOM_SEED)

# train model with split data
full_pipeline.fit(X_train, y_train)

# predict on evaluation set
eval_predict = full_pipeline.predict_proba(X_eval)
eval_predict_set = pd.DataFrame({"h1n1_vaccine": eval_predict[0][:, 1], "seasonal_vaccine": eval_predict[1][:, 1]},
                                index=y_eval.index)


sns.pairplot(data=df1, diag_kind='kde')
plt.show()


# AUC - ROC curve
def plot_roc(y_true, y_score, label, ax):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], color='black', linestyle='--')
    ax.set_ylabel('TPR')
    ax.set_xlabel('FPR')
    ax.set_title(f"{label} AUC = {roc_auc_score(y_true, y_score):.2f}")


# retain model with full training set
full_pipeline.fit(df1, df2)

# load data from test set
test_features_df = pd.read_csv("test_set_features.csv", index_col="respondent_id")

# replace non numeric values for test set
test_features_df['age_group'].replace(['18 - 34 Years', '35 - 44 Years', '45 - 54 Years', '55 - 64 Years', '65+ Years'],
                                      [0, 1, 2, 3, 4], inplace=True)
test_features_df['education'].replace(['< 12 Years', '12 Years', 'Some College', 'College Graduate'], [0, 1, 2, 3],
                                      inplace=True)
test_features_df['race'].replace(['White', 'Black', 'Hispanic', 'Other or Multiple'], [0, 1, 2, 3], inplace=True)
test_features_df['sex'].replace(['Male', 'Female'], [0, 1], inplace=True)
test_features_df['income_poverty'].replace(['Below Poverty', '<= $75,000, Above Poverty', '> $75,000'], [0, 1, 2],
                                           inplace=True)
test_features_df['marital_status'].replace(['Not Married', 'Married'], [0, 1], inplace=True)
test_features_df['rent_or_own'].replace(['Rent', 'Own'], [0, 1], inplace=True)
test_features_df['employment_status'].replace(['Not in Labor Force', 'Unemployed', 'Employed'], [0, 1, 2], inplace=True)
test_features_df['census_msa'].replace(['Non-MSA', 'MSA, Not Principle  City', 'MSA, Principle City'], [0, 1, 2],
                                       inplace=True)

# predict on test set
test_predict = full_pipeline.predict_proba(test_features_df)
# set dataframe to round the prediction
test_predict_set = pd.DataFrame({"h1n1_vaccine": test_predict[0][:, 1], "seasonal_vaccine": test_predict[1][:, 1]},
                                index=test_features_df.index)
test_predict_set["h1n1_vaccine"] = test_predict_set["h1n1_vaccine"].round(0)
test_predict_set["seasonal_vaccine"] = test_predict_set["seasonal_vaccine"].round(0)


# individual search
def search_participant(index):
    result = test_predict_set.loc[[index]]

    print(result, "\n", "1.0 = WILL Likely VACCINATE : 0.0 = WILL Likely NOT Vaccinate", "\n")


def test_set():
    fig, ax = plt.subplots(1, 2)
    n_obs = test_predict_set.shape[0]

    (test_predict_set['h1n1_vaccine'].value_counts().div(n_obs).plot(kind='pie', title="Expected H1N1 Vaccine rate",
                                                                     ax=ax[0]))
    ax[0].legend(loc='best')

    (test_predict_set['seasonal_vaccine'].value_counts().div(n_obs).plot(kind='pie',
                                                                         title="Expected Seasonal Vaccine rate",
                                                                         ax=ax[1]))
    ax[1].legend(loc='best')
    fig.tight_layout()
    plt.show()
