# import pandas
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

col_names = ["respondent_id",
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

# distribution of the two target variables
# display in plt graph
fig, ax = plt.subplots(2, 1)
n_obs = df_all.shape[0]

(df_all['h1n1_vaccine'].value_counts().div(n_obs).plot.barh(title="Proportion of H1N1 Vaccine", ax=ax[0]))
ax[0].set_ylabel("h1n1_vaccine")

(df_all['seasonal_vaccine'].value_counts().div(n_obs).plot.barh(title="Proportion of Seasonal Vaccine", ax=ax[1]))
ax[1].set_ylabel("seasonal_vaccine")

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
                "hhs_geo_region",
                "census_msa",
                "household_adults",
                "household_children",
                "employment_industry",
                "employment_occupation"
                ]

# # fill df_all NAN value with a 0
# df_all.fillna(value="0", inplace=True)

# count of observations for each combination of  two variables
counts = (
    df_all[['h1n1_concern', 'h1n1_vaccine']].groupby(['h1n1_concern', 'h1n1_vaccine']).size().unstack('h1n1_vaccine'))
ax = counts.plot.barh()
ax.invert_yaxis()
ax.set_xlabel('number of h1n1_vaccine')
ax.legend(loc='best', title='h1n1_vaccine')
plt.show()

# rate of vaccination for each level of h1n1_concern
# stacked
h1n1_concern_counts = counts.sum(axis='columns')
props = counts.div(h1n1_concern_counts, axis='index')
ax = props.plot.barh()
ax.invert_yaxis()
ax.set_xlabel('rate of h1n1_vaccine')
ax.legend(loc='best', title='h1n1_vaccine')
plt.show()

# rate of vaccination for each level of h1n1_concern
# side by side
ax = props.plot.barh(stacked=True)
ax.invert_yaxis()
ax.set_xlabel('rate of h1n1_vaccine')
ax.legend(loc='best', title='h1n1_vaccine')
plt.show()


def vaccination_rate_plot(col, target, df_all, ax=None):
    # Stacked bar chart of vaccination rate for `target` against`col`.
    # col (string): column name of feature variable
    # target (string): column name of target variable
    # data (pandas DataFrame): dataframe that contains columns `col` and `target`
    # ax (matplotlib axes object, optional): matplotlib axes object to attach plot to

    feature_counts = (df_all[[target, col]].groupby([target, col]).size().unstack(target))
    group_counts = feature_counts.sum(axis='columns')
    prop_bar = feature_counts.div(group_counts, axis='index')

    prop_bar.plot(kind="barh", stacked=True, ax=ax)
    ax.invert_yaxis()
    ax.legend().remove()


cols_to_plot = [
    'opinion_h1n1_vacc_effective',
    'opinion_h1n1_risk',
    'opinion_seas_vacc_effective',
    'opinion_seas_risk',
]

fig, ax = plt.subplots(len(cols_to_plot), 2, figsize=(10, len(cols_to_plot) * 3))
for idx, col in enumerate(cols_to_plot):
    vaccination_rate_plot(col, 'h1n1_vaccine', df_all, ax=ax[idx, 0])
    vaccination_rate_plot(col, 'seasonal_vaccine', df_all, ax=ax[idx, 1])

ax[0, 0].legend(loc='best', title='h1n1_vaccine')
ax[0, 1].legend(loc='best', title='seasonal_vaccine')
fig.tight_layout()
plt.show()

# train model using logistic regression
# set a random seed for reproducibility
RANDOM_SEED = 6

# process numeric values
numeric_cols = df1.columns[df1.dtypes != "object"].values

# chain preprocessing into a Pipeline object
# each step is a tuple of (name you chose, sklearn transformer)
# Z-score scaling. This scales and shifts features so that they have zero mean and unit variance
# NA median imputation, which fills missing values with the median from the training data

numeric_preprocessing_steps = Pipeline(
    [('standard_scaler', StandardScaler()), ('simple_imputer', SimpleImputer(strategy='median'))])

# create the preprocessor stage of final pipeline
# each entry in the transformer list is a tuple of (name you choose, sklearn transformer, list of columns)
preprocessor = ColumnTransformer(transformers=
                                 [("numeric", numeric_preprocessing_steps, numeric_cols)], remainder="drop")

estimators = MultiOutputClassifier(estimator=LogisticRegression(penalty="l2", C=1))

full_pipeline = Pipeline([("preprocessor", preprocessor), ("estimators", estimators)])

# non-numeric value processing


# split data
X_train, X_eval, y_train, y_eval = train_test_split(
    df1,
    df2,
    test_size=0.33,
    shuffle=True,
    stratify=df2,
    random_state=RANDOM_SEED
)

# train model
full_pipeline.fit(X_train, y_train)

# predict on evaluation set
eval_predict = full_pipeline.predict_proba(X_eval)

train_predict = pd.DataFrame({"h1n1_vaccine": eval_predict[0][:, 1], "seasonal_vaccine": eval_predict[1][:, 1]},
                             index=y_eval.index)
print("y_predict:", train_predict)
train_predict.head()


# AUC - ROC curve is a performance measurement for the classification problems at various threshold settings
def plot_roc(y_true, y_score, label_name, ax):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
    ax.set_ylabel('TPR')
    ax.set_xlabel('FPR')
    ax.set_title(
        f"{label_name}: AUC = {roc_auc_score(y_true, y_score):.4f}"
    )


fig, ax = plt.subplots(1, 2, figsize=(7, 3.5))

plot_roc(y_eval['h1n1_vaccine'], train_predict['h1n1_vaccine'], 'h1n1_vaccine', ax=ax[0])
plot_roc(y_eval['seasonal_vaccine'], train_predict['seasonal_vaccine'], 'seasonal_vaccine', ax=ax[1])
fig.tight_layout()
plt.show()

# retain on full training set
full_pipeline.fit(df1, df2)

# test on test set
test_features_df = pd.read_csv("test_set_features.csv", index_col="respondent_id")
test_predict = full_pipeline.predict_proba(test_features_df)

y_test_predict = pd.DataFrame({"h1n1_vaccine": test_predict[0][:, 1], "seasonal_vaccine": test_predict[1][:, 1]},
                              index=test_features_df.index)
print("final:", y_test_predict)
