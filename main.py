# import pandas
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

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

# load dataset
df1 = pd.read_csv('training_set_features.csv')
df2 = pd.read_csv('training_set_labels.csv')

# merge on respondent id
df_all = df1.merge(df2, how="left", on="respondent_id")
print(df_all)


# distribution of the two target variables
# plt graph
fig, ax = plt.subplots(2, 1)
n_obs = df_all.shape[0]

(df_all['h1n1_vaccine'].value_counts().div(n_obs).plot.barh(title="Proportion of H1N1 Vaccine", ax=ax[0]))
ax[0].set_ylabel("h1n1_vaccine")

(df_all['seasonal_vaccine'].value_counts().div(n_obs).plot.barh(title="Proportion of Seasonal Vaccine", ax=ax[1]))
ax[1].set_ylabel("seasonal_vaccine")

fig.tight_layout()
plt.show()


# split dataset in features and target variable
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
                "employment_occupation"]

X = df_all[feature_cols]  # Features
y1 = df_all.h1n1_vaccine  # Target variable
y2 = df_all.seasonal_vaccine

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X, y1)
logreg.fit(X, y2)

#
y_pred = logreg.predict(X)
print(y_pred)
