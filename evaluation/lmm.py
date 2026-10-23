import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# ------------------------------------------------------------
# 1. Daten vorbereiten
# ------------------------------------------------------------

# CSV laden
df = pd.read_csv("../results/artifacts_total.csv")
print(df.shape)


# Kategoriale Variablen explizit als category setzen
categorical_cols = ["field", "family", "hardware", "index", "temperature"]

for col in categorical_cols:
    df[col] = df[col].astype("category")

# Fehlende Werte in modellrelevanten Variablen entfernen
required_cols = [
    "times",
    "tokens_out",
    "field",
    "family",
    "active_parameters",
    "temperature",
    "hardware",
    "index",
    "run_uuid",
    "dataset_model"
]

df_model = df.dropna(subset=required_cols).copy()

# Optional: Running und Output-Tokens log-transformieren und standardisieren.
# Das ist häufig sinnvoll, da beides oft stark rechtsschief ist.
USE_OUTPUT_TOKENS = True

if USE_OUTPUT_TOKENS:
    df_model = df_model.dropna(subset=["tokens_out"]).copy()
    df_model["log_output_tokens"] = np.log1p(df_model["tokens_out"])
    df_model["log_output_tokens_z"] = (
        (df_model["log_output_tokens"] - df_model["log_output_tokens"].mean())
        / df_model["log_output_tokens"].std()
    )

USE_OUTPUT_TIMES = True

if USE_OUTPUT_TIMES:
    df_model = df_model.dropna(subset=["times"]).copy()
    df_model["log_times"] = np.log1p(df_model["times"])
    df_model["log_times_z"] = (
        (df_model["log_times"] - df_model["log_times"].mean())
        / df_model["log_times"].std()
    )

df_model["active_parameters_z"] = (df_model["active_parameters"] - df_model["active_parameters"].mean()) / df_model["active_parameters"].std()
# ------------------------------------------------------------
# 2. Formel definieren
# ------------------------------------------------------------

# Haupteffekte:
formula_parts = [
    "C(family, Treatment(reference='Llama'))",
    "active_parameters_z",
    "C(temperature)",
    "C(hardware, Treatment(reference='Workstation'))",
    "C(field, Treatment(reference='input'))",
]

# Interactions:
interaction_parts = [
    #"C(model):C(field, Treatment(reference='input'))",
    #"C(model):C(temperature)"
]

#formula = "times ~ " + " + ".join(formula_parts + interaction_parts)
#formula = "log_times_z ~ " + " + ".join(formula_parts + interaction_parts)
formula = "log_output_tokens_z ~ " + " + ".join(formula_parts + interaction_parts)
#formula = "tokens_out ~ " + " + ".join(formula_parts + interaction_parts)

print("Verwendete Modellformel:")
print(formula)

# ------------------------------------------------------------
# 3. Mixed-Effects-Modell schätzen
# ------------------------------------------------------------

df_model["_all_data"] = 1

vc_formula = {
    "index_re": "0 + C(index)",
    "dataset_model_re": "0 + C(dataset_model)"
}

model = smf.mixedlm(
    formula=formula,
    data=df_model,
    groups=df_model["index"],
    re_formula="1 + C(dataset_model)",
    vc_formula=vc_formula
)

result = model.fit(
    method="lbfgs",
    reml=True,
    maxiter=2000
)

# ------------------------------------------------------------
# 4. Mixed-Effects-Modell ausgeben
# ------------------------------------------------------------

r = result.summary()
x = r.tables
x[1].to_csv("out.csv", index=False)

print(r)
#print(r.as_latex())

fixed_effects = pd.DataFrame({
    "coefficient": result.fe_params,
    "std_error": result.bse_fe,
    "p_value": result.pvalues[result.fe_params.index],
    "ci_lower": result.conf_int().loc[result.fe_params.index, 0],
    "ci_upper": result.conf_int().loc[result.fe_params.index, 1],
})

fixed_effects.index.name = "category"

fixed_effects.to_csv(
    "mixedlm_fixed_effects.csv",
    encoding="utf-8",
    sep=";"
)

random_effects = pd.DataFrame({
    "component": ["residual_scale"] + ["index_re", "dataset_model_re"],
    "variance": [result.scale] + list(result.vcomp)
})

random_effects.to_csv(
    "mixedlm_variance_components.csv",
    index=False,
    encoding="utf-8",
    sep=";"
)
