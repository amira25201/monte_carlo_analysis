# What predicts second-language proficiency? A bootstrap and counterfactual analysis

Monte Carlo (bootstrap) and counterfactual analysis of second-language
proficiency in adult learners of English. Three outcomes were measured for 203 learners:
grammar, vocabulary and pronunciation scores. The question was which learner
characteristics predict those scores, how much to trust the model, and what a
typical learner would need to change to approach the best observed outcome.

Method: one random forest per outcome on an 80/20 split; uncertainty from
150 bootstrap refits of each forest; predictor ranking by permutation
importance on the held-out set; a bounded random search over four features a
learner could actually change, starting from a population-typical learner.

## Running it

The analysis was run on real data from 203 adult learners of English as a
second language. That dataset is not in this repository. The tests run
entirely on synthetic data generated within the test suite itself, so no
real data is needed to run them.

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python run_analysis.py            # about 3 to 5 minutes
python run_analysis.py --quick    # smoke test, under a minute
python run_analysis.py --row 12   # counterfactuals from one data row instead of the typical learner
```

Outputs: `figures/` (prediction intervals, importances, bootstrap metric
spreads), `results/` (every table as CSV plus `findings.md`), and `models/`
(git-ignored, because a forest trained on 162 rows effectively stores them).

Versions in `requirements.txt` are pinned to the ones the committed figures
were produced with, on Python 3.14.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

Eleven tests on synthetic data (generated within the test suite, not from
the real dataset) check that targets, identifiers, outcome
measures and quality flags never reach the predictor matrix; that the
pronunciation model cannot fit a noise target once the sub-scores are gone;
that every pipeline owns its own preprocessor; that bootstrap predictions are
reproducible; that the combined score is the mean of the three models; that
goals come from test rows; that the population baseline is median and mode;
and that the search stays inside observed bounds and touches only editable
features.

## Layout

```
config.py          paths, model settings, excluded and editable columns, with the reasons
mc_analysis.py     the analysis as functions: data, training, bootstrap, importance,
                   ceilings, counterfactual search, findings writer
run_analysis.py    runs everything top to bottom; --quick and --row options
tests/             pytest suite on synthetic data
data/README.md     what the data must look like; the data itself is excluded
figures/ results/  generated outputs, committed so the findings can be read without the data
```

## Design decisions

- **Goals come from held-out predictions.** A random forest is nearly perfect
  on its own training rows, so a ceiling taken over all rows would set an
  inflated target. The combined TOTAL_L2 goal is the best averaged prediction
  for a single real test learner, which is reachable by construction; the
  mean of three separate maxima might belong to nobody.
- **The baseline is a typical learner**, the median of every numeric feature
  and the mode of every text feature. Starting from a real row is available
  behind an explicit flag and prints no identifying information.
- **Editable means editable.** Age, non-verbal intelligence and working memory
  are traits. Self-rated proficiency is a rating of the outcome itself. None
  of them is a lever, so none is in the search. Search bounds are the observed
  range of each feature in the data.
- **One preprocessor per pipeline.** Each model clones the preprocessor before
  fitting, so nothing is shared between targets or bootstrap refits.
- **Combined predictions are vectorised.** The TOTAL_L2 predictor calls each
  model once per block of rows, which is what makes the search finish in
  seconds rather than hours.
