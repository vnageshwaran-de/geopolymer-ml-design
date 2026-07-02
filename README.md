# geopolymer-ml-design

Machine learning framework for designing one-part geopolymer composites for multifunctional defense and civil infrastructure applications.

This repository accompanies the manuscript *"Machine Learning-Assisted Design of One-Part Geopolymer Composites: A Framework for Multifunctional Defense and Civil Infrastructure Applications"* (prepared for MDPI *Materials*). It contains the LaTeX manuscript source, the machine-learning demonstration pipeline, and the underlying datasets.

## Repository structure

```
geopolymer-ml-design/
├── manuscript/          MDPI Materials LaTeX source
│   ├── main.tex             populated manuscript (compile this)
│   ├── Definitions/         official MDPI class, styles, and logos
│   ├── figures/             figures 1–4
│   ├── template_ORIGINAL_reference.tex
│   └── README_template.txt
├── ml/
│   └── geopolymer_ml_pipeline.py   Section 4 ML demonstration
├── data/                CSV datasets (mixtures, corpus, results)
└── README.md
```

## Manuscript

The manuscript is written against the official MDPI `materials` class (bundled under `manuscript/Definitions/`). References are embedded as `\bibitem`s, so no BibTeX pass is required.

Compile with pdfLaTeX (two passes):

```bash
cd manuscript
pdflatex main
pdflatex main
```

Or upload the `manuscript/` folder to Overleaf, set `main.tex` as the main document, and compile with pdfLaTeX. The pre-converted logo PDFs are included so the EPS assets render without `epstopdf`.

## Machine-learning pipeline

`ml/geopolymer_ml_pipeline.py` reproduces the Section 4 demonstration on a genuine one-part geopolymer dataset of 80 fly-ash/GGBS paste mixtures (`data/onepart_geopolymer_dataset.csv`). It:

1. Trains four regressors — Linear Regression, SVR (RBF), Random Forest, and XGBoost — to predict compressive strength.
2. Evaluates them with a held-out test split, 5-fold cross-validation, and leave-one-source-out (LOSO) validation.
3. Computes TreeSHAP feature importance for the XGBoost model.

Dependencies: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`, `matplotlib`.

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib
cd ml
python geopolymer_ml_pipeline.py --data ../data/onepart_geopolymer_dataset.csv --outdir .
```

### Headline results

| Model | Test R² | Test RMSE | LOSO R² |
|-------|:------:|:--------:|:------:|
| **XGBoost** | **0.90** | **8.34** | **0.61** |
| Random forest | 0.84 | 10.46 | 0.58 |
| SVR (RBF) | 0.71 | 14.19 | 0.54 |
| Linear regression | 0.80 | 11.85 | 0.36 |

TreeSHAP identifies slag fraction and Na₂O content as the dominant drivers of compressive strength.

## Data

The `data/` folder contains the modeling dataset plus the supporting bibliometric and results tables (candidate references, corpus, concept-network nodes/edges, model comparison, LOSO fold results, and SHAP importances) used to build the manuscript figures.

## Data source and attribution

The one-part geopolymer dataset is compiled from open-literature sources, principally Faridmehr, Sahraei, Nehdi & Valerievich, *"Optimization of Fly Ash-Slag One-Part Geopolymers with Improved Properties,"* Materials 2023, 16, 2348 (doi:10.3390/ma16062348, open access).

## License

This repository is dual-licensed:

- **Code** (the `ml/` directory and any scripts) — [MIT License](LICENSE).
- **Manuscript and data** (`manuscript/` and `data/`) — [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE-CC-BY-4.0.txt).

The one-part geopolymer dataset is compiled from open-literature sources (see *Data source and attribution* above), which retain their original authors' copyright and CC BY 4.0 terms.
