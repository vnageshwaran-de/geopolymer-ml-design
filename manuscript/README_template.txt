MDPI Materials — LaTeX submission package
=========================================

This folder is a complete, compile-ready MDPI *Materials* manuscript built on
the OFFICIAL MDPI LaTeX class (Definitions/mdpi.cls), populated with the full
paper content, three figures, Table 1, and a numbered (ACS-style) bibliography.

FILES
  main.tex                         <- the manuscript (compile THIS)
  Definitions/                     <- official MDPI class, .bst, logos (unmodified)
  figures/                         <- the three figures referenced by main.tex
  template_ORIGINAL_reference.tex  <- pristine MDPI template, for reference only

HOW TO COMPILE
  Option A — Overleaf (recommended, no local install):
    1. Zip this folder (or upload it) to a new Overleaf project.
    2. Set the main document to main.tex.
    3. Compile with pdfLaTeX. (Overleaf runs pdflatex -> bibtex -> pdflatex x2.)

  Option B — local TeX Live / MacTeX:
    cd MDPI_materials_submission
    pdflatex main
    pdflatex main        # second pass resolves refs/citations
    # (no bibtex step needed: references are embedded in a thebibliography block)

NOTES
  * documentclass: [materials,article,submit,pdftex,moreauthors]
    Change "submit" to "accept" only when the editor instructs (removes line numbers).
  * ORCID iDs are placeholders (0000-0000-0000-0000) — replace with real iDs.
  * The bibliography lists 41 candidate references drawn from the bibliometric
    corpus plus the Mendeley dataset (ref: dataset). The body currently cites the
    dataset; add \cite{refN} calls in the text where you want the others to appear
    (their keys are ref1..ref41, in candidate_references.csv order).
  * Dataset: 80 one-part fly-ash/GGBS geopolymer mixtures, Faridmehr et al., Materials 2023, 16, 2348 (doi:10.3390/ma16062348, Table 1, CC BY).
