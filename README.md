# KroMA: Knowledge Graph–grounded Medical Agent for Dravet Syndrome

This repository contains public materials associated with **KroMA**, a knowledge graph–grounded medical agent developed for question answering and clinical reasoning in **Dravet syndrome (DS)**.

KroMA was designed to support structured reasoning across multiple DS-relevant domains, including genetics, seizures, development, behavior, comorbidities, electrophysiology, pharmacology, and treatment response. The public contents of this repository include the benchmark question sets used in our evaluation and the curated article list used to identify the DS literature corpus. The full **DS knowledge graph** is **not included** in this public repository and is available only through a **Data Use Agreement (DUA)** process.

## Repository contents

### `KroMA_QA/`

This folder contains the **KroMA-QA** benchmark question sets, organized into two components:

- **`FoundationalKnowledge_QA`**  
  Contains **99 foundational knowledge questions and answer keys**, organized across the 9 Dravet syndrome axes used in the study. These items assess retrieval of established DS domain knowledge.

- **`SpecializedClinicalReasoning_QA`**  
  Contains **18 specialized clinical reasoning questions and answer keys**, also organized across the same 9 DS axes. These items assess integrative reasoning in clinically complex scenarios.

### `Articles/`

This folder contains:

- **`Articles.xlsx`**  
  A spreadsheet listing articles retrieved from PubMed Central using the search terms **“Dravet Syndrome”** and **“Severe Myoclonic Epilepsy of Infancy”**. The file includes bibliographic and indexing fields such as article title, authors, journal, year, PMCID, and related metadata.

## Benchmark organization

The KroMA-QA benchmark is organized across 9 DS-related knowledge axes:

1. Seizures  
2. Development  
3. Behavior  
4. SUDEP  
5. Genetics  
6. Comorbidities  
7. Electrophysiology  
8. Pharmacology  
9. Drug Responsiveness  

The benchmark is divided into two complementary components:

- **Foundational knowledge** items evaluate recall of established DS facts and standard domain knowledge.
- **Specialized clinical reasoning** items evaluate more complex inference requiring integration across multiple clinical and mechanistic axes.

## What is not included in this repository

The **DS knowledge graph dataset** used in KroMA is **not publicly posted in this repository at this time**. Access to the knowledge graph is contingent upon execution of a **Data Use Agreement (DUA)**.

## Intended use

These materials are provided to support transparency and reproducibility for the KroMA study, including:

- inspection of the benchmark structure
- review of the article corpus used for literature identification
- secondary analysis of question organization across DS knowledge domains

Users should note that the benchmark files and article list are public research resources, whereas the knowledge graph itself is distributed separately under controlled access.

## Related manuscript

These repository materials accompany the KroMA manuscript, which presents a knowledge graph–grounded approach for rare-disease reasoning in Dravet syndrome and evaluates performance using the KroMA-QA benchmark.

## Data availability

The curated article list and benchmark question/answer files are available in this repository.

Access to the DS knowledge graph dataset is contingent upon execution of a Data Use Agreement.

## Citation

If you use these materials, please cite the associated KroMA manuscript once citation details become available.

## Contact

For questions regarding this repository or DUA-based access to the DS knowledge graph, please contact:

**Pedram Golnari**  
**Case Western Reserve University**  
**pedram.golnari[@]case.edu**
