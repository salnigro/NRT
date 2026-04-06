# Project Report: AI for Substance Abuse Risk Detection

## Project Title
AI-Powered Dual-Reasoning Engine for Substance Abuse Risk and Temporal Analysis

## Team Members
Sal Nigro (and Team, if applicable)

## Problem Statement
The escalating public health crisis surrounding substance abuse (such as heroin, methadone, and synthetic opioids) requires dynamic, timely models that can decode diverse communication channels. Traditional models often rely purely on structured statistics or manual evaluations, leading to lagged interventions. The objective is to design a robust AI pipeline under "Track A: AI Modeling and Reasoning" capable of detecting nuanced language indicating emotional distress and behavioral changes from real-world unstructured text, while strictly correlating those signals with quantitative national demographic trends to surface early-warning indicators for public health officials.

## Dataset(s) Used
Our project entirely adheres to using anonymized, population-level public data representing structural real-world information:
1. **Unstructured Reviews (`drugsComTrain_raw.csv` and `drugsComTest_raw.csv`)**: Crowdsourced anonymous patient condition reviews encapsulating natural language indicators of relapses, distress, and side effects.
2. **CDC Provisional Overdose Deaths (`VSRR_Provisional_Drug_Overdose_Death_Counts_20260404.csv`)**: Wide-scale national tracking capturing explicit numeric year-over-year fatalities segmented by drug indicators.
3. **CDC Overdose Death Rates by Demographic (`Drug_overdose_death_rates,_by_drug_type,_sex,_age,_race,_and_Hispanic_origin__United_States_20260404.csv`)**: Deep longitudinal statistical data representing fatalities broken apart by targeted sub-population criteria (Age, Race, Sex, Hispanic origin).

## Data Preprocessing
Data ingestion processes were rigorously split to preserve experimental design:
*   **Textual Noise Reduction:** Raw patient data was filtered for HTML artifacts, decoded encoding errors (`&#039;`), and stripped of null/unusable reviews, ensuring only high signal-to-noise records remained.
*   **Table-to-Text Embedding Conversion:** Structured demographic arrays (such as year, panel, sub-population label, and estimated metrics) were intelligently parsed into semantic natural language phrases (e.g., *"CDC Demographic Stat: In 2017, the rate for All... among Male: Black was X"*). This allows quantitative data to be reliably fused into dense embedding schemas.
*   **Vector State Separation:** Preprocessed outputs were cleanly partitioned into an explicit `cleaned_train_reviews.csv` knowledge base (establishing our training schema memory), preserving out-of-index `cleaned_test_reviews.csv`.

## ML/AI Methods Used
*   **Retrieval-Augmented Generation (RAG):** The core framework to fuse localized contexts dynamically avoiding blind knowledge hallucination.
*   **Semantic Dense Embeddings:** Using HuggingFace's `sentence-transformers/all-MiniLM-L6-v2` representation schema to translate combined text reviews and stringified demographics into numerical vectors.
*   **High-Dimensional Vector Store:** Implementing Facebook AI Similarity Search (FAISS) as our high-speed, local similarity clustering database storing training contexts. 
*   **Causal Language Modeling:** Injecting contexts into an instantiated local `TinyLlama/TinyLlama-1.1B-Chat-v1.0` model. The chain runs through a customized reasoning prompt bridging numerical statistics and psychological texts concurrently.

## Experimental Design
The pipeline processes parallel intelligence streams. An incoming query (e.g., assessing methadone spikes or synthetic opioid behavioral distress patterns) concurrently triggers:
1. A rigorous time-aware algorithm mapping explicit keyword ontology (e.g., fentanyl -> T40.4) strictly directly evaluating Year-over-Year trajectory metrics in CDC statistics.
2. A semantic top-*K* nearest neighbor retrieval isolating text chunks depicting distress symptoms alongside embedded demographic sentences.
The LLM is prompted via LangChain templates to generate a holistic narrative and interpretable outputs reconciling these dual streams against the hidden test boundaries.

## Results and Discussion
By combining embedding-based topic discovery with explicit time-aligned computations, the engine generates complex answers that transcend literal keyword mapping. The local FAISS store flawlessly surfaces marginalized demographic context sentences alongside visceral descriptions of drug-induced distress, permitting the TinyLlama model to justify pipeline outputs directly with explicit CDC citations and textual review contexts. The approach confirms that table-to-text augmentation combined with deterministic time-bound data effectively creates auditable logic maps ideal for policy analysts. Spikes in quantitative fatality models correctly align with the thematic sentiments clustered in the embeddings.

## Ethical Considerations
Ensuring safety, the system explicitly strips and restricts user identifiers from all ingested sources emphasizing strict population-level insights. The pipeline prioritizes transparency; LLM-reasoned outputs are exclusively generated out of referenced evidence via RAG logic (reducing confabulation dangers on medical themes). We address demographic bias directly by explicitly indexing demographic stratification rates from CDC databases, ensuring the search algorithms neutrally map intersectional vectors across race and sex categories when analyzing substance disparities.

## Conclusion and Future Improvement
The deployed Hybrid track reasoning engine comprehensively bridges AI linguistics, numerical logic routing, and contextual data formatting. Future improvements would migrate this foundational Local-LLM logic into broader agentic architectures (such as `langgraph`) or visually interactive cloud interfaces (Streamlit deployment). Furthermore, injecting fine-tuned proprietary foundational models like FLAN-T5 architectures or extending multi-modal scraping across streaming web endpoints could scale the early-warning functionality universally.
