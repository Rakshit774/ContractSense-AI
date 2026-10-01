# ContractSenseAI --- Project Results & Viva Evidence

> **Status:** Weeks 1--6 completed/current. Weeks 7--13 are future work
> only.

This file keeps the important project numbers, where they came from, and
what to show during the viva.

## 1. Project overview

**Project:** ContractSenseAI

**Goal:** AI-powered legal contract analysis using clause risk
classification (RED/YELLOW/GREEN) and Retrieval-Augmented Generation
(RAG) for contract-grounded question answering.

Progression:

`Data → ML Baseline → DistilBERT → RAG → Application Integration`

------------------------------------------------------------------------

## 2. Week 1--2 --- Dataset EDA & Risk Mapping

### Original dataset

  Set              Contracts   QA pairs      Clauses
  -------------- ----------- ---------- ------------
  Train                  408     16,728       11,180
  Test                   102      4,182        1,244
  **Combined**           ---        ---   **12,424**

Combined dataset shape: **(12424, 6)**

### Data quality

-   Null values: **0**
-   Duplicate rows detected: **639**
-   Clause categories: **41**

Clause-length statistics: - Mean:
**~263**\ -\ Standard\ deviation:\ **~285.7** - Minimum: **1** - Median:
**197** - Maximum: **3,884**

### Cleaning

-   Original samples: **12,424**
-   Usable samples after cleaning: **11,554**
-   Difference: **870**

`12,424 − 11,554 = 870`

> **Viva wording:** "We started with 12,424 clause samples. After
> preprocessing and removing unsuitable rows, 11,554 usable samples
> remained for subsequent modelling."

### Risk distribution after cleaning

  Risk               Count   Approx. share
  ----------- ------------ ---------------
  RED            **2,465**           21.3%
  YELLOW         **4,764**           41.2%
  GREEN          **4,325**           37.4%
  **Total**     **11,554**          \~100%

Cross-check:

`2,465 + 4,764 + 4,325 = 11,554`

Average clause length: - RED: **~360**\ -\ YELLOW:\ **~371** - GREEN:
**\~134**

### Important clause categories

Most frequent: - Parties --- 1,580 - License Grant --- 683 - Cap On
Liability --- 596 - Anti-Assignment --- 583 - Audit Rights --- 575 -
Insurance --- 475 - Agreement Date --- 463 - Expiration Date --- 458 -
Governing Law --- 448 - Effective Date --- 419

Low-frequency examples: - No-Solicit Of Customers --- 50 - Most Favored
Nation --- 38 - Third Party Beneficiary --- 33 -
Unlimited/All-You-Can-Eat-License --- 29 - Price Restrictions --- 27

### Where these results came from

Main script:

`notebooks/Week1_2_EDA_and_Risk_Mapping.py`

Visual outputs:

`outputs/01_clause_distribution.png`

`outputs/02_risk_distribution.png`

`outputs/03_text_length_by_risk.png`

------------------------------------------------------------------------

## 3. Week 3 --- TF-IDF + Logistic Regression

Pipeline:

`Clause Text → TF-IDF → Logistic Regression → RED/YELLOW/GREEN`

Performance:

  Metric                         Result
  --------------- ---------------------
  Test Accuracy               **86.2%**
  Macro F1                   **0.8523**
  5-Fold CV F1      **0.8266 ± 0.0054**
  Training time        **\<10 sec CPU**

Confusion matrix:

  Actual  Predicted     RED   YELLOW   GREEN
  ------------------- ----- -------- -------
  RED                   220       28       9
  YELLOW                 55      323      25
  GREEN                   9       32     442

Saved model:

`models/baseline_lr_pipeline.pkl`

Visual outputs: - `outputs/04_confusion_matrix_baseline.png` -
`outputs/05_top_features_per_class.png`

> **Viva wording:** "The baseline was created as a fast and
> interpretable reference model before using a transformer."

------------------------------------------------------------------------

## 4. Week 4 --- DistilBERT Fine-Tuning

Base model:

`distilbert-base-uncased`

Configuration: - Epochs: **3** - Learning rate: **2e-5** - Maximum
sequence length: **128**

Approximate training history: - Train loss: **\~0.58 → \~0.31 →
~0.235**\ -\ Validation\ loss:\ **~0.335 → \~0.305 →
~0.325**\ -\ Validation\ Macro\ F1:\ **~0.85 → \~0.88 → \~0.87**

### Final test result

  Metric                Result
  --------------- ------------
  Test Accuracy      **87.8%**
  Macro F1          **0.8678**

Final test confusion matrix:

  Actual  Predicted     RED   YELLOW   GREEN
  ------------------- ----- -------- -------
  RED                   218       27      12
  YELLOW                 43      331      29
  GREEN                   9       20     454

Saved model:

`models/distilbert_risk_classifier/`

Comparison:

  Model                             Accuracy     Macro F1
  ------------------------------ ----------- ------------
  TF-IDF + Logistic Regression         86.2%       0.8523
  DistilBERT                       **87.8%**   **0.8678**

> **Viva wording:** "The final held-out test result for DistilBERT was
> 87.8% accuracy and 0.8678 Macro F1. The approximately 0.88 value was
> validation performance during training."

Visual outputs: - `outputs/06_distilbert_training_curve.png` -
`outputs/07_confusion_matrix_distilbert.png`

------------------------------------------------------------------------

## 5. Week 5 --- PDF Parsing, Chunking, Embeddings & RAG

Sample contract:

`data/SampleContract-Shuttle.pdf`

Results: - Pages: **10** - Extracted characters:
**~32,486**\ -\ Chunks:\ **66**\ -\ Average\ chunk\ size:\ **~541** -
Minimum chunk size: **~366**\ -\ Maximum\ chunk\ size:\ **~596**

Chunking: - Chunk size: **600** - Overlap: **100**

Embedding model:

`all-MiniLM-L6-v2`

Embedding dimension:

**384**

Vector database:

**ChromaDB**

Collection:

`contracts`

RAG pipeline:

`PDF → Text Extraction → Chunking → Embeddings → ChromaDB → Retrieval → Groq LLM → Answer`

### Sample contract risk predictions

  Risk           Count
  ----------- --------
  RED           **26**
  YELLOW        **36**
  GREEN          **4**
  **Total**     **66**

`26 + 36 + 4 = 66`

These are classifier predictions across the 66 chunks of the sample
contract.

### Example query

**Question:** "What is the termination notice period?"

The retrieval step found the relevant termination clause containing the
**30-day written notice** provision.

### Main technologies

-   PyMuPDF
-   LangChain
-   Sentence Transformers
-   ChromaDB
-   Groq
-   python-dotenv
-   pytesseract
-   Pillow

### Fixes made

Groq/httpx compatibility error:

`Client.__init__() got an unexpected keyword argument 'proxies'`

Fix:

`pip install httpx==0.27.2`

The initially configured Groq model was unavailable to the user's API
project. The available model selected was:

`openai/gpt-oss-20b`

**Never store or show the actual Groq API key in this file, PPT,
screenshots, GitHub, or viva.**

------------------------------------------------------------------------

## 6. Week 6 --- FastAPI Transition

The FastAPI server is currently running locally and Swagger UI is
available.

Server:

`127.0.0.1:8000`

Swagger:

`127.0.0.1:8000/docs`

Root response:

``` json
{
  "status": "running",
  "app": "ContractSenseAI API",
  "version": "1.0.0",
  "docs": "/docs"
}
```

Swagger currently shows:

### Authentication

-   POST `/auth/register`
-   POST `/auth/login`
-   GET `/auth/me`

### Contracts

-   POST `/contract/upload`
-   POST `/contract/query`
-   GET `/contract/risk/{document_id}`
-   GET `/contract/summary/{document_id}`
-   POST `/contract/counter-proposal`
-   GET `/contract/history`
-   DELETE `/contract/{document_id}`

### Health

-   GET `/`
-   GET `/health`

**Presentation rule:** Do not claim that the complete final web
application is finished. Weeks 7--13 remain future work.

------------------------------------------------------------------------

## 7. Where to prove each result in a viva

  -----------------------------------------------------------------------
  Examiner asks                       Show
  ----------------------------------- -----------------------------------
  Where did 12,424 come from?         Week 1--2 EDA script/output

  Where did 11,554 come from?         Week 1--2 preprocessing/cleaning
                                      output

  Why 870 removed?                    Week 1--2 cleaning logic

  Where did 41 categories come from?  Week 1--2 EDA

  Where did 2,465 / 4,764 / 4,325     Week 1--2 risk distribution
  come from?                          

  How did you get 86.2%?              Week 3 evaluation output +
                                      confusion matrix

  How did you get 0.8523 F1?          Week 3 evaluation

  How did you get 87.8%?              Week 4 final test evaluation

  How did you get 0.8678 F1?          Week 4 final test evaluation

  Why is DistilBERT better?           Week 3 vs Week 4 comparison

  Where did 66 chunks come from?      Week 5 RAG script/output

  Where did 384 come from?            Week 5 embedding
                                      configuration/output

  Where did 26/36/4 come from?        Week 5 classifier output on 66
                                      chunks

  How did RAG answer the termination  Week 5 retrieval output + sample
  question?                           PDF

  Is the backend running?             `127.0.0.1:8000/docs`
  -----------------------------------------------------------------------

### The key principle

The PPT contains the **summary**.

The Python scripts contain the **calculation**.

The `outputs/` folder contains the **visual evidence**.

This file contains the **quick reference** connecting the three.

------------------------------------------------------------------------

## 8. Future Work --- Weeks 7--13

These are planned, not completed.

  Week   Planned work
  ------ ----------------------------------------------
  7      FastAPI backend --- core endpoints
  8      FastAPI backend --- database and polish
  9      React frontend --- authentication and upload
  10     React frontend --- analysis and chat UI
  11     Integration, testing and bug fixes
  12     Deployment and documentation
  13     Final presentation preparation

------------------------------------------------------------------------

## 9. One-minute viva summary

> "In Weeks 1 and 2, I explored and cleaned the contract dataset. We
> started with 12,424 clause samples and obtained 11,554 usable samples
> after preprocessing, covering 41 clause categories and three risk
> levels.
>
> In Week 3, I developed a TF-IDF and Logistic Regression baseline that
> achieved 86.2% accuracy and 0.8523 Macro F1.
>
> In Week 4, I fine-tuned DistilBERT, which improved the final test
> performance to 87.8% accuracy and 0.8678 Macro F1.
>
> In Week 5, I built the RAG pipeline using PDF text extraction,
> chunking, MiniLM embeddings, ChromaDB retrieval and Groq generation. A
> 10-page sample contract produced 66 chunks, and the system
> successfully retrieved the termination clause for a question about the
> notice period.
>
> In Week 6, I started integrating the project into a FastAPI backend,
> with the API and Swagger documentation running locally. The remaining
> weeks will focus on completing the backend, building the React
> frontend, integration, testing, deployment and documentation."

------------------------------------------------------------------------

## 10. Accuracy notes

-   **12,424** = original combined clause samples.
-   **11,554** = cleaned/usable clause samples.
-   **870** = difference between original and cleaned counts.
-   **639** = duplicate rows detected; do not automatically describe all
    639 as the exact number removed unless the code explicitly proves
    that.
-   **2,465 / 4,764 / 4,325** = RED/YELLOW/GREEN counts in the cleaned
    dataset.
-   **86.2% / 0.8523** = TF-IDF + Logistic Regression test performance.
-   **87.8% / 0.8678** = DistilBERT final test performance.
-   **66** = RAG chunks generated from the sample contract.
-   **26 / 36 / 4** = risk predictions across those 66 chunks.
-   **384** = embedding vector dimension for `all-MiniLM-L6-v2`.
-   Never expose the Groq API key.
