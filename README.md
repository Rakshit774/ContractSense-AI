# ContractSenseAI 📄🤖

**AI-Powered Contract Analysis and Risk Detection**

ContractSenseAI is a web application designed to help users understand legal contracts by summarizing contract documents, answering questions using Retrieval-Augmented Generation (RAG), identifying potentially risky clauses, and explaining those clauses in simple language.

The project combines traditional Machine Learning, vector search, and Large Language Models (LLMs) to make contract analysis more accessible.

> **Disclaimer:** ContractSenseAI is an educational project. Its outputs are AI-generated assistance and should not be considered professional legal advice.

## ✨ Features

- **Contract Upload:** Upload a contract PDF for analysis.
- **Contract Summarization:** Generate a readable summary of the contract.
- **AI-Powered Q&A:** Ask questions about a contract and receive answers based on relevant contract passages.
- **Risk Analysis:** Classify contract clauses into RED, YELLOW, GREEN, and UNKNOWN categories.
- **Explain Risk:** Generate a natural-language explanation of a selected clause and its potential risks.
- **Contract History:** Backend support for retrieving contract history.
- **Contract Deletion:** Backend endpoint for deleting a contract.
- **Counter-Proposal Generation:** Backend functionality exists; frontend integration is currently pending.

Feature availability and behaviour may vary with the current implementation.

## 🛠️ Tech Stack

### Frontend
- React
- Vite
- Tailwind CSS
- JavaScript
- React Markdown

### Backend
- Python
- FastAPI
- Uvicorn
- SQLAlchemy
- SQLite
- JWT authentication infrastructure

### Machine Learning and AI
- Scikit-learn
- TF-IDF
- Logistic Regression
- DistilBERT experimentation
- Sentence Transformers
- ChromaDB
- Groq API
- `openai/gpt-oss-20b` (configured language model)

## 🧠 How It Works

The application combines two important workflows: contract risk classification and RAG-based question answering.

### 1. Contract Upload and Processing

1. The user uploads a contract PDF.
2. The backend extracts the text from the document.
3. The text is divided into smaller chunks.
4. Each chunk is processed for risk classification.
5. Embeddings are generated for semantic retrieval.
6. Chunks and their associated metadata are stored in ChromaDB for subsequent analysis.

### 2. Risk Classification

The current backend uses a TF-IDF and Logistic Regression pipeline for risk classification.

- **TF-IDF:** Converts text into numerical features based on word importance.
- **Logistic Regression:** Predicts a risk category from those features.

The application groups clauses into RED, YELLOW, GREEN, and UNKNOWN categories. These are model outputs, not guaranteed legal judgments.

DistilBERT was also explored during model development, but it should not be assumed to be the current backend inference model.

### 3. Contract Question Answering (RAG)

Retrieval-Augmented Generation combines document retrieval with language generation.

1. The user asks a question.
2. The embedding and vector-search pipeline retrieves relevant contract chunks.
3. The retrieved context is supplied to the configured language model.
4. The model generates a readable answer grounded in the retrieved contract text.

### 4. Summary and Risk Explanation

The language model is used to generate natural-language contract summaries and explanations of selected clauses. This is separate from the traditional ML classifier, which predicts risk categories.

## 🏗️ Project Structure

The repository contains the backend, frontend, and model-related project files.

```text
ContractSense-AI/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── rag.py
│   │   ├── database.py
│   │   ├── models.py
│   │   ├── schemas.py
│   │   ├── auth.py
│   │   └── routes/
│   │       ├── auth.py
│   │       └── contract.py
│   └── ...
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── index.css
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
├── models/
│   └── baseline_lr_pipeline.pkl
├── .gitignore
└── README.md
```

*This is a representative structure. Check the repository for the exact files and any additional folders.*

## 🚀 Getting Started

### Prerequisites

Install the following:

- Python
- Node.js and npm
- Git
- A Groq API key for features that use the Groq language model

### 1. Clone the Repository

```bash
git clone https://github.com/Rakshit774/ContractSense-AI.git
cd ContractSense-AI
```

### 2. Set Up the Backend

Create and activate a Python virtual environment from the project root.

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install backend dependencies using the project's existing requirements file, if provided:

```powershell
pip install -r backend/requirements.txt
```

If the requirements file is stored elsewhere, use its actual path. Do not install dependencies from an assumed file location.

Configure the environment variables required by the backend. Check the source code and any example environment file for the exact variable names. Keep API keys in a local `.env` file and never commit secrets to GitHub.

Start the backend from the appropriate directory:

```powershell
cd backend
uvicorn app.main:app --reload --port 8000
```

Open the FastAPI Swagger UI:

http://127.0.0.1:8000/docs

### 3. Set Up the Frontend

Open another terminal at the project root:

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL displayed by Vite, usually:

http://localhost:5173

Ensure that the frontend API URL matches the running backend. Follow the existing project's configuration and authentication requirements.

### 4. Test the Main Workflow

1. Start the backend and frontend.
2. Upload a sample contract PDF.
3. Generate its summary.
4. Ask a question about a clause.
5. Review the risk analysis.
6. Select a clause and request its risk explanation.

Some endpoints may require authentication. Follow the current backend's configured requirements.

## 🔌 API Endpoints

The following endpoints were implemented during development. Verify their current request schemas and authentication requirements in FastAPI Swagger before use.

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/contract/upload` | Upload and process a contract |
| POST | `/contract/query` | Ask a question about a contract |
| GET | `/contract/risk/{document_id}` | Retrieve risk analysis |
| POST | `/contract/risk/{document_id}/explain/{chunk_index}` | Explain a selected clause |
| GET | `/contract/summary/{document_id}` | Retrieve a generated summary |
| POST | `/contract/counter-proposal` | Generate a counter-proposal |
| GET | `/contract/history` | Retrieve contract history |
| DELETE | `/contract/{document_id}` | Delete a contract |

Authentication endpoints are also present in the backend. Refer to `/docs` for the exact current API specification.

## 🔐 Security and Configuration

- Never commit `.env` files, API keys, passwords, or access tokens.
- Do not commit virtual environments or `node_modules`.
- Avoid committing private contracts or personally identifiable information.
- Review the `.gitignore` file before pushing changes.
- Configure secrets locally using the environment variable names expected by the code.
- Treat generated risk labels and LLM answers as suggestions that require human review.

## 🌱 Fork This Project

Want to explore or extend ContractSenseAI?

1. Open the repository:  
   https://github.com/Rakshit774/ContractSense-AI
2. Sign in to GitHub.
3. Click **Fork** in the top-right corner.
4. Select your GitHub account as the destination.
5. GitHub will create a copy of the repository under your account.
6. Clone your fork to your computer:

```bash
git clone https://github.com/YOUR-USERNAME/ContractSense-AI.git
cd ContractSense-AI
```

Replace `YOUR-USERNAME` with your GitHub username.

You can now experiment with your own copy without directly modifying the original repository.

### Keep Your Fork Updated

If you want to receive later updates from the original project:

```bash
git remote add upstream https://github.com/Rakshit774/ContractSense-AI.git
git fetch upstream
git merge upstream/main
```

Resolve any merge conflicts if they occur.

## 🗺️ Current Development Status

ContractSenseAI is an ongoing academic project.

**Implemented or developed:**
- FastAPI backend
- PDF contract processing
- ML-based risk classification
- ChromaDB-based retrieval
- RAG-based question answering
- Contract summaries
- Risk explanations
- Backend history and deletion endpoints
- Initial React frontend

**Planned or pending:**
- Frontend login/register integration
- Counter-proposal frontend integration
- Further testing and refinement
- Evaluation of alternative language-model approaches

The exact completion status should be verified against the current codebase before treating every feature as production-ready.

## 🤝 Contributions

Fork the repository, create a feature branch, make your changes, and open a pull request if you want to contribute improvements.

Please avoid committing secrets, private contract documents, or generated local data.

## 📄 License

No license has been specified here. Before reusing or redistributing this project, check whether the repository includes a `LICENSE` file and confirm the applicable permissions.

---

**Repository:** https://github.com/Rakshit774/ContractSense-AI
