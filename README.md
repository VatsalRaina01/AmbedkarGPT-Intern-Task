# 🪶 AmbedkarGPT  
### *Kalpit Pvt Ltd – AI Intern Assignment (Phase 1)*  
**Author:** Vatsal Raina  
🎓 *Computer Science Student | AI & ML Enthusiast | Building things that think.*

---

## 💡 Overview  

**AmbedkarGPT** is a **command-line Q&A system** that answers questions based on an excerpt from Dr. B. R. Ambedkar’s *Annihilation of Caste*.  

This project demonstrates the core idea of **Retrieval-Augmented Generation (RAG)** — combining **embeddings**, **vector search**, and a **local language model** to produce meaningful answers from a given text.  

The entire setup runs **100% offline**, using **LangChain**, **ChromaDB**, and **Ollama’s Mistral-7B** model.  
No API keys. No internet. Just pure local AI 🔥  

---

## 🧠 Tech Stack  

| Component | Tool / Library | Purpose |
|------------|----------------|----------|
| **Framework** | LangChain | Orchestrates the RAG pipeline |
| **Embeddings** | HuggingFace – `all-MiniLM-L6-v2` | Converts text chunks into semantic vectors |
| **Vector Store** | ChromaDB | Local database for vector search |
| **LLM** | Ollama – `Mistral 7B` | Local language model for generating answers |
| **Language** | Python 3.8+ | Core programming language |
| **Runtime** | Fully Offline | No cloud, no API calls, no external dependencies |

---

## ⚙️ Installation & Setup  

Follow these steps to run the project smoothly on your local machine.  

### 1️⃣ Clone this Repository  
```bash
git clone https://github.com/VatsalRaina01/AmbedkarGPT-Intern-Task.git
cd AmbedkarGPT-Intern-Task
2️⃣ Create a Virtual Environment
bash
Copy code
python -m venv .venv
Activate the environment:

Windows (PowerShell): .venv\Scripts\Activate.ps1

macOS/Linux: source .venv/bin/activate

3️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
4️⃣ Install Ollama and Mistral
🧰 Step 1 – Install Ollama
Ollama lets you run large language models locally.
Install it using:

bash
Copy code
curl -fsSL https://ollama.ai/install.sh | sh
(On Windows, just download the installer from ollama.ai/download)

🧠 Step 2 – Download the Mistral Model
bash
Copy code
ollama pull mistral
Verify:

bash
Copy code
ollama list
✅ Expected Output:

nginx
Copy code
NAME      	ID      	SIZE  	MODIFIED
mistral   	...     	4.1GB 	recently
5️⃣ Run the Project
bash
Copy code
python main.py
💬 Then simply type your questions based on the text, for example:

pgsql
Copy code
What is the real remedy according to Ambedkar?
Type exit to quit.

🧩 How It Works
The system follows a mini RAG (Retrieval-Augmented Generation) pipeline.

⚙️ Step-by-Step Process
Load Data:
Reads speech.txt which contains Dr. Ambedkar’s excerpt.

Split Text:
Breaks the text into smaller, overlapping chunks for better embedding coverage.

Create Embeddings:
Uses sentence-transformers/all-MiniLM-L6-v2 from HuggingFace to convert each chunk into a vector representation.

Store Vectors:
Saves all embeddings in a ChromaDB database for local semantic retrieval.

Retrieve Relevant Chunks:
When you ask a question, the system finds the most semantically similar text chunks.

Generate Answer:
Sends those chunks + your question to the Mistral 7B model running locally in Ollama to generate an answer.

✅ Everything is processed locally — no external API keys or internet calls.

🧾 Example Interaction
User Input:

perl
Copy code
What does Ambedkar say about the shastras?
System Output:

vbnet
Copy code
Ambedkar says that people must stop believing in the sanctity of the shastras.
He argues that caste cannot end unless the authority of these scriptures is overthrown.
📁 Project Structure
graphql
Copy code
AmbedkarGPT-Intern-Task/
│
├── main.py                # Core LangChain RAG pipeline
├── requirements.txt       # Dependency list
├── speech.txt             # Input text (Ambedkar’s excerpt)
└── README.md              # Documentation
🧠 Key Learnings
During this project, I learned:

How embeddings represent text meaning in vector form.

How vector databases enable semantic similarity search.

How LangChain simplifies building AI pipelines.

How to run local LLMs using Ollama (no API keys required).

It was a fun experience to connect all the dots between NLP, embeddings, and local inference.

🌱 Future Improvements
🔍 Integrate FAISS or Pinecone for faster and larger-scale vector retrieval.

💬 Add a Streamlit or Gradio UI for a web-based chatbot interface.

📚 Expand the dataset with multiple Ambedkar speeches.

⚡ Use quantized models for faster inference on low-end systems.

🧠 Explore context summarization for long responses.

🧑‍💻 Reflection
“Building AmbedkarGPT was both exciting and educational.
It showed me how modern RAG systems actually connect retrieval and generation into one workflow.
It was amazing to see a simple command-line tool answer real questions completely offline!”

📜 License
This project is created for learning and internship evaluation purposes.
You can freely explore, fork, and build upon it for educational use.

