<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Syne&weight=800&size=40&pause=1000&color=5EEAB0&center=true&vCenter=true&width=700&lines=Cottonil+BI+%F0%9F%93%8A;Arabic+Business+Intelligence;AI-Powered+%C2%B7+Evidence+First" alt="Cottonil BI" />

<br/>

![Python](https://img.shields.io/badge/Python-3.13-5EEAB0?style=for-the-badge&logo=python&logoColor=black&labelColor=0a0c10)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-5EEAB0?style=for-the-badge&logo=streamlit&logoColor=black&labelColor=0a0c10)
![Pandas](https://img.shields.io/badge/Pandas-Analytics-5EEAB0?style=for-the-badge&logo=pandas&logoColor=black&labelColor=0a0c10)
![Plotly](https://img.shields.io/badge/Plotly-Charts-5EEAB0?style=for-the-badge&logo=plotly&logoColor=black&labelColor=0a0c10)
![Groq](https://img.shields.io/badge/Groq-AI_Assistant-5EEAB0?style=for-the-badge&labelColor=0a0c10)

<br/>

> **From business records to measurable decisions — in Arabic.**
> Every number is calculated locally. AI explains — it never invents figures.

<br/>

| 📄 Pages | 📈 Backtest Origins | ✅ Tests Passed | 🌐 Interface |
|:---:|:---:|:---:|:---:|
| **9** | **12** | **260** | **Arabic · RTL** |

<br/>

**[🚀 Try the Live Demo →](https://cotonil.streamlit.app/)**

<br/>

</div>

---

## ✦ Screenshots

[![Dashboard](photo/1.png)](https://cotonil.streamlit.app/)

---

## ✦ What is Cottonil BI?

**Cottonil BI** is an Arabic Streamlit dashboard for retail performance. It connects sales, branch profit, inventory, customers, and forecasts in one place — with an AI assistant that answers business questions in Arabic.

| # | Page | Description |
|---|------|-------------|
| 1 | 🏠 **Overview** | Sales KPIs, trends, top business issues |
| 2 | 💰 **Sales** | Branch, governorate, category, product, channel |
| 3 | 🏬 **Branches** | Branch comparison and profitability |
| 4 | 📦 **Inventory** | Stock coverage, shortage / overstock, transfers |
| 5 | 📈 **Forecasting** | Company, category, branch, product forecasts |
| 6 | 👥 **Customers** | RFM segments and purchase history |
| 7 | 💡 **Recommendations** | Evidence, actions, estimated impact |
| 8 | 🤖 **AI Assistant** | Arabic questions, comparisons, follow-ups |
| 9 | 🩺 **Data Health** | Data quality checks and rebuild controls |

---

## ✦ How the Assistant Works

| Step | What happens |
|------|--------------|
| 🧭 **Understand** | Reads the Arabic question: measures, branches, periods |
| 🧠 **Keep context** | Remembers the last answer for follow-up questions |
| 🔢 **Calculate** | Local code computes every number — never the AI |
| ✍️ **Explain** | Groq writes the interpretation, then a second review checks it |
| ✅ **Decide** | Shows evidence-backed options, even if AI is offline |

**Example prompts:**
```text
قارن مبيعات محافظة الإسكندرية آخر 60 يوم بالـ60 يوم اللي قبلهم.
طيب والربح في نفس الفترة؟
اختار أضعف فرعين في النمو واقترح خطة لكل فرع.
```

---

## ✦ Project Structure

```
HVIA_Cottonil/
│
├── 🚀 app.py                 # Streamlit entry point
│
├── 🧠 src/
│   ├── assistant.py          # Arabic intent + grounded answers
│   ├── llm.py                # Groq client
│   ├── data.py               # Loading & quality checks
│   ├── metrics.py            # Business metrics
│   ├── models.py             # Forecasting & customer models
│   ├── inventory.py          # Coverage & transfers
│   ├── insights.py           # Recommendations
│   └── ui.py                 # Shared UI helpers
│
├── 🧪 tests/                 # 260 offline tests
├── 🖼️ photo/1.png
├── Dockerfile
├── requirements.txt
└── requirements.lock
```

---

## ✦ Quick Start

**Install & run locally:**
```bash
git clone https://github.com/morad-elnahla/HVIA_Cottonil.git
cd HVIA_Cottonil
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.lock
streamlit run app.py
```

Open **http://localhost:8501**

**Rebuild data (optional):**
```bash
python -m src.data
```

---

## ✦ Environment (optional)

Groq is optional — local analysis works without a key. Copy `.env.example` to `.env`:

```dotenv
LLM_API_KEY=
LLM_MODEL=openai/gpt-oss-120b
LLM_NARRATE=auto
LLM_TPM_LIMIT=7000
```

| Setting | Purpose |
|---------|---------|
| `LLM_API_KEY` | Your Groq key (empty = local only) |
| `LLM_MODEL` | Model name |
| `LLM_NARRATE` | `auto` · `on` · `off` |
| `LLM_ENABLED` | `false` disables AI calls |

---

## ✦ Forecasting

| Feature | Detail |
|---------|--------|
| 📊 Levels | Company · Category · Branch · Product |
| 🔁 Backtesting | 12 time-based origins |
| 💵 Revenue | Quantity × price, with January price assumption |
| 📏 Uncertainty | Intervals, average error, reliability labels |

---

## ✦ Tech Stack

| Layer | Technology |
|-------|-----------|
| 🐍 Language | Python 3.13 |
| 🌐 UI | Streamlit (Arabic · RTL · Dark) |
| 📊 Data | pandas, numpy, parquet |
| 📈 Charts | Plotly |
| 🤖 AI | Groq (optional, reviewed) |
| 🧪 Tests | pytest — 260 passed |
| 🐳 Container | Docker |

---

<div align="center">

Built with 📊 by **Morad El-Nahla**

[![Live Demo](https://img.shields.io/badge/Live_Demo-5EEAB0?style=for-the-badge&logo=streamlit&logoColor=black&labelColor=0a0c10)](https://cotonil.streamlit.app/)

</div>
