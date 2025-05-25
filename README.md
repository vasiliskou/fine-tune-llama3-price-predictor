# 🔧 LLaMA 3.1–8B Fine-Tuning for Electronics Price Prediction

This project was developed with guidance from the Udemy course 'LLM Engineering: Master AI, Large Language Models, & Agents' by the Ligency Team and Ed Donner.

The goal is to fine-tune **LLaMA 3.1–8B** to take a product title or short description as input and predict its price.

A Gradio-based web interface allows users to input product descriptions and receive instant price predictions. This interface is the final step of a pipeline including data curation, fine-tuning, evaluation, deployment, and UI integration.


---

## 📷 Sample Screenshot

![App Screenshot](https://drive.google.com/uc?export=view&id=12trxkYXH45zgecZSKtiFZcuAgd8fVnRj)

---

## 📁 Notebooks Overview – Quick Access via Google Colab


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1TtIZ7WUPE0M7cel3mSAyBRXacZowdzNv?usp=sharing) **Create Dataset.ipynb** – Load, preprocess, and format Amazon Electronics data for training.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vJBtYkv6UhJx281xCrUiivI3J_Oo8104?usp=sharing) **Fine-tune LLaMA 3.1–8B.ipynb** – Fine-tune using QLoRA on 100k curated examples.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CFUADSzG5oCgfhnyOfTqKy2se2v3p01p?usp=sharing) **Test Baseline Models.ipynb** – Compare with traditional regressors.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YacTU3blaUj2kOedH7CFm2N6W-YfCCr1?usp=sharing) **Test Frontier Models.ipynb** – Evaluate GPT-4o, Gemini, Claude, DeepSeek.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CXUxE8r0iWxz8OPHEClFINkiWVvG9rTf?usp=sharing) **Test LLaMA 3.1 Base.ipynb** – Evaluate pre-trained model performance.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JuNeTv1bsesFDiF7ilmsd5BJsNbg9ZiT?usp=sharing) **Test LLaMA 3.1 Fine-Tuned.ipynb** – Evaluate the fine-tuned model.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1M7WxpPYEMOkmEvdcEmkGIBDXBDeGADam?usp=sharing) **Deploy to Modal.ipynb** – Package and deploy the model via Modal cloud.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SpB-7PHiUQD--n3dzfRvYbkt_nOl9J-O?usp=sharing) **Run Gradio App.ipynb** – Build and launch interactive UI.


---

## 📊 Performance Summary (Avg. Error on 250 Test Samples)

| Category        | Model                         | Avg. Error (\$) |
| --------------- | ----------------------------- | --------------- |
| Baselines       | Random                        | 350.64          |
|                 | Constant Avg Price            | 156.70          |
|                 | Bag of Words + Linear Regr.   | 112.00          |
|                 | Word2Vec + Linear Regr.       | 116.80          |
|                 | Word2Vec + SVR                | 114.70          |
|                 | Word2Vec + Random Forest      | 103.88          |
| Frontier Models | GPT-4o Mini                   | 80.17           |
|                 | GPT-4o                        | 76.53           |
|                 | Claude 3.7 Sonnet             | 97.45           |
|                 | Gemini 1.5 Flash              | 90.29           |
|                 | Gemini 2.0 Flash Exp          | 77.26           |
|                 | DeepSeek V3\*                 | 90.43           |
| LLaMA Models    | LLaMA 3.1–8B (Base)           | 320.16          |
|                 | **LLaMA 3.1–8B (Fine-Tuned)** | **49.94**       |

---

## 📄 Prompt Example

**Training Prompt:**

```
How much does this cost to the nearest dollar?
Protective skin decal for MacBook Pro 13-inch (Models A2338, A2289, A2251). Features a fresh marble design and full-body vinyl wrap that guards against scratches, dust, oil, water, and fingerprints—without adding bulk. Designed for durability and long-lasting style.
Price is $20.00
```

**Test Prompt:** same input but with the price line removed.

---

## 📌 Detailed Notebook Descriptions

<details>
<summary>Click to expand full walkthrough</summary>

### 1. Create Dataset

* Filtered Amazon Electronics data to retain items priced \$0–\$999.
* After filtering and balancing price brackets, final sets: **100k train / 2k test**.
* Created custom prompts for training and evaluation.

### 2. Fine-tune LLaMA 3.1–8B

* Fine-tuned using **QLoRA** (4-bit) with `trl`'s SFTTrainer.
* Trained on A100 GPU, batch size 16, 2 epochs (\~3h18m).
* Logs via Weights & Biases. Uploaded checkpoints to Hugging Face Hub.

### 3. Test Baselines

* Baseline models: Random, Constant, Bag-of-Words, Word2Vec + various regressors.
* Results used as a performance floor. Best ML model (Random Forest): \$103.88 error.

### 4. Test Frontier Models

* GPT-4o, Claude, Gemini, DeepSeek evaluated on same 250 samples.
* Not fine-tuned, just prompted with test set.
* GPT-4o best commercial model (\$76.53), but fine-tuned LLaMA **beats all** with \$49.94.

### 5. Test LLaMA 3.1 Base

* Base model (no fine-tuning) tested on 250 prompts.
* High error: \$320.16. Confirms need for fine-tuning.

### 6. Test LLaMA 3.1 Fine-Tuned

* Final evaluation confirms top performance: **\$49.94 average error**.
* Outperforms frontier models by a wide margin.

### 7. Deploy to Modal

* Model deployed as API using Modal with T4 GPU.
* Served via `@modal.cls` decorator with QLoRA-efficient inference.

### 8. Run Gradio App

* Simple, interactive UI with:

  * Textbox for product description
  * Button to trigger inference
  * Output box with predicted price
* Connected to deployed Modal function.

</details>

---

## 🛠️ Tech Stack

* **LLM**: LLaMA 3.1–8B (base & fine-tuned)
* **Fine-Tuning**: QLoRA + TRL + Hugging Face
* **Baseline ML**: scikit-learn (Linear, SVR, RF)
* **LLM APIs**: GPT-4o, Claude, Gemini, DeepSeek
* **Deployment**: Modal
* **UI**: Gradio (`gr.Blocks()`)
* **Tracking**: Weights & Biases
* **Data**: Amazon Reviews Dataset (Electronics)

---

## 🧠 Conclusions

This project demonstrates that a carefully fine-tuned open-source model (LLaMA 3.1–8B) can **outperform top frontier LLMs** like GPT-4o and Claude on a focused regression task. By crafting a clean dataset, using QLoRA for efficient training, and deploying the result via Modal + Gradio, we built a full-stack AI product with state-of-the-art performance.
