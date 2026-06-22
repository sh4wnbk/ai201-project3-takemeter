import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "./takemeter-model"   # produced by the notebook (trainer.save_model)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()
ID_TO_LABEL = model.config.id2label

def classify(comment):
    if not comment or not comment.strip():
        return {}
    inputs = tokenizer(comment, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        probs = torch.nn.functional.softmax(model(**inputs).logits, dim=-1)[0].tolist()
    return {ID_TO_LABEL[i]: float(probs[i]) for i in range(len(probs))}

demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(lines=4, label="Hacker News comment", placeholder="Paste a comment…"),
    outputs=gr.Label(num_top_classes=3, label="Predicted discourse type"),
    title="TakeMeter — HN discourse classifier",
    description="Classifies a comment as analysis, hot_take, or reaction. Fine-tuned DistilBERT.",
    examples=[
        ["The latency win isn't the Rust rewrite — they moved the hot path off the GC'd heap; you'd get the same gain in Go with a sync.Pool."],
        ["Kubernetes is wildly over-engineered for 99% of companies. Just use a VM."],
        ["This is the most beautiful codebase I've seen all year, wow."],
    ],
)

if __name__ == "__main__":
    demo.launch()