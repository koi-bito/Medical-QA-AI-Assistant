# 🧠 Learnings


## Day 1

What is a weight?
- **Weight is a value of importance of pixel for identifying the object/pattern.**

What does "training" mean in your own words?
- **Training means, model correcting itself through updating weights which are calculated through the loss function and gradients to clarify how much the model is going in the right direction.**


## Day 2

Why are embeddings useful? What does "similar meaning = similar vector" actually mean?
- **An embedding is a list of numbers (a vector) that represents a word, sentence, or piece of text in a multi-dimensional space.Computers can only work with numbers, not words. Embeddings solve this by converting text into a vector — an ordered list of numbers — that captures the meaning of that text.**
- **When two words or sentences have similar meanings, their vectors end up close together in vector space. For example: "happy" and "joyful" will have vectors that are nearly pointing in the same direction, while "happy" and "car" will be far apart.**


## Day 3

What is the difference between a base model and a fine-tuned model? Why does QLoRA exist?
- **A base model is pre-trained on general data, while a fine-tuned model is adapted for specific tasks; QLoRA exists to enable efficient fine-tuning of large models on limited hardware by using 4-bit quantization and low-rank adapters.**


## Day 4

What problem does RAG solve that fine-tuning doesn't?
- **RAG solves the knowledge‑access problem, letting a model fetch fresh, factual information from an external source at inference time, something fine‑tuning, which only rewrites static model weights can’t achieve.**


## Day 5

What does one training example look like in med_qa? What fields does it have?
- **One training example in med_qa (bigbio format) looks something like this:
{
  'id': '1',
  'question': 'A 23-year-old pregnant woman at 22 weeks gestation presents with...',
  'answer': {'text': 'Decreased fetal movement', 'label': 'A'},
  'options': [
    {'key': 'A', 'value': 'Decreased fetal movement'},
    {'key': 'B', 'value': 'Premature rupture of membranes'},
    ...
  ],
  'type': 'multiple_choice',
  'metamap_phrases': [...]
}
The key fields are: 'id', 'question' (the clinical scenario), 'options' (4–5 MCQ choices with keys A/B/C/D), 'answer' (the correct option label + text), 'type' (always multiple_choice here), and 'metamap_phrases' (medical concept tags extracted from the question).
The dataset is a multiple-choice medical licensing exam format — each example is a USMLE-style clinical question. The model's job is to pick the right answer from the options, not generate free text. This matters for how we'll format the training data later — we'll need to frame it as an instruction-following task so the fine-tuned model learns to reason and select.**
