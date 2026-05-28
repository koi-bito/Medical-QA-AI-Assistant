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
