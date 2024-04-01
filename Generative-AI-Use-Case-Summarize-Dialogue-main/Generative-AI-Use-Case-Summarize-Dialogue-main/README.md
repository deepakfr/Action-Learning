## Generative-AI-Use-Case-Summarize-Dialogue
This repository contains three main python codes or three main Labs discussed as follows:
**First Lab:** 
In this lab i will do the dialogue summarization task using generative AI. i will explore how the input text affects the output of the model, and perform prompt engineering to direct it towards the task you need. By comparing zero shot, one shot, and few shot inferences, i will take the first step towards prompt engineering and see how it can enhance the generative output of Large Language Models.
**Table of Contents**
1 - Set up Kernel and Required Dependencies
2 - Summarize Dialogue without Prompt Engineering
3 - Summarize Dialogue with an Instruction Prompt
3.1 - Zero Shot Inference with an Instruction Prompt
3.2 - Zero Shot Inference with the Prompt Template from FLAN-T5
4 - Summarize Dialogue with One Shot and Few Shot Inference
4.1 - One Shot Inference
4.2 - Few Shot Inference
5 - Generative Configuration Parameters for Inference

**Second Lab: Fine-Tune a Generative AI Model for Dialogue Summarization**
In this notebook, i will fine-tune an existing LLM from Hugging Face for enhanced dialogue summarization. I will use the FLAN-T5 model, which provides a high quality instruction tuned model and can summarize text out of the box. To improve the inferences, i will explore a full fine-tuning approach and evaluate the results with ROUGE metrics. Then i will perform Parameter Efficient Fine-Tuning (PEFT), evaluate the resulting model and see that the benefits of PEFT outweigh the slightly-lower performance metrics.
**Table of Contents**
1 - Set up Kernel, Load Required Dependencies, Dataset and LLM 
1.1 - Set up Kernel and Required Dependencies
1.2 - Load Dataset and LLM
1.3 - Test the Model with Zero Shot Inferencing
2 - Perform Full Fine-Tuning
2.1 - Preprocess the Dialog-Summary Dataset
2.2 - Fine-Tune the Model with the Preprocessed Dataset
2.3 - Evaluate the Model Qualitatively (Human Evaluation)
2.4 - Evaluate the Model Quantitatively (with ROUGE Metric)
3 - Perform Parameter Efficient Fine-Tuning (PEFT)
3.1 - Setup the PEFT/LoRA model for Fine-Tuning
3.2 - Train PEFT Adapter
3.3 - Evaluate the Model Qualitatively (Human Evaluation)
3.4 - Evaluate the Model Quantitatively (with ROUGE Metric)

**Third Lab:Fine-Tune FLAN-T5 with Reinforcement Learning (PPO) and PEFT to Generate Less-Toxic Summaries**
In this notebook, i will fine-tune a FLAN-T5 model to generate less toxic content with Meta AI's hate speech reward model. The reward model is a binary classifier that predicts either "not hate" or "hate" for the given text. I will use Proximal Policy Optimization (PPO) to fine-tune and reduce the model's toxicity.
**Table of Contents**
1 - Set up Kernel and Required Dependencies
2 - Load FLAN-T5 Model, Prepare Reward Model and Toxicity Evaluator
2.1 - Load Data and FLAN-T5 Model Fine-Tuned with Summarization Instruction
2.2 - Prepare Reward Model
2.3 - Evaluate Toxicity
3 - Perform Fine-Tuning to Detoxify the Summaries
3.1 - Initialize PPOTrainer
3.2 - Fine-Tune the Model
3.3 - Evaluate the Model Quantitatively
3.4 - Evaluate the Model Qualitatively
