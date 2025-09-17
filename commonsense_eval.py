import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer
import json
import os
from typing import Dict, List, Tuple, Any
import logging

# Disable transformers warnings
transformers.logging.set_verbosity_error()

logger = logging.getLogger(__name__)

class CommonsenseEvaluator:
    """
    Evaluator for the 8 standard commonsense reasoning datasets:
    - BoolQ: Reading Comprehension with Yes/No Questions
    - PIQA: Physical Interaction Question Answering  
    - SIQA: Social Interaction Question Answering
    - HellaSwag: Commonsense inference about physical situations
    - WinoGrande: Winograd Schema Challenge
    - ARC-Easy: AI2 Reasoning Challenge (Easy)
    - ARC-Challenge: AI2 Reasoning Challenge (Challenge)
    - OBQA: OpenBookQA
    
    Standard evaluation uses full validation/test sets:
    - BoolQ: 3,270 examples
    - PIQA: 1,838 examples
    - SIQA: 1,954 examples  
    - HellaSwag: 10,042 examples
    - WinoGrande: 1,267 examples
    - ARC-Easy: 2,376 examples
    - ARC-Challenge: 1,172 examples
    - OBQA: 500 examples
    """
    
    def __init__(self, tokenizer, device="cuda", max_length=512):
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.results = {}
    
    
    def _build_few_shot_prompt(self, train_examples, current_example, task_type, num_shots=5):
        """Build few-shot prompt with examples"""
        if num_shots == 0:
            return self._format_single_example(current_example, task_type, include_answer=False)
        
        # Select random examples from training set
        import random
        selected_examples = random.sample(list(train_examples), min(num_shots, len(train_examples)))
        
        prompt_parts = []
        for example in selected_examples:
            formatted_example = self._format_single_example(example, task_type, include_answer=True)
            prompt_parts.append(formatted_example)
        
        # Add current example without answer
        current_formatted = self._format_single_example(current_example, task_type, include_answer=False)
        prompt_parts.append(current_formatted)
        
        return "\n\n".join(prompt_parts)
    
    def _format_single_example(self, example, task_type, include_answer=False):
        """Format a single example for the given task type"""
        if task_type == "boolq":
            prompt = f"Passage: {example['passage']}\nQuestion: {example['question']}\nAnswer:"
            if include_answer:
                answer = "Yes" if example['answer'] else "No"
                prompt += f" {answer}"
            return prompt
            
        elif task_type == "siqa":
            prompt = f"Context: {example['context']}\nQuestion: {example['question']}\nAnswer:"
            if include_answer:
                choices = [example["answerA"], example["answerB"], example["answerC"]]
                correct_idx = int(example["label"]) - 1
                prompt += f" {choices[correct_idx]}"
            return prompt
            
        elif task_type == "hellaswag":
            prompt = f"{example['ctx']}"
            if include_answer:
                correct_ending = example["endings"][int(example["label"])]
                prompt += f" {correct_ending}"
            return prompt
            
        elif task_type == "piqa":
            prompt = f"Question: {example['goal']}\nAnswer:"
            if include_answer:
                choices = [example["sol1"], example["sol2"]]
                correct_idx = int(example["label"])
                prompt += f" {choices[correct_idx]}"
            return prompt
            
        elif task_type in ["arc_easy", "arc_challenge"]:
            prompt = f"Question: {example['question']}\nAnswer:"
            if include_answer:
                choices = example["choices"]["text"]
                labels = example["choices"]["label"]
                answer_key = example["answerKey"]
                try:
                    correct_idx = labels.index(answer_key)
                    prompt += f" {choices[correct_idx]}"
                except ValueError:
                    pass
            return prompt
            
        elif task_type == "obqa":
            prompt = f"Question: {example['question_stem']}\nAnswer:"
            if include_answer:
                choices = example["choices"]["text"]
                labels = example["choices"]["label"]
                answer_key = example["answerKey"]
                try:
                    correct_idx = labels.index(answer_key)
                    prompt += f" {choices[correct_idx]}"
                except ValueError:
                    pass
            return prompt
            
        elif task_type == "winogrande":
            if include_answer:
                correct_option = example["option1"] if int(example["answer"]) == 1 else example["option2"]
                return example["sentence"].replace("_", correct_option)
            else:
                return example["sentence"]  # Will be processed separately
        
        return ""
    
    def evaluate_boolq(self, model, num_shots=0):
        """Evaluate on BoolQ dataset - Yes/No questions"""
        dataset = load_dataset("boolq", split="validation", trust_remote_code=True)
        train_dataset = load_dataset("boolq", split="train", trust_remote_code=True) if num_shots > 0 else None
        
        correct = 0
        total = 0
        
        for example in tqdm(dataset, desc=f"Evaluating BoolQ ({num_shots}-shot)"):
            label = example["answer"]  # True/False
            
            if num_shots > 0:
                # Build few-shot prompt
                base_prompt = self._build_few_shot_prompt(train_dataset, example, "boolq", num_shots)
                prompt_yes = base_prompt + " Yes"
                prompt_no = base_prompt + " No"
            else:
                # 0-shot evaluation
                prompt_base = f"Passage: {example['passage']}\nQuestion: {example['question']}\nAnswer:"
                prompt_yes = prompt_base + " Yes"
                prompt_no = prompt_base + " No"
            
            score_yes = self._get_perplexity_score(model, prompt_yes)
            score_no = self._get_perplexity_score(model, prompt_no)
            
            predicted = score_yes < score_no  # Lower perplexity = better
            if predicted == label:
                correct += 1
            total += 1
        
        accuracy = correct / total
        self.results["boolq"] = {"accuracy": accuracy, "correct": correct, "total": total}
        logger.info(f"BoolQ Accuracy: {accuracy:.4f} ({correct}/{total})")
        return accuracy
    
    def evaluate_siqa(self, model, num_shots=0):
        """Evaluate on Social IQA dataset"""
        dataset = load_dataset("social_i_qa", split="validation", trust_remote_code=True)
        train_dataset = load_dataset("social_i_qa", split="train", trust_remote_code=True) if num_shots > 0 else None
        
        correct = 0
        total = 0
        
        for example in tqdm(dataset, desc="Evaluating SIQA"):
            context = example["context"]
            question = example["question"]
            choices = [example["answerA"], example["answerB"], example["answerC"]]
            label = int(example["label"]) - 1  # Convert 1,2,3 to 0,1,2
            
            scores = []
            for choice in choices:
                full_text = f"Context: {context}\nQuestion: {question}\nAnswer: {choice}"
                score = self._get_perplexity_score(model, full_text)
                scores.append(score)
            
            predicted = np.argmin(scores)
            if predicted == label:
                correct += 1
            total += 1
        
        accuracy = correct / total
        self.results["siqa"] = {"accuracy": accuracy, "correct": correct, "total": total}
        logger.info(f"SIQA Accuracy: {accuracy:.4f} ({correct}/{total})")
        return accuracy
        
    def evaluate_hellaswag(self, model, num_shots=0):
        """Evaluate on HellaSwag dataset"""
        dataset = load_dataset("hellaswag", split="validation", trust_remote_code=True)
        train_dataset = load_dataset("hellaswag", split="train", trust_remote_code=True) if num_shots > 0 else None
        
        correct = 0
        total = 0
        
        for example in tqdm(dataset, desc="Evaluating HellaSwag"):
            context = example["ctx"]
            choices = example["endings"]
            label = int(example["label"])
            
            scores = []
            for choice in choices:
                full_text = context + " " + choice
                score = self._get_perplexity_score(model, full_text)
                scores.append(score)
            
            predicted = np.argmin(scores)  # Lower perplexity = better
            if predicted == label:
                correct += 1
            total += 1
        
        accuracy = correct / total
        self.results["hellaswag"] = {"accuracy": accuracy, "correct": correct, "total": total}
        logger.info(f"HellaSwag Accuracy: {accuracy:.4f} ({correct}/{total})")
        return accuracy
    
    def evaluate_piqa(self, model, num_shots=0):
        """Evaluate on PIQA dataset"""
        dataset = load_dataset("piqa", split="validation", trust_remote_code=True)
        train_dataset = load_dataset("piqa", split="train", trust_remote_code=True) if num_shots > 0 else None
        
        correct = 0
        total = 0
        
        for example in tqdm(dataset, desc="Evaluating PIQA"):
            question = example["goal"]
            choices = [example["sol1"], example["sol2"]]
            label = int(example["label"])
            
            scores = []
            for choice in choices:
                full_text = f"Question: {question} Answer: {choice}"
                score = self._get_perplexity_score(model, full_text)
                scores.append(score)
            
            predicted = np.argmin(scores)
            if predicted == label:
                correct += 1
            total += 1
        
        accuracy = correct / total
        self.results["piqa"] = {"accuracy": accuracy, "correct": correct, "total": total}
        logger.info(f"PIQA Accuracy: {accuracy:.4f} ({correct}/{total})")
        return accuracy
    
    def evaluate_arc_easy(self, model, num_shots=0):
        """Evaluate on ARC-Easy dataset"""
        dataset = load_dataset("ai2_arc", "ARC-Easy", split="test", trust_remote_code=True)
        train_dataset = load_dataset("ai2_arc", "ARC-Easy", split="train", trust_remote_code=True) if num_shots > 0 else None
        
        correct = 0
        total = 0
        
        for example in tqdm(dataset, desc="Evaluating ARC-Easy"):
            question = example["question"]
            choices = example["choices"]["text"]
            labels = example["choices"]["label"]
            answer_key = example["answerKey"]
            
            # Find the correct answer index
            try:
                label = labels.index(answer_key)
            except ValueError:
                continue  # Skip if answer key not found
            
            scores = []
            for choice in choices:
                full_text = f"Question: {question} Answer: {choice}"
                score = self._get_perplexity_score(model, full_text)
                scores.append(score)
            
            predicted = np.argmin(scores)
            if predicted == label:
                correct += 1
            total += 1
        
        accuracy = correct / total
        self.results["arc_easy"] = {"accuracy": accuracy, "correct": correct, "total": total}
        logger.info(f"ARC-Easy Accuracy: {accuracy:.4f} ({correct}/{total})")
        return accuracy
    
    def evaluate_arc_challenge(self, model, num_shots=0):
        """Evaluate on ARC-Challenge dataset"""
        dataset = load_dataset("ai2_arc", "ARC-Challenge", split="test", trust_remote_code=True)
        train_dataset = load_dataset("ai2_arc", "ARC-Challenge", split="train", trust_remote_code=True) if num_shots > 0 else None
        
        correct = 0
        total = 0
        
        for example in tqdm(dataset, desc="Evaluating ARC-Challenge"):
            question = example["question"]
            choices = example["choices"]["text"]
            labels = example["choices"]["label"]
            answer_key = example["answerKey"]
            
            # Find the correct answer index
            try:
                label = labels.index(answer_key)
            except ValueError:
                continue  # Skip if answer key not found
            
            scores = []
            for choice in choices:
                full_text = f"Question: {question} Answer: {choice}"
                score = self._get_perplexity_score(model, full_text)
                scores.append(score)
            
            predicted = np.argmin(scores)
            if predicted == label:
                correct += 1
            total += 1
        
        accuracy = correct / total
        self.results["arc_challenge"] = {"accuracy": accuracy, "correct": correct, "total": total}
        logger.info(f"ARC-Challenge Accuracy: {accuracy:.4f} ({correct}/{total})")
        return accuracy
    
    def evaluate_obqa(self, model, num_shots=0):
        """Evaluate on OpenBookQA dataset"""
        dataset = load_dataset("openbookqa", "main", split="test", trust_remote_code=True)
        train_dataset = load_dataset("openbookqa", "main", split="train", trust_remote_code=True) if num_shots > 0 else None
        
        correct = 0
        total = 0
        
        for example in tqdm(dataset, desc="Evaluating OBQA"):
            question = example["question_stem"]
            choices = example["choices"]["text"]
            labels = example["choices"]["label"]
            answer_key = example["answerKey"]
            
            # Find the correct answer index
            try:
                label = labels.index(answer_key)
            except ValueError:
                continue  # Skip if answer key not found
            
            scores = []
            for choice in choices:
                full_text = f"Question: {question} Answer: {choice}"
                score = self._get_perplexity_score(model, full_text)
                scores.append(score)
            
            predicted = np.argmin(scores)
            if predicted == label:
                correct += 1
            total += 1
        
        accuracy = correct / total
        self.results["obqa"] = {"accuracy": accuracy, "correct": correct, "total": total}
        logger.info(f"OBQA Accuracy: {accuracy:.4f} ({correct}/{total})")
        return accuracy
        
    def evaluate_winogrande(self, model, num_shots=0):
        """Evaluate on WinoGrande dataset"""
        dataset = load_dataset("winogrande", "winogrande_debiased", split="validation", trust_remote_code=True)
        train_dataset = load_dataset("winogrande", "winogrande_debiased", split="train", trust_remote_code=True) if num_shots > 0 else None
        
        correct = 0
        total = 0
        
        for example in tqdm(dataset, desc="Evaluating WinoGrande"):
            sentence = example["sentence"]
            option1 = example["option1"]
            option2 = example["option2"]
            answer = int(example["answer"])
            
            # Replace placeholder with each option
            sent1 = sentence.replace("_", option1)
            sent2 = sentence.replace("_", option2)
            
            score1 = self._get_perplexity_score(model, sent1)
            score2 = self._get_perplexity_score(model, sent2)
            
            predicted = 1 if score1 < score2 else 2
            if predicted == answer:
                correct += 1
            total += 1
        
        accuracy = correct / total
        self.results["winogrande"] = {"accuracy": accuracy, "correct": correct, "total": total}
        logger.info(f"WinoGrande Accuracy: {accuracy:.4f} ({correct}/{total})")
        return accuracy
    
    def _get_perplexity_score(self, model, text):
        """Calculate perplexity score for a given text"""
        model.eval()
        with torch.no_grad():
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=False
            ).to(self.device)
            
            # Get logits
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            # Calculate perplexity
            perplexity = torch.exp(loss).item()
            return perplexity
    
    def evaluate_all(self, model, num_shots=0):
        """Evaluate on all 8 standard commonsense reasoning tasks"""
        results = {}
        
        # BoolQ
        logger.info(f"Starting BoolQ evaluation ({num_shots}-shot)...")
        results["boolq_accuracy"] = self.evaluate_boolq(model, num_shots)
        
        # PIQA - we'll implement a simplified version for now
        logger.info(f"Starting PIQA evaluation ({num_shots}-shot)...")
        results["piqa_accuracy"] = self._evaluate_simple_choice(model, "piqa", num_shots)
        
        # SIQA
        logger.info(f"Starting SIQA evaluation ({num_shots}-shot)...")
        results["siqa_accuracy"] = self._evaluate_simple_choice(model, "siqa", num_shots)
        
        # HellaSwag
        logger.info(f"Starting HellaSwag evaluation ({num_shots}-shot)...")
        results["hellaswag_accuracy"] = self._evaluate_simple_choice(model, "hellaswag", num_shots)
        
        # WinoGrande
        logger.info(f"Starting WinoGrande evaluation ({num_shots}-shot)...")
        results["winogrande_accuracy"] = self._evaluate_winogrande_shots(model, num_shots)
        
        # ARC-Easy
        logger.info(f"Starting ARC-Easy evaluation ({num_shots}-shot)...")
        results["arc_easy_accuracy"] = self._evaluate_simple_choice(model, "arc_easy", num_shots)
        
        # ARC-Challenge
        logger.info(f"Starting ARC-Challenge evaluation ({num_shots}-shot)...")
        results["arc_challenge_accuracy"] = self._evaluate_simple_choice(model, "arc_challenge", num_shots)
        
        # OBQA
        logger.info(f"Starting OBQA evaluation ({num_shots}-shot)...")
        results["obqa_accuracy"] = self._evaluate_simple_choice(model, "obqa", num_shots)
        
        # Calculate average accuracy
        accuracies = list(results.values())
        results["commonsense_avg_accuracy"] = np.mean(accuracies)
        logger.info(f"Average Commonsense Accuracy ({num_shots}-shot): {results['commonsense_avg_accuracy']:.4f}")
        
        return results
    
    def _evaluate_simple_choice(self, model, task_name, num_shots=0):
        """Generic evaluation for multiple choice tasks"""
        # Dataset loading configs
        dataset_configs = {
            "piqa": ("piqa", None, "validation"),
            "siqa": ("social_i_qa", None, "validation"),
            "hellaswag": ("hellaswag", None, "validation"),
            "arc_easy": ("ai2_arc", "ARC-Easy", "test"),
            "arc_challenge": ("ai2_arc", "ARC-Challenge", "test"),
            "obqa": ("openbookqa", "main", "test")
        }
        
        dataset_name, config_name, split = dataset_configs[task_name]
        
        if config_name:
            dataset = load_dataset(dataset_name, config_name, split=split, trust_remote_code=True)
            train_dataset = load_dataset(dataset_name, config_name, split="train", trust_remote_code=True) if num_shots > 0 else None
        else:
            dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
            train_dataset = load_dataset(dataset_name, split="train", trust_remote_code=True) if num_shots > 0 else None
        
        correct = 0
        total = 0
        
        for example in tqdm(dataset, desc=f"Evaluating {task_name.upper()} ({num_shots}-shot)"):
            # Get choices and correct answer based on task type
            if task_name == "piqa":
                choices = [example["sol1"], example["sol2"]]
                label = int(example["label"])
            elif task_name == "siqa":
                choices = [example["answerA"], example["answerB"], example["answerC"]]
                label = int(example["label"]) - 1
            elif task_name == "hellaswag":
                choices = example["endings"]
                label = int(example["label"])
            elif task_name in ["arc_easy", "arc_challenge"]:
                choices = example["choices"]["text"]
                labels = example["choices"]["label"]
                answer_key = example["answerKey"]
                try:
                    label = labels.index(answer_key)
                except ValueError:
                    continue
            elif task_name == "obqa":
                choices = example["choices"]["text"]
                labels = example["choices"]["label"]
                answer_key = example["answerKey"]
                try:
                    label = labels.index(answer_key)
                except ValueError:
                    continue
            
            scores = []
            for choice in choices:
                if num_shots > 0:
                    # Build few-shot prompt
                    base_prompt = self._build_few_shot_prompt(train_dataset, example, task_name, num_shots)
                    if task_name == "hellaswag":
                        full_text = base_prompt + " " + choice
                    else:
                        full_text = base_prompt + " " + choice
                else:
                    # 0-shot evaluation
                    if task_name == "piqa":
                        full_text = f"Question: {example['goal']} Answer: {choice}"
                    elif task_name == "siqa":
                        full_text = f"Context: {example['context']}\nQuestion: {example['question']}\nAnswer: {choice}"
                    elif task_name == "hellaswag":
                        full_text = example["ctx"] + " " + choice
                    elif task_name in ["arc_easy", "arc_challenge"]:
                        full_text = f"Question: {example['question']} Answer: {choice}"
                    elif task_name == "obqa":
                        full_text = f"Question: {example['question_stem']} Answer: {choice}"
                
                score = self._get_perplexity_score(model, full_text)
                scores.append(score)
            
            predicted = np.argmin(scores)
            if predicted == label:
                correct += 1
            total += 1
        
        accuracy = correct / total
        logger.info(f"{task_name.upper()} Accuracy ({num_shots}-shot): {accuracy:.4f} ({correct}/{total})")
        return accuracy
    
    def _evaluate_winogrande_shots(self, model, num_shots=0):
        """Special evaluation for WinoGrande with few-shot support"""
        dataset = load_dataset("winogrande", "winogrande_debiased", split="validation", trust_remote_code=True)
        train_dataset = load_dataset("winogrande", "winogrande_debiased", split="train", trust_remote_code=True) if num_shots > 0 else None
        
        correct = 0
        total = 0
        
        for example in tqdm(dataset, desc=f"Evaluating WinoGrande ({num_shots}-shot)"):
            sentence = example["sentence"]
            option1 = example["option1"]
            option2 = example["option2"]
            answer = int(example["answer"])
            
            if num_shots > 0:
                # Build few-shot examples
                import random
                selected_examples = random.sample(list(train_dataset), min(num_shots, len(train_dataset)))
                
                prompt_parts = []
                for ex in selected_examples:
                    correct_option = ex["option1"] if int(ex["answer"]) == 1 else ex["option2"]
                    completed_sentence = ex["sentence"].replace("_", correct_option)
                    prompt_parts.append(completed_sentence)
                
                # Add current examples
                sent1 = sentence.replace("_", option1)
                sent2 = sentence.replace("_", option2)
                
                few_shot_prefix = "\n".join(prompt_parts) + "\n"
                full_sent1 = few_shot_prefix + sent1
                full_sent2 = few_shot_prefix + sent2
            else:
                # 0-shot evaluation
                full_sent1 = sentence.replace("_", option1)
                full_sent2 = sentence.replace("_", option2)
            
            score1 = self._get_perplexity_score(model, full_sent1)
            score2 = self._get_perplexity_score(model, full_sent2)
            
            predicted = 1 if score1 < score2 else 2
            if predicted == answer:
                correct += 1
            total += 1
        
        accuracy = correct / total
        logger.info(f"WinoGrande Accuracy ({num_shots}-shot): {accuracy:.4f} ({correct}/{total})")
        return accuracy
    
    def save_results(self, save_path):
        """Save evaluation results to JSON file"""
        with open(save_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Results saved to {save_path}")

def evaluate_commonsense_reasoning(model, tokenizer, device="cuda", save_dir=None, shots=[0, 5]):
    """
    Main function to evaluate a model on the 8 standard commonsense reasoning tasks
    with both 0-shot and few-shot evaluation
    
    Args:
        model: The language model to evaluate
        tokenizer: The tokenizer
        device: Device to run evaluation on
        save_dir: Directory to save results (optional)
        shots: List of shot numbers to evaluate (default: [0, 5])
    
    Returns:
        Dictionary with evaluation results for each shot setting
    """
    evaluator = CommonsenseEvaluator(tokenizer, device)
    
    all_results = {}
    
    for num_shots in shots:
        logger.info(f"\n{'='*50}")
        logger.info(f"Starting {num_shots}-shot evaluation")
        logger.info(f"{'='*50}")
        
        results = evaluator.evaluate_all(model, num_shots)
        all_results[f"{num_shots}_shot"] = results
        
        logger.info(f"\n{num_shots}-shot Results Summary:")
        logger.info("-" * 30)
        for task, accuracy in results.items():
            if task != "commonsense_avg_accuracy":
                logger.info(f"{task.replace('_accuracy', '').upper()}: {accuracy:.4f}")
        logger.info(f"AVERAGE: {results['commonsense_avg_accuracy']:.4f}")
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "commonsense_results.json")
        with open(save_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Results saved to {save_path}")
    
    return all_results

if __name__ == "__main__":
    # Example usage
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    results = evaluate_commonsense_reasoning(
        model, 
        tokenizer, 
        device=device,
        num_samples_per_task={"hellaswag": 100, "piqa": 100, "arc_easy": 50, "winogrande": 100}
    )
    
    print("Commonsense Reasoning Results:")
    for task, accuracy in results.items():
        print(f"{task}: {accuracy:.4f}")