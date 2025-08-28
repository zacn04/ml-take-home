from huggingface_hub import login
from dotenv import load_dotenv
import transformers as tr
import torch
import os


load_dotenv()

amateur_path = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
expert_path = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

login(token=os.getenv("HF_TOKEN"))

tokenizer = tr.AutoTokenizer.from_pretrained(amateur_path)

amateur_model = tr.AutoModelForCausalLM.from_pretrained(amateur_path, torch_dtype=torch.float32).to("cpu")
expert_model = tr.AutoModelForCausalLM.from_pretrained(expert_path, torch_dtype=torch.float32).to("cpu")

user_message = """Give a very very brief docstring for the following function:\n```\nfunction updateEloScores(
	scores,
	results,
	kFactor = 4,
) {
	for (const result of results) {
		const { first, second, outcome } = result;
		const firstScore = scores[first] ?? 1000;
		const secondScore = scores[second] ?? 1000;

		const expectedScoreFirst = 1 / (1 + Math.pow(10, (secondScore - firstScore) / 400));
		const expectedScoreSecond = 1 / (1 + Math.pow(10, (firstScore - secondScore) / 400));
		let sa = 0.5;
		if (outcome === 1) {
			sa = 1;
		} else if (outcome === -1) {
			sa = 0;
		}
		scores[first] = firstScore + kFactor * (sa - expectedScoreFirst);
		scores[second] = secondScore + kFactor * (1 - sa - expectedScoreSecond);
	}
	return scores;
}\n```"""

prompt = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": user_message},
    ],
    add_generation_prompt=True,
    tokenize=False,
)


def contrastive_generation(amateur, expert, prompt, max_tokens) -> str:
	input_ids = tokenizer(prompt, return_tensors="pt").input_ids
	for _ in range(max_tokens):
		# getting the next token distributions

		with torch.no_grad():
			expert_logits = expert(input_ids).logits[0, -1, :] 
			amateur_logits = amateur(input_ids).logits[0, -1, :] 

		expert_probs = torch.softmax(expert_logits, dim=-1)
		amateur_probs = torch.softmax(amateur_logits, dim=-1)

		# applyign the plausibility constraint
		max_prob = expert_probs.max()
		valid_mask = expert_probs >= (0.1 * max_prob) #alpha=0.1

		#calculating contrastive scores
		cd_scores = torch.full_like(expert_probs, -float("inf"))
		cd_scores[valid_mask] = (
			torch.log(expert_probs[valid_mask]) - torch.log(amateur_probs[valid_mask])
		)

		#greedily select the next token
		next_token = cd_scores.argmax()

		input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

		if next_token == tokenizer.eos_token_id:
			break
	return tokenizer.decode(input_ids[0], skip_special_tokens=True)


if __name__ == "__main__": 
	result = contrastive_generation(
		amateur_model, expert_model, prompt, max_tokens=100
	)

	response_start = result.find("assistant\n") + len("assistant\n") 
	response = result[response_start:] if response_start > len("assistant\n") else result
	print("\n\nContrastive generation:")
	print(response)

	print("\n\nFor comparison, expert model only:")
	inputs = tokenizer(prompt, return_tensors="pt")
	expert_output = expert_model.generate(inputs.input_ids, max_new_tokens=100, do_sample=False)
	expert_response = tokenizer.decode(expert_output[0], skip_special_tokens=True)
	expert_response = expert_response[expert_response.find("assistant\n") + len("assistant\n"):]
	print(expert_response)