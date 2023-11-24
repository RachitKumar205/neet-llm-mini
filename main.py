import torch
import transformers

if torch.cuda.is_available():
    torch.set_default_device("cuda")
else:
    torch.set_default_device("cpu")

model = transformers.AutoModelForCausalLM.from_pretrained("casualdatauser/neet-llm")

# https://github.com/huggingface/transformers/issues/27132
# please use the slow tokenizer since fast and slow tokenizer produces different tokens
tokenizer = transformers.AutoTokenizer.from_pretrained(
    "casualdatauser/neet-llm",
    use_fast=True,
)

system_message = ""
user_message = "### HUMAN: Who is Ernst Mayr"

prompt = user_message

inputs = tokenizer(prompt, return_tensors='pt')
output_ids = model.generate(inputs["input_ids"], )
answer = tokenizer.batch_decode(output_ids)[0]

print(answer)
