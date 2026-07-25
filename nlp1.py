from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

model_name = 'distilbert-base-uncased-distilled-squad'

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForQuestionAnswering.from_pretrained(model_name)

context = 'Tomorrow is Sunday'

question = 'What day is Tomorrow?'

inputs = tokenizer(question, context, return_tensors='pt')

with torch.no_grad():
    output = model(**inputs)

start_index = torch.argmax(output.start_logits)

end_index = torch.argmax(output.end_logits) + 1

answer = tokenizer.convert_tokens_to_string(tokenizer.convert_tokens_to_ids(inputs['input_ids'][start_index:end_index],
                                                                            skip_special_tokens=True))

print(question)
print(answer)