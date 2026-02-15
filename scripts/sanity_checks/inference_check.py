from transformers import AutoTokenizer, AutoModelForCausalLM
#from vllm import LLM, SamplingParams

MODEL_NAME = "facebook/opt-125m"
PROMPTS = ["Hello, my name is", "The capital of Portugal is"]


def test_hftrans():
    print("=== Testing with HuggingFace Transformers ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    
    inputs = tokenizer(PROMPTS, return_tensors="pt", padding=True)
    outputs = model.generate(**inputs, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id)
    
    for i, prompt in enumerate(PROMPTS):
        result = tokenizer.decode(outputs[i], skip_special_tokens=True)
        print(f"Prompt: {prompt}")
        print(f"Output: {result}\n")


def test_vllm():
    print("=== Testing with vLLM ===")
    llm = LLM(model=MODEL_NAME)
    sampling_params = SamplingParams(max_tokens=20)
    
    outputs = llm.generate(PROMPTS, sampling_params)
    
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt}")
        print(f"Output: {prompt}{generated_text}\n")


if __name__ == "__main__":
    test_hftrans()
    #test_vllm()
