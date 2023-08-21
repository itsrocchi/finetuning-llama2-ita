#
# Script to perform inference on a LLM
#
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

tokenizer = AutoTokenizer.from_pretrained("seeweb/SeewebLLM-it-ver2")
model = AutoModelForCausalLM.from_pretrained(
    "seeweb/SeewebLLM-it-ver2",
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_8bit=True,
    rope_scaling={"type": "dynamic", "factor": 2} 
)

# eventualmente si possono modificare i parametri di model e tokenizer 
# inserendo il percorso assoluto della directory locale del modello
# eseguendo direttamente questo file il modello, se non presente in cache,
# verrà scaricato da internet

prompt = "### User:\nDescrivi cos' è l'intelligenza artificiale\n\n### Assistant:\n" 
#modificare ciò che è scritto tra "User" ed "assistant per personalizzare il prompt" 
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

output = model.generate(**inputs, streamer=streamer, use_cache=True, max_new_tokens=float('inf'))
output_text = tokenizer.decode(output[0], skip_special_tokens=True)
