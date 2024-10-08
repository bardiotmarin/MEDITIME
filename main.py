from transformers import AutoTokenizer, AutoModelForCausalLM
from api_handler import find_specialist
import torch
import wandb
import random
import os

access_token = os.environ.get('HF_TOKEN')


# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="MEDITIME",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.02,
    "architecture": "CNN",
    "dataset": "CIFAR-100",
    "epochs": 10,
    }
)


# simulate training
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset

    # log metrics to wandb
    wandb.log({"acc": acc, "loss": loss})

# [optional] finish the wandb run, necessary in notebooks
wandb.finish()


# Meditron le boss
def load_meditron():
    print("Loading Meditron model...")
    tokenizer = AutoTokenizer.from_pretrained("epfl-llm/meditron-70b", token=access_token)
    model = AutoModelForCausalLM.from_pretrained("epfl-llm/meditron-70b", token=access_token)
    return tokenizer, model
# gen une réponse  basée sur l'input de l'utilisateur

def generate_response(user_input, tokenizer, model):
    inputs = tokenizer.encode(user_input, return_tensors="pt")
    outputs = model.generate(inputs, max_length=100, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# on grab l'input
def get_user_input():
    print("How can I assist you with your health concern today buddy?")
    return input("Describe your health issue")

def main():
    tokenizer, model = load_meditron()

    while True:
        user_input = get_user_input()
        if user_input.lower() == "exit":
            print("Exiting chatbot. Take care!")
            break
        
        # Generate a response from Meditron
        bot_response = generate_response(user_input, tokenizer, model)
        print(f"Bot: {bot_response}")
        
        # Call API to find relevant specialists
        specialist = find_specialist(bot_response)
        print(f"Recommended Specialist: {specialist}")

if __name__ == "__main__":
    main()
