import gradio as gr
from profanity_check import predict_prob
import time

def mask_profanity(message):
    probability = predict_prob([message])[0]
    if probability > 0.7:
        masked_message = '*' * len(message)
    else:
        masked_message = message
    return masked_message

def slow_echo(message, history):
    for i in range(len(message)):
        time.sleep(0.05)
        yield "You typed: " + mask_profanity(message[: i + 1])

demo = gr.ChatInterface(slow_echo).queue()

if __name__ == "__main__":
    demo.launch()