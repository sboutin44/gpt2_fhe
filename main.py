import tiktoken

enc = tiktoken.get_encoding("gpt2")

text = """In a remote mountain village, nestled among the towering peaks and pine forests, \
a solitary storyteller sat by a crackling bonfire. Their voice rose and fell like a melodic river, \
weaving tales of ancient legends and forgotten heroes that captivated the hearts of the villagers gathered around. \
The stars above shone brightly, their twinkling adding a celestial backdrop to the mesmerizing stories, \
as the storyteller transported their audience to worlds of wonder and imagination. """

tokens = enc.encode(text)
tokens.append(enc.eot_token)
print(tokens[:10])

import torch

B,T = 5,10
data = torch.tensor(tokens[:50+1])

x = data[:-1].view(B,T)
y = data[1:].view(B,T)

print(x)
print(y)