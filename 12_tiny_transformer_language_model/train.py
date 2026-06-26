from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


DATA_PATH = Path(__file__).parent / "data" / "tiny_stories.txt"


def choose_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# A simple character-level tokenizer that maps each unique character to a unique integer ID.
class CharacterTokenizer:
    def __init__(self, text):
        self.characters = sorted(set(text))
        self.char_to_id = {character: index for index, character in enumerate(self.characters)}
        self.id_to_char = {index: character for character, index in self.char_to_id.items()}

    @property
    def vocab_size(self):
        return len(self.characters)

    def encode(self, text):
        return [self.char_to_id[character] for character in text]

    def decode(self, token_ids):
        return "".join(self.id_to_char[int(token_id)] for token_id in token_ids)


class CausalSelfAttention(nn.Module):
    #embed_dim: the dimensionality of the input embeddings
    #num_heads: the number of attention heads to use
    #block_size: the maximum sequence length that the model can process, maximum context length
    #dropout: the dropout rate to apply to the attention weights
    def __init__(self, embed_dim, num_heads, block_size, dropout):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.num_heads = num_heads
        #head_dim: the dimensionality of each attention head, which is equal to embed_dim divided by num_heads
        self.head_dim = embed_dim // num_heads
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.output = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        #create a lower triangular matrix of ones to serve as a causal mask,
        #  which prevents the model from attending to future tokens in the sequence. 
        # The mask is registered as a buffer so that it is not treated as a learnable parameter.
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, block_size, block_size))
    # The forward method takes an input tensor x of shape 
    # [batch_size, sequence_length, embed_dim] and 
    # computes the self-attention output. 
    # It first computes the query, key, and value matrices using linear transformations. 
    # Then, it reshapes these matrices to separate the attention heads and 
    # computes the attention scores using the dot product of queries and keys. 
    # The causal mask is applied to prevent attending to future tokens. 
    # The attention weights are computed using softmax, and dropout is applied for regularization.
    #  Finally, the attended values are combined and passed through a linear layer to produce the final output.
    def forward(self, x):
        batch_size, sequence_length, embed_dim = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Reshape the query, key, and value tensors to separate the attention heads.
        # The new shape is [batch_size, num_heads, sequence_length, head_dim].
        q = q.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)

        attention_scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        mask = self.causal_mask[:, :, :sequence_length, :sequence_length]
        attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        attended = attention_weights @ v
        attended = attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, embed_dim)
        return self.output(attended)

# A feed-forward network that consists of two linear layers with a ReLU activation in between,
# and dropout for regularization.
class FeedForward(nn.Module):
    def __init__(self, embed_dim, dropout):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size, dropout):
        super().__init__()
        self.attention_norm = nn.LayerNorm(embed_dim)
        self.attention = CausalSelfAttention(embed_dim, num_heads, block_size, dropout)
        self.feed_forward_norm = nn.LayerNorm(embed_dim)
        self.feed_forward = FeedForward(embed_dim, dropout)

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.feed_forward_norm(x))
        return x


class TinyTransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, block_size, embed_dim, num_heads, num_layers, dropout):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(embed_dim, num_heads, block_size, dropout)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(embed_dim)
        self.output_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, token_ids, targets=None):
        batch_size, sequence_length = token_ids.shape
        if sequence_length > self.block_size:
            raise ValueError("sequence length is larger than block_size")

        positions = torch.arange(sequence_length, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        x = self.blocks(x)
        x = self.final_norm(x)
        logits = self.output_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(batch_size * sequence_length, -1),
                targets.view(batch_size * sequence_length),
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, token_ids, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            context = token_ids[:, -self.block_size :]
            logits, _ = self(context)
            logits = logits[:, -1, :] / temperature
            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            token_ids = torch.cat([token_ids, next_token], dim=1)
        return token_ids


def make_batch(data, batch_size, block_size, device):
    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[start : start + block_size] for start in starts])
    y = torch.stack([data[start + 1 : start + block_size + 1] for start in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, device, eval_batches):
    model.eval()
    losses = {}
    for split_name, data in [("train", train_data), ("val", val_data)]:
        split_losses = []
        for _ in range(eval_batches):
            x, y = make_batch(data, batch_size, block_size, device)
            _, loss = model(x, y)
            split_losses.append(loss.item())
        losses[split_name] = sum(split_losses) / len(split_losses)
    model.train()
    return losses


def parse_args():
    parser = argparse.ArgumentParser(description="Train a tiny character transformer.")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--generate-chars", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--prompt", type=str, default="The model")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(42)

    device = choose_device()
    text = DATA_PATH.read_text()
    tokenizer = CharacterTokenizer(text)
    encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    split_index = int(len(encoded) * 0.9)
    train_data = encoded[:split_index]
    val_data = encoded[split_index:]

    model = TinyTransformerLanguageModel(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print("Tiny transformer language model")
    print(f"Device:       {device}")
    print(f"Characters:   {len(text)}")
    print(f"Vocabulary:   {tokenizer.vocab_size}")
    print(f"Parameters:   {parameter_count:,}")
    print(f"Block size:   {args.block_size}")
    print()

    for step in range(args.max_steps + 1):
        if step % args.eval_interval == 0 or step == args.max_steps:
            losses = estimate_loss(
                model,
                train_data,
                val_data,
                args.batch_size,
                args.block_size,
                device,
                args.eval_batches,
            )
            print(
                f"step {step:4d} | "
                f"train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f}"
            )

        x, y = make_batch(train_data, args.batch_size, args.block_size, device)
        _, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    prompt = args.prompt
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    generated_ids = model.generate(
        prompt_ids,
        max_new_tokens=args.generate_chars,
        temperature=args.temperature,
    )[0]

    print()
    print("Generated text")
    print(tokenizer.decode(generated_ids.tolist()))


if __name__ == "__main__":
    main()
