# Interface Contracts

Last updated: 2026-06-15

These contracts define the boundaries between data, model, training, and evaluation agents. Keep code changes compatible with these shapes unless this document is updated first.

Day1-Day2 are only the first part of the global plan, so these contracts are intentionally minimal and debug-friendly.

## Manifest

Each manifest line is JSONL:

```json
{
  "image_id": "000001",
  "image_path": "images/000001.jpg",
  "eeg_path": "eeg/000001.npy",
  "caption": "a photo of a red object",
  "label": 0
}
```

Paths are relative to the configured data root.

## Dataset Item

`Dataset.__getitem__` returns:

```python
{
    "image": FloatTensor[3, 224, 224],
    "eeg": FloatTensor[64, 250],
    "caption": str,
    "image_id": str,
    "label": int,
}
```

EEG should be z-score normalized per sample unless a later real-data preprocessing decision overrides this.

## Collated Batch

The collate function returns:

```python
batch = {
    "image": FloatTensor[B, 3, 224, 224],
    "eeg": FloatTensor[B, 64, 250],
    "caption": List[str],
    "image_id": List[str],
    "label": LongTensor[B],
}
```

If tokenization happens in collate or training, append:

```python
batch["input_ids"] = LongTensor[B, L]
batch["attention_mask"] = LongTensor[B, L]
batch["labels"] = LongTensor[B, L]
```

## Vision Encoder

Class:

```python
class FrozenCLIPVisionEncoder(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        ...
```

Input:

```python
images: FloatTensor[B, 3, 224, 224]
```

Output:

```python
image_emb: FloatTensor[B, D_img]
```

Debug target: `D_img = 512` for `openai/clip-vit-base-patch32`. All CLIP weights are frozen.

## EEG Encoder

Class:

```python
class EEGEncoder(nn.Module):
    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        ...
```

Input:

```python
eeg: FloatTensor[B, 64, 250]
```

Output:

```python
eeg_emb: FloatTensor[B, D_eeg]
```

Debug target: `D_eeg = 512`. Use a lightweight Conv1d plus small Transformer/MLP path first.

## Fusion

Class:

```python
class GatedFusion(nn.Module):
    def forward(self, image_emb: torch.Tensor, eeg_emb: torch.Tensor | None = None) -> torch.Tensor:
        ...
```

Inputs:

```python
image_emb: FloatTensor[B, 512]
eeg_emb: FloatTensor[B, 512] | None
```

Output:

```python
fused_emb: FloatTensor[B, 512]
```

Required first version:

```python
gate = sigmoid(MLP(concat(image_emb, eeg_emb)))
eeg_delta = MLP(eeg_emb)
fused_emb = image_emb + gate * eeg_delta
```

If `eeg_emb is None`, return an image-only embedding compatible with the caption model.

## Caption Model

The caption wrapper consumes an embedding and supervised captions.

Training interface:

```python
loss = caption_model(
    conditioning_emb=FloatTensor[B, 512],
    captions=List[str],
)
```

Expected output:

```python
loss: scalar Tensor
```

Generation interface:

```python
predictions = caption_model.generate(
    conditioning_emb=FloatTensor[B, 512],
    max_new_tokens=int,
)
```

Expected output:

```python
predictions: List[str]
```

Soft prompt projector output:

```python
soft_prompt: FloatTensor[B, K, D_llm]
```

`K = config.model.prompt_tokens` in the current config. `D_llm` is read from the selected LLM hidden size. The LLM is frozen by default.

## Training Entry Points

Baseline:

```bash
python -m src.train.train_baseline --config configs/debug.yaml
```

Fusion:

```bash
python -m src.train.train_fusion --config configs/debug.yaml
```

Training scripts must save checkpoints under `outputs/debug/` and log finite loss.

## Evaluation Modes

Generation mode names are part of the public interface:

```text
image_only
real_eeg
shuffled_eeg
random_eeg
```

Each generated JSONL record should use:

```json
{
  "image_id": "000001",
  "mode": "real_eeg",
  "reference": "a photo of a red object",
  "prediction": "a red object is shown"
}
```
