# hunyuanvideo-community/HunyuanVideo LoRA Cog Model

This is an implementation of [hunyuanvideo-community/HunyuanVideo](https://huggingface.co/hunyuanvideo-community/HunyuanVideo) as a [Cog](https://github.com/replicate/cog) model to explore LoRAs.

## Development

Follow the [model pushing guide](https://replicate.com/docs/guides/push-a-model) to push your own model to [Replicate](https://replicate.com).


## How to use

Make sure you have [cog](https://github.com/replicate/cog) installed.

To run a prediction:

    cog predict -i prompt="Style of snomexut, a cat-like Tuxemon creature walks in alien-world grass, and observes its surroundings" -i extra_lora="a-r-r-o-w/HunyuanVideo-tuxemons"

![Output](output.gif)