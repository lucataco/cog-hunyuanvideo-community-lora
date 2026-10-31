# hunyuanvideo-community/HunyuanVideo LoRA Cog Model

This is an implementation of [hunyuanvideo-community/HunyuanVideo](https://huggingface.co/hunyuanvideo-community/HunyuanVideo) as a [Cog](https://github.com/replicate/cog) model to explore LoRAs.

## Development

Follow the [model pushing guide](https://replicate.com/docs/guides/push-a-model) to push your own model to [Replicate](https://replicate.com).


## How to use

Make sure you have [cog](https://github.com/replicate/cog) installed.

To run a prediction:

    cog predict -i prompt="Style of snomexut, A vibrant yellow Tuxemon, wielding a fiery blue staff, joyfully manipulates swirling flames around its cheerful form." -i extra_lora="a-r-r-o-w/HunyuanVideo-tuxemons"

![Output](output.gif)

## License

The code in this repository is licensed under the [Apache License 2.0](LICENSE).

The model weights are not covered by this license. [hunyuanvideo-community/HunyuanVideo](https://huggingface.co/hunyuanvideo-community/HunyuanVideo) is released under the [Tencent Hunyuan Community License](https://huggingface.co/tencent/HunyuanVideo/blob/main/LICENSE), which doesn't apply in the EU, the UK or South Korea. Check its terms before use.
