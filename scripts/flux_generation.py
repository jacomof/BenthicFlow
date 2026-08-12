import argparse
import os
from pathlib import Path

import torch
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers import (
    Flux2KleinPipeline,
    Flux2Pipeline,
    Flux2Transformer2DModel,
    FluxImg2ImgPipeline,
)
from diffusers.utils import load_image
from PIL import Image

# Three selectable variants (see build_pipeline):
#   klein:     FLUX.2-klein (9B), Qwen3 text encoder -> Flux2KleinPipeline, bf16.
#   dev:       FLUX.2-dev, Mistral3 text encoder -> Flux2Pipeline, transformer +
#              text encoder quantized to 4-bit (bitsandbytes) to fit one GPU.
#   flux1-dev: FLUX.1-dev img2img -> FluxImg2ImgPipeline, bf16, seeded on the
#              reference crop with a `strength` knob (as in flux_test_generation.py).
# The FLUX.2 variants need different pipeline classes: the Mistral3-only
# Flux2Pipeline loads klein with just a warning, then dies on klein's Qwen3 chat
# template ("can only concatenate str (not 'list') to str") when encoding.
KLEIN_ID = "black-forest-labs/FLUX.2-klein-9B"
DEV_ID = "black-forest-labs/FLUX.2-dev"
FLUX1_DEV_ID = "black-forest-labs/FLUX.1-dev"

# Prompt for underwater environments; shared by any script that drives
# the FLUX pipeline with this style (e.g. flux_test_generation.py).
REEF_PROMPT = (
    "Top-down nadir benthic survey photo of underwater sea floor."
    "Flat orthographic perspective with artificial strobe lighting."
    "Substrate showing hard corals, rubble, and sand."
    "Very slight water attenuation and marine snow backscatter. Scientific documentary style."
    "Change the distribution of the benthic substrate, but keep the same perspective, lighting, and style."
)


def resize_to_multiple_of_16(image, target_width=1024, target_height=1024):
    """Resizes a PIL image so its dimensions are perfectly divisible by 16,
    preventing pipeline shape mismatch errors."""
    width = (target_width // 16) * 16
    height = (target_height // 16) * 16
    return image.resize((width, height), Image.Resampling.LANCZOS)


def preprocess_and_resize(image_path, target_width=1024, target_height=1024):
    """Loads an image from a path/URL and snaps it to a multiple of 16."""
    if image_path.startswith("http://") or image_path.startswith("https://"):
        image = load_image(image_path).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")
    return resize_to_multiple_of_16(image, target_width, target_height)


def build_pipeline(model="klein"):
    """Build the pipeline for `model` ("klein", "dev", or "flux1-dev").
    klein:     Flux2KleinPipeline (Qwen3 text encoder), bf16 weights.
    dev:       Flux2Pipeline (Mistral3 text encoder). The transformer and the
    large (~24B) Mistral3 text encoder are loaded in 4-bit NF4
    (bitsandbytes) so the pair fits on one GPU; the VAE stays bf16.
    """
    if model == "klein":
        print("Loading FLUX.2 klein image pipeline (bf16)...")
        # Load the transformer on CPU first, then let accelerate stream layers to
        # the GPU. klein reuses dev's Flux2Transformer2DModel; only the text
        # encoder (Qwen3) differs, which is why the klein pipeline is needed.
        transformer = Flux2Transformer2DModel.from_pretrained(
            KLEIN_ID,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )
        pipe = Flux2KleinPipeline.from_pretrained(
            KLEIN_ID, transformer=transformer, torch_dtype=torch.bfloat16
        )
    elif model == "dev":
        print("Loading FLUX.2-dev image pipeline (4-bit bnb)...")
        # 4-bit NF4 for the two large components; VAE stays bf16. bitsandbytes
        # quantizes on the GPU (needs CUDA), and 4-bit params can't be .to()-moved,
        # so we don't pass device_map here and leave placement to cpu offload.
        from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
        from transformers import Mistral3ForConditionalGeneration

        bnb_diffusers = DiffusersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        bnb_transformers = TransformersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        transformer = Flux2Transformer2DModel.from_pretrained(
            DEV_ID,
            subfolder="transformer",
            quantization_config=bnb_diffusers,
            torch_dtype=torch.bfloat16,
        )
        text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
            DEV_ID,
            subfolder="text_encoder",
            quantization_config=bnb_transformers,
            torch_dtype=torch.bfloat16,
        )
        pipe = Flux2Pipeline.from_pretrained(
            DEV_ID,
            transformer=transformer,
            text_encoder=text_encoder,
            torch_dtype=torch.bfloat16,
        )
    elif model == "flux1-dev":
        print("Loading FLUX.1-dev img2img pipeline (bf16)...")
        # FLUX.1-dev img2img baseline (as in flux_test_generation.py): full bf16
        # weights, seeded on the reference crop with a `strength` knob. Uses the
        # older FLUX.1 FluxImg2ImgPipeline, not the FLUX.2 pipelines above.
        pipe = FluxImg2ImgPipeline.from_pretrained(
            FLUX1_DEV_ID, torch_dtype=torch.bfloat16
        )
    else:
        raise ValueError(
            f"Unknown model '{model}' (expected 'klein', 'dev', or 'flux1-dev')"
        )

    pipe.enable_model_cpu_offload()
    return pipe


def generate_reef_variation(
    input_image_path,
    output_image_path="generated_reef.png",
    model="klein",
    steps=28,
    guidance_scale=4.0,
    strength=0.6,
):
    pipe = build_pipeline(model)

    print(f"📸 Preprocessing input image: {input_image_path}")
    init_image = preprocess_and_resize(input_image_path)

    print(f"🌊 Generating reef variation with {model}. Please wait...")
    if model == "flux1-dev":
        # FLUX.1-dev img2img: a single seed image + `strength` (fraction of noise
        # added to the seed); higher strength drifts further from the reference.
        generated_image = pipe(
            prompt=REEF_PROMPT,
            image=init_image,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
        ).images[0]
    else:
        # FLUX.2 (klein/dev) conditions on reference image(s) passed as a list;
        # there is no img2img `strength` knob as in FLUX.1.
        generated_image = pipe(
            prompt=REEF_PROMPT,
            image=[init_image],
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
        ).images[0]

    # Save the output
    output_image_path = Path(output_image_path)
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    generated_image.save(output_image_path)
    print(f"✨ Success! Your new reef image has been saved to: {output_image_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a reef variation image with FLUX.2 klein, 4-bit FLUX.2-dev, or FLUX.1-dev."
    )
    parser.add_argument(
        "--model",
        choices=["klein", "dev", "flux1-dev"],
        default="klein",
        help="klein: FLUX.2-klein via Flux2KleinPipeline, bf16 (default). "
        "dev: FLUX.2-dev via Flux2Pipeline, 4-bit bnb quantized. "
        "flux1-dev: FLUX.1-dev img2img via FluxImg2ImgPipeline, bf16.",
    )
    parser.add_argument(
        "--input",
        default=(
            "./data_normalized/ScottReef200907/"
            "r20090727_085810_scott_07_grids_auv2/PR_20090727_123631_934_LC16.jpg"
        ),
        help="Reference image path or URL.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path. Default: ./figs/flux/reef_output_variation_<model>.png",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=28,
        help="Denoising steps (28 is a good quality/speed trade-off; "
        "the diffusers klein default is 50).",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=None,
        help="Guidance scale. Default: 4.0 (FLUX.2 klein/dev), 3.5 (flux1-dev).",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.6,
        help="img2img strength for flux1-dev (0-1; higher drifts further "
        "from the reference). Ignored for the FLUX.2 variants.",
    )
    args = parser.parse_args()

    # Per-model guidance default (FLUX.1-dev conventionally uses 3.5, FLUX.2 4.0).
    guidance_scale = args.guidance_scale
    if guidance_scale is None:
        guidance_scale = 3.5 if args.model == "flux1-dev" else 4.0

    # Default output keeps each model's image separate so one run doesn't
    # overwrite another (all are wanted for the paper).
    output_path = args.output or f"./figs/flux/reef_output_variation_{args.model}.png"

    generate_reef_variation(
        input_image_path=args.input,
        output_image_path=output_path,
        model=args.model,
        steps=args.steps,
        guidance_scale=guidance_scale,
        strength=args.strength,
    )
