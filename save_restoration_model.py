from pathlib import Path

import torch


def save_restoration_model(
    generator, image_size, seed, output_path="restoration_generator.pt"
):
    output_path = Path(output_path)
    torch.save(
        {
            "model_state_dict": generator.state_dict(),
            "image_size": image_size,
            "seed": seed,
        },
        output_path,
    )
    print("Saved:", output_path.resolve())
    return output_path
