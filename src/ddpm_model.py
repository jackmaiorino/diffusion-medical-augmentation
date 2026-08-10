"""Class-conditional UNet for the from-scratch DDPM."""
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel

from dataset import CLASSES

NUM_CLASSES = len(CLASSES)
# extra embedding used as the unconditional token for guidance
NULL_CLASS = NUM_CLASSES

# one noise schedule shared by training and sampling
SCHEDULER_KWARGS = dict(num_train_timesteps=1000,
                        beta_schedule='squaredcos_cap_v2')


def build_train_scheduler():
    """The noise schedule a checkpoint is trained under."""
    return DDPMScheduler(**SCHEDULER_KWARGS)


def build_sample_scheduler():
    """The sampling schedule, matching what training used."""
    return DDIMScheduler(**SCHEDULER_KWARGS)


def build_unet(image_size=64):
    """Build the 37.1M-parameter conditional UNet from random init."""
    return UNet2DModel(
        sample_size=image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 256, 256, 256),
        down_block_types=('DownBlock2D', 'DownBlock2D', 'AttnDownBlock2D',
                          'AttnDownBlock2D'),
        up_block_types=('AttnUpBlock2D', 'AttnUpBlock2D', 'UpBlock2D',
                        'UpBlock2D'),
        num_class_embeds=NUM_CLASSES + 1,
    )
