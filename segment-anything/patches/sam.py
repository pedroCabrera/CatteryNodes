from typing import Dict, List, Tuple, Union
import mock
import torch
from torch.nn import functional as F
from segment_anything.modeling.sam import Sam


@torch.no_grad()
def forward(
    self,
    batched_images: torch.Tensor,
    batched_input: List[Dict[str, torch.Tensor]],
    multimask_output: bool,
) -> List[Dict[str, torch.Tensor]]:
    """
    Predicts masks end-to-end from provided images and prompts.
    If prompts are not known in advance, using SamPredictor is
    recommended over calling the model directly.

    Arguments:
        batched_images (torch.Tensor): Batched input images, in
        shape Bx3xHxW.
        batched_input (list(dict)): A list over input images, each a
        dictionary with the following keys. A prompt key can be
        excluded if it is not present.
            'original_size': (torch.Tensor) The original size of the
            image, as (H, W).
            'point_coords': (torch.Tensor) Batched point prompts for
            this image, with shape BxNx2. Already transformed to the
            input frame of the model.
            'point_labels': (torch.Tensor) Batched labels for point prompts,
            with shape BxN.
            'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
            Already transformed to the input frame of the model.
            'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
            in the form Bx1xHxW.
        the image before transformation, as (H, W).
        multimask_output (bool): Whether the model should predict multiple
        disambiguating masks, or return a single mask.

    Returns:
        (list(dict)): A list over input images, where each element is
        as dictionary with the following keys.
            'masks': (torch.Tensor) Batched binary mask predictions,
            with shape BxCxHxW, where B is the number of input prompts,
            C is determined by multimask_output, and (H, W) is the
            original size of the image.
            'iou_predictions': (torch.Tensor) The model's predictions
            of mask quality, in shape BxC.
            'low_res_logits': (torch.Tensor) Low resolution logits with
            shape BxCxHxW, where H=W=256. Can be passed as mask input
            to subsequent iterations of prediction.
    """
    # input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
    image_embeddings = self.image_encoder(batched_images)

    outputs: List[Dict[str, torch.Tensor]] = []
    for i, (image_record, curr_embedding) in enumerate(
        zip(batched_input, image_embeddings)
    ):
        if "point_coords" in image_record:
            points = (image_record["point_coords"], image_record["point_labels"])
        else:
            points = None
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=image_record["boxes"] if "boxes" in image_record else None,
            masks=image_record["mask_inputs"]
            if "mask_inputs" in image_record
            else None,
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=curr_embedding.unsqueeze(0),
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )
        img_shape = batched_images[i].shape[-2:]
        masks = self.postprocess_masks(
            low_res_masks,
            input_size=(img_shape[0], img_shape[1]),
            original_size=(
                image_record["original_size"][0],
                image_record["original_size"][1],
            ),
        )
        # masks = masks > self.mask_threshold
        masks = masks > 0.0

        outputs.append(
            {
                "masks": masks,
                "iou_predictions": iou_predictions,
                "low_res_logits": low_res_masks,
            }
        )
    return outputs


def postprocess_masks(
    self,
    masks: torch.Tensor,
    input_size: Tuple[int, int],
    original_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Remove padding and upscale masks to the original image size.

    Arguments:
        masks (torch.Tensor): Batched masks from the mask_decoder,
        in BxCxHxW format.
        input_size (tuple(int, int)): The size of the image input to the
        model, in (H, W) format. Used to remove padding.
        original_size (tuple(int, int)): The original size of the image
        before resizing for input to the model, in (H, W) format.

    Returns:
        (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
        is given by original_size.
    """
    masks = F.interpolate(
        masks,
        (self.image_encoder.img_size, self.image_encoder.img_size),
        mode="bilinear",
        align_corners=False,
    )
    masks = masks[..., : input_size[0], : input_size[1]]
    masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
    return masks


# Typing hack to make the mock work
def preprocess(self, x: torch.Tensor) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Removed for now
    # Normalize colors
    pixel_mean = torch.tensor([123.675, 116.28, 103.53],dtype =x.dtype, device=x.device)
    pixel_std = torch.tensor([58.395, 57.12, 57.375],dtype =x.dtype, device=x.device)
    x = (x - pixel_mean[:, None, None]) / pixel_std[:, None, None]
    # x = (x - self.pixel_mean) / self.pixel_std

    # Pad
    h, w = x.shape[-2:]
    padh = self.image_encoder.img_size - h
    padw = self.image_encoder.img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


@property
def device(self) -> torch.device:
    return self.pixel_mean.device


patches = (
    mock.patch.object(Sam, "forward", forward),
    mock.patch.object(Sam, "postprocess_masks", postprocess_masks),
    mock.patch.object(Sam, "preprocess", preprocess),
    mock.patch.object(Sam, "device", device),
)
