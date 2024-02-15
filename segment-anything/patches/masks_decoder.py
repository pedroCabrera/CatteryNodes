from typing import List, Tuple
import mock
import torch
from segment_anything.modeling.mask_decoder import MaskDecoder


def forward(
    self,
    image_embeddings: torch.Tensor,
    image_pe: torch.Tensor,
    sparse_prompt_embeddings: torch.Tensor,
    dense_prompt_embeddings: torch.Tensor,
    multimask_output: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Predict masks given image and prompt embeddings.

    Arguments:
      image_embeddings (torch.Tensor): the embeddings from the image encoder
      image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
      sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
      dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
      multimask_output (bool): Whether to return multiple masks or a single
        mask.

    Returns:
      torch.Tensor: batched predicted masks
      torch.Tensor: batched predictions of mask quality
    """
    masks, iou_pred = self.predict_masks(
        image_embeddings=image_embeddings,
        image_pe=image_pe,
        sparse_prompt_embeddings=sparse_prompt_embeddings,
        dense_prompt_embeddings=dense_prompt_embeddings,
    )

    # Select the correct mask or masks for output
    if multimask_output:
        # mask_slice = slice(1, None)
        masks = masks[:, slice(1, None), :, :]
        iou_pred = iou_pred[:, slice(1, None)]
    else:
        # mask_slice = slice(0, 1)
        masks = masks[:, slice(0, 1), :, :]
        iou_pred = iou_pred[:, slice(0, 1)]
    # masks = masks[:, mask_slice, :, :]
    # iou_pred = iou_pred[:, mask_slice]

    # Prepare output
    return masks, iou_pred


def predict_masks(
    self,
    image_embeddings: torch.Tensor,
    image_pe: torch.Tensor,
    sparse_prompt_embeddings: torch.Tensor,
    dense_prompt_embeddings: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Predicts masks. See 'forward' for more details."""
    # Concatenate output tokens
    output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
    output_tokens = output_tokens.unsqueeze(0).expand(
        sparse_prompt_embeddings.size(0), -1, -1
    )
    tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

    # Expand per-image data in batch direction to be per-mask
    src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
    src = src + dense_prompt_embeddings
    pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
    b, c, h, w = src.shape

    # Run the transformer
    hs, src = self.transformer(src, pos_src, tokens)
    iou_token_out = hs[:, 0, :]
    mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

    # Upscale mask embeddings and predict masks using the mask tokens
    src = src.transpose(1, 2).view(b, c, h, w)
    upscaled_embedding = self.output_upscaling(src)
    hyper_in_list: List[torch.Tensor] = []
    for i, func in enumerate(self.output_hypernetworks_mlps):
        hyper_in_list.append(func(mask_tokens_out[:, i, :]))
    hyper_in = torch.stack(hyper_in_list, dim=1)
    b, c, h, w = upscaled_embedding.shape
    masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

    # Generate mask quality predictions
    iou_pred = self.iou_prediction_head(iou_token_out)

    return masks, iou_pred


patches = (
    mock.patch.object(MaskDecoder, "forward", forward),
    mock.patch.object(MaskDecoder, "predict_masks", predict_masks),
)