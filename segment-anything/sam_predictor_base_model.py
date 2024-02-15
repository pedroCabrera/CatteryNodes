import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "segment-anything"))

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from segment_anything import sam_model_registry
from typing import Optional, Tuple

class SamPredictorBaseModel(nn.Module):

	"""
	A wrapper around the SAM model that allows it to be used as a TorchScript model.
	"""
	def __init__(
		self,
		variant: str,
		modelPath: str,
		position = torch.IntTensor([0, 0])
	) -> None:
		"""
		Uses SAM to calculate the image embedding for an image, and then
		allow repeated, efficient mask prediction given prompts.

		Arguments:
		  variant (str): The variant of the SAM model to use. Must be one of
			'vit_b', 'vit_l', or 'vit_h'.
		  modelPath (str): The path to the model checkpoint file.
		  position (torch.Tensor): The position of the prompt in the image,
			in (X, Y) format. The prompt is expected to be in the original
			image size, not the resized input size.
		"""
		super().__init__()
		# An instance of the model.
		self.model = sam_model_registry[variant](checkpoint=modelPath)	
		#self.model.to(device="cuda")
		self.target_length = self.model.image_encoder.img_size
		self.position = position
		self.original_size = (0,0)
		self.input_size = (0,0)
		self.mask_threshold = 0.0
		self.reset_image()
		self.device = torch.device('cuda')
		self.dtype = torch.float32
		
	def forward(
		self,
		image: torch.Tensor,
	):
		if(image.is_cuda):
		   self.device = torch.device('cuda')
		else:
		   self.device = torch.device('cpu')
		self.dtype = image.dtype

		input_point = self.position.to(self.device, dtype=self.dtype)
		
		# Unsqueeze center to add BxN dimension, where B=N=1
		input_label = torch.tensor([1]).to(self.device)


		image = image * 255.0
		image = torch.clip(image, 0.0, 255.0)

		self.set_torch_image(image.to(self.device))

		input_point[1] = image.shape[2] - input_point[1]

		point_coords = self.apply_coords_torch(input_point, self.original_size,self.target_length)[None, None, :]

		masks, scores, logits = self.predict_torch(
			point_coords=point_coords,
			point_labels=input_label[None, :],
			multimask_output=True,
		)
		# Sort scores and get indices
		sorted_scores, indices = torch.sort(scores[0], descending=True)

		# Use indices to sort masks
		sorted_masks = masks[0][indices]		
		catMasks = []
		# Remove batch dimensions from model ouputs
		for mask in sorted_masks:
			out = torch.zeros(size=self.original_size).to(self.device)
			value = torch.ones(size=self.original_size).to(self.device)
			out += value * mask  
			out = out[None, None, :]
			catMasks.append(out)
		cat = torch.cat(catMasks,1)    
		return  cat 

	@torch.no_grad()
	def set_torch_image(
		self,
		image: torch.Tensor,
	) -> None:
		"""
		Calculates the image embeddings for the provided image, allowing
		masks to be predicted with the 'predict' method. Expects the input
		image to be already transformed to the format expected by the model.

		Arguments:
		  transformed_image (torch.Tensor): The input image, with shape
			1x3xHxW, which has been transformed with ResizeLongestSide.
		  original_image_size (tuple(int, int)): The size of the image
			before transformation, in (H, W) format.
		"""       
		
		self.reset_image()
		self.original_size = (image.shape[-2],image.shape[-1])
		transformed_image = self.apply_image_torch(image)
		self.input_size = (transformed_image.shape[-2],transformed_image.shape[-1])
		input_image = self.preprocess(transformed_image)
		self.features = self.model.image_encoder(input_image)
		self.is_image_set = True

	# Typing hack to make the mock work
	def preprocess(self, x: torch.Tensor) -> torch.Tensor:
		"""Normalize pixel values and pad to a square input."""
		# Removed for now
		# Normalize colors
		pixel_mean = torch.tensor([123.675, 116.28, 103.53],dtype = x.dtype, device=self.device)
		pixel_std = torch.tensor([58.395, 57.12, 57.375], dtype = x.dtype, device=self.device)
		x = (x - pixel_mean[:, None, None]) / pixel_std[:, None, None]
		# x = (x - self.pixel_mean) / self.pixel_std

		# Pad
		h, w = x.shape[-2:]
		padh = self.target_length - h
		padw = self.target_length - w
		x = F.pad(x, (0, padw, 0, padh))
		return x

	@torch.no_grad()
	def predict_torch(
		self,
		point_coords: torch.Tensor=torch.Tensor(),
		point_labels: torch.Tensor=torch.Tensor(),
		boxes: Optional[torch.Tensor] = None,
		mask_input: Optional[torch.Tensor] = None,
		multimask_output: bool = True,
		return_logits: bool = False,
	) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""
		Predict masks for the given input prompts, using the currently set image.
		Input prompts are batched torch tensors and are expected to already be
		transformed to the input frame using ResizeLongestSide.

		Arguments:
		  point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the
			model. Each point is in (X,Y) in pixels.
		  point_labels (torch.Tensor or None): A BxN array of labels for the
			point prompts. 1 indicates a foreground point and 0 indicates a
			background point.
		  boxes (np.ndarray or None): A Bx4 array given a box prompt to the
			model, in XYXY format.
		  mask_input (np.ndarray): A low resolution mask input to the model, typically
			coming from a previous prediction iteration. Has form Bx1xHxW, where
			for SAM, H=W=256. Masks returned by a previous iteration of the
			predict method do not need further transformation.
		  multimask_output (bool): If true, the model will return three masks.
			For ambiguous input prompts (such as a single click), this will often
			produce better masks than a single prediction. If only a single
			mask is needed, the model's predicted quality score can be used
			to select the best mask. For non-ambiguous prompts, such as multiple
			input prompts, multimask_output=False can give better results.
		  return_logits (bool): If true, returns un-thresholded masks logits
			instead of a binary mask.

		Returns:
		  (torch.Tensor): The output masks in BxCxHxW format, where C is the
			number of masks, and (H, W) is the original image size.
		  (torch.Tensor): An array of shape BxC containing the model's
			predictions for the quality of each mask.
		  (torch.Tensor): An array of shape BxCxHxW, where C is the number
			of masks and H=W=256. These low res logits can be passed to
			a subsequent iteration as mask input.
		"""
		if not self.is_image_set:
			raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

		if point_coords is not None:
			points = (point_coords, point_labels)
		else:
			points = None

		# Embed prompts
		sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
			points=points,
			boxes=boxes,
			masks=mask_input,
		)

		# Predict masks
		low_res_masks, iou_predictions = self.model.mask_decoder(
			image_embeddings=self.features.to(self.device, dtype=self.dtype),
			image_pe=self.model.prompt_encoder.get_dense_pe().to(self.device, dtype=self.dtype),
			sparse_prompt_embeddings=sparse_embeddings.to(self.device, dtype=self.dtype),
			dense_prompt_embeddings=dense_embeddings.to(self.device, dtype=self.dtype),
			multimask_output=multimask_output,
		)

		# Upscale the masks to the original image resolution
		masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)

		if not return_logits:
			masks = masks > self.mask_threshold

		return masks, iou_predictions, low_res_masks

	def apply_image_torch(self, image: torch.Tensor, target_length: int = 1024) -> torch.Tensor:
		"""
		Expects batched images with shape BxCxHxW and float format. This
		transformation may not exactly match apply_image. apply_image is
		the transformation expected by the model.
		"""
		# Expects an image in BCHW format. May not exactly match apply_image.
		target_size = self.get_preprocess_shape(image.shape[2], image.shape[3], target_length)
		return F.interpolate(
			image, target_size, mode="bilinear", align_corners=False, antialias=True
		)

	def apply_coords_torch(self,
		coords: torch.Tensor, original_size: Tuple[int, int], target_length: int = 1024 ) -> torch.Tensor:
		"""
		Expects a torch tensor with length 2 in the last dimension. Requires the
		original image size in (H, W) format.
		"""
		old_h, old_w = original_size
		new_h, new_w = self.get_preprocess_shape(
			original_size[0], original_size[1], target_length
		)
		coords = coords.to(torch.float)
		coords[..., 0] = coords[..., 0] * (new_w / old_w)
		coords[..., 1] = coords[..., 1] * (new_h / old_h)
		return coords

	def apply_boxes_torch(self,
		boxes: torch.Tensor, original_size: Tuple[int, int] ) -> torch.Tensor:
		"""
		Expects a torch tensor with shape Bx4. Requires the original image
		size in (H, W) format.
		"""
		boxes = self.apply_coords_torch(boxes.reshape(-1, 2, 2), original_size)
		return boxes.reshape(-1, 4)

	def get_preprocess_shape(self,oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
		"""
		Compute the output size given input size and target long side length.
		"""
		scale = long_side_length * 1.0 / max(oldh, oldw)
		newh, neww = oldh * scale, oldw * scale
		neww = int(neww + 0.5)
		newh = int(newh + 0.5)
		return (newh, neww)

	def reset_image(self) -> None:
		"""Resets the currently set image."""
		self.is_image_set = False
		self.features = torch.empty((2,3), dtype=torch.int64)
