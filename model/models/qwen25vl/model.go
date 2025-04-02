package qwen25vl

import (
	"bytes"
	"image"

	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
)

type Model struct {
	model.Base
	*TextModel
	// *VisionModel         `gguf:"v,vision"`

	ImageProcessor
}

// Implement MultimodalProcessor interface
var _ model.MultimodalProcessor = (*Model)(nil)

func New(c ml.Config) (model.Model, error) {
	m := &Model{
		TextModel: NewTextModel(c),
		// VisionModel:         newVisionModel(c),
		ImageProcessor: newImageProcessor(c),
	}

	m.Cache = kvcache.NewCausalCache(m.TextModel.Shift)

	return m, nil
}

func (m *Model) EncodeMultimodal(ctx ml.Context, multimodalData []byte) (any, error) {
	// if len(m.VisionModel.Layers) == 0 {
	// 	return nil, model.ErrNoVisionModel
	// }

	image, _, err := image.Decode(bytes.NewReader(multimodalData))
	if err != nil {
		return nil, err
	}

	f32s, err := m.ImageProcessor.ProcessImage(image)
	if err != nil {
		return nil, err
	}

	_, err = ctx.Input().FromFloatSlice(f32s,
		m.ImageProcessor.imageSize,
		m.ImageProcessor.imageSize,
		m.ImageProcessor.numChannels,
	)
	if err != nil {
		return nil, err
	}

	return nil, nil

	// visionOutputs := m.VisionModel.Forward(ctx, pixelValues)
	// return visionOutputs, nil
}

// PostTokenize arranges Qwen-2.5-VL's inputs for the forward pass
func (m *Model) PostTokenize(inputs []input.Input) ([]input.Input, error) {
	var result []input.Input

	// Get image token IDs from config
	// imageToken := m.Config.Uint("image_token_id")
	// visionStartToken := m.Config.Uint("vision_start_token_id")
	// visionEndToken := m.Config.Uint("vision_end_token_id")
	imageToken := 151655
	visionStartToken := 151652
	visionEndToken := 151653

	// Get merge size from vision config
	// mergeSize := m.Config.Uint("vision_config.spatial_merge_size")
	// patchSize := m.Config.Uint("vision_config.spatial_patch_size")
	// windowSize := m.Config.Uint("vision_config.window_size")
	mergeSize := 2
	patchSize := 14
	windowSize := 112

	// Calculate grid dimensions
	patchesPerDim := windowSize / patchSize
	gridSize := patchesPerDim / mergeSize

	tokensPerGrid := gridSize * gridSize

	for _, inp := range inputs {
		if inp.Multimodal == nil {
			// If not a multimodal input, add it to the result unchanged
			result = append(result, inp)
		} else if inp.Token == int32(imageToken) {
			// This is an image token
			inputMultimodal := inp.Multimodal.(ml.Tensor)

			// Replace the image token with multiple placeholder tokens
			// First add the vision start token
			result = append(result, input.Input{Token: int32(visionStartToken)})

			// Then add the multimodal tensor data at the first position
			result = append(result,
				input.Input{
					Multimodal:     inputMultimodal,
					MultimodalHash: inp.MultimodalHash,
				})

			// Then add the placeholder tokens for the remaining positions
			// Subtract 1 from tokensPerGrid because we already added the first token
			placeholders := tokensPerGrid - 1
			for range int(placeholders) {
				result = append(result, input.Input{Token: int32(imageToken)})
			}

			// Finally add the vision end token
			result = append(result, input.Input{Token: int32(visionEndToken)})
		} else {
			// For any other token, just pass through
			result = append(result, inp)
		}
	}

	return result, nil
}

func (m *Model) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	positions, err := ctx.Input().FromIntSlice(batch.Positions, len(batch.Positions))
	if err != nil {
		return nil, err
	}

	outputs, err := ctx.Input().FromIntSlice(batch.Outputs, len(batch.Outputs))
	if err != nil {
		return nil, err
	}

	return m.TextModel.Forward(ctx, batch.Inputs, positions, outputs, batch, m.Cache)
}

func init() {
	model.Register("qwen25vl", New)
}
