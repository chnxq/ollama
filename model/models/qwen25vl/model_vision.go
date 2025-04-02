package qwen25vl

import (
	"math"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
)

var batchSize int = 1

// VisionSelfAttention implements self-attention for the Qwen vision model
type VisionSelfAttention struct {
	Query  *nn.Linear `gguf:"attn_q"`
	Key    *nn.Linear `gguf:"attn_k"`
	Value  *nn.Linear `gguf:"attn_v"`
	Output *nn.Linear `gguf:"attn_output"`
}

// Forward computes self-attention for the vision model
func (sa *VisionSelfAttention) Forward(ctx ml.Context, hiddenStates ml.Tensor, positionIDs ml.Tensor, opts *VisionModelOptions) ml.Tensor {
	query := sa.Query.Forward(ctx, hiddenStates)
	key := sa.Key.Forward(ctx, hiddenStates)
	value := sa.Value.Forward(ctx, hiddenStates)

	query = query.Reshape(ctx, opts.headDim, opts.numHeads, query.Dim(1), batchSize)
	key = key.Reshape(ctx, opts.headDim, opts.numHeads, key.Dim(1), batchSize)
	value = value.Reshape(ctx, opts.headDim, opts.numHeads, value.Dim(1), batchSize)

	// Apply rotary embeddings using RoPEMulti
	config := ml.RoPEConfig{
		Dim:        uint32(opts.headDim / 2),
		Type:       ml.RopeTypeVision,
		Base:       opts.ropeTheta,
		Scale:      1.0,
		YarnConfig: ml.DefaultYarnConfig(128000),
	}
	query = query.RoPEMulti(
		ctx,
		positionIDs,
		nil,
		[4]int{0, opts.headDim / 2, opts.headDim / 2, 0},
		config,
	)
	key = key.RoPEMulti(
		ctx,
		positionIDs,
		nil,
		[4]int{0, opts.headDim / 2, opts.headDim / 2, 0},
		config,
	)

	// Scale factor for scaled dot-product attention
	scale := 1.0 / math.Sqrt(float64(opts.headDim))

	attention := nn.Attention(ctx, query, key, value, scale, nil)
	attention = attention.Reshape(ctx, opts.hiddenSize, attention.Dim(2), batchSize)

	return sa.Output.Forward(ctx, attention)
}

// VisionMLP implements the MLP for the Qwen vision model
type VisionMLP struct {
	Gate *nn.Linear `gguf:"ffn_gate"`
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
}

// Forward computes the MLP for the vision model
func (mlp *VisionMLP) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts *VisionModelOptions) ml.Tensor {
	// Using GEGLU activation: (Gate * Up) * GELU(Gate)
	gateOutput := mlp.Gate.Forward(ctx, hiddenStates)
	upOutput := mlp.Up.Forward(ctx, hiddenStates)
	hiddenStates = gateOutput.GELU(ctx).Mul(ctx, upOutput)

	return mlp.Down.Forward(ctx, hiddenStates)
}

// VisionEncoderLayer implements an encoder layer for the Qwen vision model
type VisionEncoderLayer struct {
	AttentionNorm *nn.RMSNorm `gguf:"attn_norm"`
	SelfAttention *VisionSelfAttention
	FFNNorm       *nn.RMSNorm `gguf:"ffn_norm"`
	MLP           *VisionMLP
}

// Forward computes an encoder layer for the vision model
func (e *VisionEncoderLayer) Forward(ctx ml.Context, hiddenStates ml.Tensor, positionIDs ml.Tensor, opts *VisionModelOptions) ml.Tensor {
	residual := hiddenStates
	hiddenStates = e.AttentionNorm.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = e.SelfAttention.Forward(ctx, hiddenStates, positionIDs, opts)
	hiddenStates = hiddenStates.Add(ctx, residual)

	residual = hiddenStates
	hiddenStates = e.FFNNorm.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = e.MLP.Forward(ctx, hiddenStates, opts)

	return hiddenStates.Add(ctx, residual)
}

// VisionModelOptions contains configuration options for the Qwen vision model
type VisionModelOptions struct {
	hiddenSize       int
	numHeads         int
	headDim          int
	intermediateSize int
	imageSize        int
	patchSize        int
	numChannels      int
	eps              float32
	ropeTheta        float32
	outHiddenSize    int
}

// VisionPatchEmbedding implements patch embedding for the Qwen vision model
type VisionPatchEmbedding struct {
	PatchConv *nn.Conv2D `gguf:"patch_conv"`
}

// Forward computes patch embeddings for the vision model
func (pe *VisionPatchEmbedding) Forward(ctx ml.Context, pixelValues ml.Tensor, patchSize int) ml.Tensor {
	// Apply 2D convolution to extract patches
	embeddings := pe.PatchConv.Forward(ctx, pixelValues, patchSize, patchSize, 0, 0, 1, 1)

	// Reshape and permute as needed for the Qwen model
	height := pixelValues.Dim(0)
	width := pixelValues.Dim(1)

	numPatchesH := height / patchSize
	numPatchesW := width / patchSize
	numPatches := numPatchesH * numPatchesW

	embeddings = embeddings.Reshape(ctx, numPatches, embeddings.Dim(1))
	embeddings = embeddings.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)

	return embeddings
}

// VisionPatchMerger implements patch merging for the Qwen vision model
type VisionPatchMerger struct {
	LNQ *nn.RMSNorm `gguf:"ln_q"`
	MLP *nn.Linear  `gguf:"mlp"`
}

// Forward computes patch merging for the vision model
func (pm *VisionPatchMerger) Forward(ctx ml.Context, x ml.Tensor, outDim, contextDim, spatialMergeSize int) ml.Tensor {
	hiddenSize := contextDim * (spatialMergeSize * spatialMergeSize)

	// Normalize and reshape
	x = pm.LNQ.Forward(ctx, x, 1e-6)
	x = x.Reshape(ctx, -1, hiddenSize)

	// Apply MLP for merging
	x = pm.MLP.Forward(ctx, x)

	return x
}

// VisionModel implements the Qwen vision model
type VisionModel struct {
	PatchEmbedding *VisionPatchEmbedding
	EncoderNorm    *nn.RMSNorm          `gguf:"encoder_norm"`
	Layers         []VisionEncoderLayer `gguf:"blk"`
	PatchMerger    *VisionPatchMerger   `gguf:"patch_merger"`

	*VisionModelOptions
}

// Forward computes the vision model for an input tensor
func (m *VisionModel) Forward(ctx ml.Context, pixelValues ml.Tensor) ml.Tensor {
	// Extract patch embeddings
	hiddenStates := m.PatchEmbedding.Forward(ctx, pixelValues, m.patchSize)

	// Apply encoder normalization
	hiddenStates = m.EncoderNorm.Forward(ctx, hiddenStates, m.eps)

	// Calculate position IDs for 2D RoPE
	numPatchesH := pixelValues.Dim(0) / m.patchSize
	numPatchesW := pixelValues.Dim(1) / m.patchSize
	numPatches := numPatchesH * numPatchesW

	// Create position IDs - for 2D RoPE we need [h, w] pairs for each position
	positions := make([]int32, numPatches*2)

	for h := 0; h < numPatchesH; h++ {
		for w := 0; w < numPatchesW; w++ {
			idx := h*numPatchesW + w
			positions[idx*2] = int32(h)
			positions[idx*2+1] = int32(w)
		}
	}

	positionIDs, err := ctx.Input().FromIntSlice(positions, numPatches, 2)
	if err != nil {
		panic(err)
	}

	// Apply encoder layers
	for _, layer := range m.Layers {
		hiddenStates = layer.Forward(ctx, hiddenStates, positionIDs, m.VisionModelOptions)
	}

	// Apply patch merger if needed (for reducing dimensions to match text model)
	if m.PatchMerger != nil && m.outHiddenSize > 0 {
		hiddenStates = m.PatchMerger.Forward(ctx, hiddenStates, m.outHiddenSize, m.hiddenSize, 1)
	}

	return hiddenStates
}

// newVisionModel creates a new instance of the Qwen vision model
func newVisionModel(c ml.Config) *VisionModel {
	patchSize := int(c.Uint("vision.patch_size", 14))
	headDim := int(c.Uint("vision.attention.key_length", 64))
	ropeTheta := c.Float("vision.rope_theta", 10000.0)
	outHiddenSize := int(c.Uint("vision.out_embedding_length", 0))

	return &VisionModel{
		Layers: make([]VisionEncoderLayer, c.Uint("vision.block_count", 24)),
		VisionModelOptions: &VisionModelOptions{
			hiddenSize:       int(c.Uint("vision.embedding_length", 1152)),
			numHeads:         int(c.Uint("vision.attention.head_count", 16)),
			headDim:          headDim,
			intermediateSize: int(c.Uint("vision.feed_forward_length", 4608)),
			imageSize:        int(c.Uint("vision.image_size", 336)),
			patchSize:        patchSize,
			numChannels:      int(c.Uint("vision.num_channels", 3)),
			eps:              c.Float("vision.attention.layer_norm_epsilon", 1e-6),
			ropeTheta:        ropeTheta,
			outHiddenSize:    outHiddenSize,
		},
	}
}
