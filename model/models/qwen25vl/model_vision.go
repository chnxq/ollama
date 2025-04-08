package qwen25vl

import (
	"fmt"
	"math"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
)

var batchSize int = 1

// VisionSelfAttention implements self-attention for the Qwen vision model
type VisionSelfAttention struct {
	Query  *nn.Linear `gguf:"attn_q"`
	Key    *nn.Linear `gguf:"attn_k"`
	Value  *nn.Linear `gguf:"attn_v"`
	Output *nn.Linear `gguf:"attn_out"`
}

// Forward computes self-attention for the vision model
func (sa *VisionSelfAttention) Forward(ctx ml.Context, hiddenStates ml.Tensor, positionIDs ml.Tensor, opts *VisionModelOptions) ml.Tensor {
	query := sa.Query.Forward(ctx, hiddenStates)
	key := sa.Key.Forward(ctx, hiddenStates)
	value := sa.Value.Forward(ctx, hiddenStates)

	// Add debug prints for query dimensions
	fmt.Printf("DEBUG: query dimensions before reshape: %v\n",
		[]int{query.Dim(0), query.Dim(1), query.Dim(2), query.Dim(3)})

	query = query.Reshape(ctx, opts.headDim, opts.numHeads, query.Dim(1), batchSize)
	key = key.Reshape(ctx, opts.headDim, opts.numHeads, key.Dim(1), batchSize)
	value = value.Reshape(ctx, opts.headDim, opts.numHeads, value.Dim(1), batchSize)

	// Add debug prints after reshape
	fmt.Printf("DEBUG: query dimensions after reshape: %v\n",
		[]int{query.Dim(0), query.Dim(1), query.Dim(2), query.Dim(3)})

	// Check if positionIDs is a vector with expected dimensions
	numPositions := query.Dim(2)
	expectedPosIDLen := numPositions * 4 // 4 position IDs per token as required by mrope

	// Debug print expected vs actual dimensions
	fmt.Printf("DEBUG: Expected positionIDs length: %d, Actual: %d\n",
		expectedPosIDLen, positionIDs.Dim(0))

	// Apply rotary embeddings using RoPEMulti
	config := ml.RoPEConfig{
		Dim:        uint32(opts.headDim / 2),
		Type:       ml.RopeTypeMRoPE,
		Base:       opts.ropeTheta,
		Scale:      1.0,
		YarnConfig: ml.DefaultYarnConfig(128000),
	}

	// Ensure positionIDs is a proper 1D vector of int32 with correct length
	// This is likely where you need to fix your code
	fmt.Printf("DEBUG: About to call RoPEMulti\n")

	query = query.RoPEMulti(
		ctx,
		positionIDs,
		nil,
		[4]int{opts.headDim / 4, opts.headDim / 4, opts.headDim / 4, opts.headDim / 4},
		config,
	)
	key = key.RoPEMulti(
		ctx,
		positionIDs,
		nil,
		[4]int{opts.headDim / 4, opts.headDim / 4, opts.headDim / 4, opts.headDim / 4},
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
	MLP           *VisionMLP
}

// Forward computes an encoder layer for the vision model
func (e *VisionEncoderLayer) Forward(ctx ml.Context, hiddenStates ml.Tensor, positionIDs ml.Tensor, opts *VisionModelOptions) ml.Tensor {
	residual := hiddenStates
	if e.AttentionNorm != nil {
		hiddenStates = e.AttentionNorm.Forward(ctx, hiddenStates, opts.eps)
	}
	hiddenStates = e.SelfAttention.Forward(ctx, hiddenStates, positionIDs, opts)
	hiddenStates = hiddenStates.Add(ctx, residual)

	residual = hiddenStates
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
	PatchConv0 *nn.Conv2D `gguf:"patch_embd_0"`
	PatchConv1 *nn.Conv2D `gguf:"patch_embd_1"`
}

// Forward computes patch embeddings for the vision model
func (pe *VisionPatchEmbedding) Forward(ctx ml.Context, pixelValues ml.Tensor, patchSize int) ml.Tensor {
	// Apply both 2D convolutions
	embeddings0 := pe.PatchConv0.Forward(ctx, pixelValues, patchSize, patchSize, 0, 0, 1, 1)
	embeddings1 := pe.PatchConv1.Forward(ctx, pixelValues, patchSize, patchSize, 0, 0, 1, 1)

	// Add the two embeddings
	embeddings := embeddings0.Add(ctx, embeddings1)

	// Get dimensions
	patchesH := pixelValues.Dim(0) / patchSize
	patchesW := pixelValues.Dim(1) / patchSize
	hiddenSize := embeddings0.Dim(2) // The channel dimension after convolution

	// Print initial dimensions for debugging
	fmt.Printf("Initial embeddings shape: [%d, %d, %d, %d]\n",
		embeddings.Dim(0), embeddings.Dim(1), embeddings.Dim(2), embeddings.Dim(3))

	// Permute: [patchesW, patchesH, hiddenSize, batchSize] -> [hiddenSize, patchesW, patchesH, batchSize]
	embeddings = embeddings.Permute(ctx, 2, 0, 1, 3).Contiguous(ctx)

	// Print after permute
	fmt.Printf("After permute: [%d, %d, %d, %d]\n",
		embeddings.Dim(0), embeddings.Dim(1), embeddings.Dim(2), embeddings.Dim(3))

	// Reshape to combine patches in 2×2 groups
	// We need to make sure patchesW and patchesH are both even
	if patchesW%2 != 0 || patchesH%2 != 0 {
		panic(fmt.Sprintf("Patch dimensions must be even: patchesW=%d, patchesH=%d", patchesW, patchesH))
	}

	// Reshape: [hiddenSize, patchesW, patchesH, batchSize] -> [hiddenSize*2, patchesW/2, patchesH, batchSize]
	embeddings = embeddings.Reshape(ctx, hiddenSize*2, patchesW/2, patchesH, batchSize)
	fmt.Printf("After first reshape: [%d, %d, %d, %d]\n",
		embeddings.Dim(0), embeddings.Dim(1), embeddings.Dim(2), embeddings.Dim(3))

	// Reshape: [hiddenSize*2, patchesW/2, patchesH, batchSize] -> [hiddenSize*2, patchesW/2, 2, patchesH/2*batchSize]
	embeddings = embeddings.Reshape(ctx, hiddenSize*2, patchesW/2, 2, patchesH/2*batchSize)
	fmt.Printf("After second reshape: [%d, %d, %d, %d]\n",
		embeddings.Dim(0), embeddings.Dim(1), embeddings.Dim(2), embeddings.Dim(3))

	// Permute: [hiddenSize*2, patchesW/2, 2, patchesH/2*batchSize] -> [hiddenSize*2, 2, patchesW/2, patchesH/2*batchSize]
	embeddings = embeddings.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	fmt.Printf("After second permute: [%d, %d, %d, %d]\n",
		embeddings.Dim(0), embeddings.Dim(1), embeddings.Dim(2), embeddings.Dim(3))

	// Final reshape: [hiddenSize*2, 2, patchesW/2, patchesH/2*batchSize] -> [hiddenSize, patchesW*patchesH/4, batchSize]
	// This is an important step - it merges adjacent patches, reducing the sequence length by a factor of 4
	embeddings = embeddings.Reshape(ctx, hiddenSize, patchesW*patchesH, batchSize)
	fmt.Printf("Final embeddings shape: [%d, %d, %d]\n",
		embeddings.Dim(0), embeddings.Dim(1), embeddings.Dim(2))

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
	Layers         []VisionEncoderLayer `gguf:"blk"`
	PostLayerNorm  *nn.LayerNorm        `gguf:"post_ln"`
	PatchMerger    *VisionPatchMerger   `gguf:"patch_merger"`

	*VisionModelOptions
}

// Forward computes the vision model for an input tensor
func (m *VisionModel) Forward(ctx ml.Context, pixelValues ml.Tensor) ml.Tensor {
	// Extract patch embeddings
	hiddenStates := m.PatchEmbedding.Forward(ctx, pixelValues, m.patchSize)

	// Calculate position IDs for 2D RoPE
	numPatchesH := pixelValues.Dim(0) / m.patchSize
	numPatchesW := pixelValues.Dim(1) / m.patchSize
	numPatches := numPatchesH * numPatchesW

	// Create position IDs - for Qwen2VL mRoPE we need 4 values per position
	// The format needed is specified in the C++ code as "mrope expecting 4 position ids per token"
	positions := make([]int32, numPatches*4)

	for h := 0; h < numPatchesH; h++ {
		for w := 0; w < numPatchesW; w++ {
			idx := h*numPatchesW + w
			// For each position, store both h and w coordinates twice
			// This matches the pattern seen in the C++ implementation
			positions[idx*4] = int32(h)   // y coordinate
			positions[idx*4+1] = int32(w) // x coordinate
			positions[idx*4+2] = int32(h) // y coordinate (repeated)
			positions[idx*4+3] = int32(w) // x coordinate (repeated)
		}
	}

	// Create the position IDs tensor with correct dimensions
	positionIDs, err := ctx.Input().FromIntSlice(positions, numPatches*4)
	if err != nil {
		panic(err)
	}

	fmt.Printf("DEBUG: Created positionIDs with length: %d for %d patches\n",
		positionIDs.Dim(0), numPatches)

	// Apply encoder layers
	for i, layer := range m.Layers {
		fmt.Printf("Processing Layer %d\n", i)
		hiddenStates = layer.Forward(ctx, hiddenStates, positionIDs, m.VisionModelOptions)
	}

	hiddenStates = m.PostLayerNorm.Forward(ctx, hiddenStates, m.eps)
	return hiddenStates
}

// newVisionModel creates a new instance of the Qwen vision model
func newVisionModel(c fs.Config) *VisionModel {
	patchSize := int(c.Uint("vision.patch_size", 14))
	hiddenSize := int(c.Uint("vision.embedding_length", 1280))
	ropeTheta := c.Float("vision.rope_theta", 10000.0)             // not set
	outHiddenSize := int(c.Uint("vision.out_embedding_length", 0)) // not set
	numHeads := int(c.Uint("vision.attention.head_count", 16))

	return &VisionModel{
		Layers: make([]VisionEncoderLayer, c.Uint("vision.block_count", 24)),
		VisionModelOptions: &VisionModelOptions{
			hiddenSize:       hiddenSize,
			numHeads:         numHeads,
			headDim:          hiddenSize / numHeads,
			intermediateSize: int(c.Uint("vision.feed_forward_length", 0)),
			imageSize:        int(c.Uint("vision.image_size", 560)),
			patchSize:        patchSize,
			numChannels:      int(c.Uint("vision.num_channels", 3)), // not set
			eps:              c.Float("vision.attention.layer_norm_epsilon", 1e-6),
			ropeTheta:        ropeTheta,
			outHiddenSize:    outHiddenSize,
		},
	}
}
