package convert

import (
	"bytes"
	"fmt"
	"io"
	"log/slog"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

// Matches the structure in config.json for Qwen2.5-VL
type qwen25vlModel struct {
	ModelParameters
	HiddenSize            uint32  `json:"hidden_size"`
	IntermediateSize      uint32  `json:"intermediate_size"`
	MaxPositionEmbeddings uint32  `json:"max_position_embeddings"`
	NumAttentionHeads     uint32  `json:"num_attention_heads"`
	HiddenLayers          uint32  `json:"num_hidden_layers"`
	RopeTheta             float32 `json:"rope_theta"`
	NumKeyValueHeads      uint32  `json:"num_key_value_heads"`
	RMSNormEPS            float32 `json:"rms_norm_eps"`
	// TieWordEmbeddings is often present, even if not used directly here
	TieWordEmbeddings bool `json:"tie_word_embeddings"`

	// Vision specific parameters from its config (nested under vision_config)
	VisionConfig struct {
		HiddenSize        uint32  `json:"hidden_size"`
		ImageSize         uint32  `json:"image_size"`
		IntermediateSize  uint32  `json:"intermediate_size"`
		LayerNormEps      float32 `json:"layer_norm_eps"`
		NumAttentionHeads uint32  `json:"num_attention_heads"`
		NumChannels       uint32  `json:"num_channels"`
		NumHiddenLayers   uint32  `json:"num_hidden_layers"`
		PatchSize         uint32  `json:"patch_size"`
		// May include others like projection_dim, use_cls_token etc.
	} `json:"vision_config"`
	// Might have top-level vision params too, check config.json
	// Example: ProjectorHiddenAct string `json:"projector_hidden_act"`
}

// Compile-time check to ensure qwen25vlModel implements ModelConverter
var _ ModelConverter = (*qwen25vlModel)(nil)

// KV provides the metadata key-value pairs for the GGUF header.
func (q *qwen25vlModel) KV(t *Tokenizer) ggml.KV {
	kv := q.ModelParameters.KV(t) // Assuming ModelParameters provides defaults like general.name etc.
	kv["general.architecture"] = "qwen25vl"
	// Text model parameters
	kv["qwen25vl.block_count"] = q.HiddenLayers
	kv["qwen25vl.context_length"] = q.MaxPositionEmbeddings
	kv["qwen25vl.embedding_length"] = q.HiddenSize
	kv["qwen25vl.feed_forward_length"] = q.IntermediateSize
	kv["qwen25vl.attention.head_count"] = q.NumAttentionHeads
	kv["qwen25vl.attention.head_count_kv"] = q.NumKeyValueHeads
	kv["qwen25vl.rope.freq_base"] = q.RopeTheta
	kv["qwen25vl.attention.layer_norm_rms_epsilon"] = q.RMSNormEPS

	// Vision model parameters (prefix with 'vision.')
	kv["qwen25vl.vision.hidden_size"] = q.VisionConfig.HiddenSize
	kv["qwen25vl.vision.image_size"] = q.VisionConfig.ImageSize
	kv["qwen25vl.vision.intermediate_size"] = q.VisionConfig.IntermediateSize
	kv["qwen25vl.vision.layer_norm_eps"] = q.VisionConfig.LayerNormEps
	kv["qwen25vl.vision.attention.head_count"] = q.VisionConfig.NumAttentionHeads
	kv["qwen25vl.vision.num_channels"] = q.VisionConfig.NumChannels // Usually 3
	kv["qwen25vl.vision.patch_size"] = q.VisionConfig.PatchSize
	kv["qwen25vl.vision.block_count"] = q.VisionConfig.NumHiddenLayers

	// Add other relevant vision parameters if they exist in config.json
	// e.g., kv["qwen25vl.vision.projection_dim"] = q.VisionConfig.ProjectionDim

	// Explicitly DO NOT set general.alignment here, rely on default handling
	// if the tensor data sizes written by WriteTo are correct.

	return kv
}

// Tensors processes the list of loaded tensors, handling specific cases like splitting.
func (q *qwen25vlModel) Tensors(ts []Tensor) []ggml.Tensor {
	var out []ggml.Tensor

	for _, t := range ts {
		// Check if this tensor needs special handling
		if strings.HasSuffix(t.Name(), "patch_embed.proj.weight") {
			slog.Info("Splitting tensor", "name", t.Name())
			var buf bytes.Buffer
			// Write the original tensor data to a buffer
			if _, err := t.WriteTo(&buf); err != nil {
				panic(fmt.Sprintf("failed to read tensor %s for splitting: %v", t.Name(), err))

			}
			// Perform the split
			newTensors := splitPatchEmbed(buf, t.Kind(), t.Shape())
			out = append(out, newTensors...)
			slog.Info("Finished splitting tensor", "name", t.Name(), "output_tensors", len(newTensors))
		} else {
			// Pass through other tensors directly
			out = append(out, ggml.Tensor{
				Name:     t.Name(), // Name will be transformed by Replacements later
				Kind:     t.Kind(),
				Shape:    t.Shape(),
				WriterTo: t, // Pass the original tensor object
			})
		}
	}

	return out
}

// Replacements provides the rules to rename tensors from the source format to the GGUF convention.
func (q *qwen25vlModel) Replacements() []string {
	// Ensure these cover all transformations needed for both text and vision parts.
	// Use the list from your original code, adding vision specific ones if missing.
	return []string{
		// Text model replacements
		"lm_head", "output",
		"model.embed_tokens", "token_embd",
		"model.layers", "blk",
		"input_layernorm", "attn_norm",
		"self_attn.k_proj", "attn_k",
		"self_attn.v_proj", "attn_v",
		"self_attn.q_proj", "attn_q",
		"self_attn.o_proj", "attn_output",
		"mlp.down_proj", "ffn_down",
		"mlp.gate_proj", "ffn_gate",
		"mlp.up_proj", "ffn_up",
		"post_attention_layernorm", "ffn_norm", // Check if Qwen2.5 uses post_attention_layernorm or pre/post FFN norm
		"model.norm", "output_norm",

		// Vision model replacements (adjust based on actual HF names)
		"visual.patch_embed.proj.weight", "v.patch_embed.proj.weight", // Base name for the split target
		"visual.patch_embed.norm", "v.patch_embed.norm", // If norm exists
		"visual.embed_tokens", "v.cls_token", // If CLS token exists
		"visual.blocks", "v.blk",
		"visual.norm", "v.post_norm", // Or v.norm depending on architecture
		// Vision layer specific replacements (these should already be covered by text ones if names are consistent)
		// e.g., within v.blk.*:
		// "layer_norm1", "attn_norm",
		// "attn.qkv", ... handle QKV split if needed ...
		// "attn.proj", "attn_output",
		// "layer_norm2", "ffn_norm",
		// "mlp.fc1", "ffn_gate", // Or combine ffn_gate/ffn_up if HF uses different names
		// "mlp.fc2", "ffn_down",

		// Multi-modal projector replacements (if applicable)
		// "multi_modal_projector.linear_1", "mm_proj.0", // Example naming
		// "multi_modal_projector.linear_2", "mm_proj.2", // Example naming
	}
}

func splitPatchEmbed(buf bytes.Buffer, kind uint32, shape []uint64) []ggml.Tensor {
	// Ensure shape is as expected (5D with third dimension of 2)
	if len(shape) != 5 || shape[2] != 2 {
		panic(fmt.Sprintf("splitPatchEmbed: expected 5D tensor with shape[2]==2, got shape %v", shape))
	}

	// Calculate target shape (remove the third dimension)
	targetShape := append(shape[:2], shape[3:]...)

	// Calculate tensor sizes
	elementSize := uint32(2) // F16 = 2 bytes per element
	if kind == tensorKindF32 {
		elementSize = 4 // F32 = 4 bytes per element
	}

	// Calculate number of elements in each slice
	elementsPerSlice := uint64(1)
	for _, dim := range targetShape {
		elementsPerSlice *= dim
	}

	// Calculate total elements in original tensor
	totalElements := elementsPerSlice * shape[2] // should be 2x the slice size

	// Read all data from buffer
	data := make([]byte, totalElements*uint64(elementSize))
	if _, err := buf.Read(data); err != nil {
		panic(fmt.Sprintf("splitPatchEmbed: failed to read data: %v", err))
	}

	// Create the first tensor (slice 0)
	slice0Data := make([]byte, elementsPerSlice*uint64(elementSize))
	for i := uint64(0); i < elementsPerSlice; i++ {
		offset := i * uint64(elementSize)
		copy(slice0Data[offset:offset+uint64(elementSize)],
			data[offset:offset+uint64(elementSize)])
	}

	// Create the second tensor (slice 1)
	slice1Data := make([]byte, elementsPerSlice*uint64(elementSize))
	for i := uint64(0); i < elementsPerSlice; i++ {
		srcOffset := (elementsPerSlice + i) * uint64(elementSize)
		dstOffset := i * uint64(elementSize)
		copy(slice1Data[dstOffset:dstOffset+uint64(elementSize)],
			data[srcOffset:srcOffset+uint64(elementSize)])
	}

	// Return the two tensors with names matching the Python implementation
	return []ggml.Tensor{
		{
			Name:     "v.patch_embd.weight",
			Kind:     kind,
			Shape:    targetShape,
			WriterTo: &bytesWriterTo{data: slice0Data},
		},
		{
			Name:     "v.patch_embd.weight.1",
			Kind:     kind,
			Shape:    targetShape,
			WriterTo: &bytesWriterTo{data: slice1Data},
		},
	}
}

// Helper type for writing bytes
type bytesWriterTo struct {
	data []byte
}

func (b *bytesWriterTo) WriteTo(w io.Writer) (int64, error) {
	n, err := w.Write(b.data)
	return int64(n), err
}
