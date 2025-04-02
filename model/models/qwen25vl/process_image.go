package qwen25vl

import (
	"image"
	"math"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/imageproc"
)

type ImageProcessor struct {
	imageSize, numChannels       int
	factor, minPixels, maxPixels int
}

func newImageProcessor(c ml.Config) ImageProcessor {
	return ImageProcessor{
		imageSize:   int(c.Uint("vision.image_size")),
		numChannels: 3, // RGB channels
		factor:      28,
		minPixels:   56 * 56,
		maxPixels:   14 * 14 * 4 * 1280,
	}
}

// smartResize calculates the size of the image to resize to based on the
// factor, minPixels, and maxPixels.
func (p *ImageProcessor) smartResize(size image.Point) image.Point {
	// 1. Both dimensions of size are divisible by factor
	// 2. The area of the image is between minPixels and maxPixels
	// 3. The aspect ratio of the image is as close to 1:1 as possible

	if size.Y < p.factor || size.X < p.factor {
		panic("image is too small to resize")
	} else if max(size.X, size.Y)/min(size.X, size.Y) > 200 {
		panic("aspect ratio must be less than 200:1")
	}

	f := float64(p.factor)
	width := float64(size.X)
	height := float64(size.Y)

	xBar := math.Round(width/f) * f
	yBar := math.Round(height/f) * f

	if xBar*yBar > float64(p.maxPixels) {
		beta := math.Sqrt(height * width / float64(p.maxPixels))
		xBar = math.Floor(width/beta/f) * f
		yBar = math.Floor(height/beta/f) * f
	} else if xBar*yBar < float64(p.minPixels) {
		beta := math.Sqrt(float64(p.minPixels) / (height * width))
		xBar = math.Ceil(width*beta/f) * f
		yBar = math.Ceil(height*beta/f) * f
	}

	return image.Point{int(xBar), int(yBar)}
}

func (p *ImageProcessor) ProcessImage(img image.Image) ([]float32, error) {
	// Detect PNG by checking for alpha channel
	isPNG := false
	if _, _, _, a := img.At(0, 0).RGBA(); a < 0xffff {
		isPNG = true
	}

	size := p.smartResize(img.Bounds().Max)

	// Composite PNG images to handle transparency
	if isPNG {
		img = imageproc.Composite(img)
	}

	// Resize the image
	img = imageproc.Resize(img, size, imageproc.ResizeBilinear)

	// Use CLIP normalization values
	mean := [3]float32{0.48145466, 0.4578275, 0.40821073} // CLIP mean values
	std := [3]float32{0.26862954, 0.26130258, 0.27577711} // CLIP std values

	// Normalize and get the data
	data := imageproc.Normalize(img, mean, std, true, true)

	return data, nil
}
